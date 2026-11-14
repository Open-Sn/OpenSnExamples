#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate the multigroup cross sections of the ICF target chamber example.

OpenMC tallies the cross sections of each material in the chamber itself, the model of
openmc_reference.py with its 14.1 MeV point source, so each material is weighted by its own
spectrum in the problem: the wall by the flux just outside the cavity, the shield by the flux
slowing down and being absorbed in borated concrete, the air by what leaks from the shield and
streams through the ports. From the same run the script writes three libraries:

- materials_64g_p3.h5: the 64 groups of the Tokamak and ICF hohlraum examples, with P3
  scattering and no transport correction;
- materials_4g_p0.h5: 4 groups whose edges are 4 of the 65 edges of the 64-group structure,
  so that the 64-group results of OpenMC collapse exactly onto them, with P0 scattering. Group
  0 is the source group of the 64-group structure, 13.84 to 14.191 MeV, so its total cross
  section is that at 14.1 MeV and the uncollided flux keeps its meaning;
- materials_4g_p0tc.h5: the same with the extended transport correction: in each group g the
  within-group P1 moment sigma_1,g->g, when positive, is subtracted from the total cross
  section and from the within-group P0 moment. The transfers between groups and the
  absorption are unchanged. This changes the total cross section of group 0, so the
  "uncollided" flux of this library is attenuated by a smaller cross section.

The 4-group libraries are condensed from the 64-group tallies with the flux of each material
(openmc.mgxs.Library.get_condensed_library).

Each library holds, for each material, the total and absorption cross sections and the
scattering matrices. The scattering matrices are neutron-production matrices, so they include
the extra neutrons of (n,2n) reactions; they use OpenMC's consistent estimator, which makes the
P0 scattering cross section equal to the total minus the absorption cross section. The
(n,2n), (n,gamma) and (n,alpha) cross sections are added as the datasets n2n, ngamma and
nalpha. A "void" entry with zero cross sections is used for the cavity and the ports.

The script also writes provenance/materials.json, with the statistics of the libraries, the
first group that receives upscattering (the groupsets of chamber.py start a new groupset
there), and the SHA-256 of every input and output, including the nuclear data files.

Set OPENMC_CROSS_SECTIONS to the cross_sections.xml file of the ENDF/B-VIII.0 library, then:

    python3 generate_xs.py --particles 10000000 --batches 50 --threads 24

A run that is interrupted restarts from its last statepoint. With --edges the 4-group edges
can be changed and the libraries rewritten from the finished run without running OpenMC again.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import openmc
import openmc.mgxs

import chamber_geometry as geometry
import openmc_reference as reference

LEGENDRE_ORDER = 3
MGXS_TYPES = ["total", "absorption", "consistent scatter matrix", "consistent nu-scatter matrix"]
REACTIONS = {"(n,2n)": "n2n", "(n,gamma)": "ngamma", "(n,a)": "nalpha"}
# Default edges (eV) of the 4-group library, a subset of the 64-group edges: the source group,
# the fast group down to 0.117 MeV, the slowing-down group down to 167 eV, and a low group that
# holds every upscattering (in the chamber spectra, the highest group that receives
# upscattering in the 64-group library is the one from 61 to 167 eV, in the concrete).
COARSE_EDGES_EV = [0.0, 167.02, 116790.0, 13840000.0, 14191000.0]
CHECKPOINT_BATCHES = 10
RUN_DIR = Path("xs_generation")
LIBRARIES = {"64g_p3": "materials_64g_p3.h5", "4g_p0": "materials_4g_p0.h5",
             "4g_p0tc": "materials_4g_p0tc.h5"}
PROVENANCE = Path("provenance") / "materials.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(particles, batches, seed):
    """Return the chamber model with the tallies of the multigroup library, and the library."""
    materials = reference.build_materials()
    geometry_model, _ = reference.build_geometry(materials)
    settings = reference.settings(particles, batches, seed)
    library = openmc.mgxs.Library(geometry_model)
    library.energy_groups = openmc.mgxs.EnergyGroups(reference.GROUP_EDGES_EV)
    library.mgxs_types = MGXS_TYPES + list(REACTIONS)
    library.correction = None
    library.scatter_format = "legendre"
    library.legendre_order = LEGENDRE_ORDER
    library.by_nuclide = False
    library.domain_type = "material"
    library.domains = list(materials.values())
    library.build_library()
    tallies = openmc.Tallies()
    library.add_to_tallies(tallies, merge=True)
    model = openmc.Model(geometry_model, openmc.Materials(materials.values()), settings,
                         tallies)
    return model, library


def run(model, threads, processes=1):
    """Run the model, or restart it from its last statepoint; return the final statepoint."""
    batches = model.settings.batches
    final = RUN_DIR / f"statepoint.{batches}.h5"
    if final.exists():
        return final, None
    RUN_DIR.mkdir(exist_ok=True)
    model.settings.statepoint = {"batches": list(range(CHECKPOINT_BATCHES, batches + 1,
                                                       CHECKPOINT_BATCHES)) + [batches]}
    model.export_to_xml(directory=RUN_DIR)
    checkpoints = sorted(RUN_DIR.glob("statepoint.*.h5"), key=lambda p: int(p.suffixes[0][1:]))
    restart = checkpoints[-1].name if checkpoints else None
    start = time.perf_counter()
    mpi_args = ["mpiexec", "-n", str(processes)] if processes > 1 else None
    openmc.run(cwd=RUN_DIR, threads=threads, restart_file=restart, mpi_args=mpi_args)
    return final, time.perf_counter() - start


def void_data(groups, order):
    """Return zero cross sections for the cavity and the ports."""
    void = openmc.XSdata("void", groups, temperatures=[reference.TEMPERATURE])
    void.order = order
    zeros = np.zeros(groups.num_groups)
    void.set_total(zeros, reference.TEMPERATURE)
    void.set_absorption(zeros, reference.TEMPERATURE)
    void.set_scatter_matrix(np.zeros((groups.num_groups, groups.num_groups, order + 1)),
                            reference.TEMPERATURE)
    return void


def export(library, output):
    """Write a library with its reaction datasets and a void entry."""
    names = [material.name for material in library.domains]
    data = library.create_mg_library(xs_type="macro", xsdata_names=names,
                                     temperature=reference.TEMPERATURE)
    data.add_xsdata(void_data(library.energy_groups, library.legendre_order))
    data.export_to_hdf5(output)
    temperature = f"{round(reference.TEMPERATURE)}K"
    groups = library.energy_groups.num_groups
    with h5py.File(output, "r+") as stream:
        for material in library.domains:
            for reaction, dataset in REACTIONS.items():
                values = library.get_mgxs(material, reaction).get_xs(xs_type="macro")
                stream[f"{material.name}/{temperature}"].create_dataset(dataset, data=values)
        for dataset in REACTIONS.values():
            stream[f"void/{temperature}"].create_dataset(dataset, data=np.zeros(groups))
    return data


def rewrite_moments(source, output, order, correct):
    """Copy a library, keeping the scattering moments 0 to order.

    With correct=True, the within-group moment order + 1 of each group, when positive, is
    subtracted from the total cross section and from the within-group moments 0 to order (the
    extended transport correction). Return the largest correction relative to the total cross
    section, by material.
    """
    shutil.copyfile(source, output)
    largest = {}
    with h5py.File(output, "r+") as library:
        for name, material in library.items():
            if not isinstance(material, h5py.Group):
                continue
            old_order = int(material.attrs["order"])
            material.attrs["order"] = order
            for temperature in material.values():
                if "scatter_data" not in temperature:
                    continue
                data = temperature["scatter_data"]
                total = temperature["total"][()]
                g_min, g_max = data["g_min"][()], data["g_max"][()]
                pairs = [(gi, go) for gi in range(len(total))
                         for go in range(g_min[gi] - 1, g_max[gi])]
                moments = data["scatter_matrix"][()].reshape(len(pairs), old_order + 1)
                correction = np.zeros(len(total))
                if correct:
                    for k, (gi, go) in enumerate(pairs):
                        if gi == go:
                            correction[gi] = max(moments[k, order + 1], 0.0)
                    for k, (gi, go) in enumerate(pairs):
                        if gi == go:
                            moments[k, :order + 1] -= correction[gi]
                temperature["total"][...] = total - correction
                del data["scatter_matrix"]
                data.create_dataset("scatter_matrix", data=moments[:, :order + 1].ravel())
                largest[name] = float(np.max(correction / np.where(total > 0, total, 1.0)))
    return largest


def statistics(library, data):
    """Return the statistical quality and the properties of each material of a library."""
    result = {}
    for material in library.domains:
        relative = library.get_mgxs(material, "total").get_xs(value="rel_err")
        flux = library.get_mgxs(material, "total").tallies["flux"].mean.ravel()[::-1]
        xs = data.get_by_name(material.name)
        sigma_t, sigma_a = xs.total[0], xs.absorption[0]
        production = xs.scatter_matrix[0][:, :, 0]
        multiplicity = xs.multiplicity_matrix[0]
        scatter = np.divide(production, multiplicity, out=np.zeros_like(production),
                            where=multiplicity > 0)
        closure = np.abs(sigma_t - sigma_a - scatter.sum(axis=1)) / sigma_t
        # Upscattering: from group gi to a group go < gi, of higher energy.
        upscatter = [(gi, go) for gi in range(len(sigma_t)) for go in range(gi)
                     if production[gi, go] > 0.0]
        result[material.name] = {
            "total_xs": sigma_t.tolist(), "absorption_xs": sigma_a.tolist(),
            "flux_spectrum": (flux / flux.sum()).tolist(),
            "total_relative_std_dev_max": float(np.max(relative)),
            "total_relative_std_dev_median": float(np.median(relative)),
            "groups_with_total_relative_std_dev_above_1_percent": int(np.sum(relative > 0.01)),
            "collision_closure_max": float(np.max(closure)),
            "first_upscatter_group": min((go for _, go in upscatter), default=None),
        }
    return result


def nuclear_data_record(materials_xml):
    """Return the SHA-256 of every nuclear data file that the materials use."""
    xml = Path(os.environ["OPENMC_CROSS_SECTIONS"]).resolve()
    root = ET.parse(materials_xml).getroot()
    needed = {e.attrib["name"] for e in root.iter("nuclide")}
    needed |= {e.attrib["name"] for e in root.iter("sab")}
    library = ET.parse(xml).getroot()
    directory = library.find("directory")
    base = xml.parent / directory.text if directory is not None else xml.parent
    files = {}
    for entry in library.findall("library"):
        if needed.intersection(entry.attrib["materials"].split()):
            files[entry.attrib["path"]] = sha256(base / entry.attrib["path"])
    return {"library": "ENDF/B-VIII.0 HDF5 library distributed by the OpenMC project",
            "files_sha256": dict(sorted(files.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--particles", type=int, default=10000000, help="Histories per batch")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=None, help="OpenMP threads")
    parser.add_argument("--processes", type=int, default=1,
                        help="MPI processes, for an OpenMC built with MPI (mpiexec -n)")
    parser.add_argument("--edges", type=float, nargs="+", default=COARSE_EDGES_EV,
                        help="Edges (eV) of the few-group library, a subset of the 64 groups")
    args = parser.parse_args()
    edges = sorted(args.edges)
    missing = [e for e in edges if not any(np.isclose(e, reference.GROUP_EDGES_EV, rtol=1e-9,
                                                      atol=0.0) | (e == 0.0))]
    if missing:
        raise ValueError(f"few-group edges {missing} are not edges of the 64 groups")

    model, library = build_model(args.particles, args.batches, args.seed)
    statepoint, seconds = run(model, args.threads, args.processes)
    with openmc.StatePoint(statepoint) as sp:
        library.load_from_statepoint(sp)

    fine = export(library, LIBRARIES["64g_p3"])
    coarse = library.get_condensed_library(openmc.mgxs.EnergyGroups(edges))
    coarse_p3 = RUN_DIR / "materials_4g_p3.h5"
    coarse_data = export(coarse, coarse_p3)
    corrections = {
        "4g_p0": rewrite_moments(coarse_p3, LIBRARIES["4g_p0"], 0, correct=False),
        "4g_p0tc": rewrite_moments(coarse_p3, LIBRARIES["4g_p0tc"], 0, correct=True),
    }

    here = Path(__file__).resolve().parent
    version = subprocess.run(["openmc", "--version"], capture_output=True, text=True)
    record = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "libraries_sha256": {name: sha256(path) for name, path in LIBRARIES.items()},
        "generator_sha256": {name: sha256(here / name) for name in
                             ("generate_xs.py", "openmc_reference.py", "chamber_geometry.py")},
        "openmc_python_version": openmc.__version__,
        "openmc_executable_version": version.stdout.strip(),
        "nuclear_data": nuclear_data_record(RUN_DIR / "materials.xml"),
        "settings": {
            "mgxs_types": MGXS_TYPES, "reactions": REACTIONS,
            "legendre_order": LEGENDRE_ORDER, "transport_correction": None,
            "group_edges_eV": reference.GROUP_EDGES_EV, "few_group_edges_eV": edges,
            "few_group_transport_correction": {
                "file": LIBRARIES["4g_p0tc"],
                "correction": "extended transport correction with the within-group P1 "
                              "moment, negative values set to zero",
                "largest_correction_relative_to_total": corrections["4g_p0tc"]},
            "temperature_K": reference.TEMPERATURE,
            "source_energy_eV": reference.SOURCE_ENERGY,
            "particles_per_batch": args.particles, "batches": args.batches,
            "seed": args.seed, "threads": args.threads, "openmc_seconds": seconds},
        "inputs_sha256": {name: sha256(RUN_DIR / name) for name in
                          ("geometry.xml", "materials.xml", "settings.xml", "tallies.xml")},
        "statepoint_sha256": sha256(statepoint),
        "materials_64g": statistics(library, fine),
        "materials_4g": statistics(coarse, coarse_data),
    }
    PROVENANCE.parent.mkdir(exist_ok=True)
    PROVENANCE.write_text(json.dumps(record, indent=2) + "\n")
    for name, values in record["materials_64g"].items():
        print(f"{name:17s} median total rel. std. dev. "
              f"{values['total_relative_std_dev_median']:.2e}  max "
              f"{values['total_relative_std_dev_max']:.2e}  closure "
              f"{values['collision_closure_max']:.1e}  first upscatter group "
              f"{values['first_upscatter_group']}")
    print(f"Wrote {', '.join(LIBRARIES.values())} and {PROVENANCE}")


if __name__ == "__main__":
    main()
