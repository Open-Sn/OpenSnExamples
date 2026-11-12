#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate the 64-group cross sections of the hohlraum example: materials_64g_p5.h5, with P5
scattering, and materials_64g_p3tc.h5, the transport-corrected P3 library used by
hohlraum.py.

The target is optically thin: only about 0.3% of the 14.1 MeV source neutrons collide
before they escape, and almost none of them slow down below 100 keV. A library weighted by
the spectrum of the target itself would leave most of the lower groups empty. OpenMC
therefore weights each material in its own infinite medium, a reflecting cube filled with
that material, with a uniform isotropic source. 99% of the source is at 14.1 MeV, and 1%
is spread over the whole group structure with a 1/E spectrum, so that every group is
sampled even in gold, which captures most neutrons in its resonances before they reach
thermal energies.

For each material the library holds the total cross section, the physical absorption cross
section, and P0 to P5 scattering matrices. The scattering matrices are neutron-production
matrices, so they include the extra neutrons from (n,2n) and other multiplying reactions.
They use OpenMC's consistent estimator, which makes the P0 scattering cross section equal
to the total minus the absorption cross section. The (n,2n), (n,gamma), and (n,alpha) cross
sections are added as the datasets n2n, ngamma, and nalpha, which hohlraum.py uses to
compute reaction rates.

Elastic scattering of 14 MeV neutrons on gold is so strongly forward peaked that its P5
moment is still half of its P0 moment, and no affordable Legendre order represents it. The
transport-corrected library treats the forward peak as no collision, with the extended
transport correction: in each group g, the within-group P4 moment sigma_4,g->g is
subtracted from the total cross section and from the within-group P0 to P3 moments, and
the expansion is truncated at P3. Absorption, the transfers between groups, and the
reaction cross sections are unchanged. Negative within-group P4 moments, at most 1.1% of
the total cross section, are set to zero, so the correction never increases a cross
section.

OpenMC needs continuous-energy ENDF/B-VIII.0 data for the nuclides of the materials. Set
OPENMC_CROSS_SECTIONS to the cross_sections.xml file of that library, then run, for example:

    python3 generate_xs.py --processes 8

The script writes the OpenMC inputs and statepoints to xs_generation/, the libraries to
materials_64g_p5.h5 and materials_64g_p3tc.h5, and a record of every input and output to
provenance/materials_64g_p5.json.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import openmc
import openmc.mgxs

import openmc_reference as reference

# The 64-group structure in eV, from 0 to 14.191 MeV, also used by the
# Tokamak_Fusion_Engineering example. It is a subset of the VITAMIN-J 175-group boundaries
# with four thermal boundaries added below 0.1 eV. Group 0 (13.84 to 14.191 MeV) holds the
# 14.1 MeV source.
GROUP_EDGES_EV = [
    0.0, 0.005, 0.01, 0.0253, 0.05, 0.1, 0.53158, 1.1253, 2.3824, 5.0435, 10.677, 29.023,
    61.442, 167.02, 353.58, 961.12, 2034.7, 2746.5, 3707.4, 9118.8, 15034.0, 23579.0,
    24176.0, 24788.0, 28501.0, 46309.0, 67380.0, 86517.0, 116790.0, 142640.0, 174220.0,
    212800.0, 247240.0, 297210.0, 333730.0, 387740.0, 450490.0, 550230.0, 608100.0,
    672060.0, 742740.0, 820850.0, 907180.0, 1002600.0, 1164800.0, 1353400.0, 1572400.0,
    1826800.0, 2122500.0, 2345700.0, 2466000.0, 2865000.0, 3166400.0, 4065700.0,
    4965800.0, 5769500.0, 6592400.0, 7408200.0, 8607100.0, 10000000.0, 11052000.0,
    12214000.0, 12840000.0, 13840000.0, 14191000.0]
LEGENDRE_ORDER = 5
MGXS_TYPES = ["total", "absorption", "consistent scatter matrix", "consistent nu-scatter matrix"]
# Reaction cross sections added to the library, with the dataset names used by OpenSn.
REACTIONS = {"(n,2n)": "n2n", "(n,gamma)": "ngamma", "(n,a)": "nalpha"}

# The infinite medium is a reflecting cube. Its size does not affect the spectrum, but a
# cube much larger than the mean free path in helium avoids needless boundary crossings.
CUBE_SIDE = 1.0e5  # cm
SMOOTH_FRACTION, SMOOTH_MIN_ENERGY = 0.01, 1.0e-5  # eV

CHECKPOINT_BATCHES = 10

OUTPUT = "materials_64g_p5.h5"
CORRECTED_OUTPUT, CORRECTED_ORDER = "materials_64g_p3tc.h5", 3
PROVENANCE = Path("provenance") / "materials_64g_p5.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_infinite_medium(material, particles, batches, seed):
    """Return the infinite-medium model and the MGXS library of one material."""
    half = CUBE_SIDE / 2
    cube = openmc.model.RectangularParallelepiped(-half, half, -half, half, -half, half,
                                                  boundary_type="reflective")
    geometry = openmc.Geometry([openmc.Cell(fill=material, region=-cube)])
    space = openmc.stats.Box((-half, -half, -half), (half, half, half))
    top = GROUP_EDGES_EV[-1]
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles, settings.batches, settings.seed = particles, batches, seed
    settings.temperature = {"method": "nearest", "tolerance": 10.0}
    settings.output = {"tallies": False}
    settings.source = [
        openmc.IndependentSource(space=space, angle=openmc.stats.Isotropic(),
                                 energy=openmc.stats.Discrete([reference.SOURCE_ENERGY], [1.0]),
                                 strength=1.0 - SMOOTH_FRACTION),
        openmc.IndependentSource(space=space, angle=openmc.stats.Isotropic(),
                                 energy=openmc.stats.Tabular(
                                     [SMOOTH_MIN_ENERGY, top],
                                     [1.0 / SMOOTH_MIN_ENERGY, 1.0 / top],
                                     interpolation="log-log"),
                                 strength=SMOOTH_FRACTION)]

    library = openmc.mgxs.Library(geometry)
    library.energy_groups = openmc.mgxs.EnergyGroups(GROUP_EDGES_EV)
    library.mgxs_types = MGXS_TYPES + list(REACTIONS)
    library.correction = None
    library.scatter_format = "legendre"
    library.legendre_order = LEGENDRE_ORDER
    library.by_nuclide = False
    library.domain_type = "material"
    library.domains = [material]
    library.build_library()
    tallies = openmc.Tallies()
    library.add_to_tallies(tallies, merge=True)
    model = openmc.Model(geometry, openmc.Materials([material]), settings, tallies)
    return model, library


def run(model, run_dir, mpi_args, threads):
    """Run a model and return the path of its final statepoint.

    OpenMC writes a statepoint every CHECKPOINT_BATCHES batches. If an earlier run of this
    script was interrupted, the run restarts from its latest statepoint. A completed run is
    not repeated.
    """
    batches = model.settings.batches
    final = run_dir / f"statepoint.{batches}.h5"
    if final.exists():
        return final
    model.settings.statepoint = {"batches": list(range(CHECKPOINT_BATCHES, batches + 1,
                                                       CHECKPOINT_BATCHES)) + [batches]}
    model.export_to_xml(directory=run_dir)
    checkpoints = sorted(run_dir.glob("statepoint.*.h5"), key=lambda p: int(p.suffixes[0][1:]))
    restart = checkpoints[-1].name if checkpoints else None
    openmc.run(cwd=run_dir, mpi_args=mpi_args, threads=threads, restart_file=restart)
    return final


def load(library, statepoint):
    """Load the tallies of a library and return its macroscopic multigroup data."""
    with openmc.StatePoint(statepoint) as sp:
        library.load_from_statepoint(sp)
    names = [material.name for material in library.domains]
    return library.create_mg_library(xs_type="macro", xsdata_names=names,
                                     temperature=reference.TEMPERATURE)


def void_data(groups):
    """Return zero cross sections for the void region."""
    void = openmc.XSdata("void", groups, temperatures=[reference.TEMPERATURE])
    void.order = LEGENDRE_ORDER
    zeros = np.zeros(groups.num_groups)
    void.set_total(zeros, reference.TEMPERATURE)
    void.set_absorption(zeros, reference.TEMPERATURE)
    void.set_scatter_matrix(np.zeros((groups.num_groups, groups.num_groups,
                                      LEGENDRE_ORDER + 1)), reference.TEMPERATURE)
    return void


def add_reactions(libraries):
    """Add the reaction cross sections of every material to the library file."""
    temperature = f"{round(reference.TEMPERATURE)}K"
    with h5py.File(OUTPUT, "r+") as output:
        for library in libraries:
            material = library.domains[0]
            for reaction, dataset in REACTIONS.items():
                values = library.get_mgxs(material, reaction).get_xs(xs_type="macro")
                output[f"{material.name}/{temperature}"].create_dataset(dataset, data=values)
        for dataset in REACTIONS.values():
            output[f"void/{temperature}"].create_dataset(
                dataset, data=np.zeros(len(GROUP_EDGES_EV) - 1))


def write_transport_corrected(source, output, order):
    """Write the extended-transport-corrected P_order library made from a higher-order one.

    In each group g, c_g = max(sigma_{order+1, g->g}, 0) is subtracted from the total cross
    section and from the within-group moments 0 to order, and the expansion is truncated at
    order. Return the largest correction relative to the total cross section, by material.
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
                    continue  # The void has no scattering.
                data = temperature["scatter_data"]
                total = temperature["total"][()]
                flat = data["scatter_matrix"][()]
                g_min, g_max = data["g_min"][()], data["g_max"][()]
                pairs = [(gi, go) for gi in range(len(total))
                         for go in range(g_min[gi] - 1, g_max[gi])]
                moments = flat.reshape(len(pairs), old_order + 1)
                correction = np.zeros(len(total))
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


def diagnostics(library, xsdata):
    """Return the statistical quality and the collision closure of a material."""
    material = library.domains[0]
    relative = library.get_mgxs(material, "total").get_xs(value="rel_err")
    data = xsdata.get_by_name(material.name)
    sigma_t, sigma_a = data.total[0], data.absorption[0]
    if not (np.all(np.isfinite(sigma_t)) and np.all(sigma_t > 0)):
        raise ValueError(f"{material.name}: a group has no total cross section")
    production = data.scatter_matrix[0][:, :, 0]
    multiplicity = data.multiplicity_matrix[0]
    scatter = np.divide(production, multiplicity, out=np.zeros_like(production),
                        where=multiplicity > 0)
    closure = np.abs(sigma_t - sigma_a - scatter.sum(axis=1)) / sigma_t
    return {
        "total_relative_std_dev_max": float(np.max(relative)),
        "total_relative_std_dev_median": float(np.median(relative)),
        "groups_with_total_relative_std_dev_above_1_percent": int(np.sum(relative > 0.01)),
        "collision_closure_max": float(np.max(closure)),
        # Groups in which (n,2n) and other reactions make more neutrons than they absorb.
        "multiplying_groups": [int(g) for g in np.where(
            production.sum(axis=1) > (1 + 1e-9) * (sigma_t - sigma_a))[0]],
    }


def nuclear_data_record(materials_xml_files):
    """Return the SHA-256 of every nuclear data file that the materials use."""
    xml = Path(os.environ["OPENMC_CROSS_SECTIONS"]).resolve()
    needed = set()
    for materials_xml in materials_xml_files:
        root = ET.parse(materials_xml).getroot()
        needed |= {e.attrib["name"] for e in root.iter("nuclide")}
        needed |= {e.attrib["name"] for e in root.iter("sab")}
    # Library paths are relative to the optional <directory> element, or else to the file.
    library = ET.parse(xml).getroot()
    directory = library.find("directory")
    base = xml.parent / directory.text if directory is not None else xml.parent
    files = {}
    for entry in library.findall("library"):
        if needed.intersection(entry.attrib["materials"].split()):
            files[entry.attrib["path"]] = sha256(base / entry.attrib["path"])
    return {"library": "ENDF/B-VIII.0 HDF5 library distributed by the OpenMC project",
            "files_sha256": dict(sorted(files.items()))}


def inputs_record(run_dir, statepoint):
    names = ("geometry.xml", "materials.xml", "settings.xml", "tallies.xml")
    record = {name: sha256(run_dir / name) for name in names}
    record["statepoint"] = sha256(statepoint)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--particles", type=int, default=1000000,
                        help="Histories per batch for each material")
    parser.add_argument("--batches", type=int, default=100, help="Batches for each material")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--processes", type=int, default=1, help="MPI processes for OpenMC")
    parser.add_argument("--threads", type=int, default=None, help="OpenMP threads per process")
    args = parser.parse_args()
    mpi_args = ["mpiexec", "-n", str(args.processes)] if args.processes > 1 else None

    materials = reference.build_materials()
    libraries, run_dirs, statepoints = [], {}, {}
    for index, (name, material) in enumerate(materials.items()):
        model, library = build_infinite_medium(material, args.particles, args.batches,
                                               args.seed + index)
        run_dirs[name] = Path("xs_generation") / name
        statepoints[name] = run(model, run_dirs[name], mpi_args, args.threads)
        libraries.append(library)

    mg_library = None
    quality = {}
    for library in libraries:
        data = load(library, statepoints[library.domains[0].name])
        if mg_library is None:
            mg_library = data
        else:
            mg_library.add_xsdata(data.xsdatas[0])
        quality[library.domains[0].name] = diagnostics(library, data)
    mg_library.add_xsdata(void_data(libraries[0].energy_groups))
    mg_library.export_to_hdf5(OUTPUT)
    add_reactions(libraries)
    corrections = write_transport_corrected(OUTPUT, CORRECTED_OUTPUT, CORRECTED_ORDER)

    here = Path(__file__).resolve().parent
    version = subprocess.run(["openmc", "--version"], capture_output=True, text=True)
    record = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "library_sha256": sha256(OUTPUT),
        "transport_corrected_library_sha256": sha256(CORRECTED_OUTPUT),
        "generator_sha256": {name: sha256(here / name)
                             for name in ("generate_xs.py", "openmc_reference.py")},
        "openmc_python_version": openmc.__version__,
        "openmc_executable_version": version.stdout.strip(),
        "nuclear_data": nuclear_data_record([d / "materials.xml" for d in run_dirs.values()]),
        "settings": {
            "mgxs_types": MGXS_TYPES, "reactions": REACTIONS,
            "legendre_order": LEGENDRE_ORDER, "transport_correction": None,
            "transport_corrected_library": {
                "file": CORRECTED_OUTPUT, "legendre_order": CORRECTED_ORDER,
                "correction": "extended transport correction with the within-group "
                              f"P{CORRECTED_ORDER + 1} moment, negative values set to zero",
                "largest_correction_relative_to_total": corrections},
            "temperature_K": reference.TEMPERATURE, "source_energy_eV": reference.SOURCE_ENERGY,
            "smooth_source_fraction": SMOOTH_FRACTION,
            "smooth_source_min_energy_eV": SMOOTH_MIN_ENERGY, "cube_side_cm": CUBE_SIDE,
            "particles_per_batch": args.particles, "batches": args.batches,
            "seeds": {name: args.seed + i for i, name in enumerate(materials)},
            "mpi_processes": args.processes, "threads_per_process": args.threads},
        "inputs_sha256": {name: inputs_record(run_dirs[name], statepoints[name])
                          for name in materials},
        "materials": quality,
    }
    PROVENANCE.parent.mkdir(exist_ok=True)
    PROVENANCE.write_text(json.dumps(record, indent=2) + "\n")
    for name, values in quality.items():
        print(f"{name:10s} median total rel. std. dev. "
              f"{values['total_relative_std_dev_median']:.2e}  max "
              f"{values['total_relative_std_dev_max']:.2e}  closure "
              f"{values['collision_closure_max']:.1e}")
    print(f"Wrote {OUTPUT}, {CORRECTED_OUTPUT}, and {PROVENANCE}")


if __name__ == "__main__":
    main()
