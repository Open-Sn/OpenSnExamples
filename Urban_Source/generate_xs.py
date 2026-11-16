#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate materials_25g_p7.h5, the 25-group photon cross sections of the urban source
example, with P0 to P7 scattering.

The groups span 10 keV to 670 keV. Group 0, from 650 to 670 keV, holds the 661.657 keV
gamma ray of Cs-137, and the 24 groups below it are equally spaced in lethargy down to
10 keV. Photons below 10 keV are killed in both the cross sections and the OpenMC
reference: in air they travel less than 2 m, and the ICRP 116 dose coefficients start at
10 keV.

OpenMC weights each material of city.toml in its own infinite medium, a reflecting cube
filled with that material, with continuous-energy ENDF/B-VIII.0 photon data. 99% of the
source is at 661.657 keV, and 1% is spread over the group structure with a 1/E spectrum,
so that every group is sampled. The electrons are deposited where they are born, so no
bremsstrahlung is produced; below the pair-production threshold the only secondary photons
are fluorescence x rays, which in these materials are below 10 keV.

For each material and group g the script tallies, with track-length estimators, the flux,
the total cross section sigma_t,g, and the scattering cross section sigma_s,g (coherent and
incoherent). An analog tally gives the probabilities P_l(g -> g') of scattering from g to
g', with Legendre moments l = 0 to 7 of the scattering cosine, including scattering to
below 10 keV. The library holds

    sigma_s,l,g->g' = sigma_s,g P_l(g -> g')    for g' in the structure
    sigma_a,g = sigma_t,g - sum_g' sigma_s,0,g->g'

so the absorption includes photoelectric absorption and scattering to below 10 keV, and
P0 scattering plus absorption equals the total cross section exactly. The library also
holds the group-averaged ICRP 116 effective-dose coefficients for the AP and isotropic
(ISO) geometries in pSv cm^2, weighted by the infinite-medium flux in air, as the datasets
dose_ap and dose_iso of every material. urban_source.py uses them to compute dose.

Set OPENMC_CROSS_SECTIONS to the cross_sections.xml file of the ENDF/B-VIII.0 HDF5 library
of the OpenMC project, then run, for example:

    python3 generate_xs.py --processes 8

The script writes the OpenMC inputs and statepoints to xs_generation/, the library to
materials_25g_p7.h5, and a record of every input and output to
provenance/materials_25g_p7.json.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import openmc
import openmc.data
import openmc.mgxs

import city as citylib
import openmc_reference as reference

TOP_GROUP = (650.0e3, 670.0e3)  # eV
LOWEST_ENERGY = reference.PHOTON_CUTOFF  # eV
NUM_LOWER_GROUPS = 24
LEGENDRE_ORDER = 7
DOSE_GEOMETRIES = {"dose_ap": "AP", "dose_iso": "ISO"}
WEIGHTING_MATERIAL = "air"  # Its spectrum weights the dose coefficients.

CUBE_SIDE = 1.0e7  # cm. Many mean free paths of air, to avoid needless reflections.
SMOOTH_FRACTION = 0.01
CHECKPOINT_BATCHES = 10

OUTPUT = Path(reference.LIBRARY)
PROVENANCE = Path("provenance") / (OUTPUT.stem + ".json")


def group_edges():
    """Return the group boundaries in eV, increasing."""
    lower = np.geomspace(LOWEST_ENERGY, TOP_GROUP[0], NUM_LOWER_GROUPS + 1)
    return np.append(lower, TOP_GROUP[1])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_infinite_medium(material, particles, batches, seed, edges):
    """Return the infinite-medium model of one material."""
    half = CUBE_SIDE / 2
    cube = openmc.model.RectangularParallelepiped(-half, half, -half, half, -half, half,
                                                  boundary_type="reflective")
    geometry = openmc.Geometry([openmc.Cell(fill=material, region=-cube)])
    space = openmc.stats.Box((-half, -half, -half), (half, half, half))
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles, settings.batches, settings.seed = particles, batches, seed
    settings.photon_transport = True
    settings.electron_treatment = "led"
    settings.cutoff = {"energy_photon": LOWEST_ENERGY}
    settings.temperature = {"method": "nearest", "tolerance": 1000.0}
    settings.output = {"tallies": False}
    top = edges[-1]
    settings.source = [
        openmc.IndependentSource(space=space, angle=openmc.stats.Isotropic(),
                                 energy=openmc.stats.Discrete([reference.SOURCE_ENERGY], [1.0]),
                                 strength=1.0 - SMOOTH_FRACTION, particle="photon"),
        openmc.IndependentSource(space=space, angle=openmc.stats.Isotropic(),
                                 energy=openmc.stats.Tabular(
                                     [LOWEST_ENERGY, top], [1.0 / LOWEST_ENERGY, 1.0 / top],
                                     interpolation="log-log"),
                                 strength=SMOOTH_FRACTION, particle="photon")]

    photon = openmc.ParticleFilter(["photon"])
    energy = openmc.EnergyFilter(edges)
    # The extra outgoing bin collects scattering to below the lowest group.
    energy_out = openmc.EnergyoutFilter(np.concatenate(([0.0], edges)))
    tallies = openmc.Tallies()
    rates = openmc.Tally(name="rates")
    rates.filters = [photon, energy]
    rates.scores = ["flux", "total", "scatter", "absorption"]
    rates.estimator = "tracklength"
    transfer = openmc.Tally(name="transfer")
    transfer.filters = [photon, energy, energy_out, openmc.LegendreFilter(LEGENDRE_ORDER)]
    transfer.scores = ["scatter"]
    transfer.estimator = "analog"
    tallies += [rates, transfer]
    for name, geometry_name in DOSE_GEOMETRIES.items():
        dose_energy, dose_values = openmc.data.dose_coefficients("photon", geometry_name)
        dose = openmc.Tally(name=name)
        dose.filters = [photon, energy, openmc.EnergyFunctionFilter(
            dose_energy, dose_values, interpolation="log-log")]
        dose.scores = ["flux"]
        dose.estimator = "tracklength"
        tallies.append(dose)
    return openmc.Model(geometry, openmc.Materials([material]), settings, tallies)


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
    model.settings.statepoint = {"batches": sorted(set(range(CHECKPOINT_BATCHES, batches + 1,
                                                             CHECKPOINT_BATCHES)) | {batches})}
    model.export_to_xml(directory=run_dir)
    checkpoints = sorted(run_dir.glob("statepoint.*.h5"), key=lambda p: int(p.suffixes[0][1:]))
    restart = checkpoints[-1].name if checkpoints else None
    openmc.run(cwd=run_dir, mpi_args=mpi_args, threads=threads, restart_file=restart)
    return final


def group_data(statepoint, num_groups):
    """Return the group cross sections, scattering matrix, and dose coefficients of one
    material, in decreasing energy order (group 0 is the highest)."""
    with openmc.StatePoint(statepoint) as sp:
        rates = sp.get_tally(name="rates")
        mean = rates.mean.reshape(num_groups, 4)[::-1]
        std = rates.std_dev.reshape(num_groups, 4)[::-1]
        transfer = sp.get_tally(name="transfer").mean.reshape(
            num_groups, num_groups + 1, LEGENDRE_ORDER + 1)
        doses = {name: sp.get_tally(name=name).mean.reshape(num_groups)[::-1]
                 for name in DOSE_GEOMETRIES}
    flux = mean[:, 0]
    if np.any(flux <= 0):
        raise ValueError(f"{statepoint}: a group has no flux")
    sigma_t, sigma_s, sigma_pe = mean[:, 1] / flux, mean[:, 2] / flux, mean[:, 3] / flux
    # transfer[g_in, g_out, l] with energies increasing; g_out bin 0 is below the structure.
    # Reverse both group axes so that group 0 is the highest.
    probabilities = transfer / transfer[:, :, 0].sum(axis=1)[:, None, None]
    below = probabilities[::-1, 0, 0]
    probabilities = probabilities[::-1, :0:-1, :]
    scatter = sigma_s[:, None, None] * probabilities
    sigma_a = sigma_t - scatter[:, :, 0].sum(axis=1)
    relative = std[:, 1] / mean[:, 1]
    return {
        "flux": flux, "total": sigma_t, "absorption": sigma_a, "scatter": scatter,
        "doses": {name: values / flux for name, values in doses.items()},
        "quality": {
            "total_relative_std_dev_max": float(np.max(relative)),
            "total_relative_std_dev_median": float(np.median(relative)),
            # The removal by scattering to below 10 keV, relative to the total.
            "scatter_below_structure_max": float(np.max(sigma_s * below / sigma_t)),
            # The track-length total cross section, compared with the sum of the
            # track-length scattering and absorption (photoelectric) cross sections.
            "total_closure_max": float(np.max(
                np.abs(sigma_a - sigma_s * below - sigma_pe) / sigma_t)),
        },
    }


def nuclear_data_record(materials_xml_files):
    """Return the SHA-256 of every photon and neutron data file that the materials use."""
    xml = Path(os.environ["OPENMC_CROSS_SECTIONS"]).resolve()
    nuclides = set()
    for materials_xml in materials_xml_files:
        root = ET.parse(materials_xml).getroot()
        nuclides |= {e.attrib["name"] for e in root.iter("nuclide")}
    elements = {re.match(r"[A-Z][a-z]?", nuclide).group(0) for nuclide in nuclides}
    library = ET.parse(xml).getroot()
    directory = library.find("directory")
    base = xml.parent / directory.text if directory is not None else xml.parent
    files = {}
    for entry in library.findall("library"):
        names = set(entry.attrib["materials"].split())
        needed = nuclides if entry.attrib["type"] == "neutron" else elements
        if entry.attrib["type"] in ("neutron", "photon") and names & needed:
            files[entry.attrib["path"]] = sha256(base / entry.attrib["path"])
    return {"library": "ENDF/B-VIII.0 HDF5 library distributed by the OpenMC project",
            "files_sha256": dict(sorted(files.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, default=Path("city.toml"))
    parser.add_argument("--particles", type=int, default=1000000,
                        help="Histories per batch for each material")
    parser.add_argument("--batches", type=int, default=100, help="Batches for each material")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--processes", type=int, default=1, help="MPI processes for OpenMC")
    parser.add_argument("--threads", type=int, default=None, help="OpenMP threads per process")
    args = parser.parse_args()
    mpi_args = ["mpiexec", "-n", str(args.processes)] if args.processes > 1 else None

    city = citylib.load(args.layout)
    materials = reference.build_materials(city)
    edges = group_edges()
    num_groups = len(edges) - 1
    groups = openmc.mgxs.EnergyGroups(edges)
    run_dirs, data = {}, {}
    for index, (name, material) in enumerate(materials.items()):
        model = build_infinite_medium(material, args.particles, args.batches,
                                      args.seed + index, edges)
        run_dirs[name] = Path("xs_generation") / name
        statepoint = run(model, run_dirs[name], mpi_args, args.threads)
        data[name] = group_data(statepoint, num_groups)

    library = openmc.MGXSLibrary(groups)
    for name, values in data.items():
        xsdata = openmc.XSdata(name, groups, temperatures=[reference.TEMPERATURE])
        xsdata.order = LEGENDRE_ORDER
        xsdata.set_total(values["total"], reference.TEMPERATURE)
        xsdata.set_absorption(values["absorption"], reference.TEMPERATURE)
        xsdata.set_scatter_matrix(values["scatter"], reference.TEMPERATURE)
        library.add_xsdata(xsdata)
    library.export_to_hdf5(str(OUTPUT))
    doses = data[WEIGHTING_MATERIAL]["doses"]
    temperature = f"{round(reference.TEMPERATURE)}K"
    with h5py.File(OUTPUT, "r+") as output:
        for name in data:
            for dataset, values in doses.items():
                output[f"{name}/{temperature}"].create_dataset(dataset, data=values)

    here = Path(__file__).resolve().parent
    version = subprocess.run(["openmc", "--version"], capture_output=True, text=True)
    record = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "library_sha256": sha256(OUTPUT),
        "generator_sha256": {name: sha256(here / name) for name in
                             ("generate_xs.py", "openmc_reference.py", "city.py")},
        "layout_sha256": sha256(args.layout),
        "openmc_python_version": openmc.__version__,
        "openmc_executable_version": version.stdout.strip(),
        "nuclear_data": nuclear_data_record([d / "materials.xml" for d in run_dirs.values()]),
        "settings": {
            "group_edges_eV": edges.tolist(), "legendre_order": LEGENDRE_ORDER,
            "photon_cutoff_eV": LOWEST_ENERGY, "electron_treatment": "led",
            "source_energy_eV": reference.SOURCE_ENERGY,
            "smooth_source_fraction": SMOOTH_FRACTION, "cube_side_cm": CUBE_SIDE,
            "dose_coefficients": {k: f"ICRP 116 effective dose, {v}, log-log, weighted by "
                                  f"the infinite-medium flux in {WEIGHTING_MATERIAL}"
                                  for k, v in DOSE_GEOMETRIES.items()},
            "particles_per_batch": args.particles, "batches": args.batches,
            "seeds": {name: args.seed + i for i, name in enumerate(materials)},
            "mpi_processes": args.processes, "threads_per_process": args.threads},
        "inputs_sha256": {
            name: {**{f: sha256(run_dirs[name] / f) for f in
                      ("geometry.xml", "materials.xml", "settings.xml", "tallies.xml")},
                   "statepoint": sha256(run_dirs[name] / f"statepoint.{args.batches}.h5")}
            for name in materials},
        "materials": {name: values["quality"] for name, values in data.items()},
    }
    PROVENANCE.parent.mkdir(exist_ok=True)
    PROVENANCE.write_text(json.dumps(record, indent=2) + "\n")
    for name, values in data.items():
        quality = values["quality"]
        print(f"{name:10s} median total rel. std. dev. "
              f"{quality['total_relative_std_dev_median']:.2e}  max "
              f"{quality['total_relative_std_dev_max']:.2e}  total closure "
              f"{quality['total_closure_max']:.1e}")
    print(f"Wrote {OUTPUT} and {PROVENANCE}")


if __name__ == "__main__":
    main()
