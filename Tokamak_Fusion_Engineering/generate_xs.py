#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate materials_64g_p3.h5, the 64-group P3 cross sections used by tokamak.py.

OpenMC tallies multigroup cross sections in the tungsten, steel/water, steel, and
borated-polyethylene regions of the continuous-energy tokamak model in openmc_reference.py.
Each of these materials is therefore weighted by its own neutron spectrum in the reactor.
The copper coils sit behind the shield, where too few neutrons arrive to tally all 64
groups. Copper is weighted instead in a stochastic slab: 1 cm layers of all five materials
in a fixed random order, inside a reflecting cube with a 14.1 MeV source. One percent of the
slab source is spread equally over the 64 groups, so every copper group is sampled.

For each material the library holds the total cross section, the physical absorption cross
section, and P0 to P3 scattering matrices. The scattering matrices are neutron-production
matrices, so they include the extra neutrons from (n,2n) and other multiplying reactions.
They use OpenMC's consistent estimator, which makes the P0 scattering cross section equal
to the total minus the absorption cross section. No transport correction is applied. Both
calculations are analog, without variance reduction.

OpenMC needs continuous-energy ENDF/B-VIII.0 data for the nuclides and thermal scattering
tables of the model. Set OPENMC_CROSS_SECTIONS to the cross_sections.xml file of that
library, then run, for example:

    python3 generate_xs.py --processes 32

The script writes the OpenMC inputs and statepoints to xs_generation/, the library to
materials_64g_p3.h5, and a record of every input and output to
provenance/materials_64g_p3.json.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import openmc
import openmc.mgxs

import openmc_reference as reference

# The 64-group structure in eV, from 0 to 14.191 MeV. It is a subset of the VITAMIN-J
# 175-group boundaries with four thermal boundaries added below 0.1 eV. Group 0
# (13.84 to 14.191 MeV) holds the 14.1 MeV source.
GROUP_EDGES_EV = [
    0.0, 0.005, 0.01, 0.0253, 0.05, 0.1, 0.53158, 1.1253, 2.3824, 5.0435, 10.677, 29.023,
    61.442, 167.02, 353.58, 961.12, 2034.7, 2746.5, 3707.4, 9118.8, 15034.0, 23579.0,
    24176.0, 24788.0, 28501.0, 46309.0, 67380.0, 86517.0, 116790.0, 142640.0, 174220.0,
    212800.0, 247240.0, 297210.0, 333730.0, 387740.0, 450490.0, 550230.0, 608100.0,
    672060.0, 742740.0, 820850.0, 907180.0, 1002600.0, 1164800.0, 1353400.0, 1572400.0,
    1826800.0, 2122500.0, 2345700.0, 2466000.0, 2865000.0, 3166400.0, 4065700.0,
    4965800.0, 5769500.0, 6592400.0, 7408200.0, 8607100.0, 10000000.0, 11052000.0,
    12214000.0, 12840000.0, 13840000.0, 14191000.0]
LEGENDRE_ORDER = 3
MGXS_TYPES = ["total", "absorption", "consistent scatter matrix", "consistent nu-scatter matrix"]

CHECKPOINT_BATCHES = 10

# Copper weighting slab: each material fills SLAB_REPEATS layers of SLAB_LAYER cm.
SLAB_LAYER, SLAB_REPEATS, SLAB_ORDER_SEED = 1.0, 100, 42

OUTPUT = "materials_64g_p3.h5"
PROVENANCE = Path("provenance") / "materials_64g_p3.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def mgxs_library(geometry, materials):
    """Return an MGXS library that tallies the given materials."""
    library = openmc.mgxs.Library(geometry)
    library.energy_groups = openmc.mgxs.EnergyGroups(GROUP_EDGES_EV)
    library.mgxs_types = MGXS_TYPES
    library.correction = None
    library.scatter_format = "legendre"
    library.legendre_order = LEGENDRE_ORDER
    library.by_nuclide = False
    library.domain_type = "material"
    library.domains = list(materials)
    library.build_library()
    return library


def build_tokamak(particles, batches, seed):
    """Return the tokamak model that weights every material except copper."""
    model = reference.build_model(particles, batches, seed, GROUP_EDGES_EV)
    model.tallies = openmc.Tallies()
    library = mgxs_library(model.geometry, [m for m in model.materials if m.name != "copper"])
    library.add_to_tallies(model.tallies, merge=True)
    return model, library


def build_copper_slab(particles, batches, seed):
    """Return the stochastic-slab model that weights copper."""
    materials = list(reference.build_materials().values())
    universes = [openmc.Universe(cells=[openmc.Cell(fill=m)]) for m in materials]
    order = list(range(len(materials))) * SLAB_REPEATS
    random.Random(SLAB_ORDER_SEED).shuffle(order)
    width = len(order) * SLAB_LAYER
    lattice = openmc.RectLattice()
    lattice.pitch = (SLAB_LAYER, width, width)
    lattice.lower_left = (0.0, 0.0, 0.0)
    lattice.universes = [[[universes[i] for i in order]]]
    lattice.outer = universes[0]
    box = openmc.model.RectangularParallelepiped(0.0, width, 0.0, width, 0.0, width,
                                                 boundary_type="reflective")
    geometry = openmc.Geometry([openmc.Cell(fill=lattice, region=-box)])

    space = openmc.stats.Box((0.0, 0.0, 0.0), (width, width, width))
    edges = np.array(GROUP_EDGES_EV)
    midpoints = np.sqrt(np.maximum(edges[:-1], edges[1] / 2) * edges[1:])
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles, settings.batches, settings.seed = particles, batches, seed
    settings.temperature = {"method": "nearest", "tolerance": 10.0}
    settings.output = {"tallies": False}
    settings.source = [
        openmc.IndependentSource(space=space, angle=openmc.stats.Isotropic(),
                                 energy=openmc.stats.Discrete([14.1e6], [1.0]), strength=0.99),
        openmc.IndependentSource(space=space, angle=openmc.stats.Isotropic(),
                                 energy=openmc.stats.Discrete(midpoints, np.ones(len(midpoints))),
                                 strength=0.01)]
    library = mgxs_library(geometry, [m for m in materials if m.name == "copper"])
    tallies = openmc.Tallies()
    library.add_to_tallies(tallies, merge=True)
    return openmc.Model(geometry, openmc.Materials(materials), settings, tallies), library


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


def mg_data(library, statepoint):
    """Load the tallies of a library and return its macroscopic multigroup data."""
    with openmc.StatePoint(statepoint) as sp:
        library.load_from_statepoint(sp)
    names = [material.name for material in library.domains]
    return library.create_mg_library(xs_type="macro", xsdata_names=names,
                                     temperature=reference.TEMPERATURE)


def void_data(groups):
    """Return zero cross sections for the void regions."""
    void = openmc.XSdata("void", groups, temperatures=[reference.TEMPERATURE])
    void.order = LEGENDRE_ORDER
    zeros = np.zeros(groups.num_groups)
    void.set_total(zeros, reference.TEMPERATURE)
    void.set_absorption(zeros, reference.TEMPERATURE)
    void.set_scatter_matrix(np.zeros((groups.num_groups, groups.num_groups,
                                      LEGENDRE_ORDER + 1)), reference.TEMPERATURE)
    return void


def diagnostics(library, xsdata):
    """Return the statistical quality and the collision closure of each material."""
    result = {}
    for material in library.domains:
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
        result[material.name] = {
            "total_relative_std_dev_max": float(np.max(relative)),
            "total_relative_std_dev_median": float(np.median(relative)),
            "groups_with_total_relative_std_dev_above_1_percent": int(np.sum(relative > 0.01)),
            "collision_closure_max": float(np.max(closure)),
            "multiplying_groups": [int(g) for g in np.where(
                production.sum(axis=1) > sigma_t - sigma_a)[0]],
        }
    return result


def nuclear_data_record(materials_xml):
    """Return the SHA-256 of every nuclear data file that the materials use."""
    xml = Path(os.environ["OPENMC_CROSS_SECTIONS"]).resolve()
    root = ET.parse(materials_xml).getroot()
    needed = {e.attrib["name"] for e in root.iter("nuclide")}
    needed |= {e.attrib["name"] for e in root.iter("sab")}
    files = {}
    for entry in ET.parse(xml).getroot().findall("library"):
        if needed.intersection(entry.attrib["materials"].split()):
            files[entry.attrib["path"]] = sha256(xml.parent / entry.attrib["path"])
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
    parser.add_argument("--particles", type=int, default=10000000,
                        help="Tokamak histories per batch")
    parser.add_argument("--batches", type=int, default=100, help="Tokamak batches")
    parser.add_argument("--slab-particles", type=int, default=1000000,
                        help="Copper slab histories per batch")
    parser.add_argument("--slab-batches", type=int, default=100, help="Copper slab batches")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--processes", type=int, default=1, help="MPI processes for OpenMC")
    parser.add_argument("--threads", type=int, default=1, help="OpenMP threads per process")
    args = parser.parse_args()
    mpi_args = ["mpiexec", "-n", str(args.processes)] if args.processes > 1 else None

    tokamak_dir = Path("xs_generation") / "tokamak"
    slab_dir = Path("xs_generation") / "copper_slab"
    tokamak, tokamak_library = build_tokamak(args.particles, args.batches, args.seed)
    slab, slab_library = build_copper_slab(args.slab_particles, args.slab_batches,
                                           args.seed + 1)
    tokamak_statepoint = run(tokamak, tokamak_dir, mpi_args, args.threads)
    slab_statepoint = run(slab, slab_dir, mpi_args, args.threads)

    library = mg_data(tokamak_library, tokamak_statepoint)
    copper = mg_data(slab_library, slab_statepoint)
    library.add_xsdata(copper.get_by_name("copper"))
    library.add_xsdata(void_data(tokamak_library.energy_groups))
    library.export_to_hdf5(OUTPUT)
    quality = diagnostics(tokamak_library, library) | diagnostics(slab_library, library)

    here = Path(__file__).resolve().parent
    version = subprocess.run(["openmc", "--version"], capture_output=True, text=True)
    record = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "library_sha256": sha256(OUTPUT),
        "generator_sha256": {name: sha256(here / name)
                             for name in ("generate_xs.py", "openmc_reference.py")},
        "openmc_python_version": openmc.__version__,
        "openmc_executable_version": version.stdout.strip(),
        "nuclear_data": nuclear_data_record(tokamak_dir / "materials.xml"),
        "settings": {
            "mgxs_types": MGXS_TYPES, "legendre_order": LEGENDRE_ORDER,
            "transport_correction": None, "temperature_K": reference.TEMPERATURE,
            "mpi_processes": args.processes, "threads_per_process": args.threads,
            "tokamak": {"particles_per_batch": args.particles, "batches": args.batches,
                        "seed": args.seed, "materials": [m.name for m in
                                                         tokamak_library.domains]},
            "copper_slab": {"particles_per_batch": args.slab_particles,
                            "batches": args.slab_batches, "seed": args.seed + 1,
                            "layer_cm": SLAB_LAYER, "repeats": SLAB_REPEATS,
                            "order_seed": SLAB_ORDER_SEED}},
        "inputs_sha256": {"tokamak": inputs_record(tokamak_dir, tokamak_statepoint),
                          "copper_slab": inputs_record(slab_dir, slab_statepoint)},
        "materials": quality,
    }
    PROVENANCE.parent.mkdir(exist_ok=True)
    PROVENANCE.write_text(json.dumps(record, indent=2) + "\n")
    for name, values in quality.items():
        print(f"{name:22s} median total rel. std. dev. "
              f"{values['total_relative_std_dev_median']:.2e}  max "
              f"{values['total_relative_std_dev_max']:.2e}  closure "
              f"{values['collision_closure_max']:.1e}")
    print(f"Wrote {OUTPUT} and {PROVENANCE}")


if __name__ == "__main__":
    main()
