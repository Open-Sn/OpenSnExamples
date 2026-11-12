#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Continuous-energy OpenMC reference for the ICF hohlraum example.

This script builds the same target as hohlraum.py from analytic surfaces, runs OpenMC, and
writes the region-integrated flux and reaction rates per source neutron, the leakage
spectrum, and an (r, z) map of the collided flux to results.json. Lengths are in cm.

OpenMC tallies every quantity twice: for all neutrons, and for source neutrons that have
not collided. A statepoint is written after every batch, so each batch gives one sample of
the collided part, the difference of the two. The standard errors of the collided results
come from these batch samples and include the correlation between the two tallies.

OpenMC needs continuous-energy ENDF/B-VIII.0 data for the nuclides listed in
provenance/materials_64g_p5.json. Set OPENMC_CROSS_SECTIONS to the cross_sections.xml file
of that library. The defaults are those of the published reference, 10^10 histories in
100 batches with seed 20261111, which took 76 minutes on 48 processes:

    python3 openmc_reference.py --processes 48
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import openmc

TEMPERATURE = 293.6  # K
SOURCE_ENERGY = 14.1e6  # eV

# Geometry in cm. generate_mesh.py meshes the same target.
CAPSULE_RADIUS = 0.01
GAS_RADIUS, GAS_HALF_HEIGHT = 0.2, 0.3507
HOHLRAUM_RADIUS, HOHLRAUM_HALF_HEIGHT = 0.205, 0.3557
LEH_RADIUS = 0.12
TMP_RADIUS = 0.215
FLANGE_RADIUS, FLANGE_HEIGHT = 0.255, 0.02

REGIONS = {1: "capsule", 2: "fill_gas", 3: "hohlraum_wall", 4: "leh_windows", 5: "tmp",
           6: "exterior"}
BLOCK_MATERIALS = {1: "helium", 2: "helium", 3: "gold", 4: "polyimide", 5: "aluminum"}

# Reaction rates reported for every region. The OpenSn library holds the same reactions.
REACTIONS = {"absorption": "absorption", "(n,2n)": "n2n", "(n,gamma)": "ngamma",
             "(n,a)": "nalpha"}

# Sign of the outward direction relative to the normal of the outer cylinder and the top
# and bottom planes returned by build_geometry.
OUTWARD_SIGNS = [1.0, 1.0, -1.0]

# (r, z) bins of the collided-flux map.
MAP_DR = MAP_DZ = 0.005


def build_materials():
    """Return the materials by name. Densities are in g/cm^3."""
    # The hohlraum is gold at full density.
    gold = openmc.Material(material_id=3, name="gold", temperature=TEMPERATURE)
    gold.add_element("Au", 1.0)
    gold.set_density("g/cm3", 19.32)
    # The thermal-mechanical package is aluminum.
    aluminum = openmc.Material(material_id=5, name="aluminum", temperature=TEMPERATURE)
    aluminum.add_element("Al", 1.0)
    aluminum.set_density("g/cm3", 2.699)
    # Each LEH is covered by a 0.5 um polyimide (C22H10N2O5, 1.42 g/cm^3) window. The window
    # is spread over the 50 um thick hole in the wall, which keeps its areal density.
    polyimide = openmc.Material(material_id=4, name="polyimide", temperature=TEMPERATURE)
    for element, atoms in [("C", 22), ("H", 10), ("N", 2), ("O", 5)]:
        polyimide.add_element(element, atoms)
    polyimide.set_density("g/cm3", 1.42 * 0.5 / 50.0)
    # The hohlraum is filled with natural helium at 0.3 mg/cm^3. The capsule has the same
    # helium in this example.
    helium = openmc.Material(material_id=2, name="helium", temperature=TEMPERATURE)
    helium.add_element("He", 1.0)
    helium.set_density("g/cm3", 0.3e-3)
    return {m.name: m for m in (gold, aluminum, polyimide, helium)}


def build_geometry(materials):
    """Return the geometry, with cell IDs equal to the block IDs of the mesh."""
    capsule = openmc.Sphere(r=CAPSULE_RADIUS)
    leh = openmc.ZCylinder(r=LEH_RADIUS)
    gas_side = openmc.ZCylinder(r=GAS_RADIUS)
    wall_side = openmc.ZCylinder(r=HOHLRAUM_RADIUS)
    tmp_side = openmc.ZCylinder(r=TMP_RADIUS)
    outer = openmc.ZCylinder(r=FLANGE_RADIUS, boundary_type="vacuum")
    gas_top, gas_bottom = openmc.ZPlane(GAS_HALF_HEIGHT), openmc.ZPlane(-GAS_HALF_HEIGHT)
    top = openmc.ZPlane(HOHLRAUM_HALF_HEIGHT, boundary_type="vacuum")
    bottom = openmc.ZPlane(-HOHLRAUM_HALF_HEIGHT, boundary_type="vacuum")
    flange_top = openmc.ZPlane(HOHLRAUM_HALF_HEIGHT - FLANGE_HEIGHT)
    flange_bottom = openmc.ZPlane(-HOHLRAUM_HALF_HEIGHT + FLANGE_HEIGHT)

    cavity = -gas_side & +gas_bottom & -gas_top
    hole = -leh & ((+gas_top & -top) | (-gas_bottom & +bottom))
    hohlraum = -wall_side & +bottom & -top
    regions = {
        1: -capsule,
        2: cavity & +capsule,
        3: hohlraum & ~cavity & ~hole,
        4: hole,
        5: (+wall_side & -tmp_side & +bottom & -top)
        | (+tmp_side & -outer & ((+flange_top & -top) | (-flange_bottom & +bottom))),
        6: +tmp_side & -outer & +flange_bottom & -flange_top,
    }
    cells = [openmc.Cell(cell_id=block, name=REGIONS[block], region=region,
                         fill=materials.get(BLOCK_MATERIALS.get(block)))
             for block, region in regions.items()]
    return openmc.Geometry(cells), [outer, top, bottom]


def build_model(particles, batches, seed, group_edges):
    materials = build_materials()
    geometry, boundaries = build_geometry(materials)

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles, settings.batches, settings.seed = particles, batches, seed
    settings.temperature = {"method": "nearest", "tolerance": 10.0}
    settings.output = {"tallies": False}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 0.0)), angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([SOURCE_ENERGY], [1.0]))
    settings.statepoint = {"batches": list(range(1, batches + 1))}

    # Source neutrons are born in the capsule. Neutrons from (n,2n) and other multiplying
    # reactions start with zero collisions, but they are born elsewhere, because helium-4
    # has no such reaction at 14.1 MeV. A neutron born in the capsule with zero collisions
    # is therefore uncollided.
    cells = list(geometry.get_all_cells().values())
    uncollided = [openmc.CellBornFilter([1]), openmc.CollisionFilter([0])]
    energy = openmc.EnergyFilter(group_edges)
    tallies = openmc.Tallies()
    for suffix, extra in (("", []), ("_uncollided", uncollided)):
        regions = openmc.Tally(name="regions" + suffix)
        regions.filters = [openmc.CellFilter(cells), energy] + extra
        regions.scores = ["flux"] + list(REACTIONS)
        regions.estimator = "tracklength"
        leakage = openmc.Tally(name="leakage" + suffix)
        leakage.filters = [openmc.SurfaceFilter(boundaries), energy] + extra
        leakage.scores = ["current"]
        mesh = openmc.CylindricalMesh(
            r_grid=np.linspace(0.0, FLANGE_RADIUS, round(FLANGE_RADIUS / MAP_DR) + 1),
            z_grid=np.linspace(-HOHLRAUM_HALF_HEIGHT, HOHLRAUM_HALF_HEIGHT,
                               round(2 * HOHLRAUM_HALF_HEIGHT / MAP_DZ) + 1),
            phi_grid=[0.0, 2 * math.pi])
        flux_map = openmc.Tally(name="map" + suffix)
        flux_map.filters = [openmc.MeshFilter(mesh)] + extra
        flux_map.scores = ["flux"]
        tallies += [regions, leakage, flux_map]
    return openmc.Model(geometry, openmc.Materials(materials.values()), settings, tallies)


def batch_samples(run_dir, batches, name):
    """Return the result of each batch of a tally, as an array [batch, bins...]."""
    previous, samples = 0.0, []
    width = len(str(batches))
    for batch in range(1, batches + 1):
        with openmc.StatePoint(run_dir / f"statepoint.{batch:0{width}d}.h5") as sp:
            tally = sp.get_tally(name=name)
            total = np.array(tally.sum)
            shape = tally.shape
        samples.append(total - previous)
        previous = total
    return np.array(samples), shape


def summarize(samples):
    """Return the mean and the standard error of the mean of batch samples."""
    mean = samples.mean(axis=0)
    error = samples.std(axis=0, ddof=1) / math.sqrt(len(samples))
    return mean, error


def collect(run_dir, batches):
    """Return the results of a completed run."""
    results = {}
    with openmc.StatePoint(run_dir / f"statepoint.{batches}.h5") as sp:
        groups = len(sp.get_tally(name="regions").find_filter(openmc.EnergyFilter).bins)
        leakage_total = float(sp.global_tallies["mean"][3])
    for name in ("regions", "leakage", "map"):
        total, shape = batch_samples(run_dir, batches, name)
        uncollided, _ = batch_samples(run_dir, batches, name + "_uncollided")
        uncollided = uncollided.reshape(total.shape)
        for part, values in (("total", total), ("uncollided", uncollided),
                             ("collided", total - uncollided)):
            results[f"{name}_{part}"] = values
    report = {"batches": batches, "global_leakage": leakage_total}
    scores = ["flux"] + list(REACTIONS)
    for part in ("total", "uncollided", "collided"):
        # regions: [batch, cell, group, score]. Group 0 of OpenMC is the lowest energy.
        regions = results[f"regions_{part}"].reshape(batches, len(REGIONS), groups,
                                                     len(scores))
        regions = regions[:, :, ::-1, :]
        integrated = {}
        for i, block in enumerate(REGIONS):
            entry = {}
            for k, score in enumerate(scores):
                key = "flux" if score == "flux" else REACTIONS[score]
                mean, error = summarize(regions[:, i, :, k].sum(axis=1))
                entry[key] = [float(mean), float(error)]
                mean, error = summarize(regions[:, i, :, k])
                entry[key + "_groups"] = [mean.tolist(), error.tolist()]
            integrated[REGIONS[block]] = entry
        report[part] = integrated
        # leakage: [batch, surface, group]. OpenMC's current is positive along the normal of
        # the surface, which points inward on the bottom plane. A vacuum surface has no
        # inflow, so the outward current is the outflow.
        leakage = results[f"leakage_{part}"].reshape(batches, -1, groups)[:, :, ::-1]
        leakage = leakage * np.array(OUTWARD_SIGNS)[None, :, None]
        mean, error = summarize(leakage.sum(axis=(1, 2)))
        groups_mean, groups_error = summarize(leakage.sum(axis=1))
        report[part]["leakage"] = [float(mean), float(error)]
        report[part]["leakage_groups"] = [groups_mean.tolist(), groups_error.tolist()]
        # map: [batch, r, z] in OpenMC's mesh ordering (r fastest).
        nr = round(FLANGE_RADIUS / MAP_DR)
        nz = round(2 * HOHLRAUM_HALF_HEIGHT / MAP_DZ)
        flux_map = results[f"map_{part}"].reshape(batches, nz, nr)
        mean, error = summarize(flux_map)
        report[part]["map"] = [mean.tolist(), error.tolist()]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--particles", type=int, default=100000000,
                        help="Histories per batch")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--processes", type=int, default=1, help="MPI processes for OpenMC")
    parser.add_argument("--threads", type=int, default=None, help="OpenMP threads per process")
    parser.add_argument("--export-only", action="store_true",
                        help="Write the OpenMC XML input without running it")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    library = openmc.MGXSLibrary.from_hdf5(str(here / "materials_64g_p5.h5"))
    group_edges = [float(e) for e in library.energy_groups.group_edges]
    model = build_model(args.particles, args.batches, args.seed, group_edges)
    run_dir = Path(f"openmc_seed{args.seed}")
    run_dir.mkdir(exist_ok=True)
    model.export_to_xml(directory=run_dir)
    if args.export_only:
        return

    mpi_args = ["mpiexec", "-n", str(args.processes)] if args.processes > 1 else None
    openmc.run(cwd=run_dir, mpi_args=mpi_args, threads=args.threads)
    report = collect(run_dir, args.batches)
    report["particles_per_batch"], report["seed"] = args.particles, args.seed
    (run_dir / "results.json").write_text(json.dumps(report) + "\n")

    print(f"{'Region':14s} {'Total flux [cm]':>24s} {'Collided flux [cm]':>24s}")
    for name in REGIONS.values():
        total, collided = report["total"][name]["flux"], report["collided"][name]["flux"]
        print(f"{name:14s} {total[0]:12.6e} +- {total[1]:7.1e} "
              f"{collided[0]:12.6e} +- {collided[1]:7.1e}")
    leakage = report["total"]["leakage"]
    print(f"Leakage per source neutron: {leakage[0]:.8e} +- {leakage[1]:.1e} "
          f"(global tally {report['global_leakage']:.8e})")
    print(f"Wrote {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
