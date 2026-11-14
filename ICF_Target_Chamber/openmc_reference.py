#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Continuous-energy OpenMC reference for the ICF target chamber example.

The script builds the whole chamber of chamber_geometry.py from analytic surfaces, with no
symmetry plane, runs OpenMC with an isotropic 14.1 MeV point source at the centre, and writes
the results per source neutron to results.json in its run directory. Lengths are in cm.
generate_xs.py builds the same model to tally the multigroup cross sections.

Tallies, each for all neutrons and for the uncollided ones (born in the cavity, block 1, with
no collision; neutrons from (n,2n) reactions are born in the wall or the shield):

- flux and reaction rates (absorption, (n,2n), (n,gamma), (n,alpha)) in each block, in the
  64 groups of the example;
- leakage through the outer sphere, by group;
- the flux along the three line-outs of chamber_geometry.line_points: along the axis of a
  port of the 23.5 degree cone ("port"), along a radius away from every port ("shield"), and
  across the beam of that port on the sphere of radius 650 cm ("arc"). Each is tallied in
  small spherical-mesh bins around the 8 directions that the symmetry of the chamber makes
  equivalent (the azimuths 45, 135, 225 and 315 degrees, above and below the equator), and
  the 8 are averaged;
- the flux in a thin wedge around the half-planes of azimuth 45, 135, 225 and 315 degrees,
  in (r, z) bins, for the maps of the figures.

OpenMC writes a statepoint after every batch, so each batch gives one sample of every
result, and of the collided part, the difference of the total and the uncollided parts. The
standard errors come from these batch samples and include the correlation between the two.

Set OPENMC_CROSS_SECTIONS to the cross_sections.xml file of the ENDF/B-VIII.0 library
distributed by the OpenMC project, then run, for example:

    python3 openmc_reference.py --particles 10000000 --batches 100 --threads 24
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import openmc

import chamber_geometry as geometry

TEMPERATURE = 293.6  # K
SOURCE_ENERGY = 14.1e6  # eV

# The 64-group structure in eV of the Tokamak and ICF hohlraum examples: a subset of the
# VITAMIN-J 175-group boundaries with four thermal boundaries added below 0.1 eV. Group 0
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

# Reaction rates reported for every block, with the dataset names of the OpenSn library.
REACTIONS = {"absorption": "absorption", "(n,2n)": "n2n", "(n,gamma)": "ngamma",
             "(n,a)": "nalpha"}
BLOCK_MATERIALS = {1: None, 2: "aluminum", 3: "borated_concrete", 4: "air"}

# Line-outs. Radial bins of the "port" and "shield" lines (cm), and polar-angle bins of the
# "arc" line (degrees from the port axis) at r = 650 cm.
LINE_DR = 10.0
ARC_RADIUS, ARC_HALF_THICKNESS, ARC_DTHETA = 650.0, 2.5, 0.25
LINE_HALF_WIDTH = 0.5  # degrees: half-width of the bins across a line
# Map: (r, z) bins of the wedge.
MAP_DR = MAP_DZ = 10.0
MAP_HALF_WIDTH = 2.0  # degrees


def build_materials():
    """Return the materials by name.

    Compositions are elemental weight fractions from PNNL-15870 Rev. 2, "Compendium of
    Material Composition Data for Radiation Transport Modeling" (2021).
    """
    # Wall: aluminum alloy 5086-O (compendium material 12, 2.66 g/cm^3), an Al-Mg alloy of the
    # 5xxx series.
    aluminum = openmc.Material(material_id=2, name="aluminum", temperature=TEMPERATURE)
    for element, fraction in [("Mg", 0.040000), ("Al", 0.946499), ("Si", 0.002143),
                              ("Ti", 0.000804), ("Cr", 0.001500), ("Mn", 0.004500),
                              ("Fe", 0.002679), ("Cu", 0.000536), ("Zn", 0.001339)]:
        aluminum.add_element(element, fraction, "wo")
    aluminum.set_density("g/cm3", 2.66)
    # Shield: Portland concrete (compendium material 108, 2.3 g/cm^3) with 1% by weight of
    # natural boron added. The boron content of the NIF shield is not given in the sources
    # of the README; 1% is the choice of this example.
    concrete = openmc.Material(material_id=3, name="borated_concrete",
                               temperature=TEMPERATURE)
    boron = 0.01
    for element, fraction in [("H", 0.010000), ("C", 0.001000), ("O", 0.529107),
                              ("Na", 0.016000), ("Mg", 0.002000), ("Al", 0.033872),
                              ("Si", 0.337021), ("K", 0.013000), ("Ca", 0.044000),
                              ("Fe", 0.014000)]:
        concrete.add_element(element, fraction * (1.0 - boron), "wo")
    concrete.add_element("B", boron, "wo")
    concrete.set_density("g/cm3", 2.3)
    concrete.add_s_alpha_beta("c_H_in_H2O")
    # Air: dry air near sea level (compendium material 4, 0.001205 g/cm^3).
    air = openmc.Material(material_id=4, name="air", temperature=TEMPERATURE)
    for element, fraction in [("C", 0.000124), ("N", 0.755268), ("O", 0.231781),
                              ("Ar", 0.012827)]:
        air.add_element(element, fraction, "wo")
    air.set_density("g/cm3", 0.001205)
    return {m.name: m for m in (aluminum, concrete, air)}


# Cell ids: the cavity, the wall, the shield and the air have the ids of their blocks; the
# ports are cells PORT_CELL_OFFSET + 1, ..., which belong to block 1.
PORT_CELL_OFFSET = 100


def block_of_cell(cell_id):
    return 1 if cell_id > PORT_CELL_OFFSET else cell_id


def build_geometry(materials):
    """Return the geometry and the outer sphere.

    The ports come in pairs on opposite sides of the centre, on one line. Each line has one
    infinite cylinder and one plane through the centre normal to it, and the two ports of the
    line are on the two sides of the plane. Separate surfaces for the two ports would coincide,
    which can trap a particle in a loop of zero-length steps.
    """
    cavity = openmc.Sphere(r=geometry.R_CAVITY)
    wall = openmc.Sphere(r=geometry.R_WALL)
    shield = openmc.Sphere(r=geometry.R_SHIELD)
    outer = openmc.Sphere(r=geometry.R_AIR, boundary_type="vacuum")
    shell = +cavity & -shield
    lines = []  # (axis, cylinder, plane)
    port_regions = []
    for axis, radius in zip(*geometry.ports()):
        line = next((entry for entry in lines if abs(abs(entry[0] @ axis) - 1.0) < 1.0e-12),
                    None)
        if line is None:
            line = (axis, openmc.Cylinder(x0=0.0, y0=0.0, z0=0.0, r=radius,
                                          dx=axis[0], dy=axis[1], dz=axis[2]),
                    openmc.Plane(a=axis[0], b=axis[1], c=axis[2], d=0.0))
            lines.append(line)
        side = +line[2] if line[0] @ axis > 0.0 else -line[2]
        port_regions.append(-line[1] & side & shell)
    ports = openmc.Union(port_regions)
    cells = [openmc.Cell(cell_id=1, name="cavity", region=-cavity),
             openmc.Cell(cell_id=2, name="wall", region=+cavity & -wall & ~ports,
                         fill=materials[BLOCK_MATERIALS[2]]),
             openmc.Cell(cell_id=3, name="shield", region=+wall & -shield & ~ports,
                         fill=materials[BLOCK_MATERIALS[3]]),
             openmc.Cell(cell_id=4, name="air", region=+shield & -outer,
                         fill=materials[BLOCK_MATERIALS[4]])]
    cells += [openmc.Cell(cell_id=PORT_CELL_OFFSET + 1 + k, name=f"port_{k}", region=region)
              for k, region in enumerate(port_regions)]
    return openmc.Geometry(cells), outer


def symmetric_angles(polar, azimuth):
    """The 8 (polar, azimuth) pairs equivalent to one by the symmetry of the chamber."""
    azimuths = sorted({azimuth % 360.0, (180.0 - azimuth) % 360.0,
                       (180.0 + azimuth) % 360.0, (360.0 - azimuth) % 360.0})
    return sorted({polar, 180.0 - polar}), azimuths


def angular_bins(centres, half_width):
    """Edges of bins of a half-width around sorted centres, in radians, with the gaps."""
    edges = []
    for centre in centres:
        edges += [centre - half_width, centre + half_width]
    return np.radians(edges)


def line_meshes():
    """Spherical meshes of the line-outs: name -> (mesh, abscissa, info for the reduction)."""
    meshes = {}
    for name, polar, azimuth in (("port", 23.5, 45.0), ("shield", 70.0, 65.0)):
        polars, azimuths = symmetric_angles(polar, azimuth)
        # The azimuthal half-width is widened by 1 / sin(polar) to keep the bins about as wide
        # as they are tall.
        width = LINE_HALF_WIDTH / math.sin(math.radians(polar))
        mesh = openmc.SphericalMesh(
            r_grid=np.arange(0.0, geometry.R_AIR + 0.5 * LINE_DR, LINE_DR),
            theta_grid=angular_bins(polars, LINE_HALF_WIDTH),
            phi_grid=angular_bins(azimuths, width), name=f"line_{name}")
        meshes[name] = mesh
    polars, azimuths = symmetric_angles(23.5, 45.0)
    upper = np.arange(23.5 - 12.0, 23.5 + 12.0 + 0.5 * ARC_DTHETA, ARC_DTHETA)
    theta = np.radians(np.concatenate([upper, 180.0 - upper[::-1]]))
    width = LINE_HALF_WIDTH / math.sin(math.radians(23.5))
    meshes["arc"] = openmc.SphericalMesh(
        r_grid=[ARC_RADIUS - ARC_HALF_THICKNESS, ARC_RADIUS + ARC_HALF_THICKNESS],
        theta_grid=theta, phi_grid=angular_bins(azimuths, width), name="line_arc")
    return meshes


def map_mesh():
    """Cylindrical mesh of the (r, z) map in the wedges of azimuth 45 + k 90 degrees."""
    azimuths = [45.0, 135.0, 225.0, 315.0]
    nr = round(geometry.R_AIR / MAP_DR)
    nz = round(2 * geometry.R_AIR / MAP_DZ)
    return openmc.CylindricalMesh(
        r_grid=np.linspace(0.0, geometry.R_AIR, nr + 1),
        z_grid=np.linspace(-geometry.R_AIR, geometry.R_AIR, nz + 1),
        phi_grid=angular_bins(azimuths, MAP_HALF_WIDTH), name="map")


def source():
    return openmc.IndependentSource(space=openmc.stats.Point((0.0, 0.0, 0.0)),
                                    angle=openmc.stats.Isotropic(),
                                    energy=openmc.stats.Discrete([SOURCE_ENERGY], [1.0]))


def settings(particles, batches, seed):
    result = openmc.Settings()
    result.run_mode = "fixed source"
    result.particles, result.batches, result.seed = particles, batches, seed
    result.temperature = {"method": "nearest", "tolerance": 10.0}
    result.output = {"tallies": False}
    result.source = source()
    return result


def build_model(particles, batches, seed):
    materials = build_materials()
    geometry_model, outer = build_geometry(materials)
    model_settings = settings(particles, batches, seed)
    model_settings.statepoint = {"batches": list(range(1, batches + 1))}

    cells = sorted(geometry_model.get_all_cells())
    # Source neutrons are born in the cavity, cell 1.
    uncollided = [openmc.CellBornFilter([1]), openmc.CollisionFilter([0])]
    energy = openmc.EnergyFilter(GROUP_EDGES_EV)
    tallies = openmc.Tallies()
    for suffix, extra in (("", []), ("_uncollided", uncollided)):
        regions = openmc.Tally(name="regions" + suffix)
        regions.filters = [openmc.CellFilter(cells), energy] + extra
        regions.scores = ["flux"] + list(REACTIONS)
        regions.estimator = "tracklength"
        leakage = openmc.Tally(name="leakage" + suffix)
        leakage.filters = [openmc.SurfaceFilter([outer]), energy] + extra
        leakage.scores = ["current"]
        tallies += [regions, leakage]
        for name, mesh in line_meshes().items():
            line = openmc.Tally(name=f"line_{name}" + suffix)
            line.filters = [openmc.MeshFilter(mesh)] + extra
            line.scores = ["flux"]
            line.estimator = "tracklength"
            tallies.append(line)
        flux_map = openmc.Tally(name="map" + suffix)
        flux_map.filters = [openmc.MeshFilter(map_mesh())] + extra
        flux_map.scores = ["flux"]
        flux_map.estimator = "tracklength"
        tallies.append(flux_map)
    return openmc.Model(geometry_model, openmc.Materials(materials.values()), model_settings,
                        tallies)


def batch_samples(run_dir, batches, name):
    """Return the result of each batch of a tally, as an array [batch, bins...]."""
    previous, samples = 0.0, []
    # The statepoint names are padded to the number of digits of the batches of the run.
    width = next(w for w in range(1, 8) if (run_dir / f"statepoint.{1:0{w}d}.h5").exists())
    for batch in range(1, batches + 1):
        with openmc.StatePoint(run_dir / f"statepoint.{batch:0{width}d}.h5") as sp:
            total = np.array(sp.get_tally(name=name).sum)
        samples.append(total - previous)
        previous = total
    return np.array(samples)


def summarize(samples, axis=0):
    """Return the mean and the standard error of the mean of batch samples."""
    mean = samples.mean(axis=axis)
    error = samples.std(axis=axis, ddof=1) / math.sqrt(samples.shape[axis])
    return mean, error


def spherical_volumes(mesh):
    """Volumes [r, theta, phi] of the bins of a spherical mesh."""
    r, theta, phi = (np.asarray(g) for g in (mesh.r_grid, mesh.theta_grid, mesh.phi_grid))
    dr3 = np.diff(r**3) / 3.0
    dcos = -np.diff(np.cos(theta))
    return dr3[:, None, None] * dcos[None, :, None] * np.diff(phi)[None, None, :]


def line_values(samples, mesh, name):
    """Average flux of the 8 equivalent bins of a line, for each batch: [batch, points]."""
    nr, nt, nphi = (len(g) - 1 for g in (mesh.r_grid, mesh.theta_grid, mesh.phi_grid))
    # OpenMC orders spherical mesh bins with r fastest, then theta, then phi.
    values = samples.reshape(len(samples), nphi, nt, nr).transpose(0, 3, 2, 1)
    values = values / spherical_volumes(mesh)[None]
    # The bins of the lines are the even ones in theta and phi; the odd ones are the gaps.
    phis = slice(0, nphi, 2)
    if name == "arc":
        # theta bins: the upper half then the mirrored lower half, both on the arc.
        upper = values[:, 0, : nt // 2, phis]
        lower = values[:, 0, nt // 2 + 1:, phis][:, ::-1]
        return 0.5 * (upper + lower).mean(axis=2)
    lines = values[:, :, 0:nt:2, phis]
    return lines.mean(axis=(2, 3))


def collect(run_dir, batches):
    """Return the results of a completed run."""
    model_meshes = line_meshes()
    report = {"batches": batches, "group_edges_eV": GROUP_EDGES_EV}
    groups = len(GROUP_EDGES_EV) - 1
    scores = ["flux"] + list(REACTIONS)
    parts = {}
    for name in ["regions", "leakage", "map"] + [f"line_{n}" for n in model_meshes]:
        total = batch_samples(run_dir, batches, name)
        uncollided = batch_samples(run_dir, batches, name + "_uncollided").reshape(total.shape)
        parts[name] = {"total": total, "uncollided": uncollided,
                       "collided": total - uncollided}
    cell_ids = sorted(build_geometry(build_materials())[0].get_all_cells())
    blocks = np.array([block_of_cell(c) for c in cell_ids])
    for part in ("total", "uncollided", "collided"):
        # regions: [batch, cell, group, score], summed into the blocks. Group 0 of OpenMC is
        # the lowest energy.
        by_cell = parts["regions"][part].reshape(batches, len(cell_ids), groups, len(scores))
        regions = np.stack([by_cell[:, blocks == b].sum(axis=1) for b in geometry.BLOCKS],
                           axis=1)[:, :, ::-1, :]
        entry = {}
        for i, block in enumerate(geometry.BLOCKS):
            values = {}
            for k, score in enumerate(scores):
                key = "flux" if score == "flux" else REACTIONS[score]
                mean, error = summarize(regions[:, i, :, k].sum(axis=1))
                values[key] = [float(mean), float(error)]
                mean, error = summarize(regions[:, i, :, k])
                values[key + "_groups"] = [mean.tolist(), error.tolist()]
            entry[geometry.BLOCKS[block]] = values
        # The outer sphere has no inflow, so its net current is the leakage.
        leakage = parts["leakage"][part].reshape(batches, groups)[:, ::-1]
        mean, error = summarize(leakage.sum(axis=1))
        entry["leakage"] = [float(mean), float(error)]
        mean, error = summarize(leakage)
        entry["leakage_groups"] = [mean.tolist(), error.tolist()]
        lines = {}
        for name, mesh in model_meshes.items():
            mean, error = summarize(line_values(parts[f"line_{name}"][part], mesh, name))
            lines[name] = [mean.tolist(), error.tolist()]
        entry["lines"] = lines
        mesh = map_mesh()
        nr, nphi, nz = len(mesh.r_grid) - 1, len(mesh.phi_grid) - 1, len(mesh.z_grid) - 1
        # Cylindrical mesh bins: r fastest, then phi, then z. Volumes of the wedges.
        flux_map = parts["map"][part].reshape(batches, nz, nphi, nr)[:, :, 0::2, :]
        r = np.asarray(mesh.r_grid)
        volume = (0.5 * np.diff(r**2) * math.radians(2 * MAP_HALF_WIDTH) * MAP_DZ)
        mean, error = summarize(flux_map.mean(axis=2) / volume[None, None, :])
        entry["map"] = [mean.tolist(), error.tolist()]
        report[part] = entry
    report["line_abscissae"] = {
        "port": (0.5 * (np.asarray(model_meshes["port"].r_grid[1:])
                        + np.asarray(model_meshes["port"].r_grid[:-1]))).tolist(),
        "shield": (0.5 * (np.asarray(model_meshes["shield"].r_grid[1:])
                          + np.asarray(model_meshes["shield"].r_grid[:-1]))).tolist(),
        "arc": (np.arange(-12.0, 12.0, ARC_DTHETA) + 0.5 * ARC_DTHETA).tolist()}
    report["map_grid"] = {"r": list(map(float, map_mesh().r_grid)),
                          "z": list(map(float, map_mesh().z_grid))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--particles", type=int, default=10000000, help="Histories per batch")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--threads", type=int, default=None, help="OpenMP threads")
    parser.add_argument("--export-only", action="store_true",
                        help="Write the OpenMC XML input without running it")
    parser.add_argument("--collect-only", type=int, default=None, metavar="N",
                        help="Summarize the first N batches of an existing run, without running")
    args = parser.parse_args()

    run_dir = Path(f"openmc_seed{args.seed}")
    if args.collect_only:
        args.batches, seconds = args.collect_only, None
    else:
        model = build_model(args.particles, args.batches, args.seed)
        run_dir.mkdir(exist_ok=True)
        model.export_to_xml(directory=run_dir)
        if args.export_only:
            return
        start = time.perf_counter()
        openmc.run(cwd=run_dir, threads=args.threads)
        seconds = time.perf_counter() - start
    report = collect(run_dir, args.batches)
    report.update(particles_per_batch=args.particles, seed=args.seed, threads=args.threads,
                  seconds=seconds, openmc_version=openmc.__version__)
    name = f"results_{args.batches}.json" if args.collect_only else "results.json"
    (run_dir / name).write_text(json.dumps(report) + "\n")

    print(f"{'Block':17s} {'Total flux [cm]':>24s} {'Uncollided flux [cm]':>24s}")
    for block in geometry.BLOCKS.values():
        total, uncollided = report["total"][block]["flux"], report["uncollided"][block]["flux"]
        print(f"{block:17s} {total[0]:12.6e} +- {total[1]:7.1e} "
              f"{uncollided[0]:12.6e} +- {uncollided[1]:7.1e}")
    leakage = report["total"]["leakage"]
    print(f"Leakage per source neutron: {leakage[0]:.6e} +- {leakage[1]:.1e}")
    print(f"Wrote {run_dir / name}" + (f" ({seconds:.0f} s)" if seconds else ""))


if __name__ == "__main__":
    main()
