#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Continuous-energy OpenMC reference for the urban source example.

This script builds the city of city.toml from analytic surfaces, runs OpenMC with the same
Cs-137 point source as urban_source.py, and writes to results.json the flux and dose in
each region, the dose map, and the leakage, per source photon. Lengths are in cm.

OpenMC transports the photons with continuous-energy ENDF/B-VIII.0 photon data. As in the
group cross sections of generate_xs.py, photons below 10 keV are killed, and the electrons
that they set in motion deposit their energy where they are born, so no bremsstrahlung is
produced. Below the 1.022 MeV threshold of pair production, the only secondary photons are
fluorescence x rays, which in these materials are all below 10 keV. Every photon tracked
is therefore a source photon, and a photon with no collision is uncollided.

OpenMC tallies every quantity twice: for all photons and for photons that have not
collided. A statepoint is written after every batch, so each batch gives one sample of the
collided part, the difference of the two. The standard errors of the collided results come
from these batch samples and include the correlation between the two tallies.

The dose is the ICRP Publication 116 effective dose for the anterior-posterior (AP)
irradiation geometry, interpolated log-log in energy.

OpenMC needs the ENDF/B-VIII.0 HDF5 library of the OpenMC project. Set
OPENMC_CROSS_SECTIONS to its cross_sections.xml file, then run, for example:

    python3 openmc_reference.py --processes 8
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import openmc
import openmc.data

import city as citylib

CM = 100.0  # cm per m
TEMPERATURE = 294.0  # K. Photon interaction data do not depend on temperature.
SOURCE_ENERGY = citylib.CS137_GAMMA_ENERGY  # eV
PHOTON_CUTOFF = 1.0e4  # eV
DOSE_GEOMETRY = "AP"
# Cell IDs of the ground cells start here. Each belongs to the ground block of its material.
FIRST_GROUND_CELL = 1001
LIBRARY = "materials_25g_p7.h5"


def build_materials(city):
    """Return the OpenMC materials of the layout, by name."""
    materials = {}
    for index, (name, spec) in enumerate(city.materials.items()):
        material = openmc.Material(material_id=index + 1, name=name, temperature=TEMPERATURE)
        for element, fraction in spec["weight_fractions"].items():
            material.add_element(element, fraction, "wo")
        material.set_density("g/cm3", spec["density"])
        materials[name] = material
    return materials


class SurfaceCache:
    """Planes shared by all the regions that they bound.

    Two surfaces that coincide only to round-off can trap a particle that crosses them. The
    edges of concave footprints that are shared by two of their triangles, collinear edges of
    different buildings, and the roofs of buildings of equal height are therefore each one
    surface, used with whichever sense each region needs.
    """

    def __init__(self):
        self.planes = {}
        self.heights = {}

    def edge(self, a, b):
        """Return the region on the inner side of the edge a -> b of a counterclockwise
        polygon, unbounded in z."""
        normal = np.array([b[1] - a[1], a[0] - b[0]])  # Outward
        normal /= np.linalg.norm(normal)
        offset = float(normal @ a)
        # The same line has the normal of either sign; key it by one of them.
        sign = 1.0 if (normal[0], normal[1]) > (0.0, 0.0) else -1.0
        key = tuple(np.round(sign * np.array([normal[0], normal[1], offset]), 9))
        if key not in self.planes:
            self.planes[key] = openmc.Plane(a=sign * normal[0], b=sign * normal[1], c=0.0,
                                            d=sign * offset)
        plane = self.planes[key]
        return -plane if sign > 0 else +plane

    def height(self, z):
        if z not in self.heights:
            self.heights[z] = openmc.ZPlane(z)
        return self.heights[z]


def polygon_region(vertices, surfaces):
    """Return the region inside a convex counterclockwise polygon, unbounded in z."""
    region = openmc.Intersection([])
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        region &= surfaces.edge(a, b)
    return region


def footprint_region(footprint, surfaces):
    """Return the region inside a footprint, unbounded in z, from analytic surfaces."""
    if footprint.kind == "polygon":
        # A convex footprint is the intersection of the inner sides of its edges. A concave one
        # is the union of the triangles that city.py splits it into.
        vertices = footprint.vertices * CM
        pieces = [vertices] if citylib.is_convex(footprint.vertices) else [
            piece * CM for piece in citylib.triangulate(footprint.vertices)]
        regions = [polygon_region(piece, surfaces) for piece in pieces]
        return regions[0] if len(regions) == 1 else openmc.Union(regions)
    cx, cy = footprint.center * CM
    a, b = footprint.a * CM, footprint.b * CM
    if footprint.kind == "circle":
        return -openmc.ZCylinder(x0=cx, y0=cy, r=a)
    # Ellipse: (u/a)^2 + (v/b)^2 < 1, with u and v along the rotated semi-axes.
    c, s = math.cos(footprint.angle), math.sin(footprint.angle)
    qa = c * c / a**2 + s * s / b**2
    qb = s * s / a**2 + c * c / b**2
    qd = 2 * c * s * (1 / a**2 - 1 / b**2)
    return -openmc.Quadric(a=qa, b=qb, c=0.0, d=qd, e=0.0, f=0.0,
                           g=-2 * qa * cx - qd * cy, h=-2 * qb * cy - qd * cx, j=0.0,
                           k=qa * cx * cx + qb * cy * cy + qd * cx * cy - 1.0)


def build_geometry(city, materials):
    """Return the geometry, the vacuum surfaces with their outward signs, and the block of
    every cell. Air and building cells have the ID of their block."""
    xmin, xmax, ymin, ymax, zmin, zmax = (v * CM for v in city.extent)
    x_planes = [openmc.XPlane(x * CM) for x in city.x_edges]
    y_planes = [openmc.YPlane(y * CM) for y in city.y_edges]
    for plane in (x_planes[0], x_planes[-1], y_planes[0], y_planes[-1]):
        plane.boundary_type = "vacuum"
    bottom = openmc.ZPlane(zmin, boundary_type="vacuum")
    top = openmc.ZPlane(zmax, boundary_type="vacuum")
    ground_surface = openmc.ZPlane(0.0)
    column = {x: i for i, x in enumerate(city.x_edges)}
    row = {y: i for i, y in enumerate(city.y_edges)}

    cells, cell_blocks = [], {}
    building_regions = []
    surfaces = SurfaceCache()
    for building in city.buildings:
        region = (footprint_region(building.footprint, surfaces) & +ground_surface
                  & -surfaces.height(building.height * CM))
        building_regions.append(region)
        cells.append(openmc.Cell(cell_id=building.block, name=building.name, region=region,
                                 fill=materials[building.material]))
        cell_blocks[building.block] = building.block
    for index, ground in enumerate(city.ground):
        x0, x1, y0, y1 = ground.bounds
        region = (+x_planes[column[x0]] & -x_planes[column[x1]] & +y_planes[row[y0]]
                  & -y_planes[row[y1]] & +bottom & -ground_surface)
        cell_id = FIRST_GROUND_CELL + index
        cells.append(openmc.Cell(cell_id=cell_id, name=f"ground_r{ground.cell[0]}c{ground.cell[1]}",
                                 region=region, fill=materials[ground.material]))
        cell_blocks[cell_id] = ground.block
    air = (+x_planes[0] & -x_planes[-1] & +y_planes[0] & -y_planes[-1] & +ground_surface
           & -top)
    for region in building_regions:
        air &= ~region
    cells.append(openmc.Cell(cell_id=citylib.AIR_BLOCK, name="air", region=air,
                             fill=materials["air"]))
    cell_blocks[citylib.AIR_BLOCK] = citylib.AIR_BLOCK
    # OpenMC's current is positive along the normal of a surface, which points outward on
    # the east, north, and top boundaries and inward on the west, south, and bottom ones.
    boundaries = [x_planes[0], x_planes[-1], y_planes[0], y_planes[-1], bottom, top]
    signs = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    return openmc.Geometry(cells), boundaries, signs, cell_blocks


def dose_filter():
    energy, coefficient = openmc.data.dose_coefficients("photon", DOSE_GEOMETRY)
    return openmc.EnergyFunctionFilter(energy, coefficient, interpolation="log-log")


def build_model(city, particles, batches, seed, group_edges):
    materials = build_materials(city)
    geometry, boundaries, _, _ = build_geometry(city, materials)

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles, settings.batches, settings.seed = particles, batches, seed
    settings.photon_transport = True
    settings.electron_treatment = "led"
    settings.cutoff = {"energy_photon": PHOTON_CUTOFF}
    settings.temperature = {"method": "nearest", "tolerance": 1000.0}
    settings.output = {"tallies": False}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(tuple(city.source * CM)), angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([SOURCE_ENERGY], [1.0]), particle="photon")
    settings.statepoint = {"batches": list(range(1, batches + 1))}

    photon = openmc.ParticleFilter(["photon"])
    energy = openmc.EnergyFilter(group_edges)
    dose = dose_filter()
    x_edges, y_edges, z_range = city.dose_map_boxes()
    mesh = openmc.RectilinearMesh()
    mesh.x_grid, mesh.y_grid = x_edges * CM, y_edges * CM
    mesh.z_grid = np.array(z_range) * CM
    cells = list(geometry.get_all_cells().values())
    tallies = openmc.Tallies()
    for suffix, extra in (("", []), ("_uncollided", [openmc.CollisionFilter([0])])):
        regions = openmc.Tally(name="regions" + suffix)
        regions.filters = [photon, openmc.CellFilter(cells), energy] + extra
        regions.scores = ["flux"]
        regions_dose = openmc.Tally(name="regions_dose" + suffix)
        regions_dose.filters = [photon, openmc.CellFilter(cells), dose] + extra
        regions_dose.scores = ["flux"]
        leakage = openmc.Tally(name="leakage" + suffix)
        leakage.filters = [photon, openmc.SurfaceFilter(boundaries), energy] + extra
        leakage.scores = ["current"]
        dose_map = openmc.Tally(name="dose_map" + suffix)
        dose_map.filters = [photon, openmc.MeshFilter(mesh), dose] + extra
        dose_map.scores = ["flux"]
        tallies += [regions, regions_dose, leakage, dose_map]
    return openmc.Model(geometry, openmc.Materials(materials.values()), settings, tallies)


def batch_samples(run_dir, batches, name):
    """Return the result of each batch of a tally, as an array [batch, bins...]."""
    previous, samples = 0.0, []
    width = len(str(batches))
    for batch in range(1, batches + 1):
        with openmc.StatePoint(run_dir / f"statepoint.{batch:0{width}d}.h5") as sp:
            total = np.array(sp.get_tally(name=name).sum)
        samples.append(total - previous)
        previous = total
    return np.array(samples)


def summarize(samples):
    """Return the mean and the standard error of the mean of batch samples."""
    mean = samples.mean(axis=0)
    error = samples.std(axis=0, ddof=1) / math.sqrt(len(samples))
    return mean, error


def collect(city, run_dir, batches, num_groups):
    """Return the results of a completed run, per source photon."""
    materials = build_materials(city)
    geometry, _, signs, cell_blocks = build_geometry(city, materials)
    cell_ids = list(geometry.get_all_cells())
    blocks = list(city.block_materials())
    # Matrix that sums the cells of each block.
    to_block = np.zeros((len(cell_ids), len(blocks)))
    for i, cell_id in enumerate(cell_ids):
        to_block[i, blocks.index(cell_blocks[cell_id])] = 1.0

    report = {"batches": batches, "blocks": {str(b): city.names[b] for b in blocks}}
    parts = {}
    for name in ("regions", "regions_dose", "leakage", "dose_map"):
        total = batch_samples(run_dir, batches, name)
        uncollided = batch_samples(run_dir, batches, name + "_uncollided").reshape(total.shape)
        parts[name] = {"total": total, "uncollided": uncollided,
                       "collided": total - uncollided}
    x_edges, y_edges, _ = city.dose_map_boxes()
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    for part in ("total", "uncollided", "collided"):
        entry = {}
        # regions: [batch, cell, group]. Group 0 of OpenMC is the lowest energy.
        flux = parts["regions"][part].reshape(batches, len(cell_ids), num_groups)[:, :, ::-1]
        flux = np.einsum("bcg,ck->bkg", flux, to_block)
        dose = parts["regions_dose"][part].reshape(batches, len(cell_ids)) @ to_block
        for k, block in enumerate(blocks):
            mean, error = summarize(flux[:, k, :].sum(axis=1))
            groups_mean, groups_error = summarize(flux[:, k, :])
            dose_mean, dose_error = summarize(dose[:, k])
            entry[city.names[block]] = {
                "flux": [float(mean), float(error)],
                "flux_groups": [groups_mean.tolist(), groups_error.tolist()],
                "dose": [float(dose_mean), float(dose_error)]}
        # leakage: [batch, surface, group], made outward.
        leakage = parts["leakage"][part].reshape(batches, len(signs), num_groups)[:, :, ::-1]
        leakage = leakage * np.array(signs)[None, :, None]
        mean, error = summarize(leakage.sum(axis=(1, 2)))
        groups_mean, groups_error = summarize(leakage.sum(axis=1))
        entry["leakage"] = [float(mean), float(error)]
        entry["leakage_groups"] = [groups_mean.tolist(), groups_error.tolist()]
        # dose_map: [batch, box] with x fastest. Divide by the box volume for the average.
        dose_map = parts["dose_map"][part].reshape(batches, ny, nx)
        mean, error = summarize(dose_map)
        entry["dose_map_integral"] = [mean.tolist(), error.tolist()]
        report[part] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, default=Path("city.toml"))
    parser.add_argument("--library", type=Path,
                        default=Path(__file__).resolve().parent / LIBRARY,
                        help="Group library, whose group structure the tallies use")
    parser.add_argument("--particles", type=int, default=10000000, help="Histories per batch")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--processes", type=int, default=1, help="MPI processes for OpenMC")
    parser.add_argument("--threads", type=int, default=None, help="OpenMP threads per process")
    parser.add_argument("--export-only", action="store_true",
                        help="Write the OpenMC XML input without running it")
    parser.add_argument("--collect-only", action="store_true",
                        help="Collect the results of a completed run without running it")
    args = parser.parse_args()

    city = citylib.load(args.layout)
    library = openmc.MGXSLibrary.from_hdf5(str(args.library))
    group_edges = [float(e) for e in library.energy_groups.group_edges]
    run_dir = Path(f"openmc_seed{args.seed}")
    if not args.collect_only:
        model = build_model(city, args.particles, args.batches, args.seed, group_edges)
        run_dir.mkdir(exist_ok=True)
        model.export_to_xml(directory=run_dir)
        if args.export_only:
            return
        mpi_args = ["mpiexec", "-n", str(args.processes)] if args.processes > 1 else None
        openmc.run(cwd=run_dir, mpi_args=mpi_args, threads=args.threads)
    report = collect(city, run_dir, args.batches, len(group_edges) - 1)
    report["particles_per_batch"], report["seed"] = args.particles, args.seed
    report["dose_geometry"] = DOSE_GEOMETRY
    (run_dir / "results.json").write_text(json.dumps(report) + "\n")

    print(f"{'Region':18s} {'Total flux [cm]':>24s} {'Collided flux [cm]':>24s}")
    for name in report["blocks"].values():
        total, collided = report["total"][name]["flux"], report["collided"][name]["flux"]
        print(f"{name:18s} {total[0]:12.6e} +- {total[1]:7.1e} "
              f"{collided[0]:12.6e} +- {collided[1]:7.1e}")
    leakage = report["total"]["leakage"]
    print(f"Leakage per source photon: {leakage[0]:.8e} +- {leakage[1]:.1e}")
    print(f"Wrote {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
