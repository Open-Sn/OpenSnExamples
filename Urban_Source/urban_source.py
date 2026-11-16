#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Gamma-ray transport from a Cs-137 point source in a city.

The city of city.toml is a lattice of streets, plazas, a park, and homogenized concrete
buildings of several shapes and heights, on a ground slab, in air. A Cs-137 source 1 m
above a street emits the 661.657 keV gamma ray of Ba-137m isotropically. The outer
boundary is vacuum. The photons are transported with 25 energy groups from 670 keV down to
10 keV, below which they are absorbed.

The point source is treated with OpenSn's uncollided-flux method in two stages. The first
stage computes the uncollided flux by ray tracing from the source and writes it to
urban_source_uncollided.h5. It runs on one MPI rank. The second stage solves for the
collided flux with the discrete-ordinates method, using the first collisions of the
uncollided flux as its source, and adds the uncollided flux back to the result. It runs on
any number of ranks. This input runs the first stage if urban_source_uncollided.h5 does
not exist and the second stage otherwise:

    gunzip -k urban_source.msh.gz
    OPENSN_NUM_THREADS=32 opensn -i urban_source.py
    export OPENSN_NUM_THREADS=1
    mpirun -n 24 opensn -i urban_source.py

With the supplied mesh and the default settings, the first stage takes a few minutes on 32
threads, and the second stage about 85 minutes on 24 ranks with 230 GB of memory.

The settings below can be changed without editing this file with the -p option of the
console, for example -p "scattering_order=5". Use the same mesh_file, library,
scattering_order, and prefix in both stages.

Outputs, all per source photon:
    The first stage writes the uncollided flux and dose in each region to
    urban_source_uncollided.json. The second stage prints the particle balance, the
    flux and dose in each region, and the largest dose rates on the dose map, writes them,
    with the group spectra and the leakage spectrum, to urban_source_results.json, and
    writes the scalar flux in every group to urban_source_flux.pvtu.
"""

import json
import math
import os

import numpy as np

if "opensn_console" not in globals():
    from mpi4py import MPI

    size, rank = MPI.COMM_WORLD.size, MPI.COMM_WORLD.rank
    from pyopensn.aquad import GLCProductQuadrature3DXYZ, SLDFEsqQuadrature3DXYZ
    from pyopensn.fieldfunc import FieldFunctionGridBased
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.math import Vector3
    from pyopensn.mesh import DistributedMeshGenerator, FromFileMeshGenerator
    from pyopensn.post import VolumePostprocessor
    from pyopensn.solver import (
        DiscreteOrdinatesProblem,
        SteadyStateSourceSolver,
        UncollidedProblem,
        UncollidedSolver,
    )
    from pyopensn.source import PointSource
    from pyopensn.xs import MultiGroupXS

# Settings. Each can be set with the -p option of the console, for example
# -p "scattering_order=5".
mesh_file = globals().get("mesh_file", "urban_source.msh")
library = globals().get("library", "materials_25g_p7.h5")
scattering_order = globals().get("scattering_order", 3)
# Angular quadrature for the collided flux: "sldfesq", the piecewise-linear quadrature of
# the given level whose directions within horizon_cone_deg of the horizon are refined
# horizon_refinements times, or "glc", the Gauss-Legendre-Chebyshev product quadrature with
# n_polar x n_azimuthal directions. The default, level 1 refined once within 10 degrees of
# the horizon, has 576 directions.
quadrature_type = globals().get("quadrature_type", "sldfesq")
n_polar = globals().get("n_polar", 8)
n_azimuthal = globals().get("n_azimuthal", 32)
sldfesq_level = globals().get("sldfesq_level", 1)
horizon_refinements = globals().get("horizon_refinements", 1)
horizon_cone_deg = globals().get("horizon_cone_deg", 10.0)
groups_per_groupset = globals().get("groups_per_groupset", 5)
tolerance = globals().get("tolerance", 1.0e-7)
# GMRES keeps up to this many Krylov vectors per groupset. Fewer save memory and cost
# iterations.
gmres_restart_interval = globals().get("gmres_restart_interval", 5)
# Half width in cm of the cube around the point source in which the uncollided flux is ray
# traced. The default, 0, ray traces it in the whole domain.
near_source_half_width = globals().get("near_source_half_width", 0.0)
prefix = globals().get("prefix", "urban_source")
settings = {"mesh_file": mesh_file, "library": library, "scattering_order": scattering_order,
            "quadrature_type": quadrature_type, "n_polar": n_polar,
            "n_azimuthal": n_azimuthal, "sldfesq_level": sldfesq_level,
            "horizon_refinements": horizon_refinements, "horizon_cone_deg": horizon_cone_deg,
            "groups_per_groupset": groups_per_groupset, "tolerance": tolerance,
            "gmres_restart_interval": gmres_restart_interval,
            "near_source_half_width": near_source_half_width, "prefix": prefix}

uncollided_file = f"{prefix}_uncollided.h5"
uncollided_results = f"{prefix}_uncollided.json"
dose_name = "dose_ap"
hours = 3600.0

with open(mesh_file.removesuffix(".msh") + "_mesh.json") as stream:
    description = json.load(stream)
blocks = {int(block): entry for block, entry in description["blocks"].items()}
source_position = description["source_cm"]
# Source photons per second: the activity times the 661.657 keV photons per decay.
photons_per_second = description["activity_Bq"] * description["photons_per_decay"]

# Mesh. Rank 0 reads the Gmsh file and distributes the cells to the other ranks.
grid = DistributedMeshGenerator(inputs=[FromFileMeshGenerator(filename=mesh_file)]).Execute()

# Materials. Each block gets the macroscopic cross sections of its material from the
# 25-group OpenMC library, with the AP effective-dose coefficients [pSv cm^2].
xs_map, materials = [], {}
for block, entry in blocks.items():
    materials.setdefault(entry["material"], []).append(block)
for name, block_ids in materials.items():
    xs = MultiGroupXS()
    xs.LoadFromOpenMC(library, name, 294.0, [dose_name])
    xs_map.append({"block_ids": block_ids, "xs": xs})
    num_groups = xs.num_groups
    dose_coefficients = np.array(xs.get_custom_xs(dose_name))


def region_values(problem):
    """Return the group flux [cm] and the dose integral [pSv cm^3] of every block."""
    values = {}
    for block, entry in blocks.items():
        result = {}
        for quantity, options in (("flux", {}), ("dose", {"xs_multiplier": dose_name})):
            post = VolumePostprocessor(problem=problem, block_ids=[block],
                                       value_type="integral", **options)
            post.Execute()
            result[quantity] = np.asarray(post.GetValue()).reshape(-1, num_groups).sum(axis=0)
        values[entry["name"]] = result
    return values


def dose_map(problem):
    """Return the average dose [pSv] in every box of the dose map, as [y, x].

    Each box average is a 2 x 2 x 2 Gauss-Legendre rule over the box.
    """
    x_edges = np.array(description["dose_map_cm"]["x_edges"])
    y_edges = np.array(description["dose_map_cm"]["y_edges"])
    z_range = description["dose_map_cm"]["z_range"]
    offsets = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    xc, yc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])
    zc = 0.5 * (z_range[0] + z_range[1])
    hx, hy, hz = 0.5 * np.diff(x_edges), 0.5 * np.diff(y_edges), 0.5 * (z_range[1] - z_range[0])
    points = []
    for j in range(len(yc)):
        for i in range(len(xc)):
            for a in offsets:
                for b in offsets:
                    for c in offsets:
                        points.append(Vector3(xc[i] + a * hx[i], yc[j] + b * hy[j],
                                              zc + c * hz))
    fields = problem.GetScalarFluxFieldFunction()
    values = FieldFunctionGridBased.GetPointValues(fields, points)
    flux = np.array([[point[0] for point in field] for field in values])  # [group, point]
    dose = (dose_coefficients @ flux).reshape(len(yc), len(xc), 8).mean(axis=2)
    return dose


def compute_uncollided():
    """First stage: compute and write the uncollided flux on one rank."""
    # Group 0 (650 to 670 keV) holds the 661.657 keV source. The uncollided flux is ray
    # traced from the source to the quadrature points of every cell of the near-source
    # region, and outside it computed by a sweep along the directions from the source. The
    # buildings are many mean free paths thick, and the sweep cannot attenuate the flux
    # across their coarse cells, so by default the near-source region is the whole domain.
    # The cells around the source cell are integrated exactly over direction.
    source = PointSource(location=source_position,
                         strength=[1.0] + [0.0] * (num_groups - 1))
    if near_source_half_width > 0.0:
        x, y, z = source_position
        h = near_source_half_width
        near_source = RPPLogicalVolume(xmin=x - h, xmax=x + h, ymin=y - h, ymax=y + h,
                                       zmin=z - h, zmax=z + h)
    else:
        near_source = RPPLogicalVolume(infx=True, infy=True, infz=True)
    problem = UncollidedProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[{"groups_from_to": (0, num_groups - 1)}],
        xs_map=xs_map,
        point_sources=[source],
        near_source=[near_source],
        exact_near_source_leakage=True,
        scattering_order=scattering_order,
    )
    solver = UncollidedSolver(problem=problem, file_name=uncollided_file)
    solver.Initialize()
    solver.Execute()
    # The uncollided dose map is not evaluated here. uncollided_reference.py computes it
    # semi-analytically.
    values = region_values(problem)
    with open(uncollided_results, "w") as stream:
        json.dump({"regions": {name: {k: v.tolist() for k, v in entry.items()}
                               for name, entry in values.items()}}, stream)
    print(f"Wrote {uncollided_file}. Run this input again, on any number of ranks, "
          "to compute the collided flux.")


def compute_collided():
    """Second stage: solve for the collided flux on any number of ranks."""
    with open(uncollided_results) as stream:
        uncollided = json.load(stream)

    if quadrature_type == "glc":
        quadrature = GLCProductQuadrature3DXYZ(n_polar=n_polar, n_azimuthal=n_azimuthal,
                                               scattering_order=scattering_order)
    elif quadrature_type == "sldfesq":
        quadrature = SLDFEsqQuadrature3DXYZ(level=sldfesq_level,
                                            scattering_order=scattering_order)
        for _ in range(horizon_refinements):
            quadrature.LocallyRefine(Vector3(0.0, 0.0, 1.0), math.radians(horizon_cone_deg),
                                     True)
    else:
        raise ValueError(f"Unknown quadrature_type {quadrature_type!r}")

    # Compton scattering only lowers the photon energy, and coherent scattering keeps it in
    # its group, so there is no upscattering. The groupsets are solved in order of
    # decreasing energy, and one pass over them converges the scattering between groups.
    groupsets = []
    for first in range(0, num_groups, groups_per_groupset):
        groupsets.append({
            "groups_from_to": (first, min(first + groups_per_groupset, num_groups) - 1),
            "angular_quadrature": quadrature,
            "inner_linear_method": "petsc_gmres",
            "l_abs_tol": tolerance,
            "l_max_its": 300,
            "gmres_restart_interval": gmres_restart_interval,
        })

    # The first collisions of the uncollided flux are the only source of this problem.
    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=groupsets,
        xs_map=xs_map,
        uncollided_flux=uncollided_file,
    )
    solver = SteadyStateSourceSolver(problem=problem)
    solver.Initialize()
    solver.Execute()

    # Particle balance. The production rate is the source strength, 1 photon/s.
    # ComputeLeakage returns the collided outflow in each group. The balance table also
    # includes the uncollided outflow, which is all in group 0.
    balance = solver.ComputeBalanceTable()
    source_rate = balance["production_rate"]
    collided_leakage = sum(np.asarray(v) for v in problem.ComputeLeakage([]).values())
    leakage = collided_leakage.copy()
    leakage[0] += balance["outflow_rate"] - collided_leakage.sum()
    values = region_values(problem)
    total_map = dose_map(problem)
    if rank == 0:
        to_rate = photons_per_second * hours * 1.0e-6  # pSv per photon -> uSv/h
        print(f"Converged: {solver.IsConverged()}")
        print(f"Source rate [photon/s]: {source_rate:.10e}")
        print(f"Relative balance: {balance['balance'] / source_rate:.3e}")
        print(f"Leakage per source photon: {leakage.sum() / source_rate:.10e}")
        print(f"Collided leakage per source photon: "
              f"{collided_leakage.sum() / source_rate:.6e}")
        print(f"{'Region':18s} {'Flux [cm]':>13s} {'Collided':>13s} "
              f"{'Dose [pSv cm3]':>15s} {'Collided':>13s}")
        for name, entry in values.items():
            flux, dose = entry["flux"].sum(), entry["dose"].sum()
            unc = uncollided["regions"][name]
            row = [flux, flux - sum(unc["flux"]), dose, dose - sum(unc["dose"])]
            print(f"{name:18s} " + " ".join(f"{v / source_rate:13.6e}" for v in row))
        rates = total_map / source_rate * to_rate
        j, i = np.unravel_index(np.argmax(rates), rates.shape)
        print(f"Dose-rate map: maximum {rates[j, i]:.4e} uSv/h in box (row {j}, column {i}), "
              f"median {np.median(rates):.4e} uSv/h for the source activity "
              f"{description['activity_Bq']:.4g} Bq")
        results = {
            "converged": bool(solver.IsConverged()),
            "settings": settings,
            "num_groups": num_groups,
            "dose_coefficients": dose_coefficients.tolist(),
            "source_rate": source_rate,
            "balance": {k: float(v) for k, v in balance.items()},
            "leakage_groups": (leakage / source_rate).tolist(),
            "collided_leakage_groups": (collided_leakage / source_rate).tolist(),
            "total": {name: {k: (v / source_rate).tolist() for k, v in entry.items()}
                      for name, entry in values.items()},
            "uncollided": uncollided["regions"],
            "dose_map_total": (total_map / source_rate).tolist(),
        }
        with open(f"{prefix}_results.json", "w") as stream:
            json.dump(results, stream)

    # Scalar flux in every group, for visualization in ParaView.
    FieldFunctionGridBased.ExportMultipleToPVTU(problem.GetScalarFluxFieldFunction(),
                                                f"{prefix}_flux")


if os.path.exists(uncollided_file) and os.path.exists(uncollided_results):
    compute_collided()
elif size == 1:
    compute_uncollided()
else:
    raise RuntimeError(f"{uncollided_file} does not exist. Run this input on one rank "
                       "first to compute the uncollided flux.")
