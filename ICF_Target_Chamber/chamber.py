#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Neutron transport in an ICF target chamber: uncollided flux by ray tracing, then collided
flux by discrete ordinates.

The chamber of chamber_geometry.py, a 10 m vacuum sphere with an aluminum wall, a borated
concrete shield, air outside and 52 ports, holds an isotropic 14.1 MeV point source of one
neutron per second at its centre. The model is the octant x, y, z > 0 made by
generate_mesh.py. The source is at its corner, on the three reflecting planes XMIN, YMIN and
ZMIN; OUTER, the sphere of radius 800 cm, is vacuum. The source is given the strength of the
whole source, and OpenSn emits one eighth of it into the octant, so the production rate of
both stages is 1/8 and every result divided by it is per neutron of the whole source, the
value of the whole chamber.

Stages, selected with the argument stage:

- "uncollided": the uncollided flux of the source, ray traced from every cell through the
  material hull of the mesh, with the conservation correction, on any number of ranks. Writes
  <prefix>_uncollided.h5, read by the collided stage, and <prefix>_uncollided.json, with the
  flux and reaction rates in each block and the flux along the line-outs.
- "collided": the collided flux, by discrete ordinates, with the first collisions of the
  uncollided flux as its source; the result is the total flux, collided plus uncollided.
  Writes <prefix>_results.json and <prefix>_flux.pvtu.
- "all" (default): both, one after the other.
- "sn_only": the same problem by discrete ordinates alone, with the point source of strength
  1/8 in the octant and no uncollided flux, to show the ray effects that the uncollided flux
  removes. Writes <prefix>_sn_only.json.

Other arguments (opensn --py name=value):

    mesh_file     mesh of generate_mesh.py (default "chamber_coarse.msh")
    library       "4g_p0" (default), "4g_p0tc" or "64g_p3": materials_<library>.h5
    n_polar, n_azimuthal   Gauss-Legendre-Chebyshev product quadrature (default 6 and 24,
                  144 directions)
    prefix        prefix of the output files (default "chamber_<library>")

For example, on 32 ranks:

    mpiexec -n 32 python3 chamber.py --py mesh_file=\\"chamber_fine.msh\\" \\
        --py library=\\"4g_p0\\"

The 4-group P0 library uses one groupset solved by GMRES. The 64-group P3 library uses two
groupsets, the groups without upscattering and the thermal groups that have it, which share
one quadrature, each solved by GMRES, in one pass over the groupsets: no neutron goes back
from the thermal groupset to the first one. Source iteration is not used: with scattering
ratios up to 0.99 and no diffusion acceleration (the cavity is a void), it converges by about
5% per iteration on this problem.
"""

import json
import os
import sys
import time

import numpy as np

if "opensn_console" not in globals():
    from mpi4py import MPI

    size, rank = MPI.COMM_WORLD.size, MPI.COMM_WORLD.rank
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.fieldfunc import FieldFunctionGridBased
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chamber_geometry as geometry  # noqa: E402

if "mesh_file" not in globals():
    mesh_file = "chamber_coarse.msh"
if "library" not in globals():
    library = "4g_p0"
if "n_polar" not in globals():
    n_polar = 6
if "n_azimuthal" not in globals():
    n_azimuthal = 24
if "stage" not in globals():
    stage = "all"
if "prefix" not in globals():
    prefix = f"chamber_{library}"

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY_FILE = os.path.join(HERE, f"materials_{library}.h5")
TEMPERATURE = 293.6
MATERIALS = {"void": [1], "aluminum": [2], "borated_concrete": [3], "air": [4]}
REACTIONS = ["absorption", "n2n", "ngamma", "nalpha"]
REFLECTING = ["XMIN", "YMIN", "ZMIN"]
BOUNDARY_CONDITIONS = [{"name": name, "type": "reflecting"} for name in REFLECTING]
# The strength of the whole source, of which one eighth is emitted into the octant.
STRENGTH = 1.0
OCTANT_FRACTION = 0.125


def log(message):
    if rank == 0:
        print(message)
        sys.stdout.flush()


def load_cross_sections():
    xs_map = []
    for name, blocks in MATERIALS.items():
        xs = MultiGroupXS()
        xs.LoadFromOpenMC(LIBRARY_FILE, name, TEMPERATURE, REACTIONS)
        xs_map.append({"block_ids": blocks, "xs": xs})
    first = xs_map[1]["xs"]
    return xs_map, first.num_groups, first.scattering_order


def first_upscatter_group(num_groups):
    """First group that receives upscattering, from the provenance of the libraries."""
    with open(os.path.join(HERE, "provenance", "materials.json")) as stream:
        record = json.load(stream)
    key = "materials_64g" if num_groups == 64 else "materials_4g"
    groups = [m["first_upscatter_group"] for m in record[key].values()
              if m["first_upscatter_group"] is not None]
    return min(groups) if groups else num_groups


def region_values(problem, num_groups):
    """Group-wise flux and reaction rates integrated over each block of the octant."""
    values = {}
    for block, name in geometry.BLOCKS.items():
        entry = {}
        for quantity in ["flux"] + REACTIONS:
            options = {} if quantity == "flux" else {"xs_multiplier": quantity}
            post = VolumePostprocessor(problem=problem, block_ids=[block],
                                       value_type="integral", **options)
            post.Execute()
            entry[quantity] = np.asarray(post.GetValue()).reshape(-1, num_groups).sum(axis=0)
        values[name] = entry
    return values


def line_values(problem):
    """Group-wise scalar flux along the line-outs: name -> [points, groups]."""
    fluxes = problem.GetScalarFluxFieldFunction()
    result = {}
    for name, (_, points) in geometry.line_points().items():
        values = FieldFunctionGridBased.GetPointValues(
            fluxes, [Vector3(*map(float, p)) for p in points])
        result[name] = np.array([[v[0] for v in group] for group in values]).T
    return result


def per_source(entries, scale):
    """Region values or lines as lists, scaled to one neutron of the whole source."""
    return {name: ({k: (v * scale).tolist() for k, v in entry.items()}
                   if isinstance(entry, dict) else (entry * scale).tolist())
            for name, entry in entries.items()}


def compute_uncollided(grid, xs_map, num_groups, scattering_order):
    start = time.perf_counter()
    source = PointSource(location=[0.0, 0.0, 0.0],
                         strength=[STRENGTH] + [0.0] * (num_groups - 1))
    problem = UncollidedProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[{"groups_from_to": (0, num_groups - 1)}],
        xs_map=xs_map,
        boundary_conditions=BOUNDARY_CONDITIONS,
        point_sources=[source],
        method="ray_trace",
        ray_tracer="material_hull",
        exact_near_source_leakage=True,
        scattering_order=scattering_order,
    )
    solver = UncollidedSolver(problem=problem, file_name=f"{prefix}_uncollided.h5",
                              diagnostics_file=f"{prefix}_uncollided_diagnostics.h5")
    solver.Initialize()
    solver.Execute()
    solve_seconds = time.perf_counter() - start
    # Per neutron of the whole source: the octant receives one eighth of it.
    scale = 1.0 / (STRENGTH * OCTANT_FRACTION)
    results = {
        "regions": per_source(region_values(problem, num_groups), scale),
        "lines": per_source(line_values(problem), 1.0 / STRENGTH),
        "seconds": solve_seconds, "ranks": size,
    }
    if rank == 0:
        with open(f"{prefix}_uncollided.json", "w") as stream:
            json.dump(results, stream)
    log(f"Uncollided stage: {solve_seconds:.1f} s on {size} ranks.")
    FieldFunctionGridBased.ExportMultipleToPVTU(problem.GetScalarFluxFieldFunction(),
                                                f"{prefix}_uncollided")


def make_groupsets(num_groups, scattering_order):
    quadrature = GLCProductQuadrature3DXYZ(n_polar=n_polar, n_azimuthal=n_azimuthal,
                                           scattering_order=scattering_order)
    if num_groups <= 8:
        return [{"groups_from_to": (0, num_groups - 1), "angular_quadrature": quadrature,
                 "inner_linear_method": "petsc_gmres", "gmres_restart_interval": 30,
                 "l_abs_tol": 1.0e-8, "l_max_its": 500}], {}
    cut = first_upscatter_group(num_groups)
    ranges = [(0, cut - 1), (cut, num_groups - 1)] if 0 < cut < num_groups else \
        [(0, num_groups - 1)]
    # Source iteration converges too slowly here: the within-group scattering ratio reaches 0.93
    # and the void cavity rules out diffusion acceleration. GMRES converges in a few tens of
    # iterations; its Krylov vectors are copies of the groupset's flux moments.
    groupsets = [{"groups_from_to": r, "angular_quadrature": quadrature,
                  "inner_linear_method": "petsc_gmres", "gmres_restart_interval": 10,
                  "l_abs_tol": 1.0e-8, "l_max_its": 500} for r in ranges]
    # No neutron goes from the thermal groupset back to the first one: one pass suffices.
    return groupsets, {"max_ags_iterations": 1}


def compute_collided(grid, xs_map, num_groups, scattering_order, sn_only=False):
    start = time.perf_counter()
    groupsets, options = make_groupsets(num_groups, scattering_order)
    arguments = {}
    if sn_only:
        # A plain discrete-ordinates point source puts all of its strength into the octant.
        arguments["point_sources"] = [PointSource(
            location=[0.0, 0.0, 0.0],
            strength=[STRENGTH * OCTANT_FRACTION] + [0.0] * (num_groups - 1))]
    else:
        arguments["uncollided_flux"] = f"{prefix}_uncollided.h5"
    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=groupsets,
        xs_map=xs_map,
        boundary_conditions=BOUNDARY_CONDITIONS,
        options=options,
        **arguments,
    )
    solver = SteadyStateSourceSolver(problem=problem)
    solver.Initialize()
    solver.Execute()
    solve_seconds = time.perf_counter() - start

    balance = solver.ComputeBalanceTable()
    production = balance["production_rate"]
    # ComputeLeakage holds the collided outflow through OUTER, by group. The balance table
    # counts the outgoing current of a reflecting face as both an outflow and an inflow, so
    # the net leakage is the outflow minus the inflow; it also holds the uncollided outflow,
    # which is all in group 0.
    leakage = np.asarray(problem.ComputeLeakage(["OUTER"])["OUTER"])
    collided_leakage = leakage.sum()
    if not sn_only:
        leakage[0] += balance["outflow_rate"] - balance["inflow_rate"] - collided_leakage
    scale = 1.0 / production
    results = {
        "production_rate": production,
        "balance": {k: float(v) for k, v in balance.items()},
        "relative_balance": balance["balance"] / production,
        "leakage_groups": (leakage * scale).tolist(),
        "collided_leakage": collided_leakage * scale,
        "regions": per_source(region_values(problem, num_groups), scale),
        "lines": per_source(line_values(problem), 1.0 / STRENGTH),
        "seconds": solve_seconds, "ranks": size,
        "quadrature": {"n_polar": n_polar, "n_azimuthal": n_azimuthal},
        "groupsets": [list(g["groups_from_to"]) for g in groupsets],
        "library": library, "mesh_file": mesh_file,
    }
    name = f"{prefix}_sn_only" if sn_only else f"{prefix}_results"
    if rank == 0:
        with open(f"{name}.json", "w") as stream:
            json.dump(results, stream)
    log(f"Production rate: {production:.10e} (1/8 of the whole source)")
    log(f"Relative balance: {results['relative_balance']:.3e}")
    log(f"Leakage per source neutron: {leakage.sum() * scale:.6e}")
    log(f"{'Block':17s} {'Flux [cm]':>13s} {'Absorption':>13s}")
    for block, entry in results["regions"].items():
        log(f"{block:17s} {sum(entry['flux']):13.6e} {sum(entry['absorption']):13.6e}")
    log(f"{'SN-only' if sn_only else 'Collided'} stage: {solve_seconds:.1f} s on {size} ranks.")
    if not sn_only:
        FieldFunctionGridBased.ExportMultipleToPVTU(problem.GetScalarFluxFieldFunction(),
                                                    f"{prefix}_flux")


if __name__ == "__main__":
    grid = DistributedMeshGenerator(
        inputs=[FromFileMeshGenerator(filename=mesh_file)]).Execute()
    xs_map, num_groups, scattering_order = load_cross_sections()
    log(f"Library {LIBRARY_FILE}: {num_groups} groups, P{scattering_order} scattering; "
        f"mesh {mesh_file}; {n_polar * n_azimuthal} directions.")
    if stage in ("uncollided", "all"):
        compute_uncollided(grid, xs_map, num_groups, scattering_order)
    if stage in ("collided", "all"):
        compute_collided(grid, xs_map, num_groups, scattering_order)
    if stage == "sn_only":
        compute_collided(grid, xs_map, num_groups, scattering_order, sn_only=True)
    if stage not in ("uncollided", "collided", "all", "sn_only"):
        raise ValueError(f"Unknown stage {stage}")
