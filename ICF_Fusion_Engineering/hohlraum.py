#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Fixed-source neutron transport through an indirect-drive ICF target.

The target is a gold hohlraum with a polyimide window over the laser entrance hole (LEH)
in each end, filled with helium, inside an aluminum thermal-mechanical package (TMP).
Isotropic 14.1 MeV DT neutrons are emitted from a point at the center of the capsule. In
this example the capsule holds the same helium as the hohlraum. The outer boundary is
vacuum.

The point source is treated with OpenSn's uncollided-flux method in two stages. The first
stage computes the uncollided flux by ray tracing from the source and writes it to
hohlraum_uncollided.h5. It runs on one MPI rank. The second stage solves for the collided
flux with the discrete-ordinates method, using the first collisions of the uncollided flux
as its source, and adds the uncollided flux back to the result. It runs on any number of
ranks. This input runs the first stage if hohlraum_uncollided.h5 does not exist and the
second stage otherwise.

The calculation uses 64 energy groups, transport-corrected P3 scattering, and 1,024
directions on 989,832 tetrahedra. The second stage is estimated to need about 1.3 TB of
memory in total, so it needs cluster-scale resources. For example:

    gunzip -k hohlraum.msh.gz
    OPENSN_NUM_THREADS=16 opensn -i hohlraum.py
    export OPENSN_NUM_THREADS=1
    srun -n 512 opensn -i hohlraum.py

Outputs:
    The first stage writes the uncollided flux and reaction rates in each region to
    hohlraum_uncollided.json and the uncollided flux to hohlraum_uncollided.pvtu. The
    second stage prints the particle balance and the flux and reaction rates per source
    neutron in each region, writes them, with their group spectra and the leakage
    spectrum, to hohlraum_results.json, and writes the scalar flux in every group to
    hohlraum_flux.pvtu.
"""

import json
import os

import numpy as np

if "opensn_console" not in globals():
    from mpi4py import MPI

    size, rank = MPI.COMM_WORLD.size, MPI.COMM_WORLD.rank
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.fieldfunc import FieldFunctionGridBased
    from pyopensn.logvol import RPPLogicalVolume
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

uncollided_file = "hohlraum_uncollided.h5"
uncollided_results = "hohlraum_uncollided.json"
num_groups = 64
scattering_order = 3

# Mesh. Rank 0 reads the Gmsh file and distributes the cells to the other ranks.
grid = DistributedMeshGenerator(
    inputs=[FromFileMeshGenerator(filename="hohlraum.msh")]
).Execute()

# Materials. Each block gets the macroscopic cross sections of one material from the
# 64-group transport-corrected P3 OpenMC library. The capsule (1) has the same helium as the
# fill gas (2). The reaction cross sections are used only to compute reaction rates.
regions = {1: "capsule", 2: "fill_gas", 3: "hohlraum_wall", 4: "leh_windows", 5: "tmp",
           6: "exterior"}
materials = {"helium": [1, 2], "gold": [3], "polyimide": [4], "aluminum": [5], "void": [6]}
reactions = ["absorption", "n2n", "ngamma", "nalpha"]
xs_map = []
for name, blocks in materials.items():
    xs = MultiGroupXS()
    xs.LoadFromOpenMC("materials_64g_p3tc.h5", name, 293.6, reactions)
    xs_map.append({"block_ids": blocks, "xs": xs})


def region_values(problem):
    """Return the group-wise flux and reaction rates integrated over each region."""
    values = {}
    for block, name in regions.items():
        entry = {}
        for quantity in ["flux"] + reactions:
            options = {} if quantity == "flux" else {"xs_multiplier": quantity}
            post = VolumePostprocessor(problem=problem, block_ids=[block],
                                       value_type="integral", **options)
            post.Execute()
            entry[quantity] = np.asarray(post.GetValue()).reshape(-1, num_groups).sum(axis=0)
        values[name] = entry
    return values


def compute_uncollided():
    """First stage: compute and write the uncollided flux on one rank."""
    # Group 0 (13.84 to 14.191 MeV) holds the 14.1 MeV source. Inside the near-source
    # region, the cube around the capsule, the flux is ray traced to the quadrature points
    # of every cell. Outside it the uncollided flux is computed by a sweep along the
    # directions from the source. The cells around the source cell are integrated exactly
    # over direction.
    source = PointSource(location=[0.0, 0.0, 0.0], strength=[1.0] + [0.0] * (num_groups - 1))
    near_source = RPPLogicalVolume(xmin=-0.01, xmax=0.01, ymin=-0.01, ymax=0.01,
                                   zmin=-0.01, zmax=0.01)
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
    values = region_values(problem)
    with open(uncollided_results, "w") as stream:
        json.dump({name: {k: v.tolist() for k, v in entry.items()}
                   for name, entry in values.items()}, stream)
    flux = problem.GetScalarFluxFieldFunction()
    FieldFunctionGridBased.ExportMultipleToPVTU([flux[0]], "hohlraum_uncollided")
    print(f"Wrote {uncollided_file}. Run this input again, on any number of ranks, "
          "to compute the collided flux.")


def compute_collided():
    """Second stage: solve for the collided flux on any number of ranks."""
    with open(uncollided_results) as stream:
        uncollided = {name: {k: np.array(v) for k, v in entry.items()}
                      for name, entry in json.load(stream).items()}

    # A Gauss-Legendre-Chebyshev product quadrature with 16 polar and 64 azimuthal angles
    # has 1,024 directions. The collided flux inside the 50 um gold wall and the 100 um TMP
    # comes mostly from neutrons that scattered in the same thin layer and travel nearly
    # parallel to it. These grazing directions converge slowly with the number of polar
    # angles.
    quadrature = GLCProductQuadrature3DXYZ(n_polar=16, n_azimuthal=64,
                                           scattering_order=scattering_order)

    # All groups are solved together on one groupset by source iteration. Only 0.3% of the
    # source neutrons collide, so the scattering source converges quickly. Most of the
    # iterations converge the angular fluxes that the sweep lags on cycles of the
    # tetrahedral mesh and on partition boundaries. GMRES needs fewer iterations, but each
    # of its Krylov vectors is a full copy of the flux moments, which more than doubles the
    # memory.
    groupset = {
        "groups_from_to": (0, num_groups - 1),
        "angular_quadrature": quadrature,
        "inner_linear_method": "classic_richardson",
        "l_abs_tol": 1.0e-7,
        "l_max_its": 200,
    }

    # The first collisions of the uncollided flux are the only source of this problem.
    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[groupset],
        xs_map=xs_map,
        uncollided_flux=uncollided_file,
    )
    solver = SteadyStateSourceSolver(problem=problem)
    solver.Initialize()
    solver.Execute()

    # Particle balance. The production rate is the source strength, 1 neutron/s.
    # ComputeLeakage returns the collided outflow in each group. The balance table also
    # includes the uncollided outflow, which is all in group 0.
    balance = solver.ComputeBalanceTable()
    source_rate = balance["production_rate"]
    collided_leakage = sum(np.asarray(v) for v in problem.ComputeLeakage([]).values())
    leakage = collided_leakage.copy()
    leakage[0] += balance["outflow_rate"] - collided_leakage.sum()
    values = region_values(problem)
    if rank == 0:
        print(f"Converged: {solver.IsConverged()}")
        print(f"Source rate [neutron/s]: {source_rate:.10e}")
        print(f"Relative balance: {balance['balance'] / source_rate:.3e}")
        print(f"Leakage per source neutron: {leakage.sum() / source_rate:.10e}")
        print(f"Collided leakage per source neutron: "
              f"{collided_leakage.sum() / source_rate:.6e}")

        # Region-integrated flux [cm] and reaction rates per source neutron, in total and
        # for the collided neutrons alone.
        print(f"{'Region':14s} {'Flux [cm]':>13s} {'Collided':>13s} {'Absorption':>13s} "
              f"{'(n,2n)':>13s} {'(n,gamma)':>13s} {'(n,alpha)':>13s}")
        for name, entry in values.items():
            collided = entry["flux"].sum() - uncollided[name]["flux"].sum()
            row = [entry["flux"].sum(), collided] + [entry[k].sum() for k in reactions]
            print(f"{name:14s} " + " ".join(f"{v / source_rate:13.6e}" for v in row))
        results = {
            "source_rate": source_rate,
            "balance": {k: float(v) for k, v in balance.items()},
            "leakage_groups": (leakage / source_rate).tolist(),
            "collided_leakage_groups": (collided_leakage / source_rate).tolist(),
            "total": {name: {k: (v / source_rate).tolist() for k, v in entry.items()}
                      for name, entry in values.items()},
            "uncollided": {name: {k: v.tolist() for k, v in entry.items()}
                           for name, entry in uncollided.items()},
        }
        with open("hohlraum_results.json", "w") as stream:
            json.dump(results, stream)

    # Scalar flux in every group, for visualization in ParaView.
    FieldFunctionGridBased.ExportMultipleToPVTU(problem.GetScalarFluxFieldFunction(),
                                                "hohlraum_flux")


if os.path.exists(uncollided_file) and os.path.exists(uncollided_results):
    compute_collided()
elif size == 1:
    compute_uncollided()
else:
    raise RuntimeError(f"{uncollided_file} does not exist. Run this input on one rank "
                       "first to compute the uncollided flux.")
