#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Weekly regression test of the urban source example at low resolution.

This input runs urban_source.py, with the same city, materials, source, 25 groups, P3
scattering, and solvers, on urban_source_coarse.msh, a mesh of 122,490 cells made with

    python3 generate_mesh.py --size-factor 3 --output urban_source_coarse.msh

and with a Gauss-Legendre-Chebyshev product quadrature of 4 polar and 16 azimuthal angles,
64 directions, for the collided flux. On one rank it removes the output of an earlier run
and computes the uncollided flux. On more ranks it computes the collided flux, and rank 0
prints one line of results that test.py compares with stored values. It unpacks
urban_source_coarse.msh.gz the first time it runs. Run it from this directory:

    mpirun -n 1 opensn -i urban_source_coarse.py
    mpirun -n 32 opensn -i urban_source_coarse.py
"""

import gzip
import json
import os
import shutil

if "opensn_console" not in globals():
    from mpi4py import MPI

    size, rank = MPI.COMM_WORLD.size, MPI.COMM_WORLD.rank

mesh_file = "urban_source_coarse.msh"
prefix = "urban_source_coarse"
quadrature_type = "glc"
n_polar = 4
n_azimuthal = 16

if size == 1:
    # Start from the uncollided stage, so that an uncollided flux left by an earlier run is
    # not reused.
    for name in (f"{prefix}_uncollided.h5", f"{prefix}_uncollided.json",
                 f"{prefix}_results.json"):
        if os.path.exists(name):
            os.remove(name)
    # A partly written mesh is never left under the final name.
    if not os.path.exists(mesh_file):
        with gzip.open(f"{mesh_file}.gz", "rb") as source, \
                open(f"{mesh_file}.part", "wb") as target:
            shutil.copyfileobj(source, target)
        os.replace(f"{mesh_file}.part", mesh_file)

with open("urban_source.py") as stream:
    exec(compile(stream.read(), "urban_source.py", "exec"))

# One line of results per source photon for test.py. test.py uses its relative tolerance as
# an absolute tolerance too, so values much smaller than 1 are printed in scaled units.
if size > 1 and rank == 0:
    with open(f"{prefix}_results.json") as stream:
        results = json.load(stream)

    def region_total(region, quantity):
        return sum(results["total"][region][quantity])

    summary = {
        "converged:": "yes" if results["converged"] else "no",
        "balance-ok:": "yes" if abs(results["balance"]["balance"]) < 1.0e-6 else "no",
        "air-dose-x1e-4:": 1.0e-4 * region_total("air", "dose"),
        "air-flux-x1e-3:": 1.0e-3 * region_total("air", "flux"),
        "streets-dose-x1e-1:": 1.0e-1 * region_total("ground_asphalt", "dose"),
        "plaza-dose:": region_total("ground_concrete", "dose"),
        "building-r20-dose-x1e-1:": 1.0e-1 * region_total("R20_r5c3", "dose"),
        "leakage-x10:": 1.0e1 * sum(results["leakage_groups"]),
    }
    print("urban-source-coarse " + " ".join(
        f"{key} {value}" if isinstance(value, str) else f"{key} {value:.8e}"
        for key, value in summary.items()))
