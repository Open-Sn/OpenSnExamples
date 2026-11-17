#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Weekly regression test of the tokamak example at low resolution.

This input runs tokamak.py with the same materials, source, boundary conditions, 64 groups,
P3 scattering, and iteration tolerance on tokamak_coarse.msh, a mesh of 130,046 tetrahedra
made with

    python3 generate_mesh.py --coarse --output tokamak_coarse.msh

and with a Gauss-Legendre-Chebyshev product quadrature of 4 polar and 16 azimuthal angles,
64 directions. It uses GMRES without diffusion synthetic acceleration, the solver of the
GPU build, because the flexible GMRES and diffusion solves of the CPU default need about
twice the memory. Rank 0 prints one line of results that test.py compares with stored
values. The input unpacks tokamak_coarse.msh.gz the first time it runs. Run it from this
directory on 32 ranks:

    mpirun -n 32 opensn -i tokamak_coarse.py
"""

import gzip
import os
import shutil

if "opensn_console" not in globals():
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank

mesh_file = "tokamak_coarse.msh"
n_polar = 4
n_azimuthal = 16
apply_wgdsa = False
output_prefix = "tokamak_coarse"

# Rank 0 reads the mesh, so only rank 0 unpacks it. A partly written file is never left
# under the final name.
if rank == 0 and not os.path.exists(mesh_file):
    with gzip.open(f"{mesh_file}.gz", "rb") as source, open(f"{mesh_file}.part", "wb") as target:
        shutil.copyfileobj(source, target)
    os.replace(f"{mesh_file}.part", mesh_file)

# tokamak.py runs in this namespace, so its results are read from it below.
namespace = globals()
with open("tokamak.py") as stream:
    exec(compile(stream.read(), "tokamak.py", "exec"), namespace)

# One line of results for test.py. test.py uses its relative tolerance as an absolute
# tolerance too, so values much smaller than 1 are printed in scaled units.
if rank == 0:
    region_values = namespace["region_results"]
    summary = {
        "converged:": "yes" if namespace["solver"].IsConverged() else "no",
        "balance-ok:": "yes" if abs(namespace["relative_balance"]) < 1.0e-6 else "no",
        "first-wall-flux:": region_values["first_wall"]["flux"],
        "first-wall-absorption:": region_values["first_wall"]["absorption"],
        "blanket-flux:": region_values["blanket"]["flux"],
        "vessel-flux-x1e2:": 1.0e2 * region_values["vessel"]["flux"],
        "shield-flux-x1e3:": 1.0e3 * region_values["shield"]["flux"],
        "leakage-x1e7:": 1.0e7 * namespace["leakage"],
    }
    print("tokamak-coarse " + " ".join(
        f"{key} {value}" if isinstance(value, str) else f"{key} {value:.8e}"
        for key, value in summary.items()))
