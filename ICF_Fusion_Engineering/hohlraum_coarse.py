#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Weekly regression test of the hohlraum example at low resolution.

This input runs both stages of hohlraum.py, with the same materials, point source, 64
groups, transport-corrected P3 scattering, and solvers, on hohlraum_coarse.msh, a mesh of
99,633 tetrahedra made with

    python3 generate_mesh.py --size-factor 2 --output hohlraum_coarse.msh

and with a Gauss-Legendre-Chebyshev product quadrature of 4 polar and 16 azimuthal angles,
64 directions, for the collided flux. The uncollided stage must run on one MPI rank, so this
input runs both stages on one rank. It prints one line of results that test.py compares with
stored values. It unpacks hohlraum_coarse.msh.gz the first time it runs. Run it from this
directory:

    mpirun -n 1 opensn -i hohlraum_coarse.py
"""

import gzip
import json
import os
import shutil

mesh_file = "hohlraum_coarse.msh"
n_polar = 4
n_azimuthal = 16
output_prefix = "hohlraum_coarse"

# A partly written mesh is never left under the final name.
if not os.path.exists(mesh_file):
    with gzip.open(f"{mesh_file}.gz", "rb") as source, open(f"{mesh_file}.part", "wb") as target:
        shutil.copyfileobj(source, target)
    os.replace(f"{mesh_file}.part", mesh_file)

# Start from the uncollided stage, so that an uncollided flux left by an earlier run is not
# reused.
for name in (f"{output_prefix}_uncollided.h5", f"{output_prefix}_uncollided.json",
             f"{output_prefix}_results.json"):
    if os.path.exists(name):
        os.remove(name)

with open("hohlraum.py") as stream:
    code = compile(stream.read(), "hohlraum.py", "exec")
# The first run computes the uncollided flux and the second the collided flux.
exec(code)
exec(code)

# One line of results per source neutron for test.py. test.py uses its relative tolerance as
# an absolute tolerance too, so values much smaller than 1 are printed in scaled units. The
# leakage exceeds 1 because (n,2n) reactions outweigh absorption, so its excess is printed.
with open(f"{output_prefix}_results.json") as stream:
    results = json.load(stream)


def region_flux(region, collided=False):
    total = sum(results["total"][region]["flux"])
    if collided:
        total -= sum(results["uncollided"][region]["flux"]) / results["source_rate"]
    return total


summary = {
    "converged:": "yes" if results["converged"] else "no",
    "balance-ok:": "yes" if abs(results["balance"]["balance"] / results["source_rate"]) < 1.0e-10
    else "no",
    "leakage-excess-x1e4:": 1.0e4 * (sum(results["leakage_groups"]) - 1.0),
    "fill-gas-flux-x10:": 1.0e1 * region_flux("fill_gas"),
    "wall-flux-x1e3:": 1.0e3 * region_flux("hohlraum_wall"),
    "tmp-flux-x1e2:": 1.0e2 * region_flux("tmp"),
    "wall-collided-flux-x1e5:": 1.0e5 * region_flux("hohlraum_wall", collided=True),
    "tmp-collided-flux-x1e5:": 1.0e5 * region_flux("tmp", collided=True),
    "exterior-collided-flux-x1e4:": 1.0e4 * region_flux("exterior", collided=True),
    "wall-n2n-x1e4:": 1.0e4 * sum(results["total"]["hohlraum_wall"]["n2n"]),
    "tmp-absorption-x1e4:": 1.0e4 * sum(results["total"]["tmp"]["absorption"]),
}
print("hohlraum-coarse " + " ".join(
    f"{key} {value}" if isinstance(value, str) else f"{key} {value:.8e}"
    for key, value in summary.items()))
