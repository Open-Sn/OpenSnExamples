#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Fixed-source neutron shielding calculation for an illustrative tokamak.

The model is a full-height 90 degree quarter of the reactor. Isotropic 14.1 MeV neutrons
are born uniformly in the plasma region at 1 neutron/(cm^3 s). They stream through a
vacuum chamber and then cross a tungsten first wall, a steel/water blanket, a steel
vessel, and a borated-polyethylene shield. Copper toroidal coils, poloidal coils, and a
central solenoid sit outside the shield. The planes x = 0 and y = 0 reflect. The outer
cylinder and the top and bottom planes are vacuum.

The calculation uses 64 energy groups, P3 scattering, and 144 discrete directions on
1,074,332 tetrahedra. It needs cluster-scale resources. Unpack the mesh first and run
with, for example, one MPI rank per CPU core on 480 cores:

    gunzip -k tokamak.msh.gz
    srun -n 480 opensn -i tokamak.py

For a GPU build, set use_gpus = True below and run one MPI rank per GPU, for example
srun -n 40 --gpus-per-task=1 opensn -i tokamak.py.

Outputs:
    The particle balance and the flux and absorption per source neutron in each region
    are printed by rank 0. The scalar flux in every group is written to tokamak_flux.pvtu.
"""

if "opensn_console" not in globals():
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.fieldfunc import FieldFunctionGridBased
    from pyopensn.mesh import DistributedMeshGenerator, FromFileMeshGenerator
    from pyopensn.post import VolumePostprocessor
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.source import VolumetricSource
    from pyopensn.xs import MultiGroupXS

# Set this to True to use the fused GPU sweep on a GPU build.
use_gpus = False

# Mesh. Rank 0 reads the Gmsh file and distributes the cells to the other ranks.
grid = DistributedMeshGenerator(
    inputs=[FromFileMeshGenerator(filename="tokamak.msh")]
).Execute()

# Materials. Each block of the mesh gets the macroscopic cross sections of one material
# from the 64-group P3 OpenMC library. The plasma, vacuum chamber, and exterior are void.
# The extra "absorption" cross section is used only to report absorption rates.
materials = {
    "void": [1, 2, 10],
    "tungsten": [3],
    "steel_water": [4],
    "steel": [5],
    "borated_polyethylene": [6],
    "copper": [7, 8, 9],
}
xs_map = []
for name, blocks in materials.items():
    xs = MultiGroupXS()
    xs.LoadFromOpenMC("materials_64g_p3.h5", name, 293.6, ["absorption"])
    xs_map.append({"block_ids": blocks, "xs": xs})
num_groups = 64

# Source. Group 0 (13.84 to 14.191 MeV) holds the 14.1 MeV neutrons.
source = VolumetricSource(block_ids=[1], group_strength=[1.0] + [0.0] * (num_groups - 1))

# Angular quadrature. A Gauss-Legendre-Chebyshev product set with 6 polar and 24 azimuthal
# angles has 144 directions. It is symmetric about the two reflecting planes. With 288 or
# 480 directions the first wall, blanket, vessel, and shield results change by less than
# 0.15%. The coil results, which OpenMC cannot resolve, change by up to 4%.
quadrature = GLCProductQuadrature3DXYZ(n_polar=6, n_azimuthal=24, scattering_order=3)

# Groupset. All groups are solved together with a Krylov method on one groupset. A tighter
# tolerance than 1e-8 changes the first wall, blanket, vessel, and shield results by less
# than 0.01%.
groupset = {
    "groups_from_to": (0, num_groups - 1),
    "angular_quadrature": quadrature,
    "inner_linear_method": "petsc_gmres",
    "gmres_restart_interval": 10,
    "l_abs_tol": 1.0e-8,
    "l_max_its": 300,
}

# On CPUs, within-group diffusion synthetic acceleration halves the number of iterations and
# roughly halves the solve time. The diffusion solves are inexact, so the preconditioner
# changes slightly between iterations and needs flexible GMRES. On GPUs the diffusion solves
# run on the host and make each iteration slower than the iterations they save.
if not use_gpus:
    groupset.update(
        {
            "inner_linear_method": "petsc_fgmres",
            "apply_wgdsa": True,
            "wgdsa_l_abs_tol": 1.0e-1,
            "wgdsa_l_max_its": 5,
        }
    )

# Transport problem.
problem = DiscreteOrdinatesProblem(
    mesh=grid,
    num_groups=num_groups,
    groupsets=[groupset],
    xs_map=xs_map,
    volumetric_sources=[source],
    boundary_conditions=[
        {"name": "SYM_LOWER", "type": "reflecting"},
        {"name": "SYM_UPPER", "type": "reflecting"},
        {"name": "VACUUM_OUTER", "type": "vacuum"},
        {"name": "VACUUM_TOP", "type": "vacuum"},
        {"name": "VACUUM_BOTTOM", "type": "vacuum"},
    ],
    sweep_type="AAH",
    use_gpus=use_gpus,
    options={"fused_aao_sweep": use_gpus},
)

solver = SteadyStateSourceSolver(problem=problem)
solver.Initialize()
solver.Execute()

# Particle balance. The source rate equals the meshed volume of block 1 because the
# source density is 1 neutron/(cm^3 s). Equal inflow and outflow through the reflecting
# planes cancel, so the leakage is the net outflow through the vacuum boundaries.
balance = solver.ComputeBalanceTable()
source_rate = balance["production_rate"]
if rank == 0:
    print(f"Converged: {solver.IsConverged()}")
    print(f"Source rate [neutron/s]: {source_rate:.10e}")
    print(f"Relative balance: {balance['balance'] / source_rate:.3e}")
    leakage = (balance["outflow_rate"] - balance["inflow_rate"]) / source_rate
    print(f"Leakage per source neutron: {leakage:.6e}")

# Region-integrated flux [cm] and absorption per source neutron. These are the quantities
# compared with OpenMC in the README.
regions = {
    "plasma_source": 1,
    "vacuum_chamber": 2,
    "first_wall": 3,
    "blanket": 4,
    "vessel": 5,
    "shield": 6,
    "toroidal_coils": 7,
    "poloidal_coils": 8,
    "central_solenoid": 9,
    "exterior": 10,
}
if rank == 0:
    print(f"{'Region':18s} {'Flux [cm]':>14s} {'Absorption':>14s}")
for name, block in regions.items():
    flux = VolumePostprocessor(problem=problem, block_ids=[block], value_type="integral")
    flux.Execute()
    absorption = VolumePostprocessor(
        problem=problem, block_ids=[block], value_type="integral", xs_multiplier="absorption"
    )
    absorption.Execute()
    if rank == 0:
        total_flux = flux.GetValue().sum() / source_rate
        total_absorption = absorption.GetValue().sum() / source_rate
        print(f"{name:18s} {total_flux:14.6e} {total_absorption:14.6e}")

# Scalar flux in every group, for visualization in ParaView.
FieldFunctionGridBased.ExportMultipleToPVTU(problem.GetScalarFluxFieldFunction(), "tokamak_flux")
