#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Forty-group electron-photon CSDA transport on the WATERBLOCK assembly."""

import math
import os
from pathlib import Path

if "opensn_console" not in globals():
    from mpi4py import MPI
    from pyopensn.aquad import LebedevQuadrature3DXYZ
    from pyopensn.fieldfunc import FieldFunctionGridBased, FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.math import AngularFluxFunction
    from pyopensn.mesh import FromFileMeshGenerator
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.xs import MultiGroupXS
    if os.environ.get("OPENSN_PRECICE_CONFIG"):
        from pyopensn.coupling import PreciceAdapter


ELECTRON_GROUPS = 20
TOTAL_GROUPS = 40
BEAM_AREA_M2 = 0.008 * 0.008
NOMINAL_BEAM_POWER_W = 0.1
BEAM_ENERGY_MEV = 1.0
MEV_TO_J = 1.602176634e-13
MATERIALS = (
    ("tungsten", 101),
    ("copper", 102),
    ("aluminum", 103),
    ("air", 104),
    ("steel", 105),
    ("water", 106),
)


def make_normal_beam(quadrature):
    """Put a unit electron current on the single ordinate along +x."""
    angles = [
        angle
        for angle, omega in enumerate(quadrature.omegas)
        if abs(float(omega.x) - 1.0) < 1.0e-12
    ]
    if len(angles) != 1:
        raise RuntimeError("The angular quadrature must contain exactly one ordinate along +x")
    beam_angle = angles[0]
    # The beam is normally incident, so the current per unit area is weight * psi.
    angular_flux = 1.0 / (BEAM_AREA_M2 * float(quadrature.weights[beam_angle]))

    def boundary_flux(group, angle):
        return angular_flux if group == 0 and angle == beam_angle else 0.0

    return AngularFluxFunction(boundary_flux)


def integrate(field_function):
    interpolation = FieldFunctionInterpolationVolume()
    interpolation.SetOperationType("sum")
    interpolation.SetLogicalVolume(RPPLogicalVolume(infx=True, infy=True, infz=True))
    interpolation.AddFieldFunction(field_function)
    interpolation.Execute()
    return float(interpolation.GetValue())


def precice_averaging_group(block_id, x, _y, _z):
    """Return the conservative deposition control volume of a cell.

    Tungsten, three axial copper regions, and the combined low-deposition
    surroundings are averaged separately so the transferred heating is
    nonnegative without clipping energy.
    """
    if block_id == 101:
        return 0
    if block_id != 102:
        return 4
    if x < 0.006:
        return 1
    if x < 0.024:
        return 2
    return 3


def main():
    run_rank = globals()["rank"] if "opensn_console" in globals() else MPI.COMM_WORLD.rank
    case_dir = Path(__file__).resolve().parent
    mesh = FromFileMeshGenerator(
        filename=str(case_dir / "waterblock.msh"), replicated_mesh=False
    ).Execute()

    cross_sections = {}
    for name, _ in MATERIALS:
        library_name = "stainless_steel_304" if name == "steel" else name
        xs = MultiGroupXS()
        xs.LoadFromCEPXS(
            str(case_dir / "xs" / f"{library_name}.bxslib"),
            material_id=0,
            csda_format=True,
            particle_order=["electron", "photon"],
        )
        xs.Scale(100.0)
        if xs.num_groups != TOTAL_GROUPS:
            raise RuntimeError(f"{name} does not contain {TOTAL_GROUPS} groups")
        cross_sections[name] = xs

    quadrature = LebedevQuadrature3DXYZ(quadrature_order=7, scattering_order=2)
    beam = make_normal_beam(quadrature)
    groupsets = [
        {
            "groups_from_to": (0, TOTAL_GROUPS - 1),
            "angular_quadrature": quadrature,
            "angle_aggregation_type": "single",
            "inner_linear_method": "petsc_gmres",
            "gmres_restart_interval": 300,
            "l_abs_tol": 1.0e-7,
            "l_max_its": 1200,
            "apply_wgdsa": False,
            "apply_tgdsa": False,
        }
    ]
    problem = DiscreteOrdinatesProblem(
        mesh=mesh,
        num_groups=TOTAL_GROUPS,
        groupsets=groupsets,
        xs_map=[
            {"block_ids": [block_id], "xs": cross_sections[name]}
            for name, block_id in MATERIALS
        ],
        boundary_conditions=[
            {"name": "beamInlet", "type": "arbitrary", "function": beam},
            {"name": "waterInlet", "type": "vacuum"},
            {"name": "waterOutlet", "type": "vacuum"},
            {"name": "vacuum", "type": "vacuum"},
        ],
        options={
            "csda_enabled": True,
            "verbose_inner_iterations": True,
        },
        sweep_type="AAH",
    )
    solver = SteadyStateSourceSolver(problem=problem, compute_balance=True)
    solver.Initialize()
    solver.Execute()

    balance = solver.ComputeBalanceTable()
    deposition = problem.CreateFieldFunction(
        "waterblock_energy_deposition", "csda_energy_deposition"
    )
    FieldFunctionGridBased.ExportMultipleToPVTU(
        [deposition], str(case_dir / "waterblock_energy_deposition")
    )
    total_deposition = integrate(deposition)
    balance_deposition = (
        balance["csda_energy_production_rate"]
        + balance["csda_energy_inflow_rate"]
        - balance["csda_energy_outflow_rate"]
    )
    deposition_error = abs(total_deposition - balance_deposition) / abs(
        balance["csda_energy_inflow_rate"]
    )
    electron_rate = NOMINAL_BEAM_POWER_W / (BEAM_ENERGY_MEV * MEV_TO_J)
    deposited_power = total_deposition * electron_rate * MEV_TO_J

    leakage = problem.ComputeLeakage(
        ["beamInlet", "waterInlet", "waterOutlet", "vacuum"]
    )
    metrics = (
        balance["csda_particle_balance"],
        balance["csda_energy_balance"],
        deposition_error,
        deposited_power,
    )
    if not all(math.isfinite(value) for value in metrics):
        raise RuntimeError("Non-finite transport conservation metric")
    if abs(balance["csda_energy_balance"]) > 5.0e-4 or deposition_error > 5.0e-4:
        raise RuntimeError("CSDA energy/deposition balance differs by more than 0.05%")

    if run_rank == 0:
        print(f"WATERBLOCK_ENERGY_INFLOW_MEV_S={balance['csda_energy_inflow_rate']:.12e}")
        print(f"WATERBLOCK_ENERGY_OUTFLOW_MEV_S={balance['csda_energy_outflow_rate']:.12e}")
        print(f"WATERBLOCK_DEPOSITION_MEV_S={total_deposition:.12e}")
        print(f"WATERBLOCK_DEPOSITION_W={deposited_power:.15e}")
        for boundary, values in leakage.items():
            print(f"WATERBLOCK_{boundary.upper()}_ELECTRONS_S={sum(values[:ELECTRON_GROUPS]):.12e}")
            print(f"WATERBLOCK_{boundary.upper()}_PHOTONS_S={sum(values[ELECTRON_GROUPS:]):.12e}")
        print(f"WATERBLOCK_PARTICLE_BALANCE={balance['csda_particle_balance']:.12e}")
        print(f"WATERBLOCK_ENERGY_BALANCE={balance['csda_energy_balance']:.12e}")
        print(f"WATERBLOCK_DEPOSITION_CLOSURE={deposition_error:.12e}")

    precice_config = os.environ.get("OPENSN_PRECICE_CONFIG")
    if precice_config:
        adapter_type = globals().get("PreciceAdapter")
        if adapter_type is None:
            raise RuntimeError("OpenSn was not built with OPENSN_WITH_PRECICE=ON")
        adapter = adapter_type(configuration_file=str(Path(precice_config).resolve()))
        adapter.WriteSteadyCellPower(
            deposition,
            source_rate=electron_rate,
            joules_per_energy_unit=MEV_TO_J,
            averaging_group=precice_averaging_group,
        )


if __name__ == "__main__":
    main()
