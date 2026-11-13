#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Coupled photon-electron CSDA transport in an insulated copper coil.

A broad, parallel photon beam travels along -y through an air box that holds a
two-turn coil of polyethylene-insulated copper wire. The photons set electrons
in motion by Compton scattering, photoelectric absorption, and pair
production. OpenSn transports the electrons with the restricted
continuous-slowing-down approximation and the photons and electrons fully
coupled, including bremsstrahlung and fluorescence. The results are energy and
charge deposition in the copper, the insulation, and the air.

All results are per incident photon. Lengths are in cm and energies in MeV.
"""

import json
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
    from pyopensn.post import VolumePostprocessor
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.xs import MultiGroupXS

# The coarse mesh "f4" is the default and the regression test. The production
# mesh "f2" gives the results and figures of the README. Select it with
#   opensn -i coil.py -p "mesh = 'f2'"
# or, in module mode, with  python3 coil.py --mesh f2.
MESH = globals().get("mesh", "f4")
QUADRATURE_ORDER = 23
# Must equal the Legendre order of the cross-section libraries in xs/.
SCATTERING_ORDER = 7
L_ABS_TOL = 1.0e-9
OUTPUT_PREFIX = "coil"
# All groups form one groupset solved with restarted GMRES, whose Krylov
# vectors dominate memory. Within-group DSA (WGDSA) preconditions it and
# cuts the iteration count by 2-4x. Splitting photons and electrons into
# separate groupsets takes more iterations.
INNER_METHOD = "petsc_gmres"
GMRES_RESTART = 10
APPLY_WGDSA = True
SPLIT_GROUPSETS = False
AGS_TOLERANCE = 1.0e-8

# Block IDs of generate_mesh.py, with the library of each material.
REGIONS = {"insulation": (1, "polyethylene"), "copper": (2, "copper"), "air": (3, "air")}
# Material densities of the cross-section libraries, in g/cm^3.
DENSITY = {"insulation": 0.94, "copper": 8.96, "air": 0.001205}
BOUNDARIES = ("inlet", "outlet", "xmin", "xmax", "zmin", "zmax")
INLET_AREA = 2.4 * 2.2

# The incident spectrum is proportional to E exp(-E / T) between the 1 keV
# cutoff and the upper bound of the highest photon group.
SPECTRUM_TEMPERATURE = 0.7


def source_spectrum(photon_groups):
    """Return the fraction of incident photons in each photon group."""

    def cumulative(energy):
        t = SPECTRUM_TEMPERATURE
        return -t * (energy + t) * math.exp(-energy / t)

    weights = [cumulative(upper) - cumulative(lower) for upper, lower in photon_groups]
    total = sum(weights)
    return [w / total for w in weights]


def make_beam(quadrature, spectrum):
    """Put one incident photon per second on the single ordinate along -y."""
    beam = [n for n, omega in enumerate(quadrature.omegas)
            if abs(float(omega.y) + 1.0) < 1.0e-12]
    if len(beam) != 1:
        raise RuntimeError("The angular quadrature must contain exactly one ordinate along -y")
    beam_angle = beam[0]
    # The beam is normal to the inlet, so the incident current is A * w * psi.
    psi = 1.0 / (INLET_AREA * float(quadrature.weights[beam_angle]))
    num_photon_groups = len(spectrum)

    def boundary_flux(group, angle):
        if angle != beam_angle or group >= num_photon_groups:
            return 0.0
        return psi * spectrum[group]

    return AngularFluxFunction(boundary_flux)


def integrate_by_region(field_function):
    """Integrate a field function over each material region."""
    totals = {}
    for name, (block, _) in REGIONS.items():
        interpolation = FieldFunctionInterpolationVolume()
        interpolation.SetOperationType("sum_func")
        interpolation.SetOperationFunction(
            lambda value, block_id, block=block: value if block_id == block else 0.0)
        interpolation.SetLogicalVolume(RPPLogicalVolume(infx=True, infy=True, infz=True))
        interpolation.AddFieldFunction(field_function)
        interpolation.Execute()
        totals[name] = float(interpolation.GetValue())
    return totals


def main():
    rank = globals()["rank"] if "opensn_console" in globals() else MPI.COMM_WORLD.rank
    here = Path(os.path.abspath(__file__)).parent if "__file__" in globals() else Path.cwd()
    groups = json.loads((here / "xs" / "groups.json").read_text())
    num_photon_groups = len(groups["photon"])
    num_groups = num_photon_groups + len(groups["electron"])

    mesh_file = here / f"coil_{MESH}.msh"
    if not mesh_file.is_file():
        raise RuntimeError(f"No mesh {mesh_file.name}. The meshes are coil_f4.msh and coil_f2.msh.")
    mesh = FromFileMeshGenerator(filename=str(mesh_file)).Execute()

    xs_map = []
    for name, (block, library) in REGIONS.items():
        xs = MultiGroupXS()
        xs.LoadFromCEPXS(str(here / "xs" / f"{library}.bxslib"), material_id=0,
                         csda_format=True, particle_order=groups["order"])
        if xs.num_groups != num_groups or xs.scattering_order != SCATTERING_ORDER:
            raise RuntimeError(f"{library}.bxslib does not have {num_groups} groups and "
                               f"Legendre order {SCATTERING_ORDER}")
        xs_map.append({"block_ids": [block], "xs": xs})

    spectrum = source_spectrum(groups["photon"])
    quadrature = LebedevQuadrature3DXYZ(quadrature_order=QUADRATURE_ORDER,
                                        scattering_order=SCATTERING_ORDER)
    boundary_conditions = [{"name": "inlet", "type": "arbitrary",
                            "function": make_beam(quadrature, spectrum)}]
    boundary_conditions += [{"name": name, "type": "vacuum"} for name in BOUNDARIES[1:]]

    def groupset(first, last):
        return {"groups_from_to": (first, last), "angular_quadrature": quadrature,
                "angle_aggregation_type": "single", "inner_linear_method": INNER_METHOD,
                "gmres_restart_interval": GMRES_RESTART, "apply_wgdsa": APPLY_WGDSA,
                "l_abs_tol": L_ABS_TOL, "l_max_its": 1000}

    if SPLIT_GROUPSETS:
        groupsets = [groupset(0, num_photon_groups - 1),
                     groupset(num_photon_groups, num_groups - 1)]
        options = {"ags_tolerance": AGS_TOLERANCE, "max_ags_iterations": 50}
    else:
        groupsets = [groupset(0, num_groups - 1)]
        options = {}

    problem = DiscreteOrdinatesProblem(
        mesh=mesh,
        num_groups=num_groups,
        groupsets=groupsets,
        xs_map=xs_map,
        boundary_conditions=boundary_conditions,
        options={"csda_enabled": True, "verbose_inner_iterations": True, **options},
        sweep_type="AAH",
    )
    solver = SteadyStateSourceSolver(problem=problem, compute_balance=True)
    solver.Initialize()
    solver.Execute()
    balance = solver.ComputeBalanceTable()

    energy = problem.CreateFieldFunction("energy_deposition", "csda_energy_deposition")
    charge = problem.CreateFieldFunction("charge_deposition", "csda_charge_deposition")
    FieldFunctionGridBased.ExportMultipleToPVTU([energy, charge], f"{OUTPUT_PREFIX}_deposition")
    region_energy = integrate_by_region(energy)
    region_charge = integrate_by_region(charge)

    region_flux = {}
    region_volume = {}
    for name, (block, _) in REGIONS.items():
        flux = VolumePostprocessor(problem=problem, block_ids=[block], value_type="integral")
        flux.Execute()
        region_flux[name] = [float(v) for v in flux.GetValue()[0]]
        # The meshed volume is the flux integral over the flux average. It is
        # smaller than the true volume because flat facets cut the curved
        # surfaces of the wire and the jacket.
        average = VolumePostprocessor(problem=problem, block_ids=[block], value_type="avg", group=0)
        average.Execute()
        region_volume[name] = region_flux[name][0] / float(average.GetValue()[0][0])
    leakage = {name: [float(v) for v in values]
               for name, values in problem.ComputeLeakage(list(BOUNDARIES)).items()}

    # Energy conservation: the deposited energy equals the energy that enters
    # minus the energy that leaves.
    deposited = sum(region_energy.values())
    expected = (balance["csda_energy_production_rate"] + balance["csda_energy_inflow_rate"]
                - balance["csda_energy_outflow_rate"])
    energy_closure = (deposited - expected) / balance["csda_energy_inflow_rate"]
    # Charge conservation: deposition counts electrons that stop as positive
    # and electrons that are set in motion as negative, so the net deposited
    # charge is minus the electrons that leave the box.
    electron_leakage = sum(sum(values[num_photon_groups:]) for values in leakage.values())
    charge_closure = sum(region_charge.values()) + electron_leakage

    if rank == 0:
        print(f"COIL_INCIDENT_ENERGY_MEV={balance['csda_energy_inflow_rate']:.10e}")
        for name in REGIONS:
            print(f"COIL_{name.upper()}_ENERGY_DEPOSITION_MEV={region_energy[name]:.10e}")
            print(f"COIL_{name.upper()}_CHARGE_DEPOSITION={region_charge[name]:.10e}")
            mass = DENSITY[name] * region_volume[name]
            print(f"COIL_{name.upper()}_MASS_G={mass:.10e}")
            print(f"COIL_{name.upper()}_ENERGY_DEPOSITION_MEV_PER_G="
                  f"{region_energy[name] / mass:.10e}")
        for name, values in leakage.items():
            print(f"COIL_{name.upper()}_PHOTON_LEAKAGE={sum(values[:num_photon_groups]):.10e}")
            print(f"COIL_{name.upper()}_ELECTRON_LEAKAGE={sum(values[num_photon_groups:]):.10e}")
        print(f"COIL_PARTICLE_BALANCE={balance['csda_particle_balance']:.10e}")
        print(f"COIL_ENERGY_BALANCE={balance['csda_energy_balance']:.10e}")
        print(f"COIL_ENERGY_CLOSURE={energy_closure:.10e}")
        print(f"COIL_CHARGE_CLOSURE={charge_closure:.10e}")
        results = {
            "settings": {"mesh": MESH, "quadrature_order": QUADRATURE_ORDER,
                         "scattering_order": SCATTERING_ORDER, "l_abs_tol": L_ABS_TOL,
                         "apply_wgdsa": APPLY_WGDSA, "gmres_restart": GMRES_RESTART},
            "source_spectrum": spectrum,
            "balance": {key: float(value) for key, value in balance.items()},
            "energy_deposition": region_energy,
            "charge_deposition": region_charge,
            "region_volume_cm3": region_volume,
            "density_g_per_cm3": DENSITY,
            "region_flux": region_flux,
            "leakage": leakage,
        }
        Path(f"{OUTPUT_PREFIX}_results.json").write_text(json.dumps(results, indent=1) + "\n")

    # Acceptance checks, relative to one incident photon. The electron flux is
    # a small part of the GMRES residual norm, so the particle balance, which
    # only electrons affect, shows whether the electrons have converged.
    metrics = (balance["csda_particle_balance"], balance["csda_energy_balance"],
               energy_closure, charge_closure)
    if not all(math.isfinite(value) for value in metrics):
        raise RuntimeError("Non-finite conservation metric")
    if abs(balance["csda_energy_balance"]) > 1.0e-6 or abs(energy_closure) > 1.0e-6:
        raise RuntimeError("Energy is not conserved to 1e-6 of the incident energy")
    if abs(balance["csda_particle_balance"]) > 1.0e-6 or abs(charge_closure) > 1.0e-6:
        raise RuntimeError("Particles and charge are not conserved to 1e-6 per incident "
                           "photon. Tighten L_ABS_TOL.")


if __name__ == "__main__":
    if "opensn_console" not in globals():
        import argparse
        parser = argparse.ArgumentParser(description="Coupled photon-electron coil example.")
        parser.add_argument("--mesh", default=MESH, choices=("f4", "f2"))
        MESH = parser.parse_args().mesh
    main()
