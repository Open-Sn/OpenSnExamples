#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-dependent Kobayashi dog-leg benchmark (Problem 3, case ii) with discrete ordinates.

Benchmark:
    I. Variansyah, "Time-Dependent Kobayashi Dog-Leg Benchmark for Neutron Transport",
    Zenodo (2025), https://doi.org/10.5281/zenodo.15069882
Steady-state problem:
    K. Kobayashi, N. Sugimura, and Y. Nagaya, Progress in Nuclear Energy 39(2), 119-144 (2001)

The Kobayashi Problem 3 octant (reflecting planes at x=0, y=0, z=0) with one group,
sigma_t = 0.1 /cm in the source and shield, 1e-4 /cm in the dog-leg duct, scattering
ratio 0.5 everywhere, and speed 1 cm/s. A 1 n/cm^3/s source in [0,10]^3 is on for
t in [0,50] s, and the benchmark reports flux in 2 s bins up to t = 200 s.

The transient uses the theta scheme with sources evaluated at t^{n+theta}: Crank-Nicolson
steps through the benchmark window, then backward-Euler steps, which damp the remaining
flux, to TAIL_END. Summing the steps gives the discrete identity

    sum_n dt_n phi^{n+theta_n} = phi_ss[sum_n dt_n q(t^{n+theta_n})] - L^{-1} (1/v) psi^N,

where phi_ss[Q] is the steady-state solution, on the same mesh and quadrature, for a
constant source Q. By TAIL_END the remaining population (and with it the last term) is
negligible, and the time-integrated flux is checked against a steady-state solve. The
time-integrated flux divided by 50 s also tends to the published steady-state case-ii fluxes.

Outputs (rank 0):
    kobayashi_td_results.csv  OpenSn 2 s bin averages in the format of openmc_reference.csv
    kobayashi_td_summary.csv  time-integral identity and steady-state reference ratios
"""

import csv
import math
import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.aquad import GLCProductQuadrature3DXYZ, SLDFEsqQuadrature3DXYZ
    from pyopensn.fieldfunc import (
        FieldFunctionGridBased,
        FieldFunctionInterpolationPoint,
        FieldFunctionInterpolationVolume,
    )
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.math import Vector3
    from pyopensn.mesh import FromFileMeshGenerator, PETScGraphPartitioner
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver, TransientSolver
    from pyopensn.source import VolumetricSource
    from pyopensn.xs import MultiGroupXS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kobayashi_td_points import REFERENCE_POINTS, STEADY_CASE_II_FLUX  # noqa: E402

# Problem definition (from the benchmark's build-xml.py and mgxs.h5)
MATERIALS = {  # block id: (sigma_t, scattering ratio)
    1: (0.1, 0.5),  # source
    2: (1.0e-4, 0.5),  # duct
    3: (0.1, 0.5),  # shield
}
SPEED = 1.0
SOURCE_STRENGTH = 1.0
SOURCE_END = 50.0
BENCHMARK_END = 200.0
BIN_WIDTH = 2.0

# Discretization. The environment variables override the defaults. The quadrature is
# "sldfe<level>" (level 0: 96 directions, level 1: 384) or "glc<polar>x<azimuthal>"
# (glc24x96: 2304 directions).
MESH_FILE = os.environ.get("KOBAYASHI_TD_MESH", "kobayashi_dog_leg.msh")
QUADRATURE = os.environ.get("KOBAYASHI_TD_QUADRATURE", "glc24x96")
# Refine an SLDFE quadrature around the six axis directions that the duct legs follow
# (level 1 refined within 20 degrees is 672 directions).
REFINE_CONE_DEGREES = float(os.environ.get("KOBAYASHI_TD_REFINE_CONE", "0.0"))
DT = float(os.environ.get("KOBAYASHI_TD_DT", "2.0"))
THETA = float(os.environ.get("KOBAYASHI_TD_THETA", "0.5"))
# After the benchmark window, take long backward-Euler steps until the population has decayed.
TAIL_DT = 20.0
TAIL_THETA = 1.0
TAIL_END = 500.0
# Export the scalar flux every VTK_INTERVAL seconds up to BENCHMARK_END (0 disables) to
# VTK_DIR, with a kobayashi_td.pvd time index.
VTK_INTERVAL = float(os.environ.get("KOBAYASHI_TD_VTK_INTERVAL", "0"))
VTK_DIR = os.environ.get("KOBAYASHI_TD_VTK_DIR", "kobayashi_td_vtk")

INNER_SETTINGS = {
    "inner_linear_method": "petsc_gmres",
    "gmres_restart_interval": 30,
    "l_abs_tol": 1.0e-7,
    "l_max_its": 500,
}
BOUNDARY_CONDITIONS = [
    {"name": "xmin", "type": "reflecting"},
    {"name": "xmax", "type": "vacuum"},
    {"name": "ymin", "type": "reflecting"},
    {"name": "ymax", "type": "vacuum"},
    {"name": "zmin", "type": "reflecting"},
    {"name": "zmax", "type": "vacuum"},
]
DOMAIN = RPPLogicalVolume(xmin=0.0, xmax=60.0, ymin=0.0, ymax=100.0, zmin=0.0, zmax=60.0)


def make_time_steps():
    """(dt, theta) for each step."""
    if not math.isclose(BIN_WIDTH / DT, round(BIN_WIDTH / DT)):
        raise ValueError("DT must divide the 2 s benchmark bins")
    steps = [(DT, THETA)] * int(round(BENCHMARK_END / DT))
    steps += [(TAIL_DT, TAIL_THETA)] * int(round((TAIL_END - BENCHMARK_END) / TAIL_DT))
    return steps


def discrete_source_integral(steps):
    """sum_n dt_n q(t^{n+theta_n}), using the inclusive source window of VolumetricSource."""
    total, t = 0.0, 0.0
    for dt, theta in steps:
        if t + theta * dt <= SOURCE_END * (1.0 + 1.0e-12):
            total += dt * SOURCE_STRENGTH
        t += dt
    return total


def make_quadrature():
    if QUADRATURE.startswith("glc"):
        n_polar, n_azimuthal = (int(n) for n in QUADRATURE[3:].split("x"))
        return GLCProductQuadrature3DXYZ(
            n_polar=n_polar, n_azimuthal=n_azimuthal, scattering_order=0
        )
    if not QUADRATURE.startswith("sldfe"):
        raise ValueError(f"Unknown quadrature {QUADRATURE!r}")
    quadrature = SLDFEsqQuadrature3DXYZ(level=int(QUADRATURE[5:]), scattering_order=0)
    if REFINE_CONE_DEGREES > 0.0:
        cone = math.radians(REFINE_CONE_DEGREES)
        for axis in range(3):
            for sign in (1.0, -1.0):
                direction = [0.0, 0.0, 0.0]
                direction[axis] = sign
                quadrature.LocallyRefine(Vector3(*direction), cone, False)
    return quadrature


def make_problem(grid, quadrature, source, time_dependent):
    xs_map = []
    for block_id, (sigma_t, c) in MATERIALS.items():
        xs = MultiGroupXS()
        xs.CreateSimpleOneGroup(sigma_t=sigma_t, c=c, velocity=SPEED)
        xs_map.append({"block_ids": [block_id], "xs": xs})
    options = {"save_angular_flux": True, "use_precursors": False} if time_dependent else {}
    return DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[
            {
                "groups_from_to": (0, 0),
                "angular_quadrature": quadrature,
                "angle_aggregation_type": "single",
                **INNER_SETTINGS,
            }
        ],
        xs_map=xs_map,
        boundary_conditions=BOUNDARY_CONDITIONS,
        volumetric_sources=[source],
        time_dependent=time_dependent,
        options=options,
    )


def sample(problem):
    """Scalar flux at the reference points, its domain integral, minimum, and maximum."""
    phi = problem.GetScalarFluxFieldFunction()[0]
    values = {}
    for name, location in REFERENCE_POINTS.items():
        point = FieldFunctionInterpolationPoint()
        point.SetPointOfInterest(Vector3(*location))
        point.AddFieldFunction(phi)
        point.Execute()
        values[name] = point.GetPointValue()
    for operation in ("sum", "min", "max"):
        volume = FieldFunctionInterpolationVolume()
        volume.SetOperationType(operation)
        volume.SetLogicalVolume(DOMAIN)
        volume.AddFieldFunction(phi)
        volume.Execute()
        # With v = 1 cm/s the flux integral is also the neutron population.
        values["population" if operation == "sum" else operation] = volume.GetValue()
    return values


if __name__ == "__main__":
    steps = make_time_steps()
    source_integral = discrete_source_integral(steps)

    grid = FromFileMeshGenerator(
        filename=MESH_FILE, partitioner=PETScGraphPartitioner(type="parmetis")
    ).Execute()
    grid.SetOrthogonalBoundaries()
    quadrature = make_quadrature()
    if rank == 0:
        print(f"Quadrature: {QUADRATURE}, {len(quadrature.omegas)} directions")

    # Steady-state solve with the discrete source integral.
    steady_source = VolumetricSource(block_ids=[1], group_strength=[source_integral])
    steady_problem = make_problem(grid, quadrature, steady_source, time_dependent=False)
    steady_solver = SteadyStateSourceSolver(problem=steady_problem)
    steady_solver.Initialize()
    steady_solver.Execute()
    steady = sample(steady_problem)
    del steady_solver, steady_problem

    # Transient solve.
    source = VolumetricSource(
        block_ids=[1], group_strength=[SOURCE_STRENGTH], start_time=0.0, end_time=SOURCE_END
    )
    problem = make_problem(grid, quadrature, source, time_dependent=True)
    solver = TransientSolver(
        problem=problem, dt=DT, theta=THETA, stop_time=TAIL_END, initial_state="zero"
    )
    solver.Initialize()

    if VTK_INTERVAL > 0.0:
        os.makedirs(VTK_DIR, exist_ok=True)
    history = [sample(problem)]
    times = [problem.GetTime()]
    frames = []
    next_export = VTK_INTERVAL
    for dt, theta in steps:
        solver.SetTimeStep(dt)
        solver.SetTheta(theta)
        solver.Advance()
        times.append(problem.GetTime())
        history.append(sample(problem))
        export_due = next_export - 1.0e-9 <= times[-1] <= BENCHMARK_END + 1.0e-9
        if VTK_INTERVAL > 0.0 and export_due:
            frame = f"kobayashi_td_{len(frames):04d}"
            FieldFunctionGridBased.ExportMultipleToPVTU(
                problem.GetScalarFluxFieldFunction(), os.path.join(VTK_DIR, frame)
            )
            frames.append((times[-1], f"{frame}.pvtu"))
            next_export += VTK_INTERVAL

    if rank == 0 and frames:
        with open(os.path.join(VTK_DIR, "kobayashi_td.pvd"), "w", encoding="utf-8") as stream:
            stream.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n')
            stream.write("  <Collection>\n")
            for t, name in frames:
                stream.write(f'    <DataSet timestep="{t:g}" part="0" file="{name}"/>\n')
            stream.write("  </Collection>\n</VTKFile>\n")

    if rank == 0:
        series = list(REFERENCE_POINTS) + ["population"]

        # phi^{n+theta} = theta phi^{n+1} + (1 - theta) phi^n is the state the theta scheme
        # solves for in step n; time integrals and bin averages are sums of dt_n phi^{n+theta}.
        def step_value(n, key):
            theta = steps[n][1]
            return theta * history[n + 1][key] + (1.0 - theta) * history[n][key]

        num_bins = int(round(BENCHMARK_END / BIN_WIDTH))
        bins = {key: [0.0] * num_bins for key in series}
        integral = {key: 0.0 for key in series}
        for n, (dt, _) in enumerate(steps):
            for key in series:
                value = dt * step_value(n, key)
                integral[key] += value
                if times[n] < BENCHMARK_END - 1.0e-9:
                    bins[key][int(times[n] / BIN_WIDTH + 1.0e-9)] += value / BIN_WIDTH

        with open("kobayashi_td_results.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["series", "t_lo", "t_hi", "mean"])
            for key in series:
                for b, value in enumerate(bins[key]):
                    t_lo = b * BIN_WIDTH
                    writer.writerow([key, f"{t_lo:g}", f"{t_lo + BIN_WIDTH:g}", f"{value:.8e}"])

        identity_errors = []
        with open("kobayashi_td_summary.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "series",
                    "time_integral",
                    "steady_state",
                    "identity_rel_error",
                    "time_integral_over_kobayashi",
                ]
            )
            for key in series:
                error = (integral[key] - steady[key]) / steady[key]
                identity_errors.append(abs(error))
                ratio = ""
                if key in STEADY_CASE_II_FLUX:
                    ratio = integral[key] / (SOURCE_END * STEADY_CASE_II_FLUX[key])
                    ratio = f"{ratio:.6f}"
                writer.writerow(
                    [key, f"{integral[key]:.10e}", f"{steady[key]:.10e}", f"{error:.3e}", ratio]
                )
                print(f"{key}: time-integral/steady-1 = {error:.3e}, Kobayashi ratio = {ratio}")

        peak_population = max(h["population"] for h in history)
        min_flux = min(h["min"] for h in history)
        max_flux = max(h["max"] for h in history)
        print(f"discrete-source-integral: {source_integral:.6f}")
        print(f"final-population-fraction: {history[-1]['population'] / peak_population:.3e}")
        print(f"min-flux-over-max-flux: {min_flux / max_flux:.3e}")
        print(f"time-integral-identity-max-rel-error: {max(identity_errors):.3e}")
