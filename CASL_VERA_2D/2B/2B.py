import os
from pathlib import Path
import sys
import math
import numpy as np
import csv
from collections import Counter
from typing import Any, cast

path = os.getcwd()

sys.path.append("../../..")

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    barrier = MPI.COMM_WORLD.Barrier
    from pyopensn.mesh import FromFileMeshGenerator, KBAGraphPartitioner
    from pyopensn.xs import MultiGroupXS
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.solver import (
        DiscreteOrdinatesProblem,
        PowerIterationKEigenSolver,
        CMFDAcceleration,
    )
    from pyopensn.logvol import RCCLogicalVolume
    from pyopensn.post import VolumePostprocessor
else:
    barrier = cast(Any, globals()["MPIBarrier"])

sys.path.append(str(Path(__file__).resolve().parent.parent))
from casl_pin_power import compute_pin_powers  # noqa: E402 -- needs sys.path set up first


def find_repo_root(start):
    for candidate in (start, *start.parents):
        if (candidate / "assets/mesh/casl").exists():
            return candidate
    raise RuntimeError("Could not locate the OpenSn repository root from " + str(start))


casename = "2B"
h5_name = "2b"

script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
repo_root = find_repo_root(script_dir)
casl_mesh_dir = repo_root / "assets/mesh/casl"
casl_xs_dir = repo_root / "assets/xs/casl"

mesh_filepath = str(casl_mesh_dir / ('lattice_' + casename + '.obj'))


def make_spatial_partitioner(filename):
    num_partitions = size
    nx = math.isqrt(num_partitions)
    while num_partitions % nx != 0:
        nx -= 1
    ny = num_partitions // nx
    if nx < ny:
        nx, ny = ny, nx

    xmin = ymin = float("inf")
    xmax = ymax = -float("inf")
    with open(filename, encoding="utf-8") as mesh_file:
        for line in mesh_file:
            if not line.startswith("v "):
                continue
            _, x, y, *_ = line.split()
            x = float(x)
            y = float(y)
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)

    xcuts = [xmin + (xmax - xmin) * i / nx for i in range(1, nx)]
    ycuts = [ymin + (ymax - ymin) * i / ny for i in range(1, ny)]
    return KBAGraphPartitioner(nx=nx, ny=ny, xcuts=xcuts, ycuts=ycuts)


meshgen = FromFileMeshGenerator(
    filename=mesh_filepath,
    partitioner=make_spatial_partitioner(mesh_filepath)
)

grid = meshgen.Execute()
grid.SetOrthogonalBoundaries()
grid.ExportToPVTU("mesh_2B")

xs_filename = 'mgxs_' + h5_name + '_one_eighth_SHEM-361.h5'
xs_filepath = str(casl_xs_dir / ('mgxs_casl_' + h5_name) / xs_filename)
xs_dict = {}
xs_list = []

h5_mat_names = [
    "fuel",
    "fuel clad",
    "fuel gap",
    "gt-clad",
    "gt-water-in",
    "gt-water-out",
    "it-clad",
    "it-water-in",
    "it-water-out",
    "moderator",
    "water_outside",
]

FUEL_XS_NAMES = {"fuel"}  # only fissionable materials carry "kappa-fission" data in the MGXS file

for name in h5_mat_names:
    xs_dict[name] = MultiGroupXS()
    extra_xs_names = ["kappa-fission"] if name in FUEL_XS_NAMES else []
    xs_dict[name].LoadFromOpenMC(xs_filepath, name, 294.0, extra_xs_names=extra_xs_names)
    xs_list = np.append(xs_list, xs_dict[name])

block_ids = [i for i in range(0, len(xs_list))]

scat_order = 3  # xs_list[0].scattering_order

pquad = GLCProductQuadrature2DXY(n_polar=4, n_azimuthal=32, scattering_order=scat_order)

num_groups = 361

group_sets = [
    {
        "groups_from_to": [0, num_groups - 1],
        "angular_quadrature": pquad,
        "angle_aggregation_type": "polar",
        "inner_linear_method": "petsc_gmres",
        "l_abs_tol": 1.0e-7,
        "l_max_its": 300,
    }
]

# fix this when automating it stops breaking
bound_conditions = [
    {"name": "xmin", "type": "reflecting"},
    {"name": "xmax", "type": "reflecting"},
    {"name": "ymin", "type": "reflecting"},
    {"name": "ymax", "type": "reflecting"},
]

# fix this when automating it stops breaking
xs_mapping = [
    {"block_ids": [0], "xs": xs_list[0]},
    {"block_ids": [1], "xs": xs_list[1]},
    {"block_ids": [2], "xs": xs_list[2]},
    {"block_ids": [3], "xs": xs_list[3]},
    {"block_ids": [4], "xs": xs_list[4]},
    {"block_ids": [5], "xs": xs_list[5]},
    {"block_ids": [6], "xs": xs_list[6]},
    {"block_ids": [7], "xs": xs_list[7]},
    {"block_ids": [8], "xs": xs_list[8]},
    {"block_ids": [9], "xs": xs_list[9]},
    {"block_ids": [10], "xs": xs_list[10]},
]

phys = DiscreteOrdinatesProblem(
    mesh=grid,
    num_groups=num_groups,
    groupsets=group_sets,
    xs_map=xs_mapping,
    boundary_conditions=bound_conditions,
    options={
        "verbose_inner_iterations": True,
        "verbose_outer_iterations": True,
        "use_precursors": False,
        "power_default_kappa": 1.0,
        "save_angular_flux": False,
        "restart_writes_enabled": True,
        "write_delayed_psi_to_restart": True,
        "write_restart_path": "./2B_",
    },
)


cmfd = CMFDAcceleration(
    problem=phys,
    current_closure="auto",
    relaxation=1.0,
    coarse_mesh="local_aggregation",
    coarse_solver_policy="direct",
    aggregation_size=64,
    # group_aggregation_size=(num_groups + 31) // 32,
    group_aggregation_size=(num_groups + 63) // 64,
    update_wgs_max_its=1,
    update_wgs_abs_tol=1.0e-12,
    balance_residual_tolerance=1.0e-5,
)

k_solver = PowerIterationKEigenSolver(problem=phys, acceleration=cmfd, k_tol=1.0e-6)
k_solver.Initialize()
k_solver.Execute()

keff = k_solver.GetEigenvalue()


def read_csv_to_2d_array(file_path):
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    return np.asarray(data)


def count_frequencies(data):
    flattened_data = [item for row in data for item in row]  # Flatten 2D array into a 1D list
    return Counter(flattened_data)


your_files = os.getcwd()
csv_filename = "FA_cell_names_1_family.csv"
csv_filepath = your_files + "/" + csv_filename
lattice_csv = read_csv_to_2d_array(csv_filepath)

cell_frequencies = count_frequencies(lattice_csv)
if rank == 0:
    print("cell name frequency:")
    total = 0
    for key, value in cell_frequencies.items():
        print(f'"{key}": {value}')
        total += value
    print("total: ", total)

num_cells = lattice_csv.shape[0]
if num_cells != lattice_csv.shape[1]:
    raise Exception("CSV array of cell names is not square.")


num_cells_quarter = np.ceil(num_cells / 2).astype(np.int64)

label_to_block_xs = {"fu": (h5_mat_names.index("fuel"), xs_dict["fuel"])}
pin_positions_csv = casl_mesh_dir / ('pin_positions_' + casename + '.csv')
val_table = compute_pin_powers(
    phys, RCCLogicalVolume, VolumePostprocessor, pin_positions_csv, label_to_block_xs,
    lattice_csv.shape,
)

# Raw (pre-normalization) values, saved for regression testing (see check_pin_powers.py) --
# unlike the peak-normalized power.txt below, these are directly comparable run-to-run: no
# division by a run-dependent "which pin is the max" statistic that a tiny amount of solve
# noise can flip between two near-tied pins, disproportionately amplifying that noise.
if rank == 0:
    np.savetxt("power_raw.txt", val_table)

val_table_ori = val_table.copy()

# quarter array
# NOTE: this used to also flip columns here (np.flip(val_table, axis=1)) before extracting A --
# that was compensating for the OLD empirical-clustering pipeline's brute-force axis/sign/offset
# search occasionally landing on a mirrored (but still label-valid) reading specifically for this
# case's own label pattern. The new pipeline's row/col convention is deterministic and uniform
# across all 17 cases (row=round(-y/pitch)+CENTER_ROW, col=round(x/pitch)+CENTER_COL, see
# generate_lattice_meshes.py's compute_pin_positions) and already matches lattice_csv's own
# indexing directly (verified for 2A via an exact positional match on all 8 "gt" positions), so
# the pre-flip is no longer needed -- keeping it actively broke the reconstruction (confirmed: it
# collapsed 264 pins down to 12 nonzero values). Confirmed removing it is correct by feeding the
# real benchmark reference through this exact reconstruction: without the flip, the output matches
# the reference exactly (max abs diff 0.000000); with it, max abs diff was >1.0.
A = val_table[:num_cells_quarter, :num_cells_quarter]
# NOTE: this used to double the last row/col here (compensating for a half-pin-at-the-cut
# under-count) -- no longer needed now that compute_pin_powers itself already scales cut pins
# up to their true full-pin-equivalent value (see casl_pin_power.py's module docstring, fix #3).
# Doing both here and there double-corrected (~2x too high at every cut position, confirmed via
# a real solve: max error was +83% before removing this).
val_table_quarter = A.copy()
A_flipped = np.flip(A, axis=1)
B = np.hstack([A, A_flipped[:, 1:]])
B_flipped = np.flip(B, axis=0)
val_table = np.vstack([B, B_flipped[1:, :]])

norm = np.sum(val_table) / cell_frequencies["fu"]
val_table /= norm

barrier()

if rank == 0:
    # print("sum=", np.sum(val_table))
    np.savetxt("power.txt", val_table)
    # np.savetxt("power_quarter.txt", val_table_quarter)
    # np.savetxt("power_ori.txt", val_table_ori)
    with open("keff.txt", "w") as file:
        file.write(str(keff))

vtk_basename = "flx_4_32_"
# FieldFunctionGridBased.ExportMultipleToVTK([fflist[g] for g in range(num_groups)], vtk_basename)
