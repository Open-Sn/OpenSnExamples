import os
from pathlib import Path
import sys
import math
import numpy as np
import csv
from collections import Counter

path = os.getcwd()

sys.path.append("../../..")

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
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

sys.path.append(str(Path(__file__).resolve().parent.parent))
from casl_pin_power import compute_pin_powers  # noqa: E402 -- needs sys.path set up first


def find_repo_root(start):
    for candidate in (start, *start.parents):
        if (candidate / "assets/mesh/casl").exists():
            return candidate
    raise RuntimeError("Could not locate the OpenSn repository root from " + str(start))


casename = '2N'
h5_name = '2n'

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
grid.ExportToPVTU('mesh' + casename)

xs_filename = 'mgxs_' + h5_name + '_one_eighth_SHEM-361.h5'
xs_filepath = str(casl_xs_dir / ('mgxs_casl_' + h5_name) / xs_filename)
xs_dict = {}
xs_list = []

h5_mat_names = ['waba_clad',
                'coated_fuel_clad',
                'coating',
                'normal_fuel',
                'normal_fuel_clad',
                'normal_fuel_gap',
                'coated_fuel',
                'waba_gap',
                'coated_fuel_gap',
                'gt-clad',
                'gt-water-in',
                'gt-water-out',
                'waba_guide',
                'it-clad',
                'it-water-in',
                'it-water-out',
                'normal_fuel_moderator',
                'coated_fuel_moderator',
                'poison',
                'waba_water',
                'water_outside'
                ]

FUEL_XS_NAMES = {"normal_fuel", "coated_fuel"}  # only fissionable materials carry kappa-fission

for name in h5_mat_names:
    xs_dict[name] = MultiGroupXS()
    extra_xs_names = ["kappa-fission"] if name in FUEL_XS_NAMES else []
    xs_dict[name].LoadFromOpenMC(xs_filepath, name, 294.0, extra_xs_names=extra_xs_names)
    xs_list = np.append(xs_list, xs_dict[name])

block_ids = [i for i in range(0, len(xs_list))]

scat_order = 3  # xs_list[0].scattering_order

pquad = GLCProductQuadrature2DXY(n_polar=4, n_azimuthal=32, scattering_order=scat_order)

num_groups = 361

group_sets = [{
    "groups_from_to": [0, num_groups - 1],
    "angular_quadrature": pquad,
    "inner_linear_method": "petsc_gmres",
    "l_abs_tol": 1.0e-7,
    "l_max_its": 300,
}]

# fix this when automating it stops breaking
bound_conditions = [
    {'name': "xmin", 'type': "reflecting"},
    {'name': "xmax", 'type': "reflecting"},
    {'name': "ymin", 'type': "reflecting"},
    {'name': "ymax", 'type': "reflecting"},
]

# fix this when automating it stops breaking
xs_mapping = [
    {'block_ids': [0], 'xs': xs_list[0]},
    {'block_ids': [1], 'xs': xs_list[1]},
    {'block_ids': [2], 'xs': xs_list[2]},
    {'block_ids': [3], 'xs': xs_list[3]},
    {'block_ids': [4], 'xs': xs_list[4]},
    {'block_ids': [5], 'xs': xs_list[5]},
    {'block_ids': [6], 'xs': xs_list[6]},
    {'block_ids': [7], 'xs': xs_list[7]},
    {'block_ids': [8], 'xs': xs_list[8]},
    {'block_ids': [9], 'xs': xs_list[9]},
    {'block_ids': [10], 'xs': xs_list[10]},
    {'block_ids': [11], 'xs': xs_list[11]},
    {'block_ids': [12], 'xs': xs_list[12]},
    {'block_ids': [13], 'xs': xs_list[13]},
    {'block_ids': [14], 'xs': xs_list[14]},
    {'block_ids': [15], 'xs': xs_list[15]},
    {'block_ids': [16], 'xs': xs_list[16]},
    {'block_ids': [17], 'xs': xs_list[17]},
    {'block_ids': [18], 'xs': xs_list[18]},
    {'block_ids': [19], 'xs': xs_list[19]},
    {'block_ids': [20], 'xs': xs_list[20]}
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
    },
)


cmfd = CMFDAcceleration(
    problem=phys,
    current_closure="auto",
    relaxation=0.5,
    coarse_mesh="local_aggregation",
    coarse_solver_policy="direct",
    aggregation_size=32,
    group_aggregation_size=(num_groups + 31) // 32,
    update_wgs_max_its=1,
    update_wgs_abs_tol=1.0e-12,
    balance_residual_tolerance=1.0e-6,
)

k_solver = PowerIterationKEigenSolver(problem=phys, acceleration=cmfd,
                                      k_tol=1.0e-6
                                      )
k_solver.Initialize()
k_solver.Execute()

keff = k_solver.GetEigenvalue()

num_cells = 17

your_files = os.getcwd()


def read_csv_to_2d_array(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    return np.asarray(data)


def count_frequencies(data):
    flattened_data = [item for row in data for item in row]  # Flatten 2D array into a 1D list
    cell_frequencies = Counter(flattened_data)
    print("cell name frequency:")
    total = 0
    for key, value in cell_frequencies.items():
        print(f'"{key}": {value}')
        total += value
    print("total: ", total)


csv_filename = 'FA_cell_names_1_family.csv'
csv_filepath = your_files + '/' + csv_filename

lattice_csv = read_csv_to_2d_array(csv_filepath)

count_frequencies(lattice_csv)

if lattice_csv.shape[0] != lattice_csv.shape[1]:
    raise Exception('CSV array of cell names is not square.')

csv_size = lattice_csv.shape[0]

label_to_block_xs = {
    "fu": (h5_mat_names.index("normal_fuel"), xs_dict["normal_fuel"]),
    "c": (h5_mat_names.index("coated_fuel"), xs_dict["coated_fuel"]),
}
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

maxes = np.zeros([num_cells])
for i in range(0, num_cells):
    maxes[i] = max(val_table[:, i])
val_max = max(maxes)

if rank == 0:
    np.savetxt("power.txt", val_table / val_max)
    with open("keff.txt", "w") as file:
        file.write(str(keff))
