# SPDX-License-Identifier: MIT
# Settings shared by the job scripts of this directory. Edit the lines marked CHANGE_ME.

# The directory of the example (ICF_Target_Chamber), where the libraries and scripts are.
export EXAMPLE_DIR=${EXAMPLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# Where the runs write their files; a scratch file system is best (CHANGE_ME).
export RUN_DIR=${RUN_DIR:-$EXAMPLE_DIR/large_runs}
# OpenSn console built from the uncollided-hull-accel branch (commit 584810e0e or later), until
# point sources on reflecting planes are merged (CHANGE_ME).
export OPENSN=${OPENSN:-$HOME/opensn/build/python/opensn}
# MPI launcher of the machine: srun under SLURM, or mpiexec -n (CHANGE_ME).
export LAUNCH=${LAUNCH:-srun}
# The "chamber" conda environment of environment.yml, for Gmsh, OpenMC and the post-processing.
export CHAMBER_PYTHON=${CHAMBER_PYTHON:-$HOME/miniforge3/envs/chamber/bin/python}
# ENDF/B-VIII.0 HDF5 library distributed by the OpenMC project (CHANGE_ME).
export OPENMC_CROSS_SECTIONS=${OPENMC_CROSS_SECTIONS:-$HOME/nuclear_data/endfb-viii.0-hdf5/cross_sections.xml}

# VisIt, for the optional three-dimensional views of 05_visit.sh: the command, after any module
# load the machine needs (CHANGE_ME).
export VISIT=${VISIT:-visit}

# The fine configuration. See README.md for the memory each choice needs.
export MESH=${MESH:-chamber_h10p3}           # file name without .msh
export MESH_SIZE=${MESH_SIZE:-10}            # generate_mesh.py --size
export PORT_SIZE=${PORT_SIZE:-3}             # generate_mesh.py --port-size
export LIBRARY=${LIBRARY:-64g_p3}
export N_POLAR=${N_POLAR:-6}                 # 6 x 24 = 144 directions
export N_AZIMUTHAL=${N_AZIMUTHAL:-24}
export GROUPSET_SIZE=${GROUPSET_SIZE:-0}
export GMRES_RESTART=${GMRES_RESTART:-5}
export PREFIX=${PREFIX:-chamber_${LIBRARY}_${MESH}_$((N_POLAR * N_AZIMUTHAL))}
