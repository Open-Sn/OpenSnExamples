#!/bin/bash
# SPDX-License-Identifier: MIT
# Stage 1: the uncollided flux, ray traced through the material hull, on many ranks. Its file
# holds every group, also those with no source: about 74 kB per tetrahedron with 64 groups and
# P3 moments, gathered on rank 0 when it is written. Give rank 0 a node to itself if needed.
#SBATCH --job-name=chamber-uncollided
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/env.sh"
cd "$RUN_DIR"
export OPENSN_NUM_THREADS=1
$LAUNCH "$OPENSN" -i "$EXAMPLE_DIR/chamber.py" --suppress-color \
    --py "mesh_file=\"$MESH.msh\"" --py "library=\"$LIBRARY\"" --py "stage=\"uncollided\"" \
    --py "prefix=\"$PREFIX\"" --py n_polar=$N_POLAR --py n_azimuthal=$N_AZIMUTHAL \
    > "$PREFIX.uncollided.log" 2>&1
