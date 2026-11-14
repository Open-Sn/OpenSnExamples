#!/bin/bash
# SPDX-License-Identifier: MIT
# Stage 2: the collided flux by discrete ordinates, GMRES in each groupset. Size the job with
# the memory model of README.md; the number of ranks does not change the result.
#SBATCH --job-name=chamber-collided
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=32
#SBATCH --time=12:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/env.sh"
cd "$RUN_DIR"
export OPENSN_NUM_THREADS=1
$LAUNCH "$OPENSN" -i "$EXAMPLE_DIR/chamber.py" --suppress-color \
    --py "mesh_file=\"$MESH.msh\"" --py "library=\"$LIBRARY\"" --py "stage=\"collided\"" \
    --py "prefix=\"$PREFIX\"" --py n_polar=$N_POLAR --py n_azimuthal=$N_AZIMUTHAL \
    --py groupset_size=$GROUPSET_SIZE --py gmres_restart=$GMRES_RESTART \
    > "$PREFIX.collided.log" 2>&1
