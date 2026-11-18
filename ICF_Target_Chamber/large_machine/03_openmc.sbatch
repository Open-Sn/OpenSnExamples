#!/bin/bash
# SPDX-License-Identifier: MIT
# Continuous-energy OpenMC reference with ten times the histories of the committed one
# (10^10 instead of 10^9): standard errors about 3 times smaller, mostly for the line-outs.
# With an OpenMC built with MPI, use --processes and one process per node or socket.
#SBATCH --job-name=chamber-openmc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/env.sh"
mkdir -p "$RUN_DIR" && cd "$RUN_DIR"
"$CHAMBER_PYTHON" "$EXAMPLE_DIR/openmc_reference.py" --particles 100000000 --batches 100 \
    --seed 20261114 --processes "$SLURM_NNODES" --threads "$SLURM_CPUS_PER_TASK"
