#!/bin/bash
# SPDX-License-Identifier: MIT
# Generate the fine mesh. Gmsh runs on one thread (the same mesh every time); the size field
# along the port beams makes this slow: 54 minutes for h10p3 (1.3 M tetrahedra) on the
# machine of the README; finer port sizes take longer.
#SBATCH --job-name=chamber-mesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/env.sh"
mkdir -p "$RUN_DIR" && cd "$RUN_DIR"
"$CHAMBER_PYTHON" "$EXAMPLE_DIR/generate_mesh.py" --size "$MESH_SIZE" --port-size "$PORT_SIZE" \
    --out "$MESH.msh"
