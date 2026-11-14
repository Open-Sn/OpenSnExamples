#!/bin/bash
# SPDX-License-Identifier: MIT
# Compare the fine run with the OpenMC reference and make the figures.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$RUN_DIR"
REFERENCE=${REFERENCE:-$EXAMPLE_DIR/openmc_reference_results.json}
"$CHAMBER_PYTHON" "$EXAMPLE_DIR/compare_openmc.py" --openmc "$REFERENCE" --opensn "$PREFIX" \
    --exact "$EXAMPLE_DIR/uncollided_reference.json" \
    --provenance "$EXAMPLE_DIR/provenance/materials.json" | tee "$PREFIX.comparison.txt"
"$CHAMBER_PYTHON" "$EXAMPLE_DIR/make_figures.py" --openmc "$REFERENCE" \
    --opensn "$PREFIX:$LIBRARY, $MESH" --exact "$EXAMPLE_DIR/uncollided_reference.json" \
    --provenance "$EXAMPLE_DIR/provenance/materials.json" --out "$RUN_DIR/assets" \
    --libraries "64 groups P3:$EXAMPLE_DIR/materials_64g_p3.h5" \
    "4 groups P0, transport corrected:$EXAMPLE_DIR/materials_4g_p0tc.h5"
