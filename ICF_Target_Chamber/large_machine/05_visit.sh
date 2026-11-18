#!/bin/bash
# SPDX-License-Identifier: MIT
# Optional: three-dimensional views of the total flux of the fine run, rendered with VisIt from
# <prefix>_flux.pvtu (light and dark PNG files in $RUN_DIR/assets). Run it where VisIt is
# available and memory is ample (a compute node rather than a login node); VISIT_ARGS can add a
# parallel engine, for example VISIT_ARGS="-np 32".
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$RUN_DIR"
# shellcheck disable=SC2086
"$VISIT" -nowin -cli ${VISIT_ARGS:-} -s "$EXAMPLE_DIR/visit_figures.py" \
    --pvtu "${PREFIX}_flux.pvtu" --out "$RUN_DIR/assets" --name "${PREFIX}_flux3d"
