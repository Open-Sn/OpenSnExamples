#!/bin/sh
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
gmsh_executable=${GMSH_EXECUTABLE:-gmsh}

command -v "$gmsh_executable" >/dev/null 2>&1 || {
    echo "Required command '$gmsh_executable' is not available" >&2
    exit 2
}

# run_coupled_test.sh converts and splits this mesh for OpenFOAM on every run.
"$gmsh_executable" "$script_dir/waterblock.geo" -3 -format msh2 \
    -o "$script_dir/waterblock.msh"
