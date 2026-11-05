#!/bin/sh
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT
set -eu

adapter_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${WM_PROJECT_VERSION:?Source OpenFOAM Foundation v14 before building}"
test "$WM_PROJECT_VERSION" = 14 || {
    echo "OpenFOAM Foundation v14 is required; found '$WM_PROJECT_VERSION'" >&2
    exit 2
}
pkg-config --atleast-version=3.4 libprecice

export FOAM_USER_LIBBIN="$adapter_dir/lib"
mkdir -p "$FOAM_USER_LIBBIN"
(cd "$adapter_dir" && wmake libso)
