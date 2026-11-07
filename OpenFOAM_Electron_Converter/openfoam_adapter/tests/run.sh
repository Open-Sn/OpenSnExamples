#!/bin/sh
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT
set -eu

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
adapter_dir=$(CDPATH= cd -- "$test_dir/.." && pwd)
case_dir="$test_dir/case"
config="$test_dir/precice-config.xml"

"$adapter_dir/build.sh"
export FOAM_USER_APPBIN="$test_dir/bin"
export LD_LIBRARY_PATH="$adapter_dir/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$FOAM_USER_APPBIN"
(cd "$test_dir" && wmake)
mpicxx -std=c++17 "$test_dir/powerSender.C" \
    $(pkg-config --cflags --libs libprecice) -o "$FOAM_USER_APPBIN/powerSender"

for region in regionA regionB regionC; do
    blockMesh -case "$case_dir" -region "$region" >/dev/null
done
rm -rf "$test_dir/precice-run"
(cd "$test_dir" && timeout 30s "$FOAM_USER_APPBIN/powerSender" "$config") >"$test_dir/sender.log" 2>&1 &
sender_pid=$!
trap 'kill "$sender_pid" 2>/dev/null || true' EXIT
(cd "$test_dir" && timeout 30s "$FOAM_USER_APPBIN/multiRegionPrecicePowerTest" \
    -case "$case_dir" -precice-config "$config") >"$test_dir/receiver.log" 2>&1
wait "$sender_pid"
trap - EXIT
grep "MULTIREGION_RECEIVED_POWER_W=6" "$test_dir/receiver.log"

decomposePar -case "$case_dir" -allRegions -force >/dev/null
rm -rf "$test_dir/precice-run"
(cd "$test_dir" && timeout 30s "$FOAM_USER_APPBIN/powerSender" "$config") >"$test_dir/sender-mpi.log" 2>&1 &
sender_pid=$!
trap 'kill "$sender_pid" 2>/dev/null || true' EXIT
(cd "$test_dir" && timeout 30s mpirun -np 2 \
    "$FOAM_USER_APPBIN/multiRegionPrecicePowerTest" -case "$case_dir" -parallel \
    -precice-config "$config") >"$test_dir/receiver-mpi.log" 2>&1
wait "$sender_pid"
trap - EXIT
grep "MULTIREGION_RECEIVED_POWER_W=6" "$test_dir/receiver-mpi.log"

rm -rf "$test_dir/precice-run"
ulimit -c 0
(cd "$test_dir" && timeout 30s "$FOAM_USER_APPBIN/powerSender" "$config") >"$test_dir/sender-mismatch.log" 2>&1 &
sender_pid=$!
trap 'kill "$sender_pid" 2>/dev/null || true' EXIT
if (cd "$test_dir" && timeout 30s "$FOAM_USER_APPBIN/multiRegionPrecicePowerTest" \
    -case "$case_dir" -precice-config "$config" \
    -expected-power 5) >"$test_dir/receiver-mismatch.log" 2>&1; then
    echo "Expected the power-conservation check to fail" >&2
    exit 1
fi
wait "$sender_pid"
trap - EXIT
grep "exceeding tolerance" "$test_dir/receiver-mismatch.log"
