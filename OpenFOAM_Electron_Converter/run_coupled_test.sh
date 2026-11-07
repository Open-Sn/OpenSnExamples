#!/bin/sh
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT
set -eu

case_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
opensn_ranks=${OPENSN_RANKS:-16}
# Wall-clock limit for each participant. A participant whose partner has failed
# otherwise waits for the preCICE connection indefinitely.
coupling_timeout=${COUPLING_TIMEOUT:-3600}

: "${OPENSN:?Set OPENSN to a preCICE-enabled OpenSn executable}"
if [ ! -x "$OPENSN" ]; then
    echo "OPENSN='$OPENSN' is not an executable file" >&2
    exit 2
fi
case "$opensn_ranks" in
    ''|*[!0-9]*|0) echo "OPENSN_RANKS must be a positive integer" >&2; exit 2 ;;
esac
case "$coupling_timeout" in
    ''|*[!0-9]*|0) echo "COUPLING_TIMEOUT must be a positive number of seconds" >&2; exit 2 ;;
esac

for command in wmake foamMultiRun gmshToFoam splitMeshRegions pkg-config mpirun timeout; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "Required command '$command' is not available" >&2
        exit 2
    }
done
pkg-config --atleast-version=3.4 libprecice
foam_version=${WM_PROJECT_VERSION:-}
case "$foam_version" in
    14) ;;
    *) echo "This example requires OpenFOAM Foundation v14; found '$foam_version'" >&2; exit 2 ;;
esac

rm -rf "$case_dir/meshCase"
mkdir -p "$case_dir/meshCase/system"
cp "$case_dir/system/controlDict" "$case_dir/meshCase/system/controlDict"
cp "$case_dir/system/fvSchemes" "$case_dir/meshCase/system/fvSchemes"
cp "$case_dir/system/fvSolution" "$case_dir/meshCase/system/fvSolution"

"$case_dir/openfoam_adapter/build.sh"
# preCICE connects through loopback 'lo' unless PRECICE_NETWORK names another
# interface, for example lo0 on macOS.
if [ -n "${PRECICE_NETWORK:-}" ]; then
    python3 "$case_dir/configure_precice.py" --network "$PRECICE_NETWORK"
else
    python3 "$case_dir/configure_precice.py" --no-network
fi
gmshToFoam "$case_dir/waterblock.msh" -case "$case_dir/meshCase"
splitMeshRegions -case "$case_dir/meshCase" \
    -cellZones '(tungsten copper aluminum air steel water)'
python3 "$case_dir/prepare_openfoam.py" --end-time 0.1
rm -rf "$case_dir/precice-run" "$case_dir/meshCase/precice-run"

openfoam_log="$case_dir/meshCase/log.precice-openfoam"
opensn_log="$case_dir/meshCase/log.precice-opensn"
foam_pid=
cleanup()
{
    if [ -n "$foam_pid" ]; then
        kill "$foam_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT HUP INT TERM

# exec makes $! the timeout process, which forwards termination to foamMultiRun.
(
    export LD_LIBRARY_PATH="$case_dir/openfoam_adapter/lib:${LD_LIBRARY_PATH:-}"
    cd "$case_dir/meshCase"
    exec timeout "$coupling_timeout" foamMultiRun
) >"$openfoam_log" 2>&1 &
foam_pid=$!

set +e
(
    cd "$case_dir/meshCase"
    export OPENSN_PRECICE_CONFIG="$case_dir/precice-config.runtime.xml"
    exec timeout "$coupling_timeout" \
        mpirun -np "$opensn_ranks" "$OPENSN" -i ../transport.py
) >"$opensn_log" 2>&1
opensn_status=$?
set -e
if [ "$opensn_status" -ne 0 ]; then
    echo "OpenSn coupling participant failed (status $opensn_status); see $opensn_log" >&2
    exit "$opensn_status"
fi

set +e
wait "$foam_pid"
openfoam_status=$?
set -e
foam_pid=
trap - EXIT HUP INT TERM
if [ "$openfoam_status" -ne 0 ]; then
    echo "OpenFOAM coupling participant failed (status $openfoam_status); see $openfoam_log" >&2
    exit "$openfoam_status"
fi

python3 "$case_dir/validate_precice_coupling.py" "$opensn_log" "$openfoam_log"
echo "Coupled 3D OpenSn-OpenFOAM smoke test passed with OpenFOAM Foundation v14."
