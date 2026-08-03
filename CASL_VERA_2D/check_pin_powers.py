# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Runs the CASL VERA Problem 2 cases and checks their pin-power output for regressions.
Pin-power-specific logic lives here instead of in test.py so it stays a small generic
runner rather than accumulating per-example special cases.

Reuses run_all_cases.py (to actually run each case) and compare_casl_pin_power.py (for the
informational benchmark comparison) rather than duplicating either -- this script's own
contribution is just the regression check (power.txt vs. power_baseline.txt) and the pass/fail
aggregation across cases.

Two checks per case:

1. Regression: power_raw.txt (freshly produced by this run, the raw kappa-fission-weighted
   VolumePostprocessor output before any normalization) must closely match power_raw_baseline.txt.

2. Benchmark comparison (informational only, printed via compare_casl_pin_power.py, never fails
   this script): how OpenSn's pin powers compare to the VERA benchmark's own reference. Some
   divergence from the benchmark is expected (e.g. 2Q's spacer grid -- see that case's own notes),
   so this is not a pass/fail test.

Exit code 0 if every requested case's regression check passes (and ran successfully, if not
--skip-run), 1 otherwise.

Examples
--------
  python3 check_pin_powers.py                      # run + check all 17 cases, 64 ranks each
  python3 check_pin_powers.py 2A 2B --ranks 8      # just these two, 8 ranks each
  python3 check_pin_powers.py --skip-run           # don't re-run -- check whatever power.txt/
                                                   # keff.txt are already sitting in each case
                                                   # directory (e.g. after run_all_cases.py, or
                                                   # after extracting a results.tar.gz)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_all_cases import CASES, find_opensn_executable, run_case  # noqa: E402
from compare_casl_pin_power import compare_case, load_reference_keff  # noqa: E402


def check_pin_power_regression(case_dir, rel_tol=1.0e-6, abs_tol=1.0e-6):
    """Compares case_dir/power_raw.txt against case_dir/power_raw_baseline.txt (both raw,
    pre-normalization values). Returns None if they match (or if there's no baseline to
    check against, i.e. the case hasn't been baselined yet), or a short string describing
    the mismatch otherwise.
    """
    power_path = case_dir / "power_raw.txt"
    baseline_path = case_dir / "power_raw_baseline.txt"
    if not baseline_path.is_file():
        return None
    if not power_path.is_file():
        return f"missing {power_path}"

    power = np.loadtxt(power_path)
    baseline = np.loadtxt(baseline_path)
    if power.shape != baseline.shape:
        return (
            f"shape mismatch: power_raw.txt {power.shape} vs. power_raw_baseline.txt "
            f"{baseline.shape}"
        )
    if np.allclose(power, baseline, rtol=rel_tol, atol=abs_tol):
        return None
    diff = np.abs(power - baseline)
    i = int(np.argmax(diff))
    return (
        f"max abs diff {diff.flat[i]:.3e} at flat index {i} "
        f"(power_raw={power.flat[i]!r}, baseline={baseline.flat[i]!r})"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cases", nargs="*", default=CASES, help="Case names (default: all 17)")
    parser.add_argument("--ranks", type=int, default=64, help="MPI ranks per case (default: 64)")
    parser.add_argument(
        "--opensn", type=Path, default=None,
        help="Path to the opensn executable (default: searched for, or $OPENSN_EXECUTABLE)",
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Don't run the cases -- check whatever power.txt/keff.txt are already present "
        "(e.g. from a prior run_all_cases.py invocation or an extracted results.tar.gz).",
    )
    args = parser.parse_args()

    unknown = [c for c in args.cases if c not in CASES]
    if unknown:
        sys.exit(f"Unknown case(s): {', '.join(unknown)} -- valid cases are {', '.join(CASES)}")

    opensn_exe = None
    if not args.skip_run:
        opensn_exe = args.opensn or find_opensn_executable()
        if not opensn_exe.exists():
            sys.exit(f"opensn executable not found at {opensn_exe}")

    log_dir = HERE / "run_logs"
    log_dir.mkdir(exist_ok=True)
    reference_keff = load_reference_keff()

    failures = []
    for case in args.cases:
        print(f"=== {case} ===", flush=True)
        case_dir = HERE / case

        if not args.skip_run:
            result = run_case(case, opensn_exe, args.ranks, log_dir, dry_run=False)
            if not result["success"]:
                print(f"  FAILED to run (see {result['log']})")
                failures.append(case)
                continue

        error = check_pin_power_regression(case_dir)
        if error is None:
            print("  regression check: OK")
        else:
            print(f"  REGRESSION: {error}")
            failures.append(case)

        compare_case(case, reference_keff)  # informational only -- never affects pass/fail

    print()
    if failures:
        print(f"FAILED: {len(failures)}/{len(args.cases)} case(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"PASSED: all {len(args.cases)} case(s)")
    sys.exit(0)


if __name__ == "__main__":
    main()
