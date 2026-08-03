# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Runs every VERA Problem 2 case (2A-2Q) one at a time, leaving power.txt/keff.txt in each
case's directory for later comparison (see compare_casl_pin_power.py).

Continues past a failed case (logs it, moves on to the next) rather than aborting the whole
batch. Progress is written incrementally to run_all_summary.json after every case, so an
interrupted batch doesn't lose what already completed.

Examples
--------
  python3 run_all_cases.py                            # all 17 cases, 64 ranks each
  python3 run_all_cases.py 2A 2B --ranks 32           # just these two, 32 ranks each
  python3 run_all_cases.py --dry-run                  # print the commands without running them
  python3 run_all_cases.py --archive results.tar.gz   # also bundle results
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).parent
CASES = [
    "2A", "2B", "2C", "2D", "2E", "2F", "2G", "2H", "2I", "2J", "2K", "2L", "2M", "2N", "2O",
    "2P", "2Q",
]


def find_opensn_executable():
    env_path = os.environ.get("OPENSN_EXECUTABLE")
    if env_path:
        return Path(env_path)
    for candidate in (HERE, *HERE.parents):
        exe = candidate / "build" / "python" / "opensn"
        if exe.exists():
            return exe
    raise RuntimeError(
        "Could not find the opensn executable by searching upward from this script for "
        "build/python/opensn -- set OPENSN_EXECUTABLE to its path."
    )


def clean_case_outputs(case_dir, case):
    """Removes stale outputs from a previous run of this case, so a rerun can't be confused by
    leftover files from an earlier failed or partial attempt."""
    patterns = [
        "power.txt", "power_raw.txt", "keff.txt", "mesh*.pvtu", "mesh*.vtu",
        f"{case}_*.restart.h5",
    ]
    for pattern in patterns:
        for f in case_dir.glob(pattern):
            f.unlink()


def run_case(case, opensn_exe, ranks, log_dir, dry_run):
    case_dir = HERE / case
    cmd = ["mpiexec", "-n", str(ranks), str(opensn_exe), "-i", f"{case}.py"]

    if dry_run:
        print(f"[dry-run] (cd {case_dir} && {' '.join(cmd)})")
        return {"case": case, "success": None, "keff": None, "elapsed_s": 0.0, "log": None}

    clean_case_outputs(case_dir, case)
    log_path = log_dir / f"{case}.log"

    start = time.time()
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, cwd=case_dir, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    power_path = case_dir / "power.txt"
    power_raw_path = case_dir / "power_raw.txt"
    keff_path = case_dir / "keff.txt"
    success = (
        result.returncode == 0
        and power_path.exists()
        and power_raw_path.exists()
        and keff_path.exists()
    )

    keff = None
    if keff_path.exists():
        try:
            keff = float(keff_path.read_text().strip())
        except ValueError:
            pass

    return {
        "case": case,
        "success": success,
        "returncode": result.returncode,
        "elapsed_s": elapsed,
        "keff": keff,
        "log": str(log_path.relative_to(HERE)),
    }


def write_archive(archive_path, summary):
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(HERE / "run_all_summary.json", arcname="run_all_summary.json")
        for result in summary:
            case = result["case"]
            case_dir = HERE / case
            for name in ("power.txt", "power_raw.txt", "keff.txt"):
                path = case_dir / name
                if path.exists():
                    tar.add(path, arcname=f"{case}/{name}")
            if result["log"]:
                log_path = HERE / result["log"]
                if log_path.exists():
                    tar.add(log_path, arcname=result["log"])
    print(f"\nArchived results to {archive_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cases", nargs="*", default=CASES, help="Case names (default: all 17)")
    parser.add_argument("--ranks", type=int, default=64, help="MPI ranks per case (default: 64)")
    parser.add_argument(
        "--opensn", type=Path, default=None,
        help="Path to the opensn executable (default: searched for as build/python/opensn "
        "relative to this script, or $OPENSN_EXECUTABLE)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run, without running them",
    )
    parser.add_argument(
        "--archive", type=Path, default=None,
        help="Also bundle power.txt/keff.txt/logs for every case into this .tar.gz for easy "
        "transport back",
    )
    args = parser.parse_args()

    unknown = [c for c in args.cases if c not in CASES]
    if unknown:
        sys.exit(f"Unknown case(s): {', '.join(unknown)} -- valid cases are {', '.join(CASES)}")

    opensn_exe = args.opensn or find_opensn_executable()
    if not args.dry_run and not opensn_exe.exists():
        sys.exit(f"opensn executable not found at {opensn_exe}")

    log_dir = HERE / "run_logs"
    log_dir.mkdir(exist_ok=True)

    summary = []
    for case in args.cases:
        print(f"=== {case}: starting (ranks={args.ranks}) ===", flush=True)
        result = run_case(case, opensn_exe, args.ranks, log_dir, args.dry_run)
        summary.append(result)
        if not args.dry_run:
            status = "OK" if result["success"] else "FAILED"
            keff_str = f"{result['keff']:.6f}" if result["keff"] is not None else "N/A"
            print(
                f"=== {case}: {status}  k_eff={keff_str}  "
                f"elapsed={result['elapsed_s'] / 60:.1f} min ===",
                flush=True,
            )
            with open(HERE / "run_all_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    if args.dry_run:
        return

    print("\nSummary:")
    for r in summary:
        status = "OK" if r["success"] else "FAILED"
        keff_str = f"{r['keff']:.6f}" if r["keff"] is not None else "N/A"
        print(f"  {r['case']:<4} {status:<7} k_eff={keff_str:<10} {r['elapsed_s'] / 60:6.1f} min")

    failed = [r["case"] for r in summary if not r["success"]]
    if failed:
        print(f"\n{len(failed)} case(s) failed: {', '.join(failed)} -- see run_logs/")

    if args.archive:
        write_archive(args.archive, summary)


if __name__ == "__main__":
    main()
