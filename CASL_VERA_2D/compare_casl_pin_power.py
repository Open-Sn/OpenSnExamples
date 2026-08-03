# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Compares OpenSn's VERA Problem 2 k-eff and pin powers against the benchmark's own published
reference (CASL-U-2012-0131-004, "VERA Core Physics Benchmark Progression Problem
Specifications", Table P2-5 for k-eff and Appendix B for pin powers -- the same ENDF/B-VII CE
KENO-VI dataset Table P2-5 comes from). See vera_reference_keff.csv and each case's own
reference_pin_power.txt for the extracted values and provenance notes.

Each case script (2A.py, ..., 2Q.py) writes power.txt (peak-normalized to 1.0, only the
positions the mesh actually models -- see casl_pin_power.py's module docstring for why most
cases only model a sub-region, not the full 17x17 assembly) and keff.txt when run. This script
renormalizes OpenSn's modeled subset to the benchmark's own core-average-normalized convention
(mean of the 264 fuel pins = 1.0) before comparing peaking factors, and restricts the comparison
to whichever positions OpenSn actually modeled.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CASES = [
    "2A", "2B", "2C", "2D", "2E", "2F", "2G", "2H", "2I", "2J", "2K", "2L", "2M", "2N", "2O",
    "2P", "2Q",
]


def load_reference_keff():
    with open(HERE / "vera_reference_keff.csv") as f:
        return {row["case"]: float(row["k_eff"]) for row in csv.DictReader(f)}


def compare_case(case, reference_keff):
    case_dir = HERE / case
    power_path = case_dir / "power.txt"
    keff_path = case_dir / "keff.txt"
    ref_path = case_dir / "reference_pin_power.txt"

    if not power_path.exists() or not keff_path.exists():
        print(f"{case}: skipped -- run {case}/{case}.py first (missing power.txt/keff.txt)")
        return None

    opensn_power = np.loadtxt(power_path)
    opensn_keff = float(keff_path.read_text().strip())
    ref_power = np.loadtxt(ref_path)

    pcm = (opensn_keff - reference_keff[case]) * 1.0e5

    modeled = opensn_power > 0
    n_modeled = int(modeled.sum())
    if n_modeled == 0:
        print(f"{case}: power.txt has no nonzero pins -- nothing to compare")
        return None

    # power.txt is peak-normalized (each case script's own convention: val_table / val_table.max());
    # the benchmark's own convention is core-average-normalized. Renormalize OpenSn's modeled
    # subset to the same average-1.0 convention before comparing peaking factors -- an inherent
    # approximation when n_modeled < 264 (the sub-region's own average need not exactly equal the
    # full assembly's), not a bug, just the cost of comparing a partial region.
    opensn_norm = opensn_power[modeled] / opensn_power[modeled].mean()
    ref_modeled = ref_power[modeled]

    rel_error = (opensn_norm - ref_modeled) / ref_modeled
    rms_pct = float(np.sqrt(np.mean(rel_error**2)) * 100)
    mean_abs_pct = float(np.mean(np.abs(rel_error)) * 100)
    worst = int(np.argmax(np.abs(rel_error)))
    rows, cols = np.where(modeled)

    coverage = "full assembly" if n_modeled == 264 else "modeled sub-region only"
    print(f"{case}: k_eff OpenSn={opensn_keff:.6f}  reference={reference_keff[case]:.6f}  "
          f"diff={pcm:+.1f} pcm")
    print(f"      pins compared: {n_modeled}/264 ({coverage})")
    print(f"      RMS peaking-factor error: {rms_pct:.3f}%   mean|error|: {mean_abs_pct:.3f}%   "
          f"max error: {rel_error[worst] * 100:+.3f}% at (row={rows[worst]}, col={cols[worst]})")

    return {
        "case": case, "pcm": pcm, "n_modeled": n_modeled, "rms_pct": rms_pct,
        "mean_abs_pct": mean_abs_pct,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cases", nargs="*", default=CASES, help="Case names (default: all 17)")
    args = parser.parse_args()

    reference_keff = load_reference_keff()
    results = [compare_case(case, reference_keff) for case in args.cases]
    results = [r for r in results if r is not None]

    if results:
        print()
        print(f"{'case':<6}{'k_eff diff (pcm)':>18}{'pins':>8}{'RMS power err':>16}")
        for r in results:
            print(f"{r['case']:<6}{r['pcm']:>+18.1f}{r['n_modeled']:>8}{r['rms_pct']:>15.3f}%")


if __name__ == "__main__":
    main()
