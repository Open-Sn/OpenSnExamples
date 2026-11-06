#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the OpenSn time-dependent Kobayashi results with the OpenMC reference.

Reads kobayashi_td_results.csv (written by kobayashi_td.py) and openmc_reference.csv
(written by extract_openmc_reference.py), prints per-series metrics, and plots the
time histories to kobayashi_td_comparison.png.

Metrics per series, over the 2 s bins in [0,200] s:
    integral_ratio  OpenSn / OpenMC time integral
    peak_ratio      OpenSn / OpenMC peak bin value
    peak_shift      OpenSn - OpenMC peak bin time [s]
    l1_rel          sum |OpenSn - OpenMC| / sum OpenMC

With --check, no plot is made, and the script fails unless the population history
agrees with OpenMC within POPULATION_L1_LIMIT. The population is an integral over the
whole problem, so it is insensitive to ray effects and tests the neutron speed, the
cross sections, and the time discretization.

Usage: python3 compare_openmc.py [--check] [kobayashi_td_results.csv] [openmc_reference.csv]
"""

import csv
import sys
from collections import defaultdict

POPULATION_L1_LIMIT = 0.01
PLOT_SERIES = ["3A_2", "3A_6", "3A_10", "3B_4", "3B_6", "3C_1", "3C_4", "3C_6", "population"]


def read_series(path):
    data = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith("#"))
        for row in rows:
            data[row["series"]].append(
                (
                    0.5 * (float(row["t_lo"]) + float(row["t_hi"])),
                    float(row["mean"]),
                    float(row.get("std") or 0.0),
                )
            )
    return data


def metrics(opensn, openmc):
    times = [t for t, _, _ in openmc]
    sn = [v for _, v, _ in opensn]
    mc = [v for _, v, _ in openmc]
    sn_peak = max(range(len(sn)), key=sn.__getitem__)
    mc_peak = max(range(len(mc)), key=mc.__getitem__)
    return {
        "integral_ratio": sum(sn) / sum(mc),
        "peak_ratio": sn[sn_peak] / mc[mc_peak],
        "peak_shift": times[sn_peak] - times[mc_peak],
        "l1_rel": sum(abs(a - b) for a, b in zip(sn, mc)) / sum(mc),
    }


def plot(opensn, openmc, file_name):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for ax, key in zip(axes.flat, PLOT_SERIES):
        t_mc, v_mc, s_mc = zip(*openmc[key])
        t_sn, v_sn, _ = zip(*opensn[key])
        ax.errorbar(t_mc, v_mc, yerr=s_mc, fmt="o", ms=2.5, color="0.35", label="OpenMC")
        ax.plot(t_sn, v_sn, "-", lw=1.6, color="tab:blue", label="OpenSn")
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(min(v for v in v_mc if v > 0.0), max(v_mc) * 1.0e-4))
        ax.set_title(key)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("2 s bin average")
    axes.flat[0].legend()
    fig.suptitle("Time-dependent Kobayashi dog-leg, case ii")
    fig.tight_layout()
    fig.savefig(file_name, dpi=150)
    print(f"Wrote {file_name}")


def main(opensn_path, openmc_path, check):
    opensn = read_series(opensn_path)
    openmc = read_series(openmc_path)
    if set(opensn) != set(openmc) or any(len(opensn[k]) != len(openmc[k]) for k in openmc):
        sys.exit("OpenSn results and the OpenMC reference have different series or bins")
    print(f"{'series':>10} {'integral':>9} {'peak':>9} {'shift[s]':>9} {'L1 rel':>9}")
    for key in openmc:
        m = metrics(opensn[key], openmc[key])
        print(
            f"{key:>10} {m['integral_ratio']:9.4f} {m['peak_ratio']:9.4f} "
            f"{m['peak_shift']:9.1f} {m['l1_rel']:9.4f}"
        )
    if check:
        l1 = metrics(opensn["population"], openmc["population"])["l1_rel"]
        if not l1 <= POPULATION_L1_LIMIT:
            sys.exit(f"Population L1 difference {l1:.4e} exceeds {POPULATION_L1_LIMIT}")
        print(f"Population L1 difference {l1:.4e} is within {POPULATION_L1_LIMIT}")
    else:
        plot(opensn, openmc, "kobayashi_td_comparison.png")


if __name__ == "__main__":
    check = "--check" in sys.argv[1:]
    paths = [arg for arg in sys.argv[1:] if arg != "--check"]
    paths += ["kobayashi_td_results.csv", "openmc_reference.csv"][len(paths):]
    main(paths[0], paths[1], check)
