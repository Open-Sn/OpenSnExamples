#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compare the OpenSn results of chamber.py with the OpenMC reference and the exact uncollided
flux.

Every quantity is per neutron of the whole source, the value of the whole chamber. When the
OpenSn library has fewer groups than the 64 of OpenMC, the OpenMC results are summed over the
64 groups of each coarse group; the coarse edges are edges of the 64 groups, so the sums are
exact. The script prints, for each OpenSn result given:

- the uncollided flux in each block against the semi-analytic value computed with the
  group-0 total cross section of the library (uncollided_reference.py), and the uncollided
  flux of OpenMC against the value computed with the continuous-energy cross sections;
- the total and collided flux and the absorption rate in each block, and the leakage, against
  OpenMC, as relative differences and in standard errors of OpenMC;
- the RMS of the relative differences of the group spectra in each block, over the groups
  that hold at least 1% of the quantity and whose OpenMC standard error is below 5%;
- the RMS and largest relative difference along each line-out, the OpenSn values averaged over
  the bins of OpenMC, over the bins whose OpenMC standard error is below 10%;
- the RMS and largest relative difference of the uncollided flux of OpenSn along each line-out
  against the exact values at the same points, computed with the group-0 total cross section of
  the library.

With --check TOL the script exits with an error if a total flux in a block or the leakage
differs from OpenMC by more than TOL (relative), or an uncollided block integral differs from
the semi-analytic value by more than TOL. For example:

    python3 compare_openmc.py --openmc openmc_seed20261113/results.json \\
        --opensn chamber_4g_p0 chamber_64g_p3 --check 0.05
"""

import argparse
import json
import math
import sys

import numpy as np

import chamber_geometry as geometry

BLOCKS = list(geometry.BLOCKS.values())


def load(path):
    with open(path) as stream:
        return json.load(stream)


def coarse_map(fine_edges, coarse_edges):
    """Index of the coarse group of each fine group, both ordered from high energy."""
    fine = sorted(fine_edges, reverse=True)
    coarse = sorted(coarse_edges, reverse=True)
    result = []
    for g in range(len(fine) - 1):
        middle = 0.5 * (fine[g] + fine[g + 1])
        result.append(next(k for k in range(len(coarse) - 1)
                           if coarse[k + 1] <= middle <= coarse[k]))
    return np.array(result)


def collapse(mean, error, mapping, num_coarse):
    """Sum fine groups into coarse groups; the errors of the fine groups are added in
    quadrature (an approximation that neglects their correlation)."""
    mean, error = np.asarray(mean), np.asarray(error)
    out_mean = np.zeros(num_coarse)
    out_var = np.zeros(num_coarse)
    np.add.at(out_mean, mapping, mean)
    np.add.at(out_var, mapping, error**2)
    return out_mean, np.sqrt(out_var)


def bin_average(points_x, values, bin_edges):
    """Average of point values inside each bin."""
    points_x, values = np.asarray(points_x), np.asarray(values)
    result = np.full(len(bin_edges) - 1, np.nan)
    for i in range(len(bin_edges) - 1):
        inside = (points_x >= bin_edges[i]) & (points_x < bin_edges[i + 1])
        if np.any(inside):
            result[i] = values[inside].mean()
    return result


def compare(name, openmc, opensn, uncollided, exact, edges):
    """Print the comparison of one OpenSn configuration; return the checked differences."""
    library = opensn["library"]
    num_groups = len(opensn["regions"][BLOCKS[0]]["flux"])
    mapping = coarse_map(openmc["group_edges_eV"], edges if num_groups < 64
                         else openmc["group_edges_eV"])
    checked = {}
    print(f"\n=== {name}: library {library}, mesh {opensn['mesh_file']}, "
          f"{opensn['quadrature']['n_polar'] * opensn['quadrature']['n_azimuthal']} directions, "
          f"{opensn['ranks']} ranks, {opensn['seconds']:.0f} s")
    exact_lib = exact[f"materials_{library}.h5"]
    exact_ce = exact["continuous_energy"]
    print(f"{'Uncollided flux [cm]':24s} {'OpenSn':>12s} {'exact (lib)':>12s} {'diff':>8s} "
          f"{'OpenMC':>12s} {'exact (CE)':>12s} {'diff':>8s} {'in sigma':>8s}")
    for block in BLOCKS:
        ours = sum(uncollided["regions"][block]["flux"])
        reference = exact_lib["regions"][block]
        mc, mc_error = openmc["uncollided"][block]["flux"]
        checked[f"uncollided {block}"] = ours / reference - 1.0
        print(f"{block:24s} {ours:12.5e} {reference:12.5e} {ours / reference - 1:+8.2%} "
              f"{mc:12.5e} {exact_ce['regions'][block]:12.5e} "
              f"{mc / exact_ce['regions'][block] - 1:+8.2%} "
              f"{(mc - exact_ce['regions'][block]) / mc_error:+8.1f}")

    print(f"{'Block':17s} {'quantity':>10s} {'OpenSn':>12s} {'OpenMC':>12s} {'+-':>8s} "
          f"{'diff':>8s} {'in sigma':>8s} {'spectrum RMS':>12s}")
    for block in BLOCKS:
        for quantity, label in (("flux", "total"), ("collided", "collided"),
                                ("absorption", "absorption")):
            if quantity == "collided":
                ours_groups = (np.asarray(opensn["regions"][block]["flux"])
                               - np.asarray(uncollided["regions"][block]["flux"]))
                source = openmc["collided"][block]["flux_groups"]
            else:
                ours_groups = np.asarray(opensn["regions"][block][quantity])
                source = openmc["total"][block][f"{quantity}_groups"]
            mc_groups, mc_errors = collapse(*source, mapping, num_groups)
            ours, mc = ours_groups.sum(), mc_groups.sum()
            mc_error = math.sqrt(np.sum(mc_errors**2))
            if mc == 0.0:
                continue
            good = ((mc_groups > 0.01 * mc_groups.sum())
                    & (mc_errors < 0.05 * np.abs(mc_groups)))
            rms = (math.sqrt(np.mean((ours_groups[good] / mc_groups[good] - 1.0)**2))
                   if np.any(good) else float("nan"))
            if quantity == "flux":
                checked[f"total flux {block}"] = ours / mc - 1.0
            print(f"{block:17s} {label:>10s} {ours:12.5e} {mc:12.5e} {mc_error:8.1e} "
                  f"{ours / mc - 1:+8.2%} {(ours - mc) / mc_error:+8.1f} {rms:12.2%}")
    ours = sum(opensn["leakage_groups"])
    mc, mc_error = openmc["total"]["leakage"]
    checked["leakage"] = ours / mc - 1.0
    print(f"{'leakage':17s} {'':>10s} {ours:12.5e} {mc:12.5e} {mc_error:8.1e} "
          f"{ours / mc - 1:+8.2%} {(ours - mc) / mc_error:+8.1f}")

    print(f"{'Line-out':10s} {'part':>10s} {'bins':>5s} {'RMS diff':>9s} {'max diff':>9s}")
    abscissae = openmc["line_abscissae"]
    for line, (x, _) in geometry.line_points().items():
        centres = np.asarray(abscissae[line])
        width = centres[1] - centres[0]
        edges_x = np.concatenate([centres - 0.5 * width, [centres[-1] + 0.5 * width]])
        for part in ("total", "uncollided"):
            source = opensn if part == "total" else uncollided
            ours_points = np.asarray(source["lines"][line]).sum(axis=1)
            ours = bin_average(x, ours_points, edges_x)
            mc, mc_error = (np.asarray(v) for v in openmc[part]["lines"][line])
            good = np.isfinite(ours) & (mc > 0) & (mc_error < 0.1 * mc)
            if not np.any(good):
                continue
            difference = ours[good] / mc[good] - 1.0
            print(f"{line:10s} {part:>10s} {int(good.sum()):5d} "
                  f"{math.sqrt(np.mean(difference**2)):9.2%} {np.max(np.abs(difference)):9.2%}")
    print(f"{'Line-out':10s} {'uncollided vs exact (library sigma_t)':>40s} {'RMS diff':>9s} "
          f"{'max diff':>9s} {'in beam (>1% of max): RMS':>26s}")
    for line in ("port", "shield", "arc"):
        ours = np.asarray(uncollided["lines"][line]).sum(axis=1)
        reference = np.asarray(exact_lib["lines"][line])
        good = reference > 1.0e-6 * reference.max()
        difference = ours[good] / reference[good] - 1.0
        beam = reference > 1.0e-2 * reference.max()
        beam_difference = ours[beam] / reference[beam] - 1.0
        print(f"{line:10s} {int(good.sum()):40d} {math.sqrt(np.mean(difference**2)):9.2%} "
              f"{np.max(np.abs(difference)):9.2%} {math.sqrt(np.mean(beam_difference**2)):26.2%}")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--openmc", required=True, help="results.json of openmc_reference.py")
    parser.add_argument("--opensn", nargs="+", required=True,
                        help="Prefixes of chamber.py outputs (<prefix>_results.json and "
                             "<prefix>_uncollided.json)")
    parser.add_argument("--exact", default="uncollided_reference.json")
    parser.add_argument("--provenance", default="provenance/materials.json")
    parser.add_argument("--check", type=float, default=None,
                        help="Largest relative difference allowed in the checked quantities")
    args = parser.parse_args()

    openmc = load(args.openmc)
    exact = load(args.exact)
    edges = load(args.provenance)["settings"]["few_group_edges_eV"]
    failures = []
    for prefix in args.opensn:
        opensn = load(f"{prefix}_results.json")
        uncollided = load(f"{prefix}_uncollided.json")
        checked = compare(prefix, openmc, opensn, uncollided, exact, edges)
        failures += [f"{prefix}: {k} {v:+.2%}" for k, v in checked.items()
                     if args.check is not None and abs(v) > args.check]
    if args.check is not None:
        if failures:
            print("\nCheck failed:\n  " + "\n  ".join(failures))
            sys.exit(1)
        print(f"\nCheck passed: every checked difference is below {args.check:.1%}.")


if __name__ == "__main__":
    main()
