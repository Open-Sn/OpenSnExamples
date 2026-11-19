#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compare the OpenSn results of the urban source example with OpenMC and with the
semi-analytic uncollided flux.

Inputs, per source photon:
    urban_source_results.json     OpenSn, written by the second stage of urban_source.py
    uncollided_reference.json     semi-analytic uncollided flux, from uncollided_reference.py
    openmc_seed*/results.json     OpenMC, from openmc_reference.py

The uncollided flux of OpenSn is compared with the semi-analytic flux computed with the
same group-0 cross sections, and that of OpenMC with the semi-analytic flux computed with
the continuous-energy cross sections at 661.657 keV. The collided flux and the dose are
compared between OpenSn and OpenMC. The collided part of the OpenSn dose map is its total
dose minus the semi-analytic uncollided dose at the same Gauss points.

Dose-map boxes are compared where the OpenMC standard error is below --max-error and the
box is at least --min-distance from the source; the box that contains the source and its
neighbors are dominated by the 1/r^2 singularity, which neither a 2 x 2 x 2 Gauss rule nor
a box tally resolves. Boxes that a building wall crosses are also left out of the main
statistics: OpenMC integrates its tally over the whole box, but the flux drops sharply
across the wall, which the 2 x 2 x 2 Gauss rule of OpenSn does not resolve. They are
reported separately, and the open boxes are split into those that see the source
(uncollided dose at least half of the total) and those in the shadow of a building.

Part of the spread of the differences is the statistical error of OpenMC. For each group of
boxes the script also reports the root mean square of the OpenMC standard error and the root
mean square difference with that error removed, sqrt(rms(difference)^2 - rms(error)^2), an
estimate of the error of OpenSn alone, and the standard error of the mean difference. It
then reports the total dose in the open boxes by distance from the source.

    python3 compare_openmc.py urban_source_results.json uncollided_reference.json \\
        openmc_seed20261112/results.json --output comparison.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import city as citylib


def relative(a, b):
    return (a - b) / b if b != 0.0 else math.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("opensn", type=Path)
    parser.add_argument("uncollided", type=Path)
    parser.add_argument("openmc", type=Path)
    parser.add_argument("--layout", type=Path, default=Path("city.toml"))
    parser.add_argument("--max-error", type=float, default=0.05,
                        help="Largest relative OpenMC standard error of a compared box")
    parser.add_argument("--min-distance", type=float, default=10.0,
                        help="Smallest distance [m] from the source to a compared box center")
    parser.add_argument("--output", type=Path, default=Path("comparison.json"))
    args = parser.parse_args()

    city = citylib.load(args.layout)
    sn = json.loads(args.opensn.read_text())
    ref = json.loads(args.uncollided.read_text())
    mc = json.loads(args.openmc.read_text())
    names = [city.names[b] for b in city.block_materials()]
    report = {"regions": {}, "dose_map": {}}

    print("Uncollided flux, relative to the semi-analytic flux")
    print(f"{'Region':18s} {'OpenSn':>10s} {'OpenMC':>18s} {'directions':>11s}")
    for name in names:
        sn_unc = sum(sn["uncollided"][name]["flux"])
        exact_g, exact_ce = ref["group_0"]["flux"][name], ref["pointwise"]["flux"][name]
        mc_unc, mc_se = mc["uncollided"][name]["flux"]
        change = ref["group_0"]["flux_change_from_half_directions"][name] / exact_g
        entry = {"opensn_uncollided_vs_exact": relative(sn_unc, exact_g),
                 "openmc_uncollided_vs_exact": relative(mc_unc, exact_ce),
                 "openmc_uncollided_relative_se": mc_se / mc_unc if mc_unc else math.nan,
                 "exact_direction_change": change}
        report["regions"][name] = entry
        print(f"{name:18s} {entry['opensn_uncollided_vs_exact']:+10.2e} "
              f"{entry['openmc_uncollided_vs_exact']:+10.2e} +- "
              f"{entry['openmc_uncollided_relative_se']:.0e} {change:+11.1e}")

    print("\nCollided flux and total dose, OpenSn relative to OpenMC")
    print(f"{'Region':18s} {'Collided flux':>22s} {'Total dose':>22s}")
    for name in names:
        sn_total = sum(sn["total"][name]["flux"])
        sn_coll = sn_total - sum(sn["uncollided"][name]["flux"])
        mc_coll, mc_coll_se = mc["collided"][name]["flux"]
        sn_dose = sum(sn["total"][name]["dose"])
        mc_dose, mc_dose_se = mc["total"][name]["dose"]
        entry = report["regions"][name]
        entry.update({
            "collided_flux": [sn_coll, mc_coll, mc_coll_se],
            "collided_flux_difference": relative(sn_coll, mc_coll),
            "dose": [sn_dose, mc_dose, mc_dose_se],
            "dose_difference": relative(sn_dose, mc_dose)})
        print(f"{name:18s} {entry['collided_flux_difference']:+10.2%} +- "
              f"{mc_coll_se / mc_coll if mc_coll else math.nan:7.2%} "
              f"{entry['dose_difference']:+10.2%} +- "
              f"{mc_dose_se / mc_dose if mc_dose else math.nan:7.2%}")

    sn_leak = sum(sn["leakage_groups"])
    mc_leak, mc_leak_se = mc["total"]["leakage"]
    report["leakage"] = {"opensn": sn_leak, "openmc": [mc_leak, mc_leak_se],
                         "difference": relative(sn_leak, mc_leak)}
    print(f"\nLeakage per source photon: OpenSn {sn_leak:.6e}, OpenMC {mc_leak:.6e} +- "
          f"{mc_leak_se:.1e} ({report['leakage']['difference']:+.3%})")

    # Dose map. OpenMC tallies the integral over each box; divide by the box volume.
    x_edges, y_edges, z_range = city.dose_map_boxes()
    volume = (np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :]
              * (z_range[1] - z_range[0])) * 1.0e6  # cm^3
    mc_total = np.array(mc["total"]["dose_map_integral"][0]) / volume
    mc_total_se = np.array(mc["total"]["dose_map_integral"][1]) / volume
    mc_coll = np.array(mc["collided"]["dose_map_integral"][0]) / volume
    mc_coll_se = np.array(mc["collided"]["dose_map_integral"][1]) / volume
    sn_total = np.array(sn["dose_map_total"])
    sn_coll = sn_total - np.array(ref["group_0"]["dose_map_gauss2"])
    xc, yc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])
    distance = np.hypot(xc[None, :] - city.source[0], yc[:, None] - city.source[1])
    # Boxes that a building covers in part or whole, tested on a 9 x 9 grid of points at
    # mid-height of the dose map.
    z_mid = 0.5 * (z_range[0] + z_range[1])
    samples = 9
    covered = city.box_coverage(x_edges, y_edges, z_mid, samples)
    crossed = (covered > 0) & (covered < samples**2)
    open_box = covered == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        uncollided_share = np.array(ref["group_0"]["dose_map_gauss2"]) / mc_total
    for label, sn_map, mc_map, mc_se in (("total", sn_total, mc_total, mc_total_se),
                                         ("collided", sn_coll, mc_coll, mc_coll_se)):
        with np.errstate(divide="ignore", invalid="ignore"):
            error = mc_se / mc_map
            diff = sn_map / mc_map - 1.0
        usable = (error < args.max_error) & (distance >= args.min_distance)
        report["dose_map"][label] = {}
        for group, mask in (("open", usable & open_box),
                            ("open, sees the source", usable & open_box
                             & (uncollided_share >= 0.5)),
                            ("open, building shadow", usable & open_box
                             & (uncollided_share < 0.05)),
                            ("crossed by a wall", usable & crossed)):
            values = diff[mask]
            if values.size == 0:
                continue
            rms = float(np.sqrt(np.mean(values**2)))
            rms_error = float(np.sqrt(np.mean(error[mask]**2)))
            report["dose_map"][label][group] = {
                "boxes_compared": int(mask.sum()),
                "median_abs_difference": float(np.median(np.abs(values))),
                "rms_difference": rms,
                "rms_openmc_error": rms_error,
                "rms_difference_without_openmc_error": math.sqrt(max(rms**2 - rms_error**2,
                                                                     0.0)),
                "percentile_95_abs_difference": float(np.percentile(np.abs(values), 95)),
                "mean_difference": float(np.mean(values)),
                "mean_difference_error": float(np.std(values) / np.sqrt(values.size))}
            r = report["dose_map"][label][group]
            print(f"Dose map, {label:8s} {group:22s}: {r['boxes_compared']:4d} boxes, median "
                  f"|diff| {r['median_abs_difference']:6.2%}, rms {r['rms_difference']:6.2%} "
                  f"(OpenMC error {r['rms_openmc_error']:5.2%}, without it "
                  f"{r['rms_difference_without_openmc_error']:5.2%}), 95th percentile "
                  f"{r['percentile_95_abs_difference']:6.2%}, mean "
                  f"{r['mean_difference']:+.2%} +- {r['mean_difference_error']:.2%}")

        # The open boxes by distance from the source, for the total dose.
        if label != "total":
            continue
        report["dose_map"]["total_by_distance"] = []
        edges = (10.0, 20.0, 40.0, 70.0, 100.0, 150.0)
        for lower, upper in zip(edges[:-1], edges[1:]):
            ring = usable & open_box & (distance >= lower) & (distance < upper)
            entry = {"from_m": lower, "to_m": upper}
            for group, mask in (("sees the source", uncollided_share >= 0.5),
                                ("building shadow", uncollided_share < 0.05)):
                values = diff[ring & mask]
                if values.size:
                    entry[group] = {
                        "boxes_compared": int(values.size),
                        "mean_difference": float(np.mean(values)),
                        "median_abs_difference": float(np.median(np.abs(values))),
                        "median_openmc_error": float(np.median(error[ring & mask]))}
            report["dose_map"]["total_by_distance"].append(entry)
            parts = [f"{lower:3.0f} to {upper:3.0f} m"]
            for group in ("sees the source", "building shadow"):
                if group in entry:
                    e = entry[group]
                    parts.append(f"{group}: mean {e['mean_difference']:+.1%}, median |diff| "
                                 f"{e['median_abs_difference']:.1%}, OpenMC error "
                                 f"{e['median_openmc_error']:.1%}")
                else:
                    parts.append(f"{group}: no boxes")
            print("Dose map, total by distance, " + "; ".join(parts))
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
