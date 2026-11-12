#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compare the OpenSn results of hohlraum.py with OpenMC and with the semi-analytic
uncollided flux.

    python3 compare_openmc.py hohlraum_results.json openmc_reference_results.json \
        uncollided_reference.json

The first table compares the uncollided flux of each region with the semi-analytic
integrals of uncollided_reference.py. OpenSn is compared with the integrals evaluated with
its own transport-corrected cross sections, and OpenMC with the integrals evaluated with
the continuous-energy cross sections.

The transport correction treats part of the elastic scattering in the source group
(group 0, 13.84 to 14.191 MeV) as no collision, so the uncollided flux of OpenSn is not
the uncollided flux of OpenMC. The other tables therefore compare quantities that have the
same meaning in both codes:

- the collided flux and reaction rates in groups 1 to 63, where there is no uncollided
  flux,
- the total reaction rates,
- the leakage, and
- with --uncorrected-uncollided, the collided flux in group 0, as the OpenSn total group-0
  flux minus the uncollided flux of a first-stage run without the transport correction on
  the same mesh. Subtracting a flux on the same mesh cancels its spatial error, which is
  larger than the group-0 collided flux.

All values are per source neutron, and the uncertainties are one standard error of the
OpenMC mean. With --check, the script exits with an error if a compared quantity differs
from OpenMC by more than the given relative tolerance plus three standard errors.
"""

import argparse
import math
import json
import sys

import numpy as np

REGIONS = ["capsule", "fill_gas", "hohlraum_wall", "leh_windows", "tmp", "exterior"]
COLLIDED_REGIONS = REGIONS[1:]
REACTIONS = {"absorption": "absorption", "n2n": "(n,2n)", "ngamma": "(n,gamma)",
             "nalpha": "(n,alpha)"}
# Regions of the reaction rates that are compared. The other regions have rates that are
# negligible or zero.
REACTION_REGIONS = {"absorption": ["hohlraum_wall", "tmp"], "n2n": ["hohlraum_wall"],
                    "ngamma": ["hohlraum_wall", "tmp"], "nalpha": ["tmp"]}
# Groups 2 and 3 (11.052 to 12.84 MeV) are the down-scattered band of the leakage spectrum.
DOWNSCATTER_GROUPS = [2, 3]
SPECTRUM_MAX_ERROR = 0.05


class Table:
    """A comparison table that remembers its rows for the check."""

    rows = []

    def __init__(self, title):
        print(f"\n{title}")
        print(f"  {'Quantity':32s} {'OpenSn':>13s} {'OpenMC':>13s} {'SE':>8s} {'Diff':>9s} "
              f"{'Diff/SE':>8s}")

    def row(self, label, opensn, mean, error):
        relative = opensn / mean - 1
        sigmas = (opensn - mean) / error if error > 0 else math.nan
        Table.rows.append((label, relative, error / mean))
        print(f"  {label:32s} {opensn:13.6e} {mean:13.6e} {100 * error / mean:7.3f}% "
              f"{100 * relative:+8.3f}% {sigmas:+8.2f}")


def openmc_sum(entry, first, last=None):
    """Return the sum over groups first..last of an OpenMC group array and its error."""
    mean, error = (np.array(v) for v in entry)
    return float(mean[first:last].sum()), float(np.sqrt((error[first:last] ** 2).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("opensn")
    parser.add_argument("openmc")
    parser.add_argument("uncollided")
    parser.add_argument("--uncorrected-uncollided", default=None, metavar="JSON",
                        help="hohlraum_uncollided.json of a first stage run with "
                             "materials_64g_p5.h5 on the same mesh")
    parser.add_argument("--check", type=float, default=None, metavar="TOLERANCE",
                        help="Fail if a difference exceeds TOLERANCE plus 3 standard errors")
    args = parser.parse_args()
    with open(args.opensn) as f:
        sn = json.load(f)
    with open(args.openmc) as f:
        mc = json.load(f)
    with open(args.uncollided) as f:
        analytic = json.load(f)

    def sn_groups(part, region, key):
        return np.array(sn[part][region][key])

    print("Uncollided flux [cm] per source neutron, relative to the semi-analytic integrals")
    print(f"  {'Region':16s} {'OpenSn':>10s} {'OpenMC':>10s} {'OpenMC SE':>10s}")
    for region in REGIONS:
        exact_sn = analytic["opensn"]["flux"][region]
        exact_mc = analytic["continuous"]["flux"][region]
        mean, error = mc["uncollided"][region]["flux"]
        opensn = sn_groups("uncollided", region, "flux").sum()
        print(f"  {region:16s} {100 * (opensn / exact_sn - 1):+9.4f}% "
              f"{100 * (mean / exact_mc - 1):+9.4f}% {100 * error / mean:9.4f}%")
    leak_sn = sn["leakage_groups"][0] - sn["collided_leakage_groups"][0]
    mean, error = mc["uncollided"]["leakage"]
    print(f"  {'leakage':16s} {100 * (leak_sn / analytic['opensn']['leakage'] - 1):+9.4f}% "
          f"{100 * (mean / analytic['continuous']['leakage'] - 1):+9.4f}% "
          f"{100 * error / mean:9.4f}%")

    table = Table("Collided flux [cm] per source neutron, groups 1 to 63")
    for region in COLLIDED_REGIONS:
        table.row(region, sn_groups("total", region, "flux")[1:].sum(),
                  *openmc_sum(mc["collided"][region]["flux_groups"], 1))

    if args.uncorrected_uncollided:
        with open(args.uncorrected_uncollided) as f:
            uncorrected = json.load(f)
        table = Table("Collided flux [cm] per source neutron, group 0")
        for region in ["fill_gas", "hohlraum_wall", "tmp", "exterior"]:
            collided = sn_groups("total", region, "flux")[0] - uncorrected[region]["flux"][0]
            table.row(region, collided, *openmc_sum(mc["collided"][region]["flux_groups"], 0, 1))

    table = Table("Total reaction rates per source neutron")
    for key, regions in REACTION_REGIONS.items():
        for region in regions:
            table.row(f"{REACTIONS[key]} {region}", sn_groups("total", region, key).sum(),
                      *mc["total"][region][key])

    table = Table("Collided reaction rates per source neutron, groups 1 to 63")
    for key, regions in REACTION_REGIONS.items():
        for region in regions:
            if key == "n2n":
                continue  # The (n,2n) threshold is above 8 MeV, so groups 1 to 63 add little.
            table.row(f"{REACTIONS[key]} {region}", sn_groups("total", region, key)[1:].sum(),
                      *openmc_sum(mc["collided"][region][f"{key}_groups"], 1))

    table = Table("Leakage per source neutron")
    table.row("total", float(np.sum(sn["leakage_groups"])), *mc["total"]["leakage"])
    table.row("collided, groups 1 to 63", float(np.sum(sn["collided_leakage_groups"][1:])),
              *openmc_sum(mc["collided"]["leakage_groups"], 1))
    band = DOWNSCATTER_GROUPS
    mean, error = (np.array(v) for v in mc["total"]["leakage_groups"])
    table.row("11.052 to 12.84 MeV", float(np.sum(np.array(sn["leakage_groups"])[band])),
              float(mean[band].sum()), float(np.sqrt((error[band] ** 2).sum())))

    print(f"\nCollided group spectra in groups 1 to 63, over the groups whose OpenMC standard "
          f"error is below {100 * SPECTRUM_MAX_ERROR:.0f}%")
    print(f"  {'Spectrum':16s} {'Groups':>6s} {'RMS diff':>9s} {'Max diff':>9s} {'at group':>8s}")
    spectra = {region: (sn_groups("total", region, "flux"),
                        *(np.array(v) for v in mc["collided"][region]["flux_groups"]))
               for region in COLLIDED_REGIONS}
    spectra["leakage"] = (np.array(sn["collided_leakage_groups"]),
                          *(np.array(v) for v in mc["collided"]["leakage_groups"]))
    for name, (opensn, mean, error) in spectra.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            keep = (mean > 0) & (error / mean < SPECTRUM_MAX_ERROR)
        keep[0] = False
        relative = opensn[keep] / mean[keep] - 1
        worst = int(np.argmax(np.abs(relative)))
        print(f"  {name:16s} {int(keep.sum()):6d} {100 * np.sqrt(np.mean(relative ** 2)):8.2f}% "
              f"{100 * relative[worst]:+8.2f}% {int(np.where(keep)[0][worst]):8d}")

    if args.check is not None:
        failed = [label for label, relative, se in Table.rows
                  if abs(relative) > args.check + 3 * se]
        if failed:
            print(f"\nFAILED: {', '.join(failed)}")
            sys.exit(1)
        print("\nAll compared quantities agree within the tolerance.")


if __name__ == "__main__":
    main()
