#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Compare the OpenSn coil solution with the OpenMC photon reference.

Prints Markdown tables of region photon flux, face photon leakage, and the
energy budget. All quantities are per incident photon.

Usage::

    python3 compare_openmc.py [--opensn coil_results.json] [--openmc FILE]

The OpenMC reference defaults to ``openmc_results_<mesh>.json`` for the mesh
recorded in the OpenSn results, so that both codes use the same faceting.
"""

import argparse
import json
import math
from pathlib import Path

REGIONS = ("copper", "insulation", "air")
FACES = ("inlet", "outlet", "xmin", "xmax", "zmin", "zmax")
# Temperature of the incident spectrum E exp(-E / T), as in coil.py.
SPECTRUM_TEMPERATURE = 0.7


def load(here, opensn_file, openmc_file):
    groups = json.loads((here / "xs" / "groups.json").read_text())
    opensn = json.loads(Path(opensn_file).read_text())
    if openmc_file is None:
        openmc_file = here / f"openmc_results_{opensn['settings']['mesh']}.json"
    openmc = json.loads(Path(openmc_file).read_text())
    return groups, opensn, openmc


def midpoints(bounds):
    return [0.5 * (upper + lower) for upper, lower in bounds]


def photon_tables(groups, opensn, openmc):
    num_photon = len(groups["photon"])
    rows = []
    print("| Region | OpenSn photon flux (cm) | OpenMC (cm) | Difference | Max group difference |")
    print("|---|---|---|---|---|")
    for region in REGIONS:
        sn = opensn["region_flux"][region][:num_photon]
        mc = openmc["region_photon_flux"][region]
        sd = openmc["region_photon_flux_sd"][region]
        total_sd = math.sqrt(sum(s * s for s in sd))
        diff = sum(sn) / sum(mc) - 1.0
        # Largest group difference among groups with at least 1% of the region flux.
        big = [g for g in range(num_photon) if mc[g] > 0.01 * sum(mc)]
        worst = max(big, key=lambda g: abs(sn[g] / mc[g] - 1.0))
        rows.append((region, diff))
        print(f"| {region} | {sum(sn):.5e} | {sum(mc):.5e} ± {total_sd:.1e} | "
              f"{100 * diff:+.2f}% | {100 * (sn[worst] / mc[worst] - 1):+.1f}% "
              f"(group {worst}, ±{100 * sd[worst] / mc[worst]:.1f}%) |")
    print()

    print("| Face | OpenSn photons out | OpenMC | Difference |")
    print("|---|---|---|---|")
    for face in FACES:
        sn = sum(opensn["leakage"][face][:num_photon])
        if face == "inlet":
            sn += 1.0  # ComputeLeakage is net of the unit incident beam.
        mc = sum(openmc["face_photon_leakage"][face])
        sd = math.sqrt(sum(s * s for s in openmc["face_photon_leakage_sd"][face]))
        print(f"| {face} | {sn:.5e} | {mc:.5e} ± {sd:.1e} | {100 * (sn / mc - 1):+.2f}% |")
    print()
    return rows


def exact_mean_source_energy(groups):
    """Mean energy of the incident spectrum E exp(-E / T) between the cutoff and
    the upper bound of the highest photon group, which OpenMC samples exactly."""
    t = SPECTRUM_TEMPERATURE
    lower, upper = groups["photon"][-1][1], groups["photon"][0][0]

    def first(energy):
        return -t * (energy + t) * math.exp(-energy / t)

    def second(energy):
        return -t * math.exp(-energy / t) * (energy * energy + 2 * t * energy + 2 * t * t)

    return (second(upper) - second(lower)) / (first(upper) - first(lower))


def energy_budget(groups, opensn, openmc):
    """Compare the energy that the photons give to the coil and the air.

    OpenSn weights each group by its center energy, in its balance and in its
    deposition alike, so its incident energy is that of the group centers. OpenMC
    samples the spectrum exactly. Each code's absorbed energy is therefore
    compared as a fraction of its own incident energy.
    """
    num_photon = len(groups["photon"])
    e_electron = midpoints(groups["electron"])
    sn_electron = sum(opensn["leakage"][f][num_photon + g] * e
                      for f in FACES for g, e in enumerate(e_electron))
    sn_incident = opensn["balance"]["csda_energy_inflow_rate"]
    mc_incident = exact_mean_source_energy(groups)

    print("| Energy per incident photon (MeV) | OpenSn | OpenMC (local electron deposition) |")
    print("|---|---|---|")
    print(f"| incident | {sn_incident:.5f} (group centers) | {mc_incident:.5f} (exact) |")
    for region in REGIONS:
        sd = openmc["region_heating_sd_mev"][region]
        print(f"| deposited in {region} | {opensn['energy_deposition'][region]:.4e} | "
              f"{openmc['region_heating_mev'][region]:.4e} ± {sd:.1e} |")
    print(f"| carried out of the box by electrons | {sn_electron:.4e} | — |")
    sn_absorbed = sum(opensn["energy_deposition"].values()) + sn_electron
    mc_absorbed = sum(openmc["region_heating_mev"].values())
    mc_sd = math.sqrt(sum(v * v for v in openmc["region_heating_sd_mev"].values()))
    print(f"| given to electrons (deposited + carried out) | {sn_absorbed:.4e} | "
          f"{mc_absorbed:.4e} ± {mc_sd:.1e} |")
    sn_fraction = sn_absorbed / sn_incident
    mc_fraction = mc_absorbed / mc_incident
    print(f"| given to electrons / incident | {sn_fraction:.4e} | {mc_fraction:.4e} "
          f"({100 * (sn_fraction / mc_fraction - 1):+.2f}%) |")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--opensn", default="coil_results.json")
    parser.add_argument("--openmc", default=None)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    groups, opensn, openmc = load(here, args.opensn, args.openmc)
    photon_tables(groups, opensn, openmc)
    energy_budget(groups, opensn, openmc)


if __name__ == "__main__":
    main()
