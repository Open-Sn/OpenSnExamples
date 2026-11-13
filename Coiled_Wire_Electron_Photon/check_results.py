#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Regression check of the coil example on the coarse mesh.

Reads ``coil_results.json`` written by ``coil.py`` with its default settings
(mesh f4, P7, Lebedev 23, GMRES(10) with WGDSA, tolerance 1e-9) and fails
unless all three of the following hold.

1. Conservation. The CSDA particle and energy balances and the charge and
   energy closures of ``coil.py`` are below CONSERVATION_LIMIT per incident
   photon. They do not depend on any stored value. At the solver tolerance of
   1e-9 the particle balance is 1e-9 to 6e-9, depending on the rank count.
2. Photons against OpenMC. The region photon flux and the outlet photon
   leakage agree with the OpenMC reference on the same faceted geometry,
   ``openmc_results_f4.json``, within PHOTON_LIMITS. The inlet and side
   leakage is checked more loosely, because the multigroup photon data
   and ENDF/B-VIII.0 differ by 1-3% for the photons scattered out of the beam.
3. Stored values. The deposition and electron leakage match GOLD within
   GOLD_RELATIVE times the value plus GOLD_ABSOLUTE. The tolerance allows the
   reduction-order differences of other rank counts and machines at the 1e-9
   solver tolerance, and it fails for any change of the transport solution.

Usage::

    python3 check_results.py [coil_results.json]
"""

import json
import math
import sys
from pathlib import Path

CONSERVATION_LIMIT = 5.0e-8
PHOTON_LIMITS = {"region_flux": 2.0e-3, "outlet": 1.0e-3, "inlet_and_sides": 5.0e-2}
GOLD_RELATIVE = 1.0e-5
GOLD_ABSOLUTE = 1.0e-9

# OpenSn results per incident photon on coil_f4.msh with the default
# settings of coil.py. Provenance is in the README.
GOLD = {
    "energy_deposition": {
        "insulation": 3.1833825950e-03,
        "copper": 1.5164972276e-03,
        "air": 9.3219429267e-06,
    },
    "charge_deposition": {
        "insulation": -2.9676175054e-03,
        "copper": -6.8271111225e-04,
        "air": -1.6842956321e-04,
    },
    "electron_leakage": {
        "inlet": 4.9390964175e-05,
        "outlet": 2.0537167494e-03,
        "xmin": 3.6988268018e-04,
        "xmax": 3.7436630685e-04,
        "zmin": 5.7304987558e-04,
        "zmax": 3.9834571078e-04,
    },
}

REGIONS = ("copper", "insulation", "air")
FACES = ("inlet", "outlet", "xmin", "xmax", "zmin", "zmax")


def main(results_path):
    here = Path(__file__).resolve().parent
    groups = json.loads((here / "xs" / "groups.json").read_text())
    num_photon = len(groups["photon"])
    opensn = json.loads(Path(results_path).read_text())
    openmc = json.loads((here / "openmc_results_f4.json").read_text())
    failures = []

    def check(name, value, limit):
        status = "ok" if math.isfinite(value) and abs(value) <= limit else "FAIL"
        print(f"{status:4s} {name:44s} {value:+.3e}  (limit {limit:.1e})")
        if status != "ok":
            failures.append(name)

    settings = opensn["settings"]
    if settings["mesh"] != "f4" or settings["scattering_order"] != 7:
        sys.exit(f"check_results.py needs the default f4 P7 run, not {settings}")

    balance = opensn["balance"]
    leakage = opensn["leakage"]
    electron_leakage = {f: sum(leakage[f][num_photon:]) for f in FACES}
    deposited = sum(opensn["energy_deposition"].values())
    expected = (balance["csda_energy_production_rate"] + balance["csda_energy_inflow_rate"]
                - balance["csda_energy_outflow_rate"])
    check("particle balance", balance["csda_particle_balance"], CONSERVATION_LIMIT)
    check("energy balance", balance["csda_energy_balance"], CONSERVATION_LIMIT)
    check("energy closure", (deposited - expected) / balance["csda_energy_inflow_rate"],
          CONSERVATION_LIMIT)
    check("charge closure",
          sum(opensn["charge_deposition"].values()) + sum(electron_leakage.values()),
          CONSERVATION_LIMIT)

    for region in REGIONS:
        sn = sum(opensn["region_flux"][region][:num_photon])
        mc = sum(openmc["region_photon_flux"][region])
        check(f"{region} photon flux / OpenMC - 1", sn / mc - 1.0, PHOTON_LIMITS["region_flux"])
    for face in FACES:
        sn = sum(leakage[face][:num_photon]) + (1.0 if face == "inlet" else 0.0)
        mc = sum(openmc["face_photon_leakage"][face])
        limit = PHOTON_LIMITS["outlet" if face == "outlet" else "inlet_and_sides"]
        check(f"{face} photon leakage / OpenMC - 1", sn / mc - 1.0, limit)

    values = {
        "energy_deposition": opensn["energy_deposition"],
        "charge_deposition": opensn["charge_deposition"],
        "electron_leakage": electron_leakage,
    }
    for kind, gold in GOLD.items():
        for key, reference in gold.items():
            limit = GOLD_RELATIVE * abs(reference) + GOLD_ABSOLUTE
            check(f"{kind} {key} - stored", values[kind][key] - reference, limit)

    if failures:
        sys.exit(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    print("All checks passed.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "coil_results.json")
