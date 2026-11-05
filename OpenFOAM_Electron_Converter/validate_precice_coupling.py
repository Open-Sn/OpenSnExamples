#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Validate deposited power and its conservative transfer in a coupled run.

The deposited power is integrated three ways: by OpenSn's volume interpolation
of the deposition field (transport.py), by the preCICE adapter's per-cell
quadrature after conservative group averaging, and by OpenFOAM after the
conservative mapping and the division by cell volume. All three must agree.

The deposited power is also compared with a baseline from this code. It is a
regression value, not an independent reference: no Monte Carlo or published
result for this geometry is available. Energy conservation of the transport
solution is checked in transport.py.
"""

import argparse
import math
from pathlib import Path
import re


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
# Integrals of the same linear finite-element field; only round-off differs.
MAX_TRANSFER_RELATIVE_ERROR = 1.0e-10
# 16-rank, 40-group result. Rank counts 8 and 16 agree to about 1e-6 relative
# with the 1e-7 inner-iteration tolerance, so 1e-5 admits decomposition and
# reduction-order changes while rejecting any change in the transport answer.
BASELINE_POWER_W = 6.868668229649e-02
BASELINE_RELATIVE_TOLERANCE = 1.0e-5


def last_match(path, pattern):
    text = path.read_text(encoding="utf-8", errors="replace")
    values = re.findall(pattern, text)
    if not values:
        raise RuntimeError(f"Did not find {pattern!r} in {path}")
    return values[-1]


def parse_float(text, path):
    value = float(text)
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite value in {path}: {text}")
    return value


def rounding_tolerance(text):
    """Return the relative half-unit in the last printed digit of text."""
    mantissa = re.split(r"[Ee]", text.lstrip("+-"))[0]
    digits = mantissa.replace(".", "").lstrip("0")
    if not digits:
        raise RuntimeError(f"Cannot derive a rounding tolerance for {text!r}")
    return 0.5 * 10.0 ** (1 - len(digits))


def relative_error(value, reference):
    return abs(value - reference) / abs(reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("opensn_log", type=Path)
    parser.add_argument("openfoam_log", type=Path)
    args = parser.parse_args()

    integrated = parse_float(
        last_match(args.opensn_log, rf"WATERBLOCK_DEPOSITION_W\s*=\s*({FLOAT})"),
        args.opensn_log,
    )
    sent_text = last_match(
        args.opensn_log,
        rf"OpenSn preCICE sending cell-integrated power\s*=\s*({FLOAT})\s*W",
    )
    sent = parse_float(sent_text, args.opensn_log)
    received = parse_float(
        last_match(
            args.openfoam_log,
            rf"OpenFOAM preCICE received cell-integrated power\s*=\s*({FLOAT})\s*W",
        ),
        args.openfoam_log,
    )
    if integrated <= 0.0:
        raise RuntimeError(f"OpenSn deposited nonpositive power: {integrated} W")

    # The adapter log may be rounded; allow its printed precision and round-off.
    sent_tolerance = max(rounding_tolerance(sent_text), MAX_TRANSFER_RELATIVE_ERROR)
    sent_error = relative_error(sent, integrated)
    received_error = relative_error(received, integrated)
    baseline_error = relative_error(integrated, BASELINE_POWER_W)

    print(f"OpenSn deposited power  = {integrated:.12e} W")
    print(f"preCICE adapter sent    = {sent:.12e} W (relative error {sent_error:.6e})")
    print(f"OpenFOAM received power = {received:.12e} W (relative error {received_error:.6e})")
    print(
        f"baseline power          = {BASELINE_POWER_W:.12e} W "
        f"(relative error {baseline_error:.6e})"
    )

    failures = []
    if sent_error > sent_tolerance:
        failures.append(
            f"adapter power differs from the OpenSn integral by more than {sent_tolerance:.1e}"
        )
    if received_error > MAX_TRANSFER_RELATIVE_ERROR:
        failures.append(
            "OpenFOAM power differs from the OpenSn integral by more than "
            f"{MAX_TRANSFER_RELATIVE_ERROR:.1e}"
        )
    if baseline_error > BASELINE_RELATIVE_TOLERANCE:
        failures.append(
            "deposited power differs from the baseline by more than "
            f"{BASELINE_RELATIVE_TOLERANCE:.1e}"
        )
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
