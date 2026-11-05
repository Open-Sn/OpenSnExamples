#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Create a runtime preCICE socket configuration."""

import argparse
from pathlib import Path
import platform
import re
import subprocess


CASE = Path(__file__).resolve().parent


def default_interface():
    if platform.system() == "Darwin":
        output = subprocess.check_output(
            ("route", "-n", "get", "default"), text=True
        )
        match = re.search(r"^\s*interface:\s*(\S+)", output, re.MULTILINE)
    else:
        output = subprocess.check_output(("ip", "route", "show", "default"), text=True)
        match = re.search(r"\bdev\s+(\S+)", output)
    if not match:
        raise RuntimeError("Could not determine the default network interface")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser()
    network_group = parser.add_mutually_exclusive_group()
    network_group.add_argument(
        "--network",
        help="socket interface (default: default-route interface)",
    )
    network_group.add_argument(
        "--no-network",
        action="store_true",
        help="omit the network attribute so preCICE uses its default, the loopback "
        "interface 'lo' (Linux); on macOS pass --network lo0 instead",
    )
    args = parser.parse_args()

    template = (CASE / "precice-config.xml").read_text(encoding="utf-8")
    old = '<m2n:sockets acceptor="OpenFOAM" connector="OpenSn" />'
    network = None if args.no_network else (args.network or default_interface())
    if network is None:
        new = old
    else:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", network):
            raise RuntimeError(f"Unsafe network interface name: {network!r}")
        new = (
            f'<m2n:sockets acceptor="OpenFOAM" connector="OpenSn" '
            f'network="{network}" />'
        )
    if template.count(old) != 1:
        raise RuntimeError("Unexpected m2n sockets entry in precice-config.xml")
    output = CASE / "precice-config.runtime.xml"
    output.write_text(template.replace(old, new), encoding="utf-8")
    description = network if network is not None else "preCICE default (lo)"
    print(f"Wrote {output} with preCICE socket interface {description}")


if __name__ == "__main__":
    main()
