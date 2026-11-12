#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Semi-analytic uncollided flux of the hohlraum example.

The uncollided scalar flux of an isotropic point source of unit strength at the origin is
exp(-tau) / (4 pi s^2), where s is the distance from the source and tau the optical depth
along the ray. In spherical coordinates the volume element is s^2 ds dOmega, so the
integral of the uncollided flux over a region is

    F_k = (1 / 4 pi) int dOmega int_{ray in region k} exp(-tau(s)) ds.

The target is axisymmetric and symmetric about z = 0, so the direction integral reduces to
the polar cosine mu in [0, 1]. Each ray crosses the regions in straight segments. On a
segment of length L in a material with total cross section Sigma, entered at optical depth
tau_0, the inner integral is exp(-tau_0) (1 - exp(-Sigma L)) / Sigma. The leakage is the
integral of exp(-tau) at the outer boundary. The remaining integral over mu is evaluated
by adaptive Gauss-Kronrod quadrature, split at the directions of the corners of the
geometry, where the integrand has kinks.

The script evaluates these integrals for three sets of total cross sections at the
14.1 MeV source energy:

- "opensn": group 0 of materials_64g_p3tc.h5, the transport-corrected cross sections that
  OpenSn uses for the uncollided flux. Its uncollided flux includes the neutrons that the
  transport correction treats as unscattered.
- "multigroup": group 0 of materials_64g_p5.h5, without the transport correction. Its
  uncollided flux is that of the physical problem with multigroup cross sections.
- "continuous": the ENDF/B-VIII.0 cross sections at 14.1 MeV, which OpenMC uses.

It needs scipy, and OPENMC_CROSS_SECTIONS for the continuous-energy data, and writes
uncollided_reference.json:

    python3 uncollided_reference.py
"""

import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import openmc
import openmc.data
from scipy.integrate import quad

import openmc_reference as ref

REACTION_MTS = {"n2n": 16, "ngamma": 102, "nalpha": 107}
# Level-wise reactions that sum to a reaction when the evaluation gives no total.
LEVELS = {16: range(875, 892), 107: range(800, 850)}
EXTRA = ["absorption", "n2n", "ngamma", "nalpha"]
LIBRARIES = {"opensn": "materials_64g_p3tc.h5", "multigroup": "materials_64g_p5.h5"}


def is_disappearance(mt):
    """Return whether a reaction has no neutron in its exit channel, as in OpenMC."""
    return (101 <= mt <= 117 or 600 <= mt <= 849
            or mt in (155, 182, 191, 192, 193, 197))


def region_of(r, z):
    """Return the block ID at cylindrical coordinates (r, |z|)."""
    if r * r + z * z < ref.CAPSULE_RADIUS**2:
        return 1
    if z < ref.GAS_HALF_HEIGHT and r < ref.GAS_RADIUS:
        return 2
    if z < ref.HOHLRAUM_HALF_HEIGHT and r < ref.HOHLRAUM_RADIUS:
        return 4 if z > ref.GAS_HALF_HEIGHT and r < ref.LEH_RADIUS else 3
    if r < ref.TMP_RADIUS or z > ref.HOHLRAUM_HALF_HEIGHT - ref.FLANGE_HEIGHT:
        return 5
    return 6


def segments(mu):
    """Return the (block, length) segments of the ray with polar cosine mu >= 0."""
    sine = math.sqrt(max(1.0 - mu * mu, 0.0))
    radii = [ref.LEH_RADIUS, ref.GAS_RADIUS, ref.HOHLRAUM_RADIUS, ref.TMP_RADIUS,
             ref.FLANGE_RADIUS]
    heights = [ref.GAS_HALF_HEIGHT, ref.HOHLRAUM_HALF_HEIGHT - ref.FLANGE_HEIGHT,
               ref.HOHLRAUM_HALF_HEIGHT]
    end = min(ref.FLANGE_RADIUS / sine if sine > 0 else math.inf,
              ref.HOHLRAUM_HALF_HEIGHT / mu if mu > 0 else math.inf)
    crossings = [ref.CAPSULE_RADIUS]
    crossings += [r / sine for r in radii if sine > 0]
    crossings += [h / mu for h in heights if mu > 0]
    points = sorted({0.0, end} | {s for s in crossings if s < end})
    result = []
    for a, b in zip(points[:-1], points[1:]):
        middle = 0.5 * (a + b)
        result.append((region_of(middle * sine, middle * mu), b - a))
    return result


def kinks():
    """Return the polar cosines of the rays through the corners of the geometry."""
    radii = [ref.LEH_RADIUS, ref.GAS_RADIUS, ref.HOHLRAUM_RADIUS, ref.TMP_RADIUS,
             ref.FLANGE_RADIUS]
    heights = [ref.GAS_HALF_HEIGHT, ref.HOHLRAUM_HALF_HEIGHT - ref.FLANGE_HEIGHT,
               ref.HOHLRAUM_HALF_HEIGHT]
    return sorted({h / math.hypot(r, h) for r in radii for h in heights})


def integrals(sigma_t):
    """Return the region-integrated uncollided flux [cm] and the leakage per source neutron.

    sigma_t maps each block ID to its total cross section in 1/cm.
    """
    def contributions(mu):
        tau, flux = 0.0, np.zeros(len(ref.REGIONS) + 1)
        for block, length in segments(mu):
            sigma = sigma_t[block]
            optical = sigma * length
            flux[block - 1] += math.exp(-tau) * (-math.expm1(-optical) / sigma
                                                 if sigma > 0 else length)
            tau += optical
        flux[-1] = math.exp(-tau)
        return flux

    points = kinks()
    values = []
    for index in range(len(ref.REGIONS) + 1):
        value, error = quad(lambda mu: contributions(mu)[index], 0.0, 1.0, points=points,
                            limit=1000, epsabs=0.0, epsrel=1e-12)
        values.append(value)
    flux = {ref.REGIONS[block]: values[block - 1] for block in ref.REGIONS}
    return flux, values[-1]


def multigroup_xs(file_name):
    """Return group-0 total and reaction cross sections of each block, from a library."""
    path = Path(__file__).resolve().parent / file_name
    library = openmc.MGXSLibrary.from_hdf5(str(path))
    data = {}
    with h5py.File(path, "r") as f:
        for block in ref.REGIONS:
            name = ref.BLOCK_MATERIALS.get(block, "void")
            group = f[f"{name}/{round(ref.TEMPERATURE)}K"]
            data[block] = {"total": float(library.get_by_name(name).total[0][0])}
            data[block].update({key: float(group[key][0]) for key in EXTRA})
    return data


def continuous_xs():
    """Return the continuous-energy total and reaction cross sections at 14.1 MeV."""
    materials = ref.build_materials()
    library = openmc.data.DataLibrary.from_xml(os.environ["OPENMC_CROSS_SECTIONS"])
    temperature = f"{round(ref.TEMPERATURE)}K"
    cache, data = {}, {}
    for block in ref.REGIONS:
        name = ref.BLOCK_MATERIALS.get(block)
        values = {key: 0.0 for key in ["total"] + EXTRA}
        if name is not None:
            for nuclide, density in materials[name].get_nuclide_atom_densities().items():
                if nuclide not in cache:
                    path = library.get_by_material(nuclide)["path"]
                    cache[nuclide] = openmc.data.IncidentNeutron.from_hdf5(path)
                data_n = cache[nuclide]

                def micro(mt):
                    return float(data_n.reactions[mt].xs[temperature](ref.SOURCE_ENERGY))

                # As in OpenMC (Nuclide::create_derived), the total is the sum of the
                # non-redundant reactions, and absorption the sum of those without a neutron
                # in the exit channel.
                values["total"] += density * sum(
                    micro(mt) for mt, rx in data_n.reactions.items() if not rx.redundant)
                values["absorption"] += density * sum(
                    micro(mt) for mt, rx in data_n.reactions.items()
                    if not rx.redundant and is_disappearance(mt))
                for key, mt in REACTION_MTS.items():
                    if mt in data_n.reactions:
                        values[key] += density * micro(mt)
                    else:
                        levels = LEVELS.get(mt, range(0))
                        values[key] += density * sum(
                            micro(m) for m in levels if m in data_n.reactions)
        data[block] = values
    return data


def main():
    report = {}
    sets = [(label, multigroup_xs(file_name)) for label, file_name in LIBRARIES.items()]
    for label, data in sets + [("continuous", continuous_xs())]:
        flux, leakage = integrals({block: values["total"] for block, values in data.items()})
        rates = {name: {key: data[block][key] * flux[name] for key in EXTRA}
                 for block, name in ref.REGIONS.items()}
        report[label] = {"flux": flux, "leakage": leakage, "reactions": rates,
                         "xs": {ref.REGIONS[b]: v for b, v in data.items()}}
        print(f"{label}: leakage {leakage:.12f}")
        for name, value in flux.items():
            print(f"  {name:14s} flux {value:.12e}  absorption "
                  f"{rates[name]['absorption']:.6e}")
    Path("uncollided_reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Wrote uncollided_reference.json")


if __name__ == "__main__":
    main()
