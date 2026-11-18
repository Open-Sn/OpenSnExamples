#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Semi-analytic uncollided flux of the ICF target chamber example.

Along a straight path from the source, the uncollided flux S exp(-tau) / (4 pi r^2) times the
volume element r^2 dr dOmega is S exp(-tau) dr dOmega / (4 pi), so the integral of the flux
over a block is (S / 4 pi) times the integral over the directions of the length-weighted
attenuation of the path in that block, which is analytic for each direction:

    integral over the block of exp(-tau(r)) dr = (exp(-tau_in) - exp(-tau_out)) / sigma_t

on the piece of the path in the block (or its length in a void). The integral over the
directions uses a midpoint rule in the cosine of the polar angle and in the azimuth over the
octant, with the true spheres and port cylinders; the integrand is continuous, and the rule
converges as the square of its step. The result is multiplied by 8, the value of the whole
chamber. The leakage through the outer sphere is the integral of exp(-tau(R)) over the
directions, divided by 4 pi.

The total cross sections are taken from a multigroup library (group 0, the source group), for
the comparison with OpenSn, and from the continuous-energy data at 14.1 MeV, for the
comparison with OpenMC. The script writes uncollided_reference.json with the block integrals,
the leakage and the flux along the three line-outs, for each set of cross sections:

    python3 uncollided_reference.py --library materials_4g_p0.h5 materials_64g_p3.h5
"""

import argparse
import json
import math
import os

import h5py
import numpy as np

import chamber_geometry as geometry
import openmc_reference as reference

MATERIALS = {2: "aluminum", 3: "borated_concrete", 4: "air"}


def library_sigma_t(path):
    """Group-0 total cross section of each block in a multigroup library."""
    with h5py.File(path, "r") as library:
        return {block: float(library[f"{name}/294K/total"][0])
                for block, name in MATERIALS.items()}


def continuous_energy_sigma_t():
    """Macroscopic total cross section at the source energy from the nuclear data."""
    import openmc.data

    materials = reference.build_materials()
    xml = os.environ["OPENMC_CROSS_SECTIONS"]
    library = openmc.data.DataLibrary.from_xml(xml)
    sigma = {}
    for block, name in MATERIALS.items():
        total = 0.0
        for nuclide, density in materials[name].get_nuclide_atom_densities().items():
            data = openmc.data.IncidentNeutron.from_hdf5(
                library.get_by_material(nuclide)["path"])
            temperature = "294K" if "294K" in data[1].xs else sorted(data[1].xs)[0]
            total += density * data[1].xs[temperature](reference.SOURCE_ENERGY)
        sigma[block] = total
    return sigma


def block_integrals(sigma_t, num_mu=2000, num_phi=2000, chunk=200000):
    """Integrals of the uncollided flux over each block of the whole chamber, per source
    neutron, and the leakage through the outer sphere."""
    mu = (np.arange(num_mu) + 0.5) / num_mu
    phi = (np.arange(num_phi) + 0.5) * (0.5 * math.pi / num_phi)
    weight = (1.0 / num_mu) * (0.5 * math.pi / num_phi)
    mu_grid, phi_grid = (g.ravel() for g in np.meshgrid(mu, phi, indexing="ij"))
    totals = {block: 0.0 for block in geometry.BLOCKS}
    leakage = 0.0
    for start in range(0, len(mu_grid), chunk):
        m, p = mu_grid[start:start + chunk], phi_grid[start:start + chunk]
        sine = np.sqrt(1.0 - m * m)
        directions = np.stack([sine * np.cos(p), sine * np.sin(p), m], axis=1)
        r_void = np.clip(geometry.port_exit_radius(directions), geometry.R_CAVITY,
                         geometry.R_SHIELD)
        totals[1] += np.sum(r_void)
        depth = np.zeros_like(r_void)
        for block, (inner, outer) in geometry.SHELLS.items():
            start_r = np.maximum(inner, r_void) if block != 4 else np.full_like(r_void, inner)
            length = np.maximum(0.0, outer - start_r)
            sigma = sigma_t[block]
            if sigma > 0.0:
                totals[block] += np.sum(np.exp(-depth) * -np.expm1(-sigma * length) / sigma)
            else:
                totals[block] += np.sum(np.exp(-depth) * length)
            depth = depth + sigma * length
        leakage += np.sum(np.exp(-depth))
    # The octant holds 1/8 of the directions; S / (4 pi) per unit solid angle.
    scale = 8.0 * weight / (4.0 * math.pi)
    return ({geometry.BLOCKS[b]: v * scale for b, v in totals.items()}, leakage * scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", nargs="*", default=["materials_4g_p0.h5"],
                        help="Multigroup libraries whose group-0 total cross section is used")
    parser.add_argument("--directions", type=int, default=2000,
                        help="Points of the midpoint rule in each angle")
    args = parser.parse_args()

    sets = {os.path.basename(path): library_sigma_t(path) for path in args.library}
    sets["continuous_energy"] = continuous_energy_sigma_t()
    report = {}
    for name, sigma_t in sets.items():
        regions, leakage = block_integrals(sigma_t, args.directions, args.directions)
        coarse, _ = block_integrals(sigma_t, args.directions // 2, args.directions // 2)
        lines = {line: geometry.exact_flux(points, sigma_t).tolist()
                 for line, (_, points) in geometry.line_points().items()}
        report[name] = {
            "sigma_t": {MATERIALS[b]: s for b, s in sigma_t.items()},
            "regions": regions, "leakage": leakage,
            # Change of each integral when the rule is halved: about 4 times its error.
            "rule_change": {k: abs(regions[k] - coarse[k]) for k in regions},
            "lines": lines}
        print(f"{name}: total cross sections (1/cm) "
              + ", ".join(f"{MATERIALS[b]} {s:.6e}" for b, s in sigma_t.items()))
        for block, value in regions.items():
            change = report[name]["rule_change"][block]
            print(f"  {block:17s} {value:.8e} cm (rule change {change:.1e})")
        print(f"  leakage           {leakage:.8e}")
    report["line_abscissae"] = {line: x.tolist()
                                for line, (x, _) in geometry.line_points().items()}
    with open("uncollided_reference.json", "w") as stream:
        json.dump(report, stream)
    print("Wrote uncollided_reference.json")


if __name__ == "__main__":
    main()
