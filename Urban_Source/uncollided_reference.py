#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Semi-analytic uncollided flux of the urban source example.

The uncollided flux of a point source of unit strength is exp(-tau) / (4 pi s^2) at
distance s from the source, where tau is the optical depth along the straight line from
the source. The buildings and the ground cells are unions of convex pieces bounded by
planes and circular or elliptic cylinders, and their materials are uniform, so tau is a
sum of exact chord lengths times total cross sections. This script computes, from
city.toml:

- the region integral of the uncollided flux in every block of the mesh, as an integral
  over the direction of (1/4 pi) times the integral of exp(-tau) along the ray, which is
  exact on each straight segment of the ray in one region;
- the uncollided leakage through the outer boundary;
- the average uncollided dose in every box of the dose map, with the 2 x 2 x 2 Gauss rule
  of urban_source.py and with a converged 8 x 8 x 4 rule.

The direction integrals use a product rule with --polar Gauss-Legendre polar cosines and
--azimuthal equally spaced azimuths. The integrand is discontinuous across the silhouettes
of the buildings, so the integrals converge only to first order in the number of
directions; the script reports the change from half as many directions in each angle.

Every quantity is computed twice: with the group-0 total cross sections and dose
coefficient of the 25-group library, which is what OpenSn uses, and with the
continuous-energy values at 661.657 keV of the ENDF/B-VIII.0 photon data, which is what
OpenMC uses. Set OPENMC_CROSS_SECTIONS to the cross_sections.xml file of that library, then
run, for example:

    python3 uncollided_reference.py --processes 8
"""

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path

import h5py
import numpy as np
import openmc.data

import city as citylib
import openmc_reference as reference

CM = 100.0  # cm per m
FINE_RULE = (8, 8, 4)  # Gauss points per box along x, y, z for the converged dose map
POLAR_PER_TASK = 4

# Set in each worker by init_worker.
city = pieces = piece_blocks = sigma_sets = None


def init_worker(layout, sigmas):
    global city, pieces, piece_blocks, sigma_sets
    city = citylib.load(layout)
    pieces = city.convex_pieces()
    piece_blocks = np.array([piece["block"] for piece in pieces])
    sigma_sets = sigmas


def group_zero(library, materials):
    """Return the group-0 total cross section [1/cm] of every material and the group-0 AP
    dose coefficient [pSv cm^2] of the library."""
    temperature = f"{round(reference.TEMPERATURE)}K"
    with h5py.File(library) as stream:
        totals = {name: float(stream[f"{name}/{temperature}/total"][0]) for name in materials}
        dose = float(stream[f"air/{temperature}/dose_ap"][0])
    return totals, dose


def pointwise(materials):
    """Return the total cross section [1/cm] of every material and the AP dose coefficient
    [pSv cm^2] at the source energy, from the ENDF/B-VIII.0 photon data of OpenMC."""
    library = Path(os.environ["OPENMC_CROSS_SECTIONS"]).resolve().parent
    energy = reference.SOURCE_ENERGY
    totals = {}
    for name, spec in materials.items():
        mu_rho = 0.0
        for element, fraction in spec["weight_fractions"].items():
            data = openmc.data.IncidentPhoton.from_hdf5(library / "photon" / f"{element}.h5")
            # Coherent, incoherent, and photoelectric; pair production is zero below 1.022 MeV.
            micro = sum(data.reactions[mt].xs(energy) for mt in (502, 504, 522)
                        if mt in data.reactions)
            mu_rho += (fraction * micro * 1.0e-24 * openmc.data.AVOGADRO
                       / openmc.data.atomic_weight(element))
        totals[name] = mu_rho * spec["density"]
    e, d = openmc.data.dose_coefficients("photon", reference.DOSE_GEOMETRY)
    dose = float(np.exp(np.interp(np.log(energy), np.log(e), np.log(d))))
    return totals, dose


def clip(start, directions, t_max):
    """Return the entry and exit distances [m] of the rays start + t direction, t in
    [0, t_max], in every piece, as arrays [ray, piece]. An empty chord has entry = exit."""
    n = len(directions)
    entry = np.zeros((n, len(pieces)))
    exit_ = np.repeat(t_max[:, None], len(pieces), axis=1)
    for k, piece in enumerate(pieces):
        t0, t1 = entry[:, k], exit_[:, k]
        planes = piece.get("planes")
        if planes is None:
            footprint, (z0, z1) = piece["ellipse"], piece["z"]
            planes = np.array([[0.0, 0.0, -1.0, -z0], [0.0, 0.0, 1.0, z1]])
            c, s = math.cos(footprint.angle), math.sin(footprint.angle)
            dx, dy = start[0] - footprint.center[0], start[1] - footprint.center[1]
            u0, v0 = (c * dx + s * dy) / footprint.a, (-s * dx + c * dy) / footprint.b
            du = (c * directions[:, 0] + s * directions[:, 1]) / footprint.a
            dv = (-s * directions[:, 0] + c * directions[:, 1]) / footprint.b
            qa, qb, qc = du * du + dv * dv, 2 * (u0 * du + v0 * dv), u0 * u0 + v0 * v0 - 1
            disc = qb * qb - 4 * qa * qc
            hit = (disc > 0.0) & (qa > 0.0)
            root = np.sqrt(np.where(hit, disc, 0.0))
            safe = np.where(hit, 2 * qa, 1.0)
            np.maximum(t0, np.where(hit, (-qb - root) / safe, np.inf), out=t0)
            np.minimum(t1, np.where(hit, (-qb + root) / safe, -np.inf), out=t1)
            # A vertical ray inside the footprint has qa = 0 and qc < 0.
            vertical = (qa == 0.0) & (qc < 0.0)
            t0[vertical] = 0.0
            t1[vertical] = t_max[vertical]
        for nx, ny, nz, c0 in planes:
            denominator = directions @ np.array([nx, ny, nz])
            numerator = c0 - (nx * start[0] + ny * start[1] + nz * start[2])
            with np.errstate(divide="ignore", invalid="ignore"):
                t = numerator / denominator
            np.minimum(t1, np.where(denominator > 0.0, t, np.inf), out=t1)
            np.maximum(t0, np.where(denominator < 0.0, t, -np.inf), out=t0)
            if numerator < 0.0:
                parallel = denominator == 0.0
                t1[parallel] = -np.inf
    empty = exit_ <= entry
    entry[empty] = exit_[empty] = np.inf
    return entry, exit_


def exit_distance(start, directions):
    """Return the distance [m] from start to the outer boundary along every direction."""
    xmin, xmax, ymin, ymax, zmin, zmax = city.extent
    distance = np.full(len(directions), np.inf)
    for axis, (lo, hi) in enumerate(((xmin, xmax), (ymin, ymax), (zmin, zmax))):
        d = directions[:, axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(d > 0.0, (hi - start[axis]) / d,
                         np.where(d < 0.0, (lo - start[axis]) / d, np.inf))
        np.minimum(distance, t, out=distance)
    return distance


def segment_integral(sigma, path, attenuation):
    """Return the integral of exp(-tau) over a segment [cm], given exp(-tau) at its start."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return attenuation * np.where(sigma > 0.0, -np.expm1(-sigma * path) / sigma, path)


def region_task(polar_values):
    """Return the direction-weighted integrals of exp(-tau) in every block and the
    transmission to the outer boundary, for the rays of some polar cosines, for every
    cross-section set."""
    mu, weight, n_azimuthal = polar_values
    blocks = list(city.block_materials())
    block_index = np.array([blocks.index(b) for b in piece_blocks])
    phi = 2 * math.pi * (np.arange(n_azimuthal) + 0.5) / n_azimuthal
    sin_theta = np.sqrt(1.0 - mu**2)
    directions = np.column_stack([np.outer(sin_theta, np.cos(phi)).ravel(),
                                  np.outer(sin_theta, np.sin(phi)).ravel(),
                                  np.repeat(mu, n_azimuthal)])
    weights = np.repeat(weight, n_azimuthal) * (2 * math.pi / n_azimuthal) / (4 * math.pi)
    t_exit = exit_distance(city.source, directions)
    entry, exit_ = clip(city.source, directions, t_exit)
    order = np.argsort(entry, axis=1)
    entry = np.take_along_axis(entry, order, axis=1)
    exit_ = np.take_along_axis(exit_, order, axis=1)
    piece_order = block_index[order]
    rows = np.arange(len(directions))
    results = {}
    for name, sigmas in sigma_sets.items():
        sigma_block = np.array([sigmas[b] for b in blocks])
        sigma_air = sigma_block[blocks.index(citylib.AIR_BLOCK)]
        integrals = np.zeros(len(blocks))
        tau = np.zeros(len(directions))
        t_now = np.zeros(len(directions))
        air = np.zeros(len(directions))
        for k in range(entry.shape[1]):
            valid = np.isfinite(entry[:, k])
            if not np.any(valid):
                break
            with np.errstate(invalid="ignore"):
                gap = np.where(valid, entry[:, k] - t_now, 0.0) * CM
            air += segment_integral(sigma_air, gap, np.exp(-tau))
            tau += sigma_air * gap
            with np.errstate(invalid="ignore"):
                path = np.where(valid, exit_[:, k] - entry[:, k], 0.0) * CM
            s = sigma_block[piece_order[:, k]]
            value = segment_integral(s, path, np.exp(-tau))
            integrals += np.bincount(piece_order[rows[valid], k],
                                     weights=(weights * value)[valid], minlength=len(blocks))
            tau += s * path
            t_now = np.where(valid, exit_[:, k], t_now)
        gap = (t_exit - t_now) * CM
        air += segment_integral(sigma_air, gap, np.exp(-tau))
        tau += sigma_air * gap
        integrals[blocks.index(citylib.AIR_BLOCK)] += weights @ air
        results[name] = (integrals, float(weights @ np.exp(-tau)))
    return results


def region_integrals(n_polar, n_azimuthal, pool, names):
    mu, w = np.polynomial.legendre.leggauss(n_polar)
    tasks = [(mu[i:i + POLAR_PER_TASK], w[i:i + POLAR_PER_TASK], n_azimuthal)
             for i in range(0, n_polar, POLAR_PER_TASK)]
    parts = pool.map(region_task, tasks)
    blocks = list(city.block_materials())
    out = {}
    for name in names:
        values = sum(part[name][0] for part in parts)
        out[name] = {"blocks": {city.names[b]: float(v) for b, v in zip(blocks, values)},
                     "leakage": float(sum(part[name][1] for part in parts))}
    return out


def point_task(points):
    """Return the uncollided flux [1/cm^2] at points [m] for every cross-section set."""
    offsets = points - city.source
    distance = np.linalg.norm(offsets, axis=1)
    directions = offsets / distance[:, None]
    entry, exit_ = clip(city.source, directions, distance)
    with np.errstate(invalid="ignore"):  # Empty chords have entry = exit = inf.
        lengths = np.where(np.isfinite(entry), exit_ - entry, 0.0)
    out = {}
    for name, sigmas in sigma_sets.items():
        sigma_piece = np.array([sigmas[b] for b in piece_blocks])
        tau = (lengths @ sigma_piece + (distance - lengths.sum(axis=1))
               * sigmas[citylib.AIR_BLOCK]) * CM
        out[name] = np.exp(-tau) / (4 * math.pi * (distance * CM) ** 2)
    return out


def gauss_box(rule, xr, yr, zr):
    """Return the points and weights (summing to 1) of a product Gauss rule over a box."""
    axes = []
    for n, (lo, hi) in zip(rule, (xr, yr, zr)):
        x, w = np.polynomial.legendre.leggauss(n)
        axes.append((0.5 * (lo + hi) + 0.5 * (hi - lo) * x, 0.5 * w))
    (px, wx), (py, wy), (pz, wz) = axes
    points = np.array([[x, y, z] for x in px for y in py for z in pz])
    weights = np.array([a * b * c for a in wx for b in wy for c in wz])
    return points, weights


def box_average_flux(rule, pool, names):
    """Return the average uncollided flux [1/cm^2] in every dose-map box, as [y, x]."""
    x_edges, y_edges, z_range = city.dose_map_boxes()
    points, weights = [], None
    for j in range(len(y_edges) - 1):
        for i in range(len(x_edges) - 1):
            box_points, weights = gauss_box(rule, (x_edges[i], x_edges[i + 1]),
                                            (y_edges[j], y_edges[j + 1]), z_range)
            points.append(box_points)
    points = np.array(points)
    flat = points.reshape(-1, 3)
    parts = pool.map(point_task, np.array_split(flat, 64))
    shape = (len(y_edges) - 1, len(x_edges) - 1)
    return {name: (np.concatenate([p[name] for p in parts]).reshape(len(points), -1)
                   @ weights).reshape(shape) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, default=Path("city.toml"))
    parser.add_argument("--library", type=Path,
                        default=Path(__file__).resolve().parent / reference.LIBRARY)
    parser.add_argument("--polar", type=int, default=4096)
    parser.add_argument("--azimuthal", type=int, default=8192)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--output", type=Path, default=Path("uncollided_reference.json"))
    args = parser.parse_args()
    init_worker(args.layout, None)

    group_totals, group_dose = group_zero(args.library, city.materials)
    ce_totals, ce_dose = pointwise(city.materials)
    materials = city.block_materials()
    sigmas = {name: {block: totals[m] for block, m in materials.items()}
              for name, totals in (("group_0", group_totals), ("pointwise", ce_totals))}
    doses = {"group_0": group_dose, "pointwise": ce_dose}

    report = {"cross_sections": {"group_0": group_totals, "pointwise": ce_totals},
              "dose_coefficient": doses, "directions": [args.polar, args.azimuthal]}
    with multiprocessing.Pool(args.processes, initializer=init_worker,
                              initargs=(args.layout, sigmas)) as pool:
        fine = region_integrals(args.polar, args.azimuthal, pool, sigmas)
        coarse = region_integrals(args.polar // 2, args.azimuthal // 2, pool, sigmas)
        maps = {label: box_average_flux(rule, pool, sigmas)
                for label, rule in (("dose_map_gauss2", (2, 2, 2)),
                                    ("dose_map_fine", FINE_RULE))}
    for name in sigmas:
        report[name] = {
            "flux": fine[name]["blocks"],
            "flux_change_from_half_directions": {
                k: v - coarse[name]["blocks"][k] for k, v in fine[name]["blocks"].items()},
            "leakage": fine[name]["leakage"],
            "leakage_change_from_half_directions":
                fine[name]["leakage"] - coarse[name]["leakage"]}
        for label, values in maps.items():
            report[name][label] = (values[name] * doses[name]).tolist()
    args.output.write_text(json.dumps(report) + "\n")
    for name in sigmas:
        change = report[name]["leakage_change_from_half_directions"]
        print(f"{name}: leakage {report[name]['leakage']:.8f} (change {change:+.1e} "
              "from half the directions)")
        for block, value in report[name]["flux"].items():
            change = report[name]["flux_change_from_half_directions"][block]
            print(f"  {block:18s} {value:.6e} cm  change {change / value:+.1e}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
