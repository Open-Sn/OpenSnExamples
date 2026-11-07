#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract the OpenMC reference used by kobayashi_td.py from a statepoint of the
time-dependent Kobayashi dog-leg benchmark:

    I. Variansyah, "Time-Dependent Kobayashi Dog-Leg Benchmark for Neutron Transport",
    Zenodo (2025), https://doi.org/10.5281/zenodo.15069882, CC-BY-4.0.

The statepoint holds two track-length tallies, both normalized per source particle:

- "flux": a 1 cm voxel mesh over [0,60]x[0,100]x[0,60] crossed with 2 s time bins
  over [0,200] (time-major bin order, x fastest within the mesh). Each bin is the
  integral of the scalar flux over the voxel and the time bin.
- "density": the inverse-velocity score over the same time bins, i.e. the integral
  of the neutron population over each time bin.

kobayashi_td.py uses a source of 1 n/cm^3/s over the 1000 cm^3 source region for
50 s, so the tallies are scaled by 5.0e4 source particles. Flux at a reference
point is the average over the eight voxels that share that point as a corner
(a 2 cm cube centered on the point). The reported standard deviation of that
average is the mean of the voxel standard deviations, an upper bound because
the statepoint does not store voxel covariances.

Usage: python3 extract_openmc_reference.py output_4.h5 [openmc_reference.csv]
"""

import csv
import hashlib
import sys

import h5py
import numpy as np

from kobayashi_td_points import REFERENCE_POINTS

SOURCE_PARTICLES = 1.0 * 1000.0 * 50.0


def find_tally(f, name):
    for key in f["tallies"]:
        if key.startswith("tally ") and f["tallies"][key]["name"][()].decode() == name:
            return f["tallies"][key]
    raise KeyError(f"No tally named {name!r}")


def mean_and_std(results, n):
    """Mean and standard deviation of the mean from OpenMC (sum, sum_sq) results."""
    mean = results[..., 0] / n
    var = (results[..., 1] / n - mean**2) / (n - 1)
    return mean, np.sqrt(np.maximum(var, 0.0))


def md5sum(path):
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(statepoint, output):
    with h5py.File(statepoint, "r") as f:
        n_particles = int(f["n_particles"][()])
        flux_tally = find_tally(f, "flux")
        density_tally = find_tally(f, "density")
        n = int(flux_tally["n_realizations"][()])

        flux_filter = f["tallies"]["filters"][f"filter {flux_tally['filters'][0]}"]
        if flux_filter["type"][()].decode() != "timedmesh":
            raise RuntimeError("Expected a timedmesh filter on the flux tally")
        time_bins = flux_filter["time_bins"][()]
        mesh = f["tallies"]["meshes"][f"mesh {flux_filter['mesh_bins'][()]}"]
        grids = [mesh[f"{axis}_grid"][()] for axis in "xyz"]
        for grid, extent in zip(grids, (60.0, 100.0, 60.0)):
            if not np.allclose(grid, np.linspace(0.0, extent, int(extent) + 1)):
                raise RuntimeError("Expected a uniform 1 cm tally mesh")
        nx, ny, nz = (len(grid) - 1 for grid in grids)
        nt = len(time_bins) - 1
        if flux_tally["results"].shape[0] != nt * nx * ny * nz:
            raise RuntimeError("Unexpected flux tally size")

        flux_mean, flux_std = mean_and_std(flux_tally["results"][:, 0, :], n)
        flux_mean = flux_mean.reshape(nt, nz, ny, nx)
        flux_std = flux_std.reshape(nt, nz, ny, nx)
        density_mean, density_std = mean_and_std(density_tally["results"][:, 0, :], n)

    dt = np.diff(time_bins)
    rows = []
    for name, (x, y, z) in REFERENCE_POINTS.items():
        i, j, k = int(round(x)), int(round(y)), int(round(z))
        if not np.allclose((x, y, z), (i, j, k)):
            raise RuntimeError(f"Reference point {name} is not a tally-mesh vertex")
        cube = (slice(None), slice(k - 1, k + 1), slice(j - 1, j + 1), slice(i - 1, i + 1))
        # Voxel volume is 1 cm^3; divide by the bin width for a time-averaged flux.
        mean = flux_mean[cube].mean(axis=(1, 2, 3)) * SOURCE_PARTICLES / dt
        std = flux_std[cube].mean(axis=(1, 2, 3)) * SOURCE_PARTICLES / dt
        for b in range(nt):
            rows.append([name, time_bins[b], time_bins[b + 1], mean[b], std[b]])
    for b in range(nt):
        rows.append(
            [
                "population",
                time_bins[b],
                time_bins[b + 1],
                density_mean[b] * SOURCE_PARTICLES / dt[b],
                density_std[b] * SOURCE_PARTICLES / dt[b],
            ]
        )

    with open(output, "w", newline="", encoding="utf-8") as stream:
        stream.write("# OpenMC reference for the time-dependent Kobayashi dog-leg benchmark\n")
        stream.write("# Source: https://doi.org/10.5281/zenodo.15069882 (CC-BY-4.0)\n")
        stream.write(f"# Statepoint md5: {md5sum(statepoint)}\n")
        stream.write(f"# Histories: {n} batches x {n_particles} particles\n")
        stream.write("# Units: source 1 n/cm^3/s in [0,10]^3 for t in [0,50] s, v = 1 cm/s\n")
        stream.write("# Point series: flux averaged over a 2 cm cube and the time bin\n")
        stream.write("# population: neutron population averaged over the time bin\n")
        writer = csv.writer(stream)
        writer.writerow(["series", "t_lo", "t_hi", "mean", "std"])
        for name, t_lo, t_hi, mean, std in rows:
            writer.writerow([name, f"{t_lo:g}", f"{t_hi:g}", f"{mean:.8e}", f"{std:.3e}"])


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "openmc_reference.csv")
