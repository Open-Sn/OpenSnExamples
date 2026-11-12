#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Continuous-energy OpenMC photon reference for the coil example.

The model has the same faceted geometry as ``coil.msh`` (through the DAGMC file
written by ``make_dagmc.py``), the same materials, and the same beam: photons
start uniformly on the inlet face y = 1.35 cm, travel along -y, and have the
spectrum E exp(-E / 0.7 MeV) between 1 keV and the 5.7765 MeV upper bound of
the photon groups.

OpenMC does not transport electrons. Its thick-target bremsstrahlung
approximation deposits each electron's kinetic energy, less the bremsstrahlung
it emits, where the electron is created. The heating tallies are therefore the
local-deposition limit. The photon flux and leakage tallies are directly
comparable with OpenSn.

Usage::

    python3 openmc_reference.py --dagmc coil.h5m --particles 10000000 --batches 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import openmc

BOX = (-1.20, -1.35, -0.45, 1.20, 1.35, 1.75)
SPECTRUM_TEMPERATURE = 0.7e6
FACES = ("inlet", "outlet", "xmin", "xmax", "zmin", "zmax")


def materials():
    copper = openmc.Material(name="copper")
    copper.add_element("Cu", 1.0)
    copper.set_density("g/cm3", 8.96)

    polyethylene = openmc.Material(name="polyethylene")
    polyethylene.add_element("H", 0.143716, "wo")
    polyethylene.add_element("C", 0.856284, "wo")
    polyethylene.set_density("g/cm3", 0.94)

    air = openmc.Material(name="air")
    for element, fraction in (("C", 0.000124), ("N", 0.755268), ("O", 0.231781),
                              ("Ar", 0.012827)):
        air.add_element(element, fraction, "wo")
    air.set_density("g/cm3", 0.001205)
    return copper, polyethylene, air


def build_model(dagmc_file, particles, batches, seed, photon_bounds_ev):
    copper, polyethylene, air = materials()
    x0, y0, z0, x1, y1, z1 = BOX
    box = openmc.model.RectangularParallelepiped(x0, x1, y0, y1, z0, z1,
                                                 boundary_type="vacuum")
    coil = openmc.DAGMCUniverse(dagmc_file, auto_geom_ids=True)
    geometry = openmc.Geometry([openmc.Cell(fill=coil, region=-box)])

    # Fine tabulation of the incident spectrum, linear-linear between points.
    energies = np.geomspace(1.0e3, photon_bounds_ev[-1], 4001)
    shape = energies * np.exp(-energies / SPECTRUM_TEMPERATURE)
    source = openmc.IndependentSource(
        space=openmc.stats.CartesianIndependent(
            openmc.stats.Uniform(x0, x1),
            openmc.stats.Discrete([y1 - 1.0e-9], [1.0]),
            openmc.stats.Uniform(z0, z1)),
        angle=openmc.stats.Monodirectional((0.0, -1.0, 0.0)),
        energy=openmc.stats.Tabular(energies, shape, "linear-linear"),
        particle="photon")

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = source
    settings.particles = particles
    settings.batches = batches
    settings.photon_transport = True
    settings.electron_treatment = "ttb"
    settings.cutoff = {"energy_photon": 1.0e3}
    settings.seed = seed
    settings.output = {"tallies": False}

    material_filter = openmc.MaterialFilter([copper, polyethylene, air])
    group_filter = openmc.EnergyFilter(photon_bounds_ev)
    photon_filter = openmc.ParticleFilter(["photon"])

    flux = openmc.Tally(name="region photon flux")
    flux.filters = [material_filter, group_filter, photon_filter]
    flux.scores = ["flux"]
    flux.estimator = "tracklength"

    heating = openmc.Tally(name="region heating")
    heating.filters = [material_filter,
                       openmc.ParticleFilter(["photon", "electron", "positron"])]
    heating.scores = ["heating"]

    surfaces = [box.ymax, box.ymin, box.xmin, box.xmax, box.zmin, box.zmax]
    surface_filter = openmc.SurfaceFilter(surfaces)
    leakage = openmc.Tally(name="face photon current")
    leakage.filters = [surface_filter, group_filter, photon_filter]
    leakage.scores = ["current"]
    energy_weight = openmc.EnergyFunctionFilter([0.0, 1.0e8], [0.0, 1.0e8])
    energy_leakage = openmc.Tally(name="face photon energy current")
    energy_leakage.filters = [surface_filter, energy_weight, photon_filter]
    energy_leakage.scores = ["current"]

    tallies = openmc.Tallies([flux, heating, leakage, energy_leakage])
    model = openmc.Model(geometry, openmc.Materials([copper, polyethylene, air]),
                         settings, tallies)
    return model, surfaces


def summarize(statepoint_path, surfaces, output):
    """Write the tallies per source photon, with standard deviations, to JSON."""
    # Surface currents are positive along the plane normal; flip the faces
    # whose outward direction is negative.
    outward = {"inlet": 1.0, "outlet": -1.0, "xmin": -1.0, "xmax": 1.0,
               "zmin": -1.0, "zmax": 1.0}
    regions = ("copper", "insulation", "air")
    with openmc.StatePoint(statepoint_path) as sp:
        flux = sp.get_tally(name="region photon flux")
        heating = sp.get_tally(name="region heating")
        leakage = sp.get_tally(name="face photon current")
        energy_leakage = sp.get_tally(name="face photon energy current")

        def per_region(tally, count):
            mean = tally.mean.reshape(len(regions), count)
            sd = tally.std_dev.reshape(len(regions), count)
            return mean, sd

        # The energy filter is ascending; OpenSn groups are descending.
        flux_mean, flux_sd = per_region(flux, flux.mean.size // 3)
        heat_mean, heat_sd = per_region(heating, 3)
        leak = leakage.mean.reshape(len(FACES), -1)
        leak_sd = leakage.std_dev.reshape(len(FACES), -1)
        eleak = energy_leakage.mean.reshape(len(FACES))
        eleak_sd = energy_leakage.std_dev.reshape(len(FACES))
        results = {
            "openmc_version": openmc.__version__,
            "particles": int(sp.n_particles),
            "batches": int(sp.n_batches),
            "seed": int(sp.seed),
            "region_photon_flux": {r: flux_mean[i, ::-1].tolist() for i, r in enumerate(regions)},
            "region_photon_flux_sd": {r: flux_sd[i, ::-1].tolist()
                                      for i, r in enumerate(regions)},
            # Heating is in eV per source photon; convert to MeV.
            "region_heating_mev": {r: float(heat_mean[i].sum() * 1.0e-6)
                                   for i, r in enumerate(regions)},
            "region_heating_sd_mev": {r: float(np.sqrt((heat_sd[i] ** 2).sum()) * 1.0e-6)
                                      for i, r in enumerate(regions)},
            "region_heating_by_particle_mev": {
                r: dict(zip(("photon", "electron", "positron"), (heat_mean[i] * 1.0e-6).tolist()))
                for i, r in enumerate(regions)},
            "face_photon_leakage": {f: (outward[f] * leak[i, ::-1]).tolist()
                                    for i, f in enumerate(FACES)},
            "face_photon_leakage_sd": {f: leak_sd[i, ::-1].tolist() for i, f in enumerate(FACES)},
            "face_photon_energy_leakage_mev": {f: float(outward[f] * eleak[i] * 1.0e-6)
                                               for i, f in enumerate(FACES)},
            "face_photon_energy_leakage_sd_mev": {f: float(eleak_sd[i] * 1.0e-6)
                                                  for i, f in enumerate(FACES)},
        }
    Path(output).write_text(json.dumps(results, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dagmc", default="coil.h5m")
    parser.add_argument("--particles", type=int, default=10_000_000)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", default="openmc_results.json")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    groups = json.loads((here / "xs" / "groups.json").read_text())
    photon = groups["photon"]
    bounds_ev = [1.0e6 * photon[-1][1]] + [1.0e6 * upper for upper, _ in reversed(photon)]

    model, surfaces = build_model(str(Path(args.dagmc).resolve()), args.particles,
                                  args.batches, args.seed, bounds_ev)
    statepoint = model.run(threads=args.threads)
    summarize(statepoint, surfaces, args.output)


if __name__ == "__main__":
    main()
