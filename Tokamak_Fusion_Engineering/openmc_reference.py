#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Continuous-energy OpenMC reference for the tokamak example.

This script builds the same quarter tokamak as tokamak.py from analytic surfaces, runs
OpenMC, and prints the region-integrated flux and absorption per source neutron in the
same form as tokamak.py. Lengths are in cm.

OpenMC needs continuous-energy ENDF/B-VIII.0 data for the nuclides and thermal scattering
tables listed in provenance/materials_64g_p3.json. Set OPENMC_CROSS_SECTIONS to the
cross_sections.xml file of that library. The published reference averages two runs of
100 million histories with seeds 491211 and 491213:

    python3 openmc_reference.py --seed 491211 --processes 32
    python3 openmc_reference.py --seed 491213 --processes 32
"""

import argparse
from functools import reduce
import math
import operator
from pathlib import Path

import openmc

TEMPERATURE = 293.6  # K

# Each D-shaped envelope is (R - R0)^2 + (z / ELONGATION)^2 < a^2 with R > R_cut, where
# R = sqrt(x^2 + y^2). Successive envelopes, listed as (a, R_cut), bound blocks 1 to 6.
R0, ELONGATION = 620.0, 1.55
ENVELOPES = [(160.0, 490.0), (190.0, 460.0), (202.0, 448.0),
             (249.0, 401.0), (266.0, 384.0), (294.0, 356.0)]

# The full reactor has 18 toroidal coils, each a D-shaped shell 0.04 rad wide.
TOROIDAL_COILS, COIL_WIDTH = 18, 0.04
COIL_OUTER, COIL_INNER = (359.0, 291.0), (324.0, 326.0)

# Poloidal coils are tori with a 22 cm minor radius, centered at (R, z).
POLOIDAL_COILS, POLOIDAL_MINOR_RADIUS = [(600.0, -610.0), (600.0, 610.0),
                                         (1040.0, -330.0), (1040.0, 330.0)], 22.0

# The model is the quarter 0 <= phi <= 90 degrees with R <= 1150 and |z| <= 690.
DOMAIN_R, DOMAIN_Z = 1150.0, 690.0

REGIONS = {1: "plasma_source", 2: "vacuum_chamber", 3: "first_wall", 4: "blanket",
           5: "vessel", 6: "shield", 7: "toroidal_coils", 8: "poloidal_coils",
           9: "central_solenoid", 10: "exterior"}


def build_materials():
    """Return the materials by name. Densities are in g/cm^3."""
    # The steel is an idealized Fe/Cr/Ni/Mo alloy given by mass fraction.
    steel = openmc.Material(material_id=5, name="steel", temperature=TEMPERATURE)
    steel.set_density("g/cm3", 8.0)
    for element, fraction in [("Fe", 0.685), ("Cr", 0.17), ("Ni", 0.12), ("Mo", 0.025)]:
        steel.add_element(element, fraction, "wo")
    water = openmc.Material(material_id=101, name="water", temperature=TEMPERATURE)
    water.add_nuclide("H1", 2.0)
    water.add_nuclide("O16", 1.0)
    water.set_density("g/cm3", 0.998)
    # The blanket is 50/50 steel and water by volume. All of its hydrogen is in water.
    blanket = openmc.Material.mix_materials(
        [steel, water], [0.5, 0.5], "vo", name="steel_water", material_id=4)
    blanket.add_s_alpha_beta("c_H_in_H2O")
    blanket.temperature = TEMPERATURE
    tungsten = openmc.Material(material_id=3, name="tungsten", temperature=TEMPERATURE)
    tungsten.add_element("W", 1.0)
    tungsten.set_density("g/cm3", 19.25)
    copper = openmc.Material(material_id=7, name="copper", temperature=TEMPERATURE)
    copper.add_element("Cu", 1.0)
    copper.set_density("g/cm3", 8.96)
    poly = openmc.Material(material_id=102, name="polyethylene", temperature=TEMPERATURE)
    poly.add_nuclide("C12", 1.0)
    poly.add_nuclide("H1", 2.0)
    poly.set_density("g/cm3", 0.94)
    boron = openmc.Material(material_id=103, name="boron", temperature=TEMPERATURE)
    boron.add_element("B", 1.0)
    boron.set_density("g/cm3", 2.34)
    # The shield is polyethylene with 5 wt% natural boron. All of its hydrogen is in the
    # polyethylene.
    shield = openmc.Material.mix_materials(
        [poly, boron], [0.95, 0.05], "wo", name="borated_polyethylene", material_id=6)
    shield.add_s_alpha_beta("c_H_in_CH2")
    shield.temperature = TEMPERATURE
    return {m.name: m for m in (tungsten, blanket, steel, shield, copper)}


def d_region(envelope):
    """Return the inside of a D-shaped envelope (a, R_cut)."""
    radius, cut = envelope
    return (-openmc.ZTorus(a=R0, b=ELONGATION * radius, c=radius)
            & +openmc.ZCylinder(r=cut))


def build_model(particles, batches, seed, group_edges):
    materials = build_materials()
    block_materials = {3: "tungsten", 4: "steel_water", 5: "steel",
                       6: "borated_polyethylene", 7: "copper", 8: "copper", 9: "copper"}

    # The phi = 0 and phi = 90 degree planes reflect. The other boundaries are vacuum.
    lower = openmc.YPlane(y0=0.0, boundary_type="reflective", name="SYM_LOWER")
    upper = openmc.XPlane(x0=0.0, boundary_type="reflective", name="SYM_UPPER")
    bottom = openmc.ZPlane(z0=-DOMAIN_Z, boundary_type="vacuum", name="VACUUM_BOTTOM")
    top = openmc.ZPlane(z0=DOMAIN_Z, boundary_type="vacuum", name="VACUUM_TOP")
    outside = openmc.ZCylinder(r=DOMAIN_R, boundary_type="vacuum", name="VACUUM_OUTER")
    domain = +lower & +upper & +bottom & -top & -outside

    # Blocks 1 to 6 are the shells between successive envelopes.
    envelopes = [d_region(envelope) for envelope in ENVELOPES]
    regions = {1: envelopes[0]}
    regions.update({i + 1: envelopes[i] & ~envelopes[i - 1] for i in range(1, 6)})
    # Build every toroidal coil of the full reactor. The domain clips them to the quarter.
    wedges = []
    for index in range(TOROIDAL_COILS):
        center = 2 * math.pi * index / TOROIDAL_COILS
        low, high = center - COIL_WIDTH / 2, center + COIL_WIDTH / 2
        lo = openmc.Plane(a=-math.sin(low), b=math.cos(low))
        hi = openmc.Plane(a=-math.sin(high), b=math.cos(high))
        wedges.append(+lo & -hi)
    regions[7] = d_region(COIL_OUTER) & ~d_region(COIL_INNER) & reduce(operator.or_, wedges)
    regions[8] = reduce(operator.or_, [
        -openmc.ZTorus(z0=z, a=r, b=POLOIDAL_MINOR_RADIUS, c=POLOIDAL_MINOR_RADIUS)
        for r, z in POLOIDAL_COILS])
    regions[9] = (+openmc.ZCylinder(r=165.0) & -openmc.ZCylinder(r=250.0)
                  & +openmc.ZPlane(z0=-450.0) & -openmc.ZPlane(z0=450.0))
    regions[10] = ~reduce(operator.or_, list(regions.values()))
    geometry = openmc.Geometry([
        openmc.Cell(cell_id=block, name=f"block_{block}", region=region & domain,
                    fill=materials.get(block_materials.get(block)))
        for block, region in regions.items()])

    # The source is uniform and isotropic at 14.1 MeV in block 1. Points are sampled in a
    # cylindrical shell around the plasma and rejected outside cell 1.
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles, settings.batches, settings.seed = particles, batches, seed
    settings.temperature = {"method": "nearest", "tolerance": 10.0}
    settings.output = {"tallies": False}
    radius, cut = ENVELOPES[0]
    settings.source = openmc.IndependentSource(
        space=openmc.stats.CylindricalIndependent(
            r=openmc.stats.PowerLaw(cut, R0 + radius, 1),
            phi=openmc.stats.Uniform(0.0, math.pi / 2),
            z=openmc.stats.Uniform(-ELONGATION * radius, ELONGATION * radius)),
        angle=openmc.stats.Isotropic(), energy=openmc.stats.Discrete([14.1e6], [1.0]),
        constraints={"domains": [geometry.get_all_cells()[1]]})

    # Track-length tallies of flux and absorption in every region, in total and in the
    # 64 groups of the OpenSn library.
    cell_filter = openmc.CellFilter(list(geometry.get_all_cells().values()))
    totals = openmc.Tally(name="material_totals")
    totals.filters, totals.scores = [cell_filter], ["flux", "absorption"]
    totals.estimator = "tracklength"
    groups = openmc.Tally(name="group_flux")
    groups.filters = [cell_filter, openmc.EnergyFilter(group_edges)]
    groups.scores, groups.estimator = ["flux", "absorption"], "tracklength"
    return openmc.Model(geometry, openmc.Materials(materials.values()), settings,
                        openmc.Tallies([totals, groups]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--particles", type=int, default=5000000, help="Histories per batch")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=491211)
    parser.add_argument("--processes", type=int, default=1, help="MPI processes for OpenMC")
    parser.add_argument("--export-only", action="store_true",
                        help="Write the OpenMC XML input without running it")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    library = openmc.MGXSLibrary.from_hdf5(str(here / "materials_64g_p3.h5"))
    group_edges = [float(e) for e in library.energy_groups.group_edges]
    model = build_model(args.particles, args.batches, args.seed, group_edges)
    run_dir = Path(f"openmc_seed{args.seed}")
    run_dir.mkdir(exist_ok=True)
    model.export_to_xml(directory=run_dir)
    if args.export_only:
        return

    mpi_args = ["mpiexec", "-n", str(args.processes)] if args.processes > 1 else None
    openmc.run(cwd=run_dir, mpi_args=mpi_args)
    with openmc.StatePoint(run_dir / f"statepoint.{args.batches}.h5") as sp:
        totals = sp.get_tally(name="material_totals")
        print(f"{'Region':18s} {'Flux [cm]':>24s} {'Absorption':>24s}")
        for block, name in REGIONS.items():
            row = [name]
            for score in ("flux", "absorption"):
                options = dict(scores=[score], filters=[openmc.CellFilter],
                               filter_bins=[(block,)])
                mean = totals.get_values(value="mean", **options).item()
                std_dev = totals.get_values(value="std_dev", **options).item()
                row.append(f"{mean:.6e} +- {std_dev:.1e}")
            print(f"{row[0]:18s} {row[1]:>24s} {row[2]:>24s}")
        print(f"Leakage per source neutron: {sp.global_tallies['mean'][3]:.6e}")


if __name__ == "__main__":
    main()
