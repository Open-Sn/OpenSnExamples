#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Mesh a two-turn insulated copper coil inside an air box with Gmsh.

The wire centerline is a helix of radius 0.6 cm and pitch 0.55 cm about the
z axis, starting at (0.6, 0, 0). A copper core of radius 0.05 cm and a
polyethylene jacket of outer radius 0.2 cm are swept along it, with the
circular cross section perpendicular to the centerline. The air box spans
[-1.2, 1.2] x [-1.35, 1.35] x [-0.45, 1.75] cm. Lengths are in cm.

Volume tags are 1 (insulation), 2 (copper), and 3 (air). The box faces are
the physical surfaces ``inlet`` (y = 1.35), ``outlet`` (y = -1.35),
``xmin``, ``xmax``, ``zmin``, and ``zmax``.

Usage::

    python3 generate_mesh.py [--size-factor F] [--output coil.msh]

``--size-factor`` scales every target cell size. The default 1.0 gives the
mesh used by ``coil.py``.
"""

import argparse
import math

import gmsh

COIL_RADIUS = 0.60
PITCH = 0.55
TURNS = 2.0
WIRE_RADIUS = 0.20
COPPER_RADIUS = 0.05
BOX = (-1.20, -1.35, -0.45, 1.20, 1.35, 1.75)

# Target tetrahedron sizes (cm) at size factor 1.
H_COPPER = 0.020
H_INSULATION = 0.040
H_AIR = 0.300
AIR_GRADING_DISTANCE = 0.60

INSULATION_TAG = 1
COPPER_TAG = 2
AIR_TAG = 3


def build_geometry(points_per_turn):
    occ = gmsh.model.occ
    npts = int(points_per_turn * TURNS)
    points = []
    for i in range(npts + 1):
        theta = 2.0 * math.pi * TURNS * i / npts
        z = PITCH * TURNS * i / npts
        points.append(occ.addPoint(COIL_RADIUS * math.cos(theta),
                                   COIL_RADIUS * math.sin(theta), z))
    spline = occ.addSpline(points)
    wire = occ.addWire([spline])

    # The helix tangent at the start is (0, cos(alpha), sin(alpha)). Rotate the
    # z-normal disks about the x axis so that their normal matches it.
    alpha = math.atan2(PITCH, 2.0 * math.pi * COIL_RADIUS)
    jacket_disk = occ.addDisk(COIL_RADIUS, 0.0, 0.0, WIRE_RADIUS, WIRE_RADIUS)
    copper_disk = occ.addDisk(COIL_RADIUS, 0.0, 0.0, COPPER_RADIUS, COPPER_RADIUS)
    occ.rotate([(2, jacket_disk), (2, copper_disk)], COIL_RADIUS, 0.0, 0.0,
               1.0, 0.0, 0.0, alpha - 0.5 * math.pi)

    jacket = occ.addPipe([(2, jacket_disk)], wire, "DiscreteTrihedron")
    copper = occ.addPipe([(2, copper_disk)], wire, "DiscreteTrihedron")
    occ.remove([(2, jacket_disk), (2, copper_disk), (1, spline)], recursive=True)

    x0, y0, z0, x1, y1, z1 = BOX
    box = occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
    occ.fragment([(3, box)], jacket + copper)
    occ.synchronize()

    volumes = [tag for _, tag in gmsh.model.getEntities(3)]
    if len(volumes) != 3:
        raise RuntimeError(f"Expected 3 volumes after fragmenting, found {len(volumes)}")
    copper_vol, insulation_vol, air_vol = sorted(
        volumes, key=lambda tag: occ.getMass(3, tag))

    expected_copper = math.pi * COPPER_RADIUS**2 * TURNS * math.hypot(
        2.0 * math.pi * COIL_RADIUS, PITCH)
    copper_volume = occ.getMass(3, copper_vol)
    if abs(copper_volume / expected_copper - 1.0) > 1.0e-3:
        raise RuntimeError(f"Copper volume {copper_volume} differs from {expected_copper}")
    return copper_vol, insulation_vol, air_vol


def add_physical_groups(copper_vol, insulation_vol, air_vol):
    gmsh.model.addPhysicalGroup(3, [insulation_vol], INSULATION_TAG, "insulation")
    gmsh.model.addPhysicalGroup(3, [copper_vol], COPPER_TAG, "copper")
    gmsh.model.addPhysicalGroup(3, [air_vol], AIR_TAG, "air")

    x0, y0, z0, x1, y1, z1 = BOX
    eps = 1.0e-6
    faces = {
        "inlet": (x0, y1, z0, x1, y1, z1),
        "outlet": (x0, y0, z0, x1, y0, z1),
        "xmin": (x0, y0, z0, x0, y1, z1),
        "xmax": (x1, y0, z0, x1, y1, z1),
        "zmin": (x0, y0, z0, x1, y1, z0),
        "zmax": (x0, y0, z1, x1, y1, z1),
    }
    for tag, (name, (a, b, c, d, e, f)) in enumerate(faces.items(), start=10):
        surfaces = gmsh.model.getEntitiesInBoundingBox(
            a - eps, b - eps, c - eps, d + eps, e + eps, f + eps, 2)
        if len(surfaces) != 1:
            raise RuntimeError(f"Box face {name} has {len(surfaces)} surfaces")
        gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], tag, name)


def set_mesh_size(copper_vol, insulation_vol, size_factor):
    field = gmsh.model.mesh.field

    copper = field.add("Constant")
    field.setNumber(copper, "VIn", H_COPPER * size_factor)
    field.setNumbers(copper, "VolumesList", [copper_vol])
    field.setNumber(copper, "IncludeBoundary", 1)

    insulation = field.add("Constant")
    field.setNumber(insulation, "VIn", H_INSULATION * size_factor)
    field.setNumbers(insulation, "VolumesList", [insulation_vol])
    field.setNumber(insulation, "IncludeBoundary", 1)

    jacket_surfaces = [tag for _, tag in gmsh.model.getBoundary(
        [(3, insulation_vol)], oriented=False)]
    distance = field.add("Distance")
    field.setNumbers(distance, "SurfacesList", jacket_surfaces)
    field.setNumber(distance, "Sampling", 200)
    grading = field.add("Threshold")
    field.setNumber(grading, "InField", distance)
    field.setNumber(grading, "SizeMin", H_INSULATION * size_factor)
    field.setNumber(grading, "SizeMax", H_AIR * size_factor)
    field.setNumber(grading, "DistMin", 0.0)
    field.setNumber(grading, "DistMax", AIR_GRADING_DISTANCE)

    minimum = field.add("Min")
    field.setNumbers(minimum, "FieldsList", [copper, insulation, grading])
    field.setAsBackgroundMesh(minimum)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size-factor", type=float, default=1.0)
    parser.add_argument("--points-per-turn", type=int, default=64,
                        help="spline points per helix turn")
    parser.add_argument("--output", default="coil.msh")
    args = parser.parse_args()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.add("coil")
    volumes = build_geometry(args.points_per_turn)
    add_physical_groups(*volumes)
    set_mesh_size(volumes[0], volumes[1], args.size_factor)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)

    counts = {}
    for name, tag in (("insulation", INSULATION_TAG), ("copper", COPPER_TAG),
                      ("air", AIR_TAG)):
        count = 0
        for entity in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
            types, tags, _ = gmsh.model.mesh.getElements(3, entity)
            count += sum(len(t) for t in tags)
        counts[name] = count
    print("Tetrahedra per region:", counts, "total", sum(counts.values()))

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(args.output)
    gmsh.finalize()


if __name__ == "__main__":
    main()
