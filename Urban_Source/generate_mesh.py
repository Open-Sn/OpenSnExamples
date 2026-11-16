#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate urban_source.msh, the mesh of the city described by city.toml, and
urban_source_mesh.json, the description of its blocks that urban_source.py reads.

Gmsh builds the city from OpenCASCADE solids. The ground is a slab of boxes, one for each
lattice cell. Each building is its footprint extruded from the ground surface to its
height. The air is a box from the ground surface to the top of the domain. The solids are
fragmented so that the mesh is conforming, and each fragment belongs to a building if it
lies in one, else to the ground, else to the air. Lengths are in cm in the mesh.

The air is meshed with linear tetrahedra. The ground is meshed with prisms: the triangles
of the ground surface are extruded down in layers that are thinnest at the surface. Most of
the gamma rays that reach the ground collide within a few cm of its surface, and the gamma
rays scattered back from that skin are most of the scattered flux in the air, so the layers
must resolve it. Tetrahedra as large as those on the ground surface would make the
scattered flux in the air about 30% too low.

The buildings are meshed in layers too. Next to the walls, each footprint is split into
rings that are thinnest at the wall, and every piece of the footprint is extruded to the
roof in equal layers and split into tetrahedra. The mean free path in a building is about
0.5 m, so the flux falls steeply across the outer meter of its walls. Tetrahedra as large as
those on the walls, without the rings, make the dose in the shadows of the buildings about
6% too low.

The point source sits in the air. Four points of a regular tetrahedron centered on the
source are embedded in the mesh, so the source lies well inside one cell rather than on or
near a face, edge, or vertex. The script checks this and stops if Gmsh did not keep that
tetrahedron.

The elements are smallest at the source, because the uncollided flux varies as 1/r^2
there, and next to the ground surface and the building walls and roofs, where most of the
gamma rays collide. Every element size is multiplied by --size-factor, for refinement
studies.

The script checks the volume of every block, that no cell is inverted or folded, and that
the source lies at the centroid of its cell. It needs the gmsh and numpy Python packages.
Another Gmsh version can give a different mesh. The supplied mesh was made with

    python3 generate_mesh.py --size-factor 2
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import gmsh
import numpy as np

import city as citylib

CM = 100.0  # cm per m

# OpenSn identifies physical names by their tag alone, so the boundary tag must differ from
# the block IDs.
BOUNDARY_ID, BOUNDARY_NAME = 100, "VACUUM"

# Element sizes in m at --size-factor 1.
SOURCE_SIZE = 0.5  # at the source
SOURCE_GROWTH = 0.3  # increase per unit distance from the source
SURFACE_SIZE = 2.0  # at the ground surface and at the walls and roofs, on both sides
SURFACE_GROWTH = 0.3  # increase per unit distance from these surfaces
BUILDING_SIZE = 4.0  # inside the buildings, away from their surfaces (largest layer)
AIR_SIZE = 12.0  # in the air, away from the source and the surfaces
# Points per direction on each surface at which Gmsh samples the distance to it.
DISTANCE_SAMPLING = 60
# Ground layers: the first is GROUND_FIRST_LAYER thick at --size-factor 1, and each layer
# below is GROUND_LAYER_GROWTH times thicker than the one above, down to the ground depth.
GROUND_FIRST_LAYER = 0.01  # m
GROUND_LAYER_GROWTH = 1.6
# Building skin layers: next to the walls, the buildings are meshed with layers that are
# WALL_FIRST_LAYER thick at the wall at --size-factor 1 and grow by WALL_LAYER_GROWTH per
# layer, to WALL_SKIN_DEPTH into the building or less in a narrow one. Vertically, the
# buildings are split into equal layers of about SURFACE_SIZE, so that the faces of the walls
# are not slivers. A circle or ellipse is replaced by a polygon whose sides are CURVED_FACET
# long at --size-factor 1.
WALL_FIRST_LAYER = 0.05  # m
WALL_LAYER_GROWTH = 1.6
WALL_SKIN_DEPTH = 2.0  # m
CURVED_FACET = 1.0  # m

# Orientation of the tetrahedron embedded around the source. A generic rotation keeps the
# edges of the cells around it off the coordinate planes through the source.
SOURCE_TETRAHEDRON_AXIS, SOURCE_TETRAHEDRON_ANGLE = (0.31, 0.57, 0.76), 0.83


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polygon_area(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def footprint_polygon(footprint, facet):
    """Return the footprint as polygon vertices in m, counterclockwise. A circle or ellipse
    is replaced by the inscribed polygon whose sides are at most facet long."""
    if footprint.kind == "polygon":
        return np.asarray(footprint.vertices, float)
    a, b = footprint.a, footprint.b
    h = ((a - b) / (a + b)) ** 2
    perimeter = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    return footprint.outline(max(12, math.ceil(perimeter / facet)))


def offset_polygon(vertices, distance):
    """Return the polygon offset inward by distance, each vertex moved along the bisector of
    its corner, or None if the offset polygon is not valid (an edge reversed, or edges that
    cross)."""
    edge = np.roll(vertices, -1, axis=0) - vertices
    tangent = edge / np.linalg.norm(edge, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))  # inward, counterclockwise
    previous = np.roll(normal, 1, axis=0)
    offset = vertices + distance * (normal + previous) / (
        1.0 + np.einsum("ij,ij->i", normal, previous))[:, None]
    new_edge = np.roll(offset, -1, axis=0) - offset
    if np.any(np.einsum("ij,ij->i", new_edge, tangent) <= 0.0):
        return None
    n = len(offset)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if citylib._segments_intersect(offset[i], offset[(i + 1) % n], offset[j],
                                           offset[(j + 1) % n]):
                return None
    return offset


def skin_levels(vertices, first, growth, depth):
    """Return the footprint polygon and its inward offsets at the skin layer interfaces."""
    levels, distance, thickness = [vertices], 0.0, first
    area = polygon_area(vertices)
    while distance + thickness <= depth:
        offset = offset_polygon(vertices, distance + thickness)
        if offset is None or polygon_area(offset) < 0.25 * area:
            break
        levels.append(offset)
        distance += thickness
        thickness *= growth
    return levels


def add_building(building, index, factor, registry):
    """Add the solids of a building and return their volume tags.

    The footprint is split into rings at the skin layer interfaces, each ring into one
    quadrilateral per side of the footprint polygon, and a core; every piece is extruded to
    the roof in equal layers. registry records the vertices and the number of segments of
    every side, for the mesh constraints set after the fragmentation.
    """
    occ = gmsh.model.occ
    polygon = footprint_polygon(building.footprint, factor * CURVED_FACET)
    levels = skin_levels(polygon, factor * WALL_FIRST_LAYER, WALL_LAYER_GROWTH,
                         WALL_SKIN_DEPTH)
    n = len(polygon)
    edge_length = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
    registry["segments"][index] = [max(1, math.ceil(length / (factor * SURFACE_SIZE)))
                                   for length in edge_length]
    registry["vertices"][index] = n
    registry["polygons"][index] = polygon
    points = []
    for k, level in enumerate(levels):
        row = []
        for i, (x, y) in enumerate(level):
            row.append(occ.addPoint(x * CM, y * CM, 0.0))
            registry["points"][(round(x * CM, 3), round(y * CM, 3))] = (index, k, i)
        points.append(row)
    along = [[occ.addLine(row[i], row[(i + 1) % n]) for i in range(n)] for row in points]
    radial = [[occ.addLine(points[k][i], points[k + 1][i]) for i in range(n)]
              for k in range(len(levels) - 1)]
    faces = []
    for k in range(len(levels) - 1):
        for i in range(n):
            loop = occ.addCurveLoop([along[k][i], radial[k][(i + 1) % n], along[k + 1][i],
                                     radial[k][i]])
            faces.append(occ.addPlaneSurface([loop]))
    faces.append(occ.addPlaneSurface([occ.addCurveLoop(along[-1])]))
    height = building.height * CM
    layers = max(1, math.ceil(building.height / (factor * SURFACE_SIZE)))
    # Without recombination, Gmsh splits each extruded prism into three tetrahedra, so the
    # walls and roofs are triangles and the air needs no pyramids against them.
    extruded = occ.extrude([(2, f) for f in faces], 0, 0, height, numElements=[layers],
                           recombine=False)
    return [tag for dim, tag in extruded if dim == 3]


def set_building_constraints(registry):
    """Mesh the rings of the building footprints with structured triangles: one segment
    across each ring, and the same segments along each side at every level."""
    model, mesh = gmsh.model, gmsh.model.mesh
    point_info = {}
    for _, tag in model.getEntities(0):
        x, y, z = model.getValue(0, tag, [])
        if abs(z) < 1e-6:
            info = registry["points"].get((round(x, 3), round(y, 3)))
            if info is not None:
                point_info[tag] = info
    kind = {}
    for _, curve in model.getEntities(1):
        ends = [abs(t) for _, t in model.getBoundary([(1, curve)], oriented=False)]
        if len(ends) != 2 or not all(e in point_info for e in ends):
            continue
        (b0, k0, i0), (b1, k1, i1) = point_info[ends[0]], point_info[ends[1]]
        if b0 != b1:
            continue
        n = registry["vertices"][b0]
        if k0 == k1 and (i1 - i0) % n in (1, n - 1):
            side = i0 if (i1 - i0) % n == 1 else i1
            mesh.setTransfiniteCurve(curve, registry["segments"][b0][side] + 1)
            kind[curve] = "along"
        elif i0 == i1 and abs(k0 - k1) == 1:
            mesh.setTransfiniteCurve(curve, 2)
            kind[curve] = "radial"
    for _, surface in model.getEntities(2):
        x0, y0, z0, x1, y1, z1 = model.getBoundingBox(2, surface)
        if abs(z0) > 1e-6 or abs(z1) > 1e-6:
            continue
        curves = [abs(t) for _, t in model.getBoundary([(2, surface)], oriented=False)]
        kinds = [kind.get(c) for c in curves]
        if len(curves) == 4 and kinds.count("radial") == 2 and kinds.count("along") == 2:
            mesh.setTransfiniteSurface(surface)


def build_city(city, factor):
    """Build the solids and return the volume tags of each block and the registry of the
    building footprints."""
    occ = gmsh.model.occ
    xmin, xmax, ymin, ymax, zmin, zmax = (v * CM for v in city.extent)
    registry = {"points": {}, "segments": {}, "vertices": {}, "polygons": {}}
    owners = []  # (volume tag, block), in order of precedence
    for index, building in enumerate(city.buildings):
        for tag in add_building(building, index, factor, registry):
            owners.append((tag, building.block))
    for cell in city.ground:
        x0, x1, y0, y1 = (v * CM for v in cell.bounds)
        owners.append((occ.addBox(x0, y0, zmin, x1 - x0, y1 - y0, -zmin), cell.block))
    owners.append((occ.addBox(xmin, ymin, 0.0, xmax - xmin, ymax - ymin, zmax),
                   citylib.AIR_BLOCK))
    inputs = [(3, tag) for tag, _ in owners]
    _, children = occ.fragment(inputs, [])
    occ.synchronize()

    # A fragment lies inside several of the overlapping solids, and belongs to the first of
    # them in the order buildings, ground, air.
    owner = {}
    for (_, block), pieces in zip(owners, children):
        for _, piece in pieces:
            owner.setdefault(piece, block)
    groups = {block: [] for block in city.block_materials()}
    for tag, block in owner.items():
        groups[block].append(tag)
    for block, tags in groups.items():
        if not tags:
            raise RuntimeError(f"Block {block} ({city.names[block]}) has no volume")
    return groups, registry


def expected_volumes(city, polygons=None):
    """Return the volume of every block in cm^3: exact, or with the building footprints
    replaced by polygons (index -> vertices) when given."""
    xmin, xmax, ymin, ymax, zmin, zmax = city.extent
    volumes = {block: 0.0 for block in city.block_materials()}
    for cell in city.ground:
        x0, x1, y0, y1 = cell.bounds
        volumes[cell.block] += (x1 - x0) * (y1 - y0) * -zmin
    buildings = 0.0
    for building in city.buildings:
        area = (building.footprint.area() if polygons is None
                else polygon_area(polygons[city.buildings.index(building)]))
        volumes[building.block] = area * building.height
        buildings += volumes[building.block]
    volumes[citylib.AIR_BLOCK] = (xmax - xmin) * (ymax - ymin) * zmax - buildings
    return {block: v * CM**3 for block, v in volumes.items()}


def check_volumes(city, groups, polygons):
    """Compare the OpenCASCADE volume of each block with its volume as built."""
    for block, expected in expected_volumes(city, polygons).items():
        volume = sum(gmsh.model.occ.getMass(3, tag) for tag in groups[block])
        if abs(volume / expected - 1) > 1e-6:
            raise RuntimeError(f"Block {block} ({city.names[block]}) volume {volume} differs "
                               f"from the exact volume {expected}")


def rotation_matrix(axis, angle):
    """Return the matrix of a rotation by angle (radians) about axis."""
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def embed_source_tetrahedron(city, air_tags, factor):
    """Embed the vertices of a regular tetrahedron centered on the source."""
    edge = factor * SOURCE_SIZE * CM
    radius = edge * math.sqrt(3 / 8)  # Circumradius of a regular tetrahedron
    vertices = radius * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / math.sqrt(3)
    vertices = vertices @ rotation_matrix(SOURCE_TETRAHEDRON_AXIS, SOURCE_TETRAHEDRON_ANGLE).T
    vertices = vertices + city.source * CM
    points = [gmsh.model.occ.addPoint(*v, edge) for v in vertices]
    gmsh.model.occ.synchronize()
    if len(air_tags) != 1:
        raise RuntimeError("The air must be a single volume")
    gmsh.model.mesh.embed(0, points, 3, air_tags[0])
    return vertices


def set_sizes(city, groups, ground_faces, factor, air_size=AIR_SIZE, wall_size=SURFACE_SIZE):
    """Set the element sizes of the air with Gmsh fields. The buildings are extruded, so
    their elements are set by their layers. air_size and wall_size [m, at size factor 1]
    override AIR_SIZE and, next to the walls and roofs, SURFACE_SIZE."""
    field = gmsh.model.mesh.field
    model = gmsh.model

    def threshold(distance, size_min, growth):
        tag = field.add("Threshold")
        field.setNumber(tag, "InField", distance)
        field.setNumber(tag, "SizeMin", factor * size_min * CM)
        field.setNumber(tag, "SizeMax", factor * air_size * CM)
        field.setNumber(tag, "DistMin", 0.0)
        field.setNumber(tag, "DistMax", (air_size - size_min) / growth * CM)
        return tag

    source = field.add("Distance")
    source_point = gmsh.model.occ.addPoint(*(city.source * CM))
    gmsh.model.occ.synchronize()
    field.setNumbers(source, "PointsList", [source_point])

    # The ground surface, z = 0, and the outer surfaces of the buildings.
    building_tags = [t for b in city.buildings for t in groups[b.block]]
    building_faces = model.getBoundary([(3, t) for t in building_tags], combined=True,
                                       oriented=False)
    walls = field.add("Distance")
    field.setNumbers(walls, "SurfacesList", sorted({abs(tag) for _, tag in building_faces}))
    field.setNumber(walls, "Sampling", DISTANCE_SAMPLING)
    ground = field.add("Distance")
    field.setNumbers(ground, "SurfacesList", sorted(ground_faces))
    field.setNumber(ground, "Sampling", DISTANCE_SAMPLING)

    smallest = field.add("Min")
    field.setNumbers(smallest, "FieldsList", [
        threshold(source, SOURCE_SIZE, SOURCE_GROWTH),
        threshold(ground, SURFACE_SIZE, SURFACE_GROWTH),
        threshold(walls, wall_size, SURFACE_GROWTH)])
    field.setAsBackgroundMesh(smallest)


def ground_surfaces(city, groups):
    """Return the block of the ground under each surface at z = 0."""
    faces = {}
    for block in city.ground_blocks:
        for _, tag in gmsh.model.getBoundary([(3, t) for t in groups[block]], combined=False,
                                             oriented=False):
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, abs(tag))
            if abs(z0) < 1e-6 and abs(z1) < 1e-6:
                faces[abs(tag)] = block
    return faces


def remove_ground_solids(city, groups):
    """Remove the ground volumes, and the surfaces, curves, and points that bound nothing
    else, so that Gmsh meshes only the air and the buildings."""
    model = gmsh.model
    model.removeEntities([(3, t) for block in city.ground_blocks for t in groups[block]])
    kept = [(3, t) for block, tags in groups.items() if block not in city.ground_blocks
            for t in tags]
    used = {3: {t for _, t in kept}}
    for dim in (2, 1, 0):
        bounds = model.getBoundary([(dim + 1, t) for t in used[dim + 1]], combined=False,
                                   oriented=False)
        used[dim] = {abs(t) for _, t in bounds}
    # The vertices of the source tetrahedron are embedded in the air, not on its boundary.
    used[0] |= {t for d, t in model.mesh.getEmbedded(3, groups[citylib.AIR_BLOCK][0])
                if d == 0}
    for dim in (2, 1, 0):
        orphans = [(dim, t) for _, t in model.getEntities(dim) if t not in used[dim]]
        if orphans:
            model.removeEntities(orphans)


def tag_groups(city, groups):
    """Add the physical groups of the air and the buildings, and return the surfaces of the
    outer boundary above the ground."""
    for block, tags in groups.items():
        if block not in city.ground_blocks:
            gmsh.model.addPhysicalGroup(3, sorted(tags), block, city.names[block])
    volumes = [(3, t) for block, tags in groups.items() if block not in city.ground_blocks
               for t in tags]
    outer = gmsh.model.getBoundary(volumes, combined=True, oriented=False)
    surfaces = []
    for _, tag in outer:
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, abs(tag))
        if not (abs(z0) < 1e-6 and abs(z1) < 1e-6):
            surfaces.append(abs(tag))
    return sorted(surfaces)


def layer_depths(city, factor):
    """Return the depths in cm of the ground layer interfaces, from 0 to the ground depth."""
    depth = city.ground_depth * CM
    first = factor * GROUND_FIRST_LAYER * CM
    depths, thickness = [0.0], first
    while depths[-1] + thickness < depth - 0.5 * thickness * GROUND_LAYER_GROWTH:
        depths.append(depths[-1] + thickness)
        thickness *= GROUND_LAYER_GROWTH
    depths.append(depth)
    return depths


def add_ground_prisms(city, faces, depths, boundary_surfaces):
    """Extrude the triangles and quadrilaterals of the ground surface down into layers of
    prisms and hexahedra, one discrete volume for each ground block, and add the outer
    boundary group. Return the number of ground cells."""
    model, mesh = gmsh.model, gmsh.model.mesh
    # Ground surface elements of each block, by number of nodes (3 or 4).
    elements = {}
    for surface, block in faces.items():
        types, _, nodes = mesh.getElements(2, surface)
        for element_type, element_nodes in zip(types, nodes):
            if element_type not in (2, 3):
                raise RuntimeError(f"Ground surface {surface} has elements of type "
                                   f"{element_type}")
            width = 3 if element_type == 2 else 4
            elements.setdefault((block, width), []).append(
                element_nodes.astype(np.int64).reshape(-1, width))
    elements = {key: np.vstack(e) for key, e in elements.items()}
    node_tags, coordinates, _ = mesh.getNodes()
    points = np.zeros((int(node_tags.max()) + 1, 3))
    points[node_tags.astype(np.int64)] = coordinates.reshape(-1, 3)
    top = np.unique(np.concatenate([e.ravel() for e in elements.values()]))
    if np.any(np.abs(points[top, 2]) > 1e-6):
        raise RuntimeError("A ground surface node is not at z = 0")
    # Node tags of each level: level 0 is the ground surface, level k is at depth depths[k].
    position = np.full(points.shape[0], -1, dtype=np.int64)
    position[top] = np.arange(len(top))
    first_new = int(node_tags.max()) + 1
    levels = [top] + [first_new + (k - 1) * len(top) + np.arange(len(top))
                      for k in range(1, len(depths))]
    new_coordinates = np.vstack([np.column_stack((points[top, 0], points[top, 1],
                                                  np.full(len(top), -d)))
                                 for d in depths[1:]])
    blocks = sorted({block for block, _ in elements})
    entities = {block: model.addDiscreteEntity(3) for block in blocks}
    mesh.addNodes(3, entities[blocks[0]], np.concatenate(levels[1:]).tolist(),
                  new_coordinates.ravel().tolist())

    def level(k, nodes):
        return levels[k][position[nodes]]

    count = 0
    for (block, width), face in elements.items():
        # Order each face counterclockwise seen from above, so that the cell with the lower
        # face first has a positive volume.
        p = points[face]
        area = (np.einsum("ij,ij->i", p[:, :, 0], np.roll(p[:, :, 1], -1, axis=1))
                - np.einsum("ij,ij->i", np.roll(p[:, :, 0], -1, axis=1), p[:, :, 1]))
        face = np.where((area < 0)[:, None], face[:, ::-1], face)
        cells = [np.column_stack([level(k + 1, face[:, i]) for i in range(width)]
                                 + [level(k, face[:, i]) for i in range(width)])
                 for k in range(len(depths) - 1)]
        mesh.addElementsByType(entities[block], 6 if width == 3 else 5, [],
                               np.vstack(cells).ravel().tolist())
        count += len(face) * (len(depths) - 1)
    for block in blocks:
        model.addPhysicalGroup(3, [entities[block]], block, city.names[block])

    # Outer boundary of the ground: the bottom faces and the quadrilaterals of the sides.
    surface = model.addDiscreteEntity(2)
    xmin, xmax, ymin, ymax, _, _ = (v * CM for v in city.extent)
    side_quads = []
    for (block, width), face in elements.items():
        bottom = np.column_stack([level(len(depths) - 1, face[:, i])
                                  for i in range(width - 1, -1, -1)])
        mesh.addElementsByType(surface, 2 if width == 3 else 3, [], bottom.ravel().tolist())
        edges = np.vstack([face[:, [i, (i + 1) % width]] for i in range(width)])
        a, b = points[edges[:, 0]], points[edges[:, 1]]
        on_side = np.zeros(len(edges), dtype=bool)
        for axis, value in ((0, xmin), (0, xmax), (1, ymin), (1, ymax)):
            on_side |= ((np.abs(a[:, axis] - value) < 1e-6)
                        & (np.abs(b[:, axis] - value) < 1e-6))
        side = edges[on_side]
        side_quads += [np.column_stack((level(k, side[:, 0]), level(k, side[:, 1]),
                                        level(k + 1, side[:, 1]), level(k + 1, side[:, 0])))
                       for k in range(len(depths) - 1)]
    mesh.addElementsByType(surface, 3, [], np.vstack(side_quads).ravel().tolist())
    model.addPhysicalGroup(2, boundary_surfaces + [surface], BOUNDARY_ID, BOUNDARY_NAME)
    return count


# Nodes of the linear Gmsh elements, and their splits into positive tetrahedra.
NODES = {4: 4, 5: 8, 6: 6, 7: 5}
SPLIT = {
    4: [(0, 1, 2, 3)],
    5: [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)],
    6: [(0, 1, 2, 5), (0, 1, 5, 4), (0, 4, 5, 3)],
    7: [(0, 1, 2, 4), (0, 2, 3, 4)],
}


# Faces of the linear Gmsh elements, as local node indices.
FACES = {
    4: [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
    5: [(0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)],
    6: [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    7: [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}


def check_folds(points, cells):
    """Check that every interior face separates its two cells: their centroids must lie on
    opposite sides of it, and no face may belong to more than two cells. cells maps a Gmsh
    element type to node tags [cell, node]."""
    for size in (3, 4):
        keys, centroids = [], []
        for element_type, nodes in cells.items():
            if len(nodes) == 0:
                continue
            centroid = points[nodes].mean(axis=1)
            for face in FACES[element_type]:
                if len(face) == size:
                    keys.append(np.sort(nodes[:, list(face)], axis=1))
                    centroids.append(centroid)
        if not keys:
            continue
        keys, centroids = np.vstack(keys), np.vstack(centroids)
        order = np.lexsort(keys.T[::-1])
        keys, centroids = keys[order], centroids[order]
        same = np.all(keys[1:] == keys[:-1], axis=1)
        if np.any(same[1:] & same[:-1]):
            raise RuntimeError("A face of the mesh belongs to more than two cells")
        first = np.where(same)[0]
        # One normal per face, from its sorted nodes, so that both cells use the same one.
        p = points[keys[first, :3]]
        normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        side_a = np.einsum("ij,ij->i", centroids[first] - p[:, 0], normal)
        side_b = np.einsum("ij,ij->i", centroids[first + 1] - p[:, 0], normal)
        folded = np.count_nonzero(side_a * side_b >= 0)
        if folded:
            raise RuntimeError(f"The mesh is folded: {folded} interior faces have both of "
                               f"their cells on the same side")


def tetrahedron_volumes(points, tetrahedra):
    """Return the signed volumes of tetrahedra given by node tags."""
    corners = points[tetrahedra]
    edges = np.stack([corners[:, i] - corners[:, 0] for i in (1, 2, 3)], axis=2)
    return np.linalg.det(edges) / 6


def check_mesh(city, vertices):
    """Check orientation and the cell that contains the source, and return statistics."""
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    points = np.zeros((int(node_tags.max()) + 1, 3))
    points[node_tags.astype(int)] = coordinates.reshape(-1, 3)
    counts, volumes = {}, {}
    source_cell = None
    all_cells = {t: [] for t in SPLIT}
    for block in city.block_materials():
        cells = {t: [] for t in SPLIT}  # Gmsh element type -> node tags
        for tag in gmsh.model.getEntitiesForPhysicalGroup(3, block):
            types, _, nodes = gmsh.model.mesh.getElements(3, tag)
            for element_type, element_nodes in zip(types, nodes):
                if element_type not in cells:
                    raise RuntimeError("The mesh must contain only linear tetrahedra, "
                                       "hexahedra, prisms, and pyramids")
                cells[element_type].append(element_nodes.astype(int).reshape(
                    -1, NODES[element_type]))
        cells = {t: np.vstack(c) if c else np.zeros((0, NODES[t]), dtype=int)
                 for t, c in cells.items()}
        # Each of the tetrahedra into which every cell is split must have a positive volume.
        signed = np.concatenate([tetrahedron_volumes(points, cells[t][:, list(tet)])
                                 for t, split in SPLIT.items() for tet in split])
        if np.any(signed <= 0):
            bad = {t: int(np.count_nonzero(np.min([tetrahedron_volumes(
                points, cells[t][:, list(tet)]) for tet in split], axis=0) <= 0))
                for t, split in SPLIT.items() if len(cells[t])}
            raise RuntimeError(f"The mesh has inverted cells in block {block}, by Gmsh "
                               f"element type: {bad}")
        counts[block] = sum(len(c) for c in cells.values())
        for t, c in cells.items():
            all_cells[t].append(c)
        volumes[block] = float(signed.sum())
        corners = points[cells[4]]
        edges = np.stack([corners[:, i] - corners[:, 0] for i in (1, 2, 3)], axis=2)
        if block == citylib.AIR_BLOCK:
            # Barycentric coordinates of the source in every air cell.
            source = city.source * CM
            weights = np.linalg.solve(edges, (source - corners[:, 0])[:, :, None])[:, :, 0]
            bary = np.column_stack((1 - weights.sum(axis=1), weights))
            inside = np.where(np.all(bary > 0, axis=1))[0]
            if len(inside) != 1:
                raise RuntimeError("The source does not lie strictly inside one cell")
            source_cell = (corners[inside[0]], bary[inside[0]])
    check_folds(points, {t: np.vstack(c) for t, c in all_cells.items()})
    corners, bary = source_cell
    if not np.allclose(np.sort(corners, axis=0), np.sort(vertices, axis=0), atol=1e-6):
        raise RuntimeError("Gmsh did not keep the embedded source tetrahedron")
    return counts, volumes, bary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, default=Path("city.toml"))
    parser.add_argument("--output", type=Path, default=Path("urban_source.msh"))
    parser.add_argument("--size-factor", type=float, default=1.0,
                        help="Factor applied to every element size, for refinement studies")
    parser.add_argument("--air-size", type=float, default=AIR_SIZE,
                        help="Element size in m in the air away from the source and surfaces")
    parser.add_argument("--wall-size", type=float, default=SURFACE_SIZE,
                        help="Element size in m in the air next to the buildings")
    parser.add_argument("--threads", type=int, default=1,
                        help="Gmsh threads. More than one can change the mesh from run to run.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists. Remove it or choose another name.")
    city = citylib.load(args.layout)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", args.threads)
        gmsh.model.add("urban_source")
        groups, registry = build_city(city, args.size_factor)
        check_volumes(city, groups, registry["polygons"])
        faces = ground_surfaces(city, groups)
        vertices = embed_source_tetrahedron(city, groups[citylib.AIR_BLOCK], args.size_factor)
        remove_ground_solids(city, groups)
        boundary_surfaces = tag_groups(city, groups)
        set_sizes(city, groups, faces, args.size_factor, args.air_size, args.wall_size)
        set_building_constraints(registry)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        depths = layer_depths(city, args.size_factor)
        num_prisms = add_ground_prisms(city, faces, depths, boundary_surfaces)
        counts, volumes, bary = check_mesh(city, vertices)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(args.output))
    finally:
        gmsh.finalize()

    exact = expected_volumes(city)
    description = {
        "layout": str(args.layout),
        "layout_sha256": sha256(args.layout),
        "mesh": str(args.output),
        "mesh_sha256": sha256(args.output),
        "size_factor": args.size_factor,
        "air_size_m": args.air_size,
        "wall_size_m": args.wall_size,
        "wall_first_layer_m": args.size_factor * WALL_FIRST_LAYER,
        "wall_skin_depth_m": WALL_SKIN_DEPTH,
        "gmsh_version": gmsh.__version__,
        "source_cm": (city.source * CM).tolist(),
        "source_tetrahedron_edge_cm": args.size_factor * SOURCE_SIZE * CM,
        "ground_layer_depths_cm": depths,
        "activity_Bq": city.activity,
        "photons_per_decay": citylib.CS137_GAMMA_YIELD,
        "dose_map_cm": {"x_edges": (city.dose_map_boxes()[0] * CM).tolist(),
                        "y_edges": (city.dose_map_boxes()[1] * CM).tolist(),
                        "z_range": [z * CM for z in city.dose_map_boxes()[2]]},
        "blocks": {str(block): {"name": city.names[block], "material": material,
                                "cells": counts[block],
                                "mesh_volume_cm3": volumes[block],
                                "exact_volume_cm3": exact[block]}
                   for block, material in city.block_materials().items()},
    }
    description_path = args.output.with_name(args.output.stem + "_mesh.json")
    description_path.write_text(json.dumps(description, indent=2) + "\n")
    for block, material in city.block_materials().items():
        print(f"Block {block:4d} {city.names[block]:18s} {counts[block]:8d} cells  "
              f"volume error {volumes[block] / exact[block] - 1:+.2e}")
    print(f"Source barycentric coordinates in its cell: {np.array2string(bary, precision=4)}")
    print(f"Wrote {args.output} with {sum(counts.values())} cells ({num_prisms} prisms in "
          f"{len(depths) - 1} ground layers), and {description_path}")


if __name__ == "__main__":
    main()
