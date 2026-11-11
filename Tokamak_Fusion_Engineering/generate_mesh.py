#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate tokamak.msh, the mesh used by tokamak.py.

Gmsh builds the upper half (z >= 0) of the 90 degree quarter tokamak from OpenCASCADE
solids and meshes it with linear tetrahedra. The lower half is the mirror image of the
upper half. Both halves share the nodes on z = 0, so the midplane is an interior surface
of the final mesh. The first wall, blanket, vessel, and shield are split into thin layers
so the mesh resolves the steep flux gradients through them.

This script needs the gmsh, meshio, and numpy Python packages. The supplied mesh was made
with Gmsh 4.15.2 and takes several minutes to generate. Another Gmsh version can give a
different mesh.

    python3 generate_mesh.py --output tokamak.msh
"""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import tempfile

import gmsh
import meshio
import numpy as np

# Geometry in cm. openmc_reference.py describes the same reactor with analytic surfaces.
# Each D-shaped envelope is (R - R0)^2 + (z / ELONGATION)^2 < a^2 with R > R_cut. The
# successive envelopes, listed as (a, R_cut), bound blocks 1 to 6.
R0, ELONGATION = 620.0, 1.55
ENVELOPES = [(160.0, 490.0), (190.0, 460.0), (202.0, 448.0),
             (249.0, 401.0), (266.0, 384.0), (294.0, 356.0)]
TOROIDAL_COILS, COIL_WIDTH = 18, 0.04
COIL_OUTER, COIL_INNER = (359.0, 291.0), (324.0, 326.0)
UPPER_POLOIDAL_COILS, POLOIDAL_MINOR_RADIUS = [(600.0, 610.0), (1040.0, 330.0)], 22.0
DOMAIN_R, DOMAIN_Z = 1150.0, 690.0
QUARTER = math.pi / 2

# Maximum distance between layer surfaces in the first wall (3), blanket (4), vessel (5),
# and shield (6), and the maximum element size in each block.
LAYER_WIDTHS = {3: 1.0, 4: 2.0, 5: 2.0, 6: 2.0}
MAX_SIZE = 90.0
BLOCK_SIZES = {1: 60.0, 2: 50.0, 3: 35.0, 4: 40.0, 5: 35.0, 6: 40.0,
               7: 25.0, 8: 12.0, 9: 30.0, 10: MAX_SIZE}

# Boundary IDs and names of the upper half. The midplane (102) becomes interior and the
# mirrored top becomes VACUUM_BOTTOM (105).
BOUNDARIES = {100: "SYM_LOWER", 101: "SYM_UPPER", 102: "SYM_MIDPLANE",
              103: "VACUUM_OUTER", 104: "VACUUM_TOP"}


def revolved(envelope, angle=QUARTER, start=0.0):
    """Revolve the upper half of a D-shaped envelope (a, R_cut) about the z axis."""
    a, cut = envelope
    occ = gmsh.model.occ
    face = [(2, occ.addDisk(R0, 0, 0, ELONGATION * a, a))]
    occ.rotate(face, R0, 0, 0, 0, 0, 1, math.pi / 2)
    left = (2, occ.addRectangle(-2000.0, -2000.0, 0.0, cut + 2000.0, 4000.0))
    below = (2, occ.addRectangle(-2000.0, -2000.0, 0.0, 4000.0, 2000.0))
    face, _ = occ.cut(face, [left, below])
    occ.rotate(face, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, math.pi / 2)
    occ.rotate(face, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, start)
    return [e for e in occ.revolve(face, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, angle) if e[0] == 3]


def cylinder(radius, height):
    """Return a quarter cylinder from z = 0 to z = height."""
    tag = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius, angle=QUARTER)
    gmsh.model.occ.rotate([(3, tag)], 0, 0, 0, 0, 0, 1, 0.0)
    return (3, tag)


def coil_wedges():
    """Return the angular intervals of the toroidal coils inside the quarter."""
    intervals = []
    for index in range(TOROIDAL_COILS):
        center = 2 * math.pi * index / TOROIDAL_COILS
        low, high = max(0.0, center - COIL_WIDTH / 2), min(QUARTER, center + COIL_WIDTH / 2)
        if high > low:
            intervals.append((low, high))
    return intervals


def boundary_id(points, tolerance=1.0e-5):
    """Return the ID of the outer boundary that contains all of the points."""
    x, y, z = np.asarray(points).T
    if np.max(abs(z - DOMAIN_Z)) < tolerance:
        return 104
    if np.max(abs(z)) < tolerance:
        return 102
    if np.max(abs(y)) < tolerance:
        return 100
    if np.max(abs(x)) < tolerance:
        return 101
    if np.max(abs(np.hypot(x, y) - DOMAIN_R)) < tolerance:
        return 103
    raise ValueError("A boundary surface is not on an outer boundary of the quarter")


def build_upper_half():
    """Build the CAD model of the upper half and return the volume tags of each block."""
    occ = gmsh.model.occ
    envelopes = [revolved(e) for e in ENVELOPES]
    bodies = {1: envelopes[0]}
    for i in range(5, 0, -1):
        bodies[i + 1], _ = occ.cut(envelopes[i], envelopes[i - 1], removeTool=False)
    bodies[7] = []
    for low, high in coil_wedges():
        coil, _ = occ.cut(revolved(COIL_OUTER, high - low, low),
                          revolved(COIL_INNER, high - low, low))
        bodies[7].extend(coil)
    domain = cylinder(DOMAIN_R, DOMAIN_Z)
    rings = [(3, occ.addTorus(0, 0, z, r, POLOIDAL_MINOR_RADIUS))
             for r, z in UPPER_POLOIDAL_COILS]
    bodies[8], _ = occ.intersect(rings, [domain], removeTool=False)
    bodies[9], _ = occ.cut([cylinder(250.0, 450.0)], [cylinder(165.0, 450.0)])

    # Split each shell into layers bounded by intermediate envelopes.
    for block in range(3, 7):
        inner, outer = ENVELOPES[block - 2], ENVELOPES[block - 1]
        count = math.ceil((outer[0] - inner[0]) / LAYER_WIDTHS[block])
        tools = []
        for j in range(1, count):
            tools.extend(revolved(tuple(a + j / count * (b - a) for a, b in zip(inner, outer))))
        if not tools:
            continue
        result, mapping = occ.fragment(bodies[block], tools)
        retained = {tag for parts in mapping[:len(bodies[block])] for dim, tag in parts
                    if dim == 3}
        occ.remove([(dim, tag) for dim, tag in result if dim == 3 and tag not in retained],
                   recursive=True)
        bodies[block] = [(3, tag) for tag in sorted(retained)]

    # Fragment everything against the domain so all blocks share conforming faces. The
    # rest of the domain is the exterior, block 10.
    objects = [v for block in sorted(bodies) for v in bodies[block]]
    owners = [block for block in sorted(bodies) for _ in bodies[block]]
    _, mapping = occ.fragment(objects, [domain])
    occ.synchronize()
    groups = {block: set() for block in range(1, 11)}
    for block, parts in zip(owners, mapping[:len(objects)]):
        groups[block].update(tag for dim, tag in parts if dim == 3)
    assigned = set().union(*groups.values())
    groups[10] = {tag for dim, tag in mapping[len(objects)] if dim == 3} - assigned

    # Remove construction surfaces, curves, and points that bound no volume.
    for dimension in (2, 1, 0):
        used = {abs(tag) for dim, tag in gmsh.model.getBoundary(
            gmsh.model.getEntities(dimension + 1), combined=False, oriented=True)
            if dim == dimension}
        unused = [(dim, tag) for dim, tag in gmsh.model.getEntities(dimension)
                  if tag not in used]
        occ.remove(unused, recursive=False)
        occ.synchronize()
    return groups


def tag_boundaries(groups):
    """Add a physical group for each outer boundary of the upper half."""
    faces = defaultdict(list)
    for block, tags in groups.items():
        for tag in tags:
            for dim, signed in gmsh.model.getBoundary([(3, tag)], oriented=True):
                faces[abs(signed)].append(signed)
    patches = defaultdict(list)
    for face, signs in faces.items():
        if len(signs) == 1:
            lo, hi = gmsh.model.getParametrizationBounds(2, face)
            uv = [(u, v) for u in np.linspace(lo[0], hi[0], 4)
                  for v in np.linspace(lo[1], hi[1], 4)]
            points = np.array(gmsh.model.getValue(2, face, np.ravel(uv))).reshape(-1, 3)
            patches[boundary_id(points)].append(face)
    for boundary, surfaces in patches.items():
        gmsh.model.addPhysicalGroup(2, surfaces, boundary, BOUNDARIES[boundary])


def mesh_upper_half(groups, filename):
    """Mesh the upper half with the element size limit of each block."""
    for block, tags in groups.items():
        gmsh.model.addPhysicalGroup(3, sorted(tags), block, f"block_{block}")
    # An entity shared by several blocks takes the smallest of their size limits.
    entity_sizes = {}
    for block, tags in groups.items():
        entities = [(3, t) for t in tags]
        while entities:
            for entity in entities:
                entity_sizes[entity] = min(entity_sizes.get(entity, MAX_SIZE),
                                           BLOCK_SIZES[block])
            if entities[0][0] == 0:
                break
            entities = list(set(gmsh.model.getBoundary(entities, combined=False,
                                                       oriented=False)))
    gmsh.model.mesh.setSizeCallback(
        lambda dim, tag, x, y, z, lc: min(lc, entity_sizes.get((dim, tag), MAX_SIZE)))
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", -2)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MAX_SIZE)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 32)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("General.NumThreads", 2)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(filename))


def mirror(upper, output):
    """Reflect the upper half through z = 0 and write the full-height mesh."""
    mesh = meshio.read(upper)
    tetrahedra = mesh.get_cells_type("tetra")
    triangles = mesh.get_cells_type("triangle")
    triangle_ids = mesh.get_cell_data("gmsh:physical", "triangle")
    blocks = mesh.get_cell_data("gmsh:physical", "tetra")
    entities = mesh.get_cell_data("gmsh:geometrical", "tetra")

    # Gmsh can return inverted tetrahedra on this geometry. Refuse such a mesh.
    points = np.asarray(mesh.points, dtype=float).copy()
    corners = points[tetrahedra]
    volumes = np.linalg.det(np.stack([corners[:, i] - corners[:, 0] for i in (1, 2, 3)],
                                     axis=2)) / 6
    if np.any(volumes <= 0):
        raise ValueError(f"Gmsh produced {np.count_nonzero(volumes <= 0)} inverted "
                         "tetrahedra. Run the script again or change the mesh settings.")

    # Nodes on the midplane are shared. Every other node gets a mirrored copy.
    on_midplane = np.zeros(len(points), dtype=bool)
    on_midplane[np.unique(triangles[triangle_ids == 102])] = True
    points[on_midplane, 2] = 0.0
    image = np.arange(len(points))
    image[~on_midplane] = len(points) + np.arange(np.count_nonzero(~on_midplane))
    all_points = np.vstack((points, points[~on_midplane] * [1.0, 1.0, -1.0]))
    # Swapping two vertices keeps the mirrored tetrahedra and triangles positively oriented.
    all_tetrahedra = np.vstack((tetrahedra, image[tetrahedra[:, [0, 2, 1, 3]]]))
    outer = triangles[triangle_ids != 102]
    outer_ids = triangle_ids[triangle_ids != 102]
    all_triangles = np.vstack((outer, image[outer[:, [0, 2, 1]]]))
    all_triangle_ids = np.r_[outer_ids, np.where(outer_ids == 104, 105, outer_ids)]

    names = {name: value for name, value in mesh.field_data.items()
             if not (value[1] == 2 and value[0] == 102)}
    names["VACUUM_BOTTOM"] = [105, 2]
    all_entities = np.r_[entities, entities + int(entities.max()) + 1]
    full = meshio.Mesh(
        all_points, [("triangle", all_triangles), ("tetra", all_tetrahedra)],
        cell_data={"gmsh:physical": [all_triangle_ids, np.tile(blocks, 2)],
                   "gmsh:geometrical": [all_triangle_ids, all_entities]},
        field_data=names)
    meshio.write(output, full, file_format="gmsh22", binary=False)
    return len(all_tetrahedra)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("tokamak.msh"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists. Remove it or choose another name.")
    with tempfile.TemporaryDirectory() as scratch:
        upper = Path(scratch) / "upper.msh"
        gmsh.initialize()
        try:
            gmsh.model.add("tokamak_upper_half")
            groups = build_upper_half()
            tag_boundaries(groups)
            mesh_upper_half(groups, upper)
        finally:
            gmsh.finalize()
        cells = mirror(upper, args.output)
    print(f"Wrote {args.output} with {cells} tetrahedra")


if __name__ == "__main__":
    main()
