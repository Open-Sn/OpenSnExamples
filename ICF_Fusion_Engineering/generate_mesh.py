#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate hohlraum.msh, the mesh used by hohlraum.py.

Gmsh builds the indirect-drive target from OpenCASCADE solids and meshes it with linear
tetrahedra. The target is a gold hohlraum with a laser entrance hole (LEH) in each end,
filled with helium, around a spherical capsule. An aluminum thermal-mechanical package
(TMP) surrounds the hohlraum wall and has a flange at each end. The space between the two
flanges, outside the TMP, is meshed as void so that the outer boundary of the mesh is a
convex cylinder. Neutrons that leave the TMP there can still enter a flange, as they can in
the OpenMC model.

The point source sits at the center of the capsule. Four points of a regular tetrahedron
centered on the origin are embedded in the capsule mesh, so the source lies well inside one
cell rather than on or near a face, edge, or vertex. The script checks this and stops if
Gmsh did not keep that tetrahedron.

The default size factor of 0.7 gives the mesh used by hohlraum.py, with 989,832
tetrahedra. Factors 2 and 1 give the coarse and medium meshes of the sensitivity studies in
the README, with 99,633 and 381,502 tetrahedra.

This script needs the gmsh and numpy Python packages. The supplied mesh was made with
Gmsh 4.15.2. Another Gmsh version can give a different mesh.

    python3 generate_mesh.py --output hohlraum.msh
"""

import argparse
import math
from pathlib import Path

import gmsh
import numpy as np

# Geometry in cm. openmc_reference.py describes the same target with analytic surfaces.
CAPSULE_RADIUS = 0.01
GAS_RADIUS, GAS_HALF_HEIGHT = 0.2, 0.3507
HOHLRAUM_RADIUS, HOHLRAUM_HALF_HEIGHT = 0.205, 0.3557
LEH_RADIUS = 0.12
TMP_RADIUS = 0.215
FLANGE_RADIUS, FLANGE_HEIGHT = 0.255, 0.02

# Block IDs and names.
BLOCKS = {1: "capsule", 2: "fill_gas", 3: "hohlraum_wall", 4: "leh_windows", 5: "tmp",
          6: "exterior"}
# OpenSn identifies physical names by their tag alone, so the boundary tag must differ from
# the block IDs.
BOUNDARY_ID, BOUNDARY_NAME = 100, "VACUUM"

# Element sizes in cm. The capsule and its surroundings are resolved finely because the
# uncollided flux varies as 1/r^2 from the source. The walls are 50 and 100 um thick.
CAPSULE_SIZE = 0.002
CAVITY_SIZE = 0.02
WALL_SIZE = 0.01
SIZE_GROWTH = 0.15  # Size increase per unit distance from the capsule

# Orientation of the tetrahedron embedded around the source.
SOURCE_TETRAHEDRON_AXIS, SOURCE_TETRAHEDRON_ANGLE = (0.31, 0.57, 0.76), 0.83


def build_target():
    """Build the solids and return the volume tags of each block."""
    occ = gmsh.model.occ
    z0, height = -HOHLRAUM_HALF_HEIGHT, 2 * HOHLRAUM_HALF_HEIGHT
    gas_z0 = -GAS_HALF_HEIGHT
    solids = {
        "capsule": [occ.addSphere(0, 0, 0, CAPSULE_RADIUS)],
        "leh": [occ.addCylinder(0, 0, GAS_HALF_HEIGHT, 0, 0,
                                HOHLRAUM_HALF_HEIGHT - GAS_HALF_HEIGHT, LEH_RADIUS),
                occ.addCylinder(0, 0, -HOHLRAUM_HALF_HEIGHT, 0, 0,
                                HOHLRAUM_HALF_HEIGHT - GAS_HALF_HEIGHT, LEH_RADIUS)],
        "gas": [occ.addCylinder(0, 0, gas_z0, 0, 0, 2 * GAS_HALF_HEIGHT, GAS_RADIUS)],
        "hohlraum": [occ.addCylinder(0, 0, z0, 0, 0, height, HOHLRAUM_RADIUS)],
        "tmp": [occ.addCylinder(0, 0, z0, 0, 0, height, TMP_RADIUS),
                occ.addCylinder(0, 0, z0, 0, 0, FLANGE_HEIGHT, FLANGE_RADIUS),
                occ.addCylinder(0, 0, HOHLRAUM_HALF_HEIGHT - FLANGE_HEIGHT, 0, 0,
                                FLANGE_HEIGHT, FLANGE_RADIUS)],
        "domain": [occ.addCylinder(0, 0, z0, 0, 0, height, FLANGE_RADIUS)],
    }
    order = ["capsule", "leh", "gas", "hohlraum", "tmp", "domain"]
    inputs = [(3, tag) for name in order for tag in solids[name]]
    _, children = occ.fragment(inputs, [])
    occ.synchronize()

    # A fragment lies inside several of the overlapping solids. The inputs are listed in
    # the order capsule, LEH, gas, hohlraum, TMP, domain, and a fragment belongs to the
    # first of them that contains it.
    owner = {}
    for (dim, tag), pieces in zip(inputs, children):
        name = next(n for n in order if tag in solids[n])
        for piece in pieces:
            owner.setdefault(piece[1], name)
    block_of = {"capsule": 1, "gas": 2, "hohlraum": 3, "leh": 4, "tmp": 5, "domain": 6}
    groups = {block: [] for block in BLOCKS}
    for tag, name in owner.items():
        groups[block_of[name]].append(tag)
    for block, tags in groups.items():
        if not tags:
            raise RuntimeError(f"Block {block} ({BLOCKS[block]}) has no volume")
    return groups


def check_volumes(groups):
    """Compare the OpenCASCADE volume of each block with its analytic volume."""
    gas_length = 2 * GAS_HALF_HEIGHT
    end_thickness = HOHLRAUM_HALF_HEIGHT - GAS_HALF_HEIGHT
    height = 2 * HOHLRAUM_HALF_HEIGHT
    capsule = 4 / 3 * math.pi * CAPSULE_RADIUS**3
    leh = 2 * math.pi * LEH_RADIUS**2 * end_thickness
    expected = {
        1: capsule,
        2: math.pi * GAS_RADIUS**2 * gas_length - capsule,
        3: math.pi * HOHLRAUM_RADIUS**2 * height - math.pi * GAS_RADIUS**2 * gas_length - leh,
        4: leh,
        5: math.pi * (TMP_RADIUS**2 - HOHLRAUM_RADIUS**2) * height
        + 2 * math.pi * (FLANGE_RADIUS**2 - TMP_RADIUS**2) * FLANGE_HEIGHT,
        6: math.pi * (FLANGE_RADIUS**2 - TMP_RADIUS**2) * (height - 2 * FLANGE_HEIGHT),
    }
    for block, tags in groups.items():
        volume = sum(gmsh.model.occ.getMass(3, tag) for tag in tags)
        if abs(volume / expected[block] - 1) > 1e-6:
            raise RuntimeError(f"Block {block} volume {volume} differs from the analytic "
                               f"volume {expected[block]}")


def rotation_matrix(axis, angle):
    """Return the matrix of a rotation by angle (radians) about axis."""
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def embed_source_tetrahedron(capsule_tags, factor):
    """Embed the vertices of a regular tetrahedron centered on the origin."""
    edge = factor * CAPSULE_SIZE
    radius = edge * math.sqrt(3 / 8)  # Circumradius of a regular tetrahedron
    vertices = radius * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / math.sqrt(3)
    # In this orientation the midpoints of the edges lie on the coordinate axes, and Gmsh
    # builds mirror-symmetric cells around the tetrahedron. Rays from their quadrature points
    # to the source then pass exactly through an edge, which OpenSn's ray tracer cannot
    # handle. A rotation about a generic axis removes these symmetries.
    vertices = vertices @ rotation_matrix(SOURCE_TETRAHEDRON_AXIS,
                                          SOURCE_TETRAHEDRON_ANGLE).T
    points = [gmsh.model.occ.addPoint(*v, edge) for v in vertices]
    gmsh.model.occ.synchronize()
    if len(capsule_tags) != 1:
        raise RuntimeError("The capsule must be a single volume")
    gmsh.model.mesh.embed(0, points, 3, capsule_tags[0])
    return vertices


def size_at(x, y, z, factor):
    """Return the element size at a point, with every size scaled by factor."""
    distance = max(math.sqrt(x * x + y * y + z * z) - CAPSULE_RADIUS, 0.0)
    size = min(CAPSULE_SIZE + SIZE_GROWTH * distance / factor, CAVITY_SIZE)
    # The hohlraum wall, its ends, the LEH windows, the TMP sleeve, and the flanges.
    r, height = math.hypot(x, y), abs(z)
    side = GAS_RADIUS - WALL_SIZE < r < TMP_RADIUS + WALL_SIZE
    ends = height > GAS_HALF_HEIGHT - WALL_SIZE and r < TMP_RADIUS + WALL_SIZE
    flanges = height > HOHLRAUM_HALF_HEIGHT - FLANGE_HEIGHT - WALL_SIZE
    if side or ends or flanges:
        size = min(size, WALL_SIZE)
    return factor * size


def tag_groups(groups):
    for block, tags in groups.items():
        gmsh.model.addPhysicalGroup(3, sorted(tags), block, BLOCKS[block])
    outer = gmsh.model.getBoundary([(3, t) for tags in groups.values() for t in tags],
                                   combined=True, oriented=False)
    gmsh.model.addPhysicalGroup(2, sorted(abs(tag) for _, tag in outer), BOUNDARY_ID,
                                BOUNDARY_NAME)


def check_mesh(vertices):
    """Check orientation, the block volumes, and the cell that contains the source."""
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    points = np.zeros((int(node_tags.max()) + 1, 3))
    points[node_tags.astype(int)] = coordinates.reshape(-1, 3)
    counts, volumes = {}, {}
    source_cell = None
    for block in BLOCKS:
        cells = []
        for tag in gmsh.model.getEntitiesForPhysicalGroup(3, block):
            types, _, nodes = gmsh.model.mesh.getElements(3, tag)
            if list(types) != [4]:
                raise RuntimeError("The mesh must contain only linear tetrahedra")
            cells.append(nodes[0].astype(int).reshape(-1, 4))
        cells = np.vstack(cells)
        corners = points[cells]
        edges = np.stack([corners[:, i] - corners[:, 0] for i in (1, 2, 3)], axis=2)
        signed = np.linalg.det(edges) / 6
        if np.any(signed <= 0):
            raise RuntimeError(f"Gmsh produced {np.count_nonzero(signed <= 0)} inverted "
                               f"tetrahedra in block {block}")
        counts[block], volumes[block] = len(cells), signed.sum()
        if block == 1:
            # Barycentric coordinates of the origin in every capsule cell.
            weights = np.linalg.solve(edges, -corners[:, 0, :, None])[:, :, 0]
            bary = np.column_stack((1 - weights.sum(axis=1), weights))
            inside = np.where(np.all(bary > 0, axis=1))[0]
            if len(inside) != 1:
                raise RuntimeError("The source does not lie strictly inside one cell")
            source_cell = (corners[inside[0]], bary[inside[0]])
    corners, bary = source_cell
    if not np.allclose(np.sort(corners, axis=0), np.sort(vertices, axis=0), atol=1e-12):
        raise RuntimeError("Gmsh did not keep the embedded source tetrahedron")
    return counts, volumes, bary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("hohlraum.msh"))
    parser.add_argument("--size-factor", type=float, default=0.7,
                        help="Factor applied to every element size, for refinement studies")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists. Remove it or choose another name.")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("hohlraum")
        groups = build_target()
        check_volumes(groups)
        vertices = embed_source_tetrahedron(groups[1], args.size_factor)
        tag_groups(groups)
        gmsh.model.mesh.setSizeCallback(
            lambda dim, tag, x, y, z, lc: size_at(x, y, z, args.size_factor))
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.mesh.generate(3)
        counts, volumes, bary = check_mesh(vertices)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(args.output))
    finally:
        gmsh.finalize()
    for block, name in BLOCKS.items():
        print(f"Block {block} {name:14s} {counts[block]:8d} tetrahedra  "
              f"volume {volumes[block]:.6e} cm^3")
    print(f"Source barycentric coordinates in its cell: {np.array2string(bary, precision=4)}")
    print(f"Wrote {args.output} with {sum(counts.values())} tetrahedra")


if __name__ == "__main__":
    main()
