#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Write the coil mesh's material interfaces as a DAGMC geometry for OpenMC.

The surfaces are the faces of the OpenSn tetrahedra that separate two
materials, so OpenMC tracks particles through the same faceted geometry that
OpenSn meshes. DAGMC volume 1 is the copper core and volume 2 the
insulation. The air is DAGMC's implicit complement. ``openmc_reference.py``
bounds the implicit complement with the air box.

Requires pymoab and meshio. Usage::

    python3 make_dagmc.py [--mesh coil.msh] [--output coil.h5m]
"""

import argparse
from collections import defaultdict

import meshio
import numpy as np
from pymoab import core, types

INSULATION, COPPER, AIR = 1, 2, 3
VOLUMES = {COPPER: (1, "copper"), INSULATION: (2, "polyethylene")}


def read_tetrahedra(path):
    mesh = meshio.read(path)
    cells, blocks = [], []
    for block, data in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if block.type == "tetra":
            cells.append(block.data)
            blocks.append(data)
    return mesh.points, np.concatenate(cells), np.concatenate(blocks)


def interface_faces(points, tets, blocks):
    """Return the oriented faces between materials, keyed by (inside, outside)."""
    owners = defaultdict(list)
    for tet, block in zip(tets, blocks):
        for opposite in range(4):
            face = [tet[i] for i in range(4) if i != opposite]
            a, b, c = (points[v] for v in face)
            # Orient the face so that its normal points away from the tet.
            if np.dot(np.cross(b - a, c - a), points[tet[opposite]] - a) > 0.0:
                face = [face[0], face[2], face[1]]
            owners[frozenset(face)].append((block, face))

    surfaces = defaultdict(list)
    for sides in owners.values():
        if len(sides) == 1:
            if sides[0][0] != AIR:
                raise RuntimeError("A wire face lies on the outer boundary")
            continue
        (block_a, face_a), (block_b, _) = sides
        if block_a == block_b:
            continue
        if block_a == AIR or (block_a == INSULATION and block_b == COPPER):
            block_a, block_b = block_b, block_a
            face_a = sides[1][1]
        surfaces[(block_a, block_b)].append(face_a)
    return surfaces


def write_dagmc(points, surfaces, output):
    mb = core.Core()
    category = mb.tag_get_handle(types.CATEGORY_TAG_NAME, types.CATEGORY_TAG_SIZE,
                                 types.MB_TYPE_OPAQUE, types.MB_TAG_SPARSE,
                                 create_if_missing=True)
    name = mb.tag_get_handle(types.NAME_TAG_NAME, types.NAME_TAG_SIZE, types.MB_TYPE_OPAQUE,
                             types.MB_TAG_SPARSE, create_if_missing=True)
    dimension = mb.tag_get_handle(types.GEOM_DIMENSION_TAG_NAME, 1, types.MB_TYPE_INTEGER,
                                  types.MB_TAG_SPARSE, create_if_missing=True)
    global_id = mb.tag_get_handle(types.GLOBAL_ID_TAG_NAME, 1, types.MB_TYPE_INTEGER,
                                  types.MB_TAG_DENSE, create_if_missing=True)
    sense = mb.tag_get_handle("GEOM_SENSE_2", 2, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE,
                              create_if_missing=True)

    def entity_set(dim, category_name, gid):
        handle = mb.create_meshset()
        mb.tag_set_data(dimension, handle, dim)
        mb.tag_set_data(category, handle, category_name)
        mb.tag_set_data(global_id, handle, gid)
        return handle

    used = sorted({v for faces in surfaces.values() for face in faces for v in face})
    vertex_of = dict(zip(used, mb.create_vertices(points[used].flatten())))

    def group(group_name, gid, volume):
        handle = entity_set(4, "Group", gid)
        mb.tag_set_data(name, handle, group_name)
        mb.add_entity(handle, volume)

    volume_sets = {}
    for block, (gid, material) in VOLUMES.items():
        volume_sets[block] = entity_set(3, "Volume", gid)
        group(f"mat:{material}", gid, volume_sets[block])
    # DAGMC reads the implicit-complement material from a "_comp" group, which
    # must contain a volume to be parsed.
    group("mat:air_comp", len(VOLUMES) + 1, volume_sets[COPPER])

    for gid, ((inside, outside), faces) in enumerate(sorted(surfaces.items()), start=1):
        surface = entity_set(2, "Surface", gid)
        connectivity = np.array([[vertex_of[v] for v in face] for face in faces], dtype="uint64")
        triangles = mb.create_elements(types.MBTRI, connectivity)
        mb.add_entities(surface, triangles)
        mb.add_entities(surface, [vertex_of[v] for v in sorted({v for f in faces for v in f})])
        # The triangle normals point out of the forward volume.
        reverse = volume_sets.get(outside, 0)
        mb.tag_set_data(sense, surface,
                        np.array([volume_sets[inside], reverse], dtype="uint64"))
        mb.add_parent_child(volume_sets[inside], surface)
        if reverse:
            mb.add_parent_child(reverse, surface)
        print(f"surface {gid}: block {inside} | block {outside}, {len(faces)} triangles")

    mb.write_file(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", default="coil.msh")
    parser.add_argument("--output", default="coil.h5m")
    args = parser.parse_args()
    points, tets, blocks = read_tetrahedra(args.mesh)
    write_dagmc(points, interface_faces(points, tets, blocks), args.output)


if __name__ == "__main__":
    main()
