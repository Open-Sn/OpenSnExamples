#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Mesh one octant of the target chamber with Gmsh, for chamber.py.

The octant x, y, z > 0 of the geometry of chamber_geometry.py is meshed with tetrahedra.
Blocks (physical volumes): 1 cavity and ports, 2 wall, 3 shield, 4 air. Boundaries (physical
surfaces): XMIN, YMIN and ZMIN, the three coordinate planes, which are reflecting in
chamber.py, and OUTER, the sphere of radius 800 cm, which is vacuum. The source is at the
origin, a vertex of the mesh.

--size H is the cell size (cm) in the wall, the shield and the ports. In the air the size
grows linearly from H at r = 570 cm to 3 H at the outer boundary. In the cavity it grows
from H at r = 480 cm towards the centre, up to 4 H, but stays below 0.15 r so that the flux,
which varies as 1 / r^2, changes by a similar fraction across every cell; it is not smaller
than H / 4.

--port-size P refines the mesh along the edge of the beam of each port, where the uncollided
flux falls by a factor of several hundred across a few centimetres. A ray from the source at
the angle g to the axis of a port of radius rho leaves the port at the radius rho / sin(g):
it stays in the port through the wall and the shield when sin(g) < rho / 550, and crosses all
of both when sin(g) > rho / 500. Beyond the shield the edge of the beam is therefore near the
cone at the distance rho t / 525 from the axis, t being the coordinate along the axis. The
cell size is P within 3 cm of that cone, and of the cylinder of the port for t < 525 cm, and
grows by 0.3 cm per cm away from it, for t > 470 cm.

The faces of the mesh are flat, so the mesh approximates the spheres and cylinders. The
script compares the volume of each block with the exact one, checks that no tetrahedron is
inverted and that the origin is a vertex of block 1, puts the nodes of the
symmetry planes exactly on them (OpenCASCADE leaves some about 1e-11 cm off, and OpenSn
requires a reflecting boundary to be planar to 1e-12), and writes the mesh in Gmsh 4.1 binary
format. Gmsh runs on one thread, so the same command gives the same mesh. For example:

    python3 generate_mesh.py --size 20 --out chamber_coarse.msh
    python3 generate_mesh.py --size 10 --port-size 5 --out chamber_fine.msh
"""

import argparse
import json
import time

import gmsh
import numpy as np

import chamber_geometry as geometry

BOUNDARIES = {"XMIN": 101, "YMIN": 102, "ZMIN": 103, "OUTER": 104}


def positive(a):
    return f"(({a})+Fabs({a}))/2"


def smaller(a, b):
    return f"((({a})+({b})-Fabs(({a})-({b})))/2)"


def larger(a, b):
    return f"((({a})+({b})+Fabs(({a})-({b})))/2)"


def build_geometry():
    """Build the octant with OpenCASCADE and return the volume tags of each block."""
    occ = gmsh.model.occ
    extent = geometry.R_AIR + 10.0

    def octant(shape):
        box = occ.addBox(0.0, 0.0, 0.0, extent, extent, extent)
        return occ.intersect([shape], [(3, box)])[0]

    spheres = []
    for radius in (geometry.R_CAVITY, geometry.R_WALL, geometry.R_SHIELD, geometry.R_AIR):
        spheres += octant((3, occ.addSphere(0.0, 0.0, 0.0, radius)))
    # The port cylinders extend 20 cm beyond the wall and the shield on both sides; the
    # pieces outside them are cavity or air.
    inner, outer = geometry.R_CAVITY - 20.0, geometry.R_SHIELD + 20.0
    cylinders = []
    for axis, radius, _ in geometry.octant_ports():
        cylinders += octant((3, occ.addCylinder(*(inner * axis), *((outer - inner) * axis),
                                                radius)))
    occ.fragment(spheres, cylinders)
    occ.synchronize()

    blocks = {1: [], 2: [], 3: [], 4: []}
    for dim, tag in gmsh.model.getEntities(3):
        box = gmsh.model.getBoundingBox(dim, tag)
        largest = max(abs(value) for value in box)
        centre = np.linalg.norm(occ.getCenterOfMass(dim, tag))
        if box[3] - box[0] > 450.0:  # one of the four large pieces
            radii = {geometry.R_CAVITY: 1, geometry.R_WALL: 2, geometry.R_SHIELD: 3,
                     geometry.R_AIR: 4}
            block = next(b for r, b in radii.items() if abs(largest - r) < 1.0)
        else:  # a piece of a port cylinder
            block = 1 if centre < geometry.R_SHIELD else 4
        blocks[block].append(tag)
    for block, tags in blocks.items():
        gmsh.model.addPhysicalGroup(3, tags, block, geometry.BLOCKS[block])

    surfaces = {name: [] for name in BOUNDARIES}
    for dim, tag in gmsh.model.getEntities(2):
        box = np.array(gmsh.model.getBoundingBox(dim, tag))
        low, high = box[:3], box[3:]
        for axis, name in enumerate(("XMIN", "YMIN", "ZMIN")):
            if high[axis] - low[axis] < 1.0e-6 and abs(high[axis]) < 1.0e-6:
                surfaces[name].append(tag)
        planar = np.any(high - low < 1.0e-6)
        if not planar and abs(high.max() - geometry.R_AIR) < 1.0e-3:
            surfaces["OUTER"].append(tag)
    for name, tags in surfaces.items():
        if not tags:
            raise RuntimeError(f"no surface found for boundary {name}")
        gmsh.model.addPhysicalGroup(2, tags, BOUNDARIES[name], name)
    return blocks, surfaces


def snap_to_planes(surfaces):
    """Put the nodes of the symmetry planes exactly on them.

    OpenCASCADE places some of them up to about 1e-11 cm off the plane, and OpenSn requires a
    reflecting boundary to be planar to 1e-12. Return the largest displacement.
    """
    largest = 0.0
    for axis, name in enumerate(("XMIN", "YMIN", "ZMIN")):
        for tag in surfaces[name]:
            node_tags, coordinates, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
            coordinates = coordinates.reshape(-1, 3)
            for node, point in zip(node_tags, coordinates):
                if point[axis] != 0.0:
                    largest = max(largest, abs(point[axis]))
                    point = point.copy()
                    point[axis] = 0.0
                    gmsh.model.mesh.setNode(int(node), point.tolist(), [])
    return largest


def size_field(size, port_size):
    """Return the Gmsh MathEval expression of the cell size."""
    r = "Sqrt(x*x+y*y+z*z)"
    inward = positive(f"{geometry.R_CAVITY - 20.0}-{r}") + "/100"
    outward = positive(f"{r}-{geometry.R_SHIELD + 20.0}") + "/230"
    near_wall = f"{size}*(1+{inward}+2*{outward})"
    near_source = larger(f"0.15*{r}", f"{size / 4.0}")
    expression = smaller(near_wall, near_source)
    if port_size:
        middle = 0.5 * (geometry.R_CAVITY + geometry.R_SHIELD)
        for axis, radius, _ in geometry.octant_ports():
            a = [float(value) for value in axis]
            t = f"({a[0]!r}*x+{a[1]!r}*y+{a[2]!r}*z)"
            from_axis = f"Sqrt({positive(f'x*x+y*y+z*z-{t}*{t}')})"
            edge = f"{float(radius) / middle!r}*{larger(t, repr(middle))}"
            away = positive(f"Fabs({from_axis}-{edge})-3") + "+" + positive(f"470-{t}")
            expression = smaller(expression, f"{port_size}+0.3*({away})")
    return expression


def check_mesh(blocks):
    """Return the statistics of the mesh; raise if a check fails."""
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    nodes = np.zeros((int(node_tags.max()) + 1, 3))
    nodes[node_tags.astype(int)] = coordinates.reshape(-1, 3)
    exact = geometry.octant_volumes()
    stats = {"blocks": {}}
    origin_in_block_1 = False
    smallest = np.inf
    for block, tags in blocks.items():
        count, volume, largest_edge = 0, 0.0, 0.0
        for tag in tags:
            _, element_nodes = gmsh.model.mesh.getElementsByType(4, tag)
            tets = element_nodes.astype(int).reshape(-1, 4)
            p = nodes[tets]
            signed = np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                               np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6.0
            smallest = min(smallest, float(signed.min()))
            count += len(tets)
            volume += float(signed.sum())
            edges = [np.linalg.norm(p[:, i] - p[:, j], axis=1)
                     for i in range(4) for j in range(i + 1, 4)]
            largest_edge = max(largest_edge, float(np.max(edges)))
            if block == 1:
                origin_in_block_1 |= bool(np.any(np.linalg.norm(p, axis=2) == 0.0))
        stats["blocks"][geometry.BLOCKS[block]] = {
            "tetrahedra": count, "volume_cm3": volume, "exact_volume_cm3": exact[block],
            "relative_volume_error": volume / exact[block] - 1.0,
            "largest_edge_cm": largest_edge}
    stats["tetrahedra"] = sum(b["tetrahedra"] for b in stats["blocks"].values())
    stats["nodes"] = int(len(node_tags))
    stats["smallest_signed_volume_cm3"] = smallest
    if smallest <= 0.0:
        raise RuntimeError(f"inverted or flat tetrahedron: signed volume {smallest}")
    if not origin_in_block_1:
        raise RuntimeError("the origin is not a vertex of block 1")
    for name, block in stats["blocks"].items():
        # The faceted spheres and cylinders are slightly smaller than the true ones.
        if abs(block["relative_volume_error"]) > 0.02:
            raise RuntimeError(f"volume of {name} differs from the exact one by "
                               f"{block['relative_volume_error']:.2%}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=float, required=True,
                        help="Cell size (cm) in the wall, the shield and the ports")
    parser.add_argument("--port-size", type=float, default=None,
                        help="Cell size (cm) along the edges of the port beams")
    parser.add_argument("--out", default=None, help="Output file (default chamber_h<H>.msh)")
    args = parser.parse_args()
    name = f"chamber_h{args.size:g}" + (f"p{args.port_size:g}" if args.port_size else "")
    out = args.out or name + ".msh"

    start = time.perf_counter()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("target_chamber")
    blocks, surfaces = build_geometry()

    field = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field, "F", size_field(args.size, args.port_size))
    gmsh.model.mesh.field.setAsBackgroundMesh(field)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 4.0 * args.size)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT Delaunay
    gmsh.option.setNumber("General.NumThreads", 1)  # one thread: the same mesh every time
    gmsh.model.mesh.generate(3)
    snapped = snap_to_planes(surfaces)
    stats = check_mesh(blocks)
    stats["largest_snap_to_plane_cm"] = snapped

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(out)
    stats["gmsh_version"] = gmsh.option.getString("General.Version")
    gmsh.finalize()
    stats["size_cm"], stats["port_size_cm"] = args.size, args.port_size
    stats["seconds"] = time.perf_counter() - start
    with open(out.rsplit(".", 1)[0] + "_stats.json", "w") as stream:
        json.dump(stats, stream, indent=2)

    print(f"Wrote {out}: {stats['tetrahedra']} tetrahedra, {stats['nodes']} nodes, "
          f"Gmsh {stats['gmsh_version']}, {stats['seconds']:.0f} s")
    for name, block in stats["blocks"].items():
        print(f"  {name:17s} {block['tetrahedra']:9d} tetrahedra, volume "
              f"{block['volume_cm3']:.6e} cm^3 ({block['relative_volume_error']:+.3%} from "
              f"exact), largest edge {block['largest_edge_cm']:.1f} cm")


if __name__ == "__main__":
    main()
