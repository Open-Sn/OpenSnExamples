#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Layout of the city in the urban source example.

This module reads city.toml, checks it, and describes the city for the other scripts of
the example: generate_mesh.py meshes it, openmc_reference.py builds it from analytic
surfaces, and uncollided_reference.py traces rays through it. The city is a rectilinear
lattice of cells. Each cell holds a building on a lot, a park, a plaza, or a road. The air
fills the space above the ground up to the top of the domain, and the ground slab lies
below it.

Lengths in city.toml are in meters. This module returns them in meters too; the scripts
convert them to centimeters, the length unit of OpenSn and OpenMC.

Block IDs of the mesh, which are also the cell IDs of the OpenMC model:
    1           the air
    2, 3, ...   the ground, one block for each ground material, in the order in which the
                materials first appear in the lattice, from north to south and west to east
    101, 102,   the buildings, one block each, in the same order
"""

import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AIR_BLOCK = 1
FIRST_BUILDING_BLOCK = 101
# A footprint must stay this far inside its lattice cell [m].
LOT_CLEARANCE = 1.0
# The source must be this far from every surface [m].
SOURCE_CLEARANCE = 0.5
# Gamma rays of 661.657 keV emitted per decay of Cs-137, through Ba-137m.
CS137_GAMMA_ENERGY = 661.657e3  # eV
CS137_GAMMA_YIELD = 0.8510


class LayoutError(ValueError):
    """A layout file that does not describe a valid city."""


@dataclass
class Footprint:
    """The footprint of a building in the x-y plane, in meters.

    kind is "polygon", with vertices listed counterclockwise, or "circle" or "ellipse",
    with a center, semi-axes a and b (a = b for a circle), and the counterclockwise angle
    in radians of the a axis from the x axis.
    """

    kind: str
    vertices: np.ndarray = None
    center: np.ndarray = None
    a: float = 0.0
    b: float = 0.0
    angle: float = 0.0

    def area(self):
        if self.kind == "polygon":
            x, y = self.vertices[:, 0], self.vertices[:, 1]
            return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
        return math.pi * self.a * self.b

    def _local(self, x, y):
        """Return coordinates in the frame of a circle or ellipse, scaled to a unit circle."""
        c, s = math.cos(self.angle), math.sin(self.angle)
        dx, dy = np.asarray(x) - self.center[0], np.asarray(y) - self.center[1]
        return (c * dx + s * dy) / self.a, (-s * dx + c * dy) / self.b

    def contains(self, x, y):
        """Return whether points lie strictly inside the footprint."""
        if self.kind == "polygon":
            return _inside_polygon(self.vertices, np.asarray(x, float), np.asarray(y, float))
        u, v = self._local(x, y)
        return u * u + v * v < 1.0

    def outline(self, segments=128):
        """Return the outline as polygon vertices, counterclockwise.

        A circle or ellipse is sampled at segments points, for plots and mesh sizing.
        """
        if self.kind == "polygon":
            return self.vertices
        t = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
        c, s = math.cos(self.angle), math.sin(self.angle)
        u, v = self.a * np.cos(t), self.b * np.sin(t)
        return np.column_stack((self.center[0] + c * u - s * v, self.center[1] + s * u + c * v))

    def chord_intervals(self, p0, p1):
        """Return the intervals of t in [0, 1] where p0 + t (p1 - p0) is inside, in 2D."""
        p0, d = np.asarray(p0[:2], float), np.asarray(p1[:2], float) - np.asarray(p0[:2], float)
        if self.kind != "polygon":
            u0, v0 = self._local(p0[0], p0[1])
            c, s = math.cos(self.angle), math.sin(self.angle)
            du, dv = (c * d[0] + s * d[1]) / self.a, (-s * d[0] + c * d[1]) / self.b
            qa, qb, qc = du * du + dv * dv, 2 * (u0 * du + v0 * dv), u0 * u0 + v0 * v0 - 1
            if qa == 0.0:
                return [(0.0, 1.0)] if qc < 0 else []
            disc = qb * qb - 4 * qa * qc
            if disc <= 0.0:
                return []
            root = math.sqrt(disc)
            t0, t1 = (-qb - root) / (2 * qa), (-qb + root) / (2 * qa)
            t0, t1 = max(t0, 0.0), min(t1, 1.0)
            return [(t0, t1)] if t1 > t0 else []
        if d[0] == 0.0 and d[1] == 0.0:
            return [(0.0, 1.0)] if self.contains(p0[0], p0[1]) else []
        # Crossings of the segment with every edge, then the inside parts between them.
        crossings = [0.0, 1.0]
        vertices = self.vertices
        for i in range(len(vertices)):
            a, b = vertices[i], vertices[(i + 1) % len(vertices)]
            e = b - a
            denominator = d[0] * e[1] - d[1] * e[0]
            if denominator == 0.0:
                continue
            w = a - p0
            t = (w[0] * e[1] - w[1] * e[0]) / denominator
            s = (w[0] * d[1] - w[1] * d[0]) / denominator
            if 0.0 < t < 1.0 and 0.0 <= s <= 1.0:
                crossings.append(t)
        crossings = sorted(crossings)
        intervals = []
        for t0, t1 in zip(crossings[:-1], crossings[1:]):
            if t1 <= t0:
                continue
            mid = p0 + 0.5 * (t0 + t1) * d
            if self.contains(mid[0], mid[1]):
                if intervals and intervals[-1][1] == t0:
                    intervals[-1] = (intervals[-1][0], t1)
                else:
                    intervals.append((t0, t1))
        return intervals

    def boundary_distance(self, x, y):
        """Return the distance from a point to the outline, in 2D (sampled for curves)."""
        outline = self.outline()
        a, b = outline, np.roll(outline, -1, axis=0)
        e = b - a
        p = np.array([x, y])
        t = np.clip(np.einsum("ij,ij->i", p - a, e) / np.einsum("ij,ij->i", e, e), 0.0, 1.0)
        nearest = a + t[:, None] * e
        return float(np.min(np.hypot(*(p - nearest).T)))


def is_convex(vertices):
    """Return whether a counterclockwise polygon is convex."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] \
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    return bool(np.all(cross >= 0.0))


def triangulate(vertices):
    """Return the triangles of a simple counterclockwise polygon, by ear clipping."""
    remaining = list(range(len(vertices)))
    triangles = []
    while len(remaining) > 3:
        for k in range(len(remaining)):
            i, j, l = remaining[k - 1], remaining[k], remaining[(k + 1) % len(remaining)]
            a, b, c = vertices[i], vertices[j], vertices[l]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0.0:
                continue  # Reflex or degenerate corner
            triangle = np.array([a, b, c])
            others = [m for m in remaining if m not in (i, j, l)]
            if others and np.any(_inside_polygon(triangle, vertices[others, 0],
                                                 vertices[others, 1])):
                continue
            triangles.append(triangle)
            remaining.pop(k)
            break
        else:
            raise LayoutError("Could not triangulate a polygon footprint")
    triangles.append(vertices[remaining])
    return triangles


def _prism_planes(polygon, z0, z1):
    """Return the half-spaces n . x <= c, as rows [nx, ny, nz, c], of a convex prism."""
    planes = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        normal = np.array([b[1] - a[1], a[0] - b[0]])  # Outward for a counterclockwise polygon
        normal /= np.linalg.norm(normal)
        planes.append([normal[0], normal[1], 0.0, normal @ a])
    planes += [[0.0, 0.0, -1.0, -z0], [0.0, 0.0, 1.0, z1]]
    return np.array(planes)


def _inside_polygon(vertices, x, y):
    """Even-odd test of points against a polygon."""
    inside = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    xi, yi = vertices[:, 0], vertices[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    for k in range(len(vertices)):
        crosses = (yi[k] > y) != (yj[k] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj[k] - xi[k]) * (y - yi[k]) / (yj[k] - yi[k]) + xi[k]
        inside ^= crosses & (x < x_cross)
    return inside


def _segments_intersect(p, q, r, s):
    """Return whether segments pq and rs intersect at a point other than a shared end."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(r, s, p), orient(r, s, q)
    d3, d4 = orient(p, q, r), orient(p, q, s)
    return d1 * d2 < 0 and d3 * d4 < 0


@dataclass
class Building:
    block: int
    name: str
    type: str
    height: float
    material: str
    footprint: Footprint
    cell: tuple  # (row from the north, column from the west)

    def chord_length(self, p0, p1):
        """Return the length of the segment p0-p1 inside the building, in meters."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        dz = p1[2] - p0[2]
        if dz == 0.0:
            if not 0.0 < p0[2] < self.height:
                return 0.0
            z_lo, z_hi = 0.0, 1.0
        else:
            ta, tb = (0.0 - p0[2]) / dz, (self.height - p0[2]) / dz
            z_lo, z_hi = max(min(ta, tb), 0.0), min(max(ta, tb), 1.0)
            if z_hi <= z_lo:
                return 0.0
        length = float(np.linalg.norm(p1 - p0))
        inside = 0.0
        for t0, t1 in self.footprint.chord_intervals(p0, p1):
            inside += max(0.0, min(t1, z_hi) - max(t0, z_lo))
        return inside * length

    def surface_distance(self, x, y, z):
        """Return the distance from a point to the surface of the building, in meters."""
        d2 = self.footprint.boundary_distance(x, y)
        if self.footprint.contains(x, y):
            if z >= self.height:
                return z - self.height
            return min(d2, self.height - z)
        return math.hypot(d2, max(z - self.height, 0.0))


@dataclass
class GroundCell:
    block: int
    material: str
    kind: str  # the ground type, or "lot" under a building
    bounds: tuple  # (xmin, xmax, ymin, ymax) in meters
    cell: tuple


@dataclass
class City:
    path: Path
    x_edges: np.ndarray
    y_edges: np.ndarray
    air_height: float
    ground_depth: float
    buildings: list
    ground: list
    ground_blocks: dict  # block -> material
    materials: dict  # name -> {"density": g/cm^3, "weight_fractions": {...}}
    source: np.ndarray
    activity: float
    dose_map: dict
    names: dict = field(default_factory=dict)  # block -> name

    @property
    def extent(self):
        """Return (xmin, xmax, ymin, ymax, zmin, zmax) of the domain in meters."""
        return (float(self.x_edges[0]), float(self.x_edges[-1]), float(self.y_edges[0]),
                float(self.y_edges[-1]), -self.ground_depth, self.air_height)

    def block_materials(self):
        """Return the material of every block."""
        blocks = {AIR_BLOCK: "air"}
        blocks.update(self.ground_blocks)
        blocks.update({b.block: b.material for b in self.buildings})
        return blocks

    def convex_pieces(self):
        """Return every building and ground cell as convex pieces, in meters.

        Each piece is a dict with its block and either "planes", the half-spaces
        n . x <= c of a convex prism, or "ellipse", a footprint (a circle or ellipse) with
        "z" = (bottom, top). Non-convex footprints are split into triangles. The pieces do
        not overlap, and the space outside them above the ground is air.
        """
        pieces = []
        for building in self.buildings:
            footprint = building.footprint
            if footprint.kind != "polygon":
                pieces.append({"block": building.block, "ellipse": footprint,
                               "z": (0.0, building.height)})
                continue
            polygons = [footprint.vertices] if is_convex(footprint.vertices) \
                else triangulate(footprint.vertices)
            for polygon in polygons:
                pieces.append({"block": building.block,
                               "planes": _prism_planes(polygon, 0.0, building.height)})
        for ground in self.ground:
            x0, x1, y0, y1 = ground.bounds
            square = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            pieces.append({"block": ground.block,
                           "planes": _prism_planes(square, -self.ground_depth, 0.0)})
        return pieces

    def building_at(self, x, y, z):
        """Return the building that contains a point, or None."""
        for building in self.buildings:
            if 0.0 < z < building.height and building.footprint.contains(x, y):
                return building
        return None

    def box_coverage(self, x_edges, y_edges, z, samples=9):
        """Return, for every box [y, x] of the grid, how many of samples x samples points
        spread evenly over the box, edges included, at height z lie in a building."""
        u = np.linspace(0.0, 1.0, samples)
        xs = x_edges[:-1, None] + u[None, :] * np.diff(x_edges)[:, None]  # [box, sample]
        ys = y_edges[:-1, None] + u[None, :] * np.diff(y_edges)[:, None]
        shape = (len(ys), samples, len(xs), samples)
        x = np.broadcast_to(xs[None, None, :, :], shape).ravel()
        y = np.broadcast_to(ys[:, :, None, None], shape).ravel()
        inside = np.zeros(x.size, dtype=bool)
        for building in self.buildings:
            if 0.0 < z < building.height:
                inside |= building.footprint.contains(x, y)
        return inside.reshape(shape).sum(axis=(1, 3))

    def dose_map_boxes(self):
        """Return the x and y edges and the z range of the dose-map boxes, in meters."""
        spacing = self.dose_map["spacing"]
        xmin, xmax, ymin, ymax, _, _ = self.extent
        nx, ny = round((xmax - xmin) / spacing), round((ymax - ymin) / spacing)
        return (np.linspace(xmin, xmax, nx + 1), np.linspace(ymin, ymax, ny + 1),
                tuple(self.dose_map["z_range"]))


def _footprint(spec, center, rotation_deg):
    shape = spec.get("shape")
    angle = math.radians(rotation_deg)
    c, s = math.cos(angle), math.sin(angle)
    rotate = np.array([[c, -s], [s, c]])
    if shape == "rectangle":
        w, d = 0.5 * float(spec["width"]), 0.5 * float(spec["depth"])
        local = np.array([[-w, -d], [w, -d], [w, d], [-w, d]])
    elif shape == "regular_polygon":
        n, r = int(spec["sides"]), float(spec["radius"])
        if n < 3:
            raise LayoutError("A regular polygon needs at least 3 sides")
        t = 2 * math.pi * np.arange(n) / n
        local = r * np.column_stack((np.cos(t), np.sin(t)))
    elif shape == "polygon":
        local = np.array(spec["points"], dtype=float)
        if local.ndim != 2 or local.shape[1] != 2 or len(local) < 3:
            raise LayoutError("A polygon needs a list of at least three [x, y] points")
    elif shape == "circle":
        r = float(spec["radius"])
        return Footprint("circle", center=np.array(center), a=r, b=r, angle=0.0)
    elif shape == "ellipse":
        a, b = (float(v) for v in spec["semi_axes"])
        return Footprint("ellipse", center=np.array(center), a=a, b=b, angle=angle)
    else:
        raise LayoutError(f"Unknown building shape {shape!r}")
    vertices = local @ rotate.T + np.asarray(center)
    footprint = Footprint("polygon", vertices=vertices)
    if footprint.area() <= 0.0:
        raise LayoutError("Polygon points must be listed counterclockwise")
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # Adjacent edges share a vertex.
            if _segments_intersect(vertices[i], vertices[(i + 1) % n], vertices[j],
                                   vertices[(j + 1) % n]):
                raise LayoutError("A polygon footprint intersects itself")
    return footprint


def _parse_token(token, building_types, ground_tokens):
    if token in ground_tokens:
        return None, ground_tokens[token], None
    match = re.fullmatch(r"([A-Za-z]+)(\d+(?:\.\d*)?)", token)
    if match and match.group(1) in building_types:
        return match.group(1), None, float(match.group(2))
    raise LayoutError(f"Lattice token {token!r} is neither a ground type nor a building type "
                      "followed by a height")


def load(path="city.toml"):
    """Read and check a layout file, and return the City it describes."""
    path = Path(path)
    with open(path, "rb") as stream:
        data = tomllib.load(stream)
    domain, lattice = data["domain"], data["lattice"]
    air_height, ground_depth = float(domain["air_height"]), float(domain["ground_depth"])
    if air_height <= 0 or ground_depth <= 0:
        raise LayoutError("air_height and ground_depth must be positive")
    x_widths = np.array(lattice["x_widths"], float)
    y_widths = np.array(lattice["y_widths"], float)
    if np.any(x_widths <= 0) or np.any(y_widths <= 0):
        raise LayoutError("Lattice widths must be positive")
    x_edges = np.concatenate(([0.0], np.cumsum(x_widths)))
    y_edges = np.concatenate(([0.0], np.cumsum(y_widths)))
    rows = [row.split() for row in lattice["rows"]]
    if len(rows) != len(y_widths):
        raise LayoutError(f"The lattice has {len(rows)} rows but {len(y_widths)} y_widths")
    for i, row in enumerate(rows):
        if len(row) != len(x_widths):
            raise LayoutError(f"Row {i} has {len(row)} tokens but there are "
                              f"{len(x_widths)} x_widths")

    materials = data["materials"]
    for name, material in materials.items():
        if float(material["density"]) <= 0:
            raise LayoutError(f"Material {name} needs a positive density")
        total = sum(material["weight_fractions"].values())
        if abs(total - 1.0) > 1e-4:
            raise LayoutError(f"The weight fractions of {name} sum to {total}, not 1")
    if "air" not in materials:
        raise LayoutError("The layout must define the material air")

    ground_tokens = {}
    for kind, spec in data["ground"].items():
        if spec["material"] not in materials:
            raise LayoutError(f"Ground type {kind} uses the undefined material "
                              f"{spec['material']}")
        ground_tokens[spec["token"]] = (kind, spec["material"])
    defaults = data.get("building_defaults", {})
    building_types = {name: {**defaults, **spec} for name, spec in data["buildings"].items()}
    for name, spec in building_types.items():
        for key in ("material", "lot_material"):
            if spec.get(key) not in materials:
                raise LayoutError(f"Building type {name} uses the undefined material "
                                  f"{spec.get(key)}")
        if not name.isalpha():
            raise LayoutError(f"Building type name {name!r} must contain only letters")

    buildings, ground, ground_blocks, names = [], [], {}, {AIR_BLOCK: "air"}
    material_block = {}
    for i, row in enumerate(rows):
        # Rows are listed from north to south; row i covers the (n - 1 - i)-th y interval.
        j_y = len(rows) - 1 - i
        ymin, ymax = y_edges[j_y], y_edges[j_y + 1]
        for j, token in enumerate(row):
            xmin, xmax = x_edges[j], x_edges[j + 1]
            type_name, ground_type, height = _parse_token(token, building_types, ground_tokens)
            if type_name is None:
                kind, ground_material = ground_type
            else:
                kind, ground_material = "lot", building_types[type_name]["lot_material"]
            if ground_material not in material_block:
                block = 2 + len(material_block)
                if block >= FIRST_BUILDING_BLOCK:
                    raise LayoutError("Too many ground materials")
                material_block[ground_material] = block
                ground_blocks[block] = ground_material
                names[block] = f"ground_{ground_material}"
            ground.append(GroundCell(material_block[ground_material], ground_material, kind,
                                     (xmin, xmax, ymin, ymax), (i, j)))
            if type_name is None:
                continue
            spec = building_types[type_name]
            if not 0.0 < height < air_height - 1.0:
                raise LayoutError(f"Building {token} in row {i}, column {j} must be lower "
                                  f"than air_height - 1 m")
            center = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
            footprint = _footprint(spec, center, float(spec.get("rotation", 0.0)))
            outline = footprint.outline(512)
            margin = min(outline[:, 0].min() - xmin, xmax - outline[:, 0].max(),
                         outline[:, 1].min() - ymin, ymax - outline[:, 1].max())
            if margin < LOT_CLEARANCE:
                raise LayoutError(f"Building {token} in row {i}, column {j} comes within "
                                  f"{margin:.2f} m of the edge of its cell; the minimum is "
                                  f"{LOT_CLEARANCE} m")
            block = FIRST_BUILDING_BLOCK + len(buildings)
            name = f"{token}_r{i}c{j}"
            buildings.append(Building(block, name, type_name, height, spec["material"],
                                      footprint, (i, j)))
            names[block] = name
    if not buildings:
        raise LayoutError("The city has no buildings")

    source = np.array(data["source"]["position"], float)
    xmin, xmax, ymin, ymax = x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]
    clearance = min(source[0] - xmin, xmax - source[0], source[1] - ymin, ymax - source[1],
                    source[2], air_height - source[2])
    for building in buildings:
        if building.footprint.contains(source[0], source[1]) and source[2] < building.height:
            raise LayoutError(f"The source lies inside building {building.name}")
        clearance = min(clearance, building.surface_distance(*source))
    if clearance < SOURCE_CLEARANCE:
        raise LayoutError(f"The source is {clearance:.3f} m from a surface; the minimum is "
                          f"{SOURCE_CLEARANCE} m")

    dose_map = dict(data["dose_map"])
    spacing = float(dose_map["spacing"])
    for extent in (xmax - xmin, ymax - ymin):
        if abs(extent / spacing - round(extent / spacing)) > 1e-9:
            raise LayoutError("The dose-map spacing must divide the city into whole boxes")
    z_range = dose_map["z_range"]
    if not 0.0 <= z_range[0] < z_range[1] < air_height:
        raise LayoutError("The dose-map z_range must lie in the air")

    return City(path=path, x_edges=x_edges, y_edges=y_edges, air_height=air_height,
                ground_depth=ground_depth, buildings=buildings, ground=ground,
                ground_blocks=ground_blocks, materials=materials, source=source,
                activity=float(data["source"]["activity_Bq"]), dose_map=dose_map,
                names=names)


if __name__ == "__main__":
    city = load(Path(__file__).resolve().parent / "city.toml")
    xmin, xmax, ymin, ymax, zmin, zmax = city.extent
    print(f"Domain [{xmin}, {xmax}] x [{ymin}, {ymax}] x [{zmin}, {zmax}] m")
    for block, material in city.block_materials().items():
        print(f"Block {block:4d} {city.names[block]:20s} {material}")
    for building in city.buildings:
        print(f"{building.name:12s} area {building.footprint.area():8.2f} m^2 "
              f"height {building.height:5.1f} m")
