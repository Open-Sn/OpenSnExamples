#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Geometry of the ICF target chamber example, and the exact uncollided flux of its source.

The chamber is a NIF-like spherical vacuum vessel with a point source at its centre, an
aluminum wall, a borated concrete shield and air outside, crossed by cylindrical ports whose
axes pass through the centre. Lengths are in cm.

    r < 500 cm            cavity, vacuum                       block 1
    500 < r < 510 cm      wall, aluminum alloy                 block 2
    510 < r < 550 cm      shield, borated concrete             block 3
    550 < r < 800 cm      air                                  block 4
    ports                 vacuum through the wall and shield   block 1

48 ports of radius 25 cm lie on eight cones, at polar angles of 23.5, 30, 44.5 and 50 degrees
from each pole (4, 4, 8 and 8 ports per cone), the angles of the laser-beam ports of the
National Ignition Facility. 4 ports of radius 40 cm lie on the equator. The arrangement is
symmetric about the three coordinate planes. The OpenSn model is the octant x, y, z > 0, with
the source at its corner; it holds four whole ports and five ports cut in half by a coordinate
plane. The OpenMC model is the whole sphere.

A straight path from the centre stays in a port until its distance to the port axis exceeds
the port radius, at the radius r* = (port radius) / sin(angle to the axis); after that it is
in the wall or the shield. Ports do not overlap, so at most one port matters for a path. The
exact uncollided flux uses the spheres and cylinders themselves, not a faceted mesh.

This module is imported by the other scripts of the example.
"""

import math

import numpy as np

R_CAVITY, R_WALL, R_SHIELD, R_AIR = 500.0, 510.0, 550.0, 800.0
BEAM_PORT_RADIUS, EQUATOR_PORT_RADIUS = 25.0, 40.0
# Port cones: polar angle from the pole (degrees), number of ports, azimuth of the first one.
CONES = [(23.5, 4, 45.0), (30.0, 4, 0.0), (44.5, 8, 22.5), (50.0, 8, 0.0)]
EQUATOR_AZIMUTHS = (45.0, 135.0, 225.0, 315.0)

BLOCKS = {1: "cavity_and_ports", 2: "wall", 3: "shield", 4: "air"}
# Shells of the blocks, for the exact flux: block -> (inner radius, outer radius).
SHELLS = {2: (R_CAVITY, R_WALL), 3: (R_WALL, R_SHIELD), 4: (R_SHIELD, R_AIR)}


def ports():
    """Unit axes [52, 3] and radii [52] of the ports."""
    axes, radii = [], []
    for polar, count, first in CONES:
        for pole in (polar, 180.0 - polar):
            for k in range(count):
                axes.append(direction(pole, first + 360.0 * k / count))
                radii.append(BEAM_PORT_RADIUS)
    for azimuth in EQUATOR_AZIMUTHS:
        axes.append(direction(90.0, azimuth))
        radii.append(EQUATOR_PORT_RADIUS)
    return np.array(axes), np.array(radii)


def octant_ports():
    """Ports in the octant x, y, z > 0: (axis, radius, fraction in the octant)."""
    result = []
    for axis, radius in zip(*ports()):
        if axis.min() > -1.0e-12:
            on_planes = int(np.sum(np.abs(axis) < 1.0e-12))
            result.append((np.maximum(axis, 0.0), radius, 0.5**on_planes))
    return result


def direction(polar, azimuth):
    """Unit vector of polar angle and azimuth in degrees."""
    theta, phi = math.radians(polar), math.radians(azimuth)
    return np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi),
                     math.cos(theta)])


def port_exit_radius(directions):
    """Radius at which the path along each unit direction leaves its port (0 if in none)."""
    axes, radii = ports()
    cosine = directions @ axes.T
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    with np.errstate(divide="ignore"):
        exit_radius = np.where(cosine > 0.0, radii / sine, 0.0)
    return exit_radius.max(axis=1)


def path_lengths(points):
    """Length of the path from the centre to each point [n, 3] in blocks 2, 3 and 4.

    Returns a dict block -> array [n]. The rest of the path is in vacuum.
    """
    radius = np.linalg.norm(points, axis=1)
    # The path is in vacuum up to r_void: the cavity, then a port if it is in one.
    r_void = np.clip(port_exit_radius(points / radius[:, None]), R_CAVITY, R_SHIELD)
    lengths = {}
    for block, (inner, outer) in SHELLS.items():
        start = np.maximum(inner, r_void) if block != 4 else np.full_like(radius, inner)
        lengths[block] = np.maximum(0.0, np.minimum(radius, outer) - start)
    return lengths


def exact_flux(points, sigma_t, strength=1.0):
    """Exact uncollided flux at points [n, 3] away from the centre.

    sigma_t maps the blocks 2, 3 and 4 to a total cross section (1/cm), or to an array of
    them, one per group; the flux then has a last axis over the groups.
    """
    points = np.atleast_2d(points)
    radius = np.linalg.norm(points, axis=1)
    lengths = path_lengths(points)
    depth = sum(np.multiply.outer(lengths[block], np.asarray(sigma_t[block], dtype=float))
                for block in SHELLS)
    kernel = strength / (4.0 * math.pi * radius**2)
    return np.exp(-depth) * (kernel if np.ndim(depth) == 1 else kernel[:, None])


def half_cylinder_volume(radius, sphere_radius):
    """Volume of the part t > 0 of an infinite cylinder of a radius through the centre,
    inside a sphere: the integral of 2 pi s sqrt(R^2 - s^2) ds from 0 to the radius."""
    return 2.0 * math.pi / 3.0 * (sphere_radius**3 - (sphere_radius**2 - radius**2)**1.5)


def octant_volumes():
    """Exact volumes (cm^3) of the blocks in the octant, with the true spheres and cylinders."""
    def sphere(r):
        return math.pi * r**3 / 6.0

    def ports_in_shell(inner, outer):
        return sum(fraction * (half_cylinder_volume(radius, outer)
                               - half_cylinder_volume(radius, inner))
                   for _, radius, fraction in octant_ports())

    wall_ports, shield_ports = ports_in_shell(R_CAVITY, R_WALL), ports_in_shell(R_WALL, R_SHIELD)
    return {
        1: sphere(R_CAVITY) + wall_ports + shield_ports,
        2: sphere(R_WALL) - sphere(R_CAVITY) - wall_ports,
        3: sphere(R_SHIELD) - sphere(R_WALL) - shield_ports,
        4: sphere(R_AIR) - sphere(R_SHIELD),
    }


def line_points():
    """Points of the three line-outs: name -> (abscissa, points [m, 3]).

    "port": the axis of a port of the 23.5 degree cone (azimuth 45 degrees), r from 20 to
    790 cm; "shield": the direction of polar angle 70 and azimuth 65 degrees, away from every
    port; "arc": an arc of radius 650 cm in the plane of azimuth 45 degrees across the beam of
    that port, whose abscissa is the polar angle minus 23.5 degrees.
    """
    radius = 20.0 + 2.0 * np.arange(386)
    angle = -12.0 + 0.1 * np.arange(241)
    return {
        "port": (radius, radius[:, None] * direction(23.5, 45.0)),
        "shield": (radius, radius[:, None] * direction(70.0, 65.0)),
        "arc": (angle, 650.0 * np.array([direction(23.5 + a, 45.0) for a in angle])),
    }


if __name__ == "__main__":
    volumes = octant_volumes()
    for block, name in BLOCKS.items():
        print(f"block {block} {name:17s} octant volume {volumes[block]:.6e} cm^3")
    print(f"{len(octant_ports())} ports in the octant, "
          f"{sum(f for _, _, f in octant_ports()):g} port equivalents")
