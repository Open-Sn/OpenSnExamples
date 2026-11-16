#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Make the figures of the README of the urban source example, in light and dark variants,
in assets/.

    python3 make_figures.py                     # geometry and cross sections
    python3 make_figures.py --opensn urban_source_results.json \\
        --uncollided uncollided_reference.json --openmc openmc_seed20261112/results.json

The second form also draws the dose-rate maps and their comparison with OpenMC.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
import openmc.data  # noqa: E402

import city as citylib  # noqa: E402
import openmc_reference as reference  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", secondary="#52514e", muted="#8a8984",
                  grid="#e4e3df", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                          "#e87ba4", "#008300", "#6250d6", "#e34948"],
                  ramp=["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                        "#184f95", "#0d366b"],
                  diverging=["#256abf", "#fcfcfb", "#d95926"],
                  ground={"road": "#c9c8c3", "plaza": "#e4e3df", "park": "#b9dfb0",
                          "lot": "#e4e3df"}),
    "dark": dict(surface="#1a1a19", text="#f0efec", secondary="#c3c2b7", muted="#8a8984",
                 grid="#33332f", series=["#3987e5", "#d95926", "#199e70", "#c98500",
                                         "#d55181", "#008300", "#9085e9", "#e66767"],
                 ramp=["#1a1a19", "#104281", "#184f95", "#1c5cab", "#2a78d6", "#5598e7",
                       "#86b6ef", "#cde2fb"],
                 diverging=["#5598e7", "#1a1a19", "#e0703f"],
                 ground={"road": "#4a4a46", "plaza": "#33332f", "park": "#2f4a2a",
                         "lot": "#33332f"}),
}
MATERIAL_LABELS = {"air": "Air", "building": "Building (homogenized)", "concrete": "Concrete",
                   "asphalt": "Asphalt", "soil": "Soil"}


def style(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["text"],
        "axes.labelcolor": t["text"], "axes.edgecolor": t["muted"],
        "xtick.color": t["secondary"], "ytick.color": t["secondary"],
        "axes.titlecolor": t["text"], "grid.color": t["grid"], "font.size": 10,
        "axes.grid": False, "legend.frameon": False, "lines.linewidth": 2.0,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return t


def save(fig, out, name, theme):
    fig.savefig(out / f"{name}-{theme}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_city(ax, city, t, fill_buildings=None, outline=True):
    """Draw the ground types and the building footprints in the x-y plane, in meters."""
    for ground in city.ground:
        x0, x1, y0, y1 = ground.bounds
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=t["ground"][ground.kind],
                               lw=0, zorder=0))
    for building in city.buildings:
        face = fill_buildings(building) if fill_buildings else "none"
        ax.add_patch(Polygon(building.footprint.outline(), closed=True, facecolor=face,
                             edgecolor=t["text"] if outline else "none", lw=0.8, zorder=3))


def label_point(footprint):
    """Return the point of a footprint farthest from its outline, on a 1 m grid."""
    outline = footprint.outline()
    xs = np.arange(outline[:, 0].min(), outline[:, 0].max(), 1.0)
    ys = np.arange(outline[:, 1].min(), outline[:, 1].max(), 1.0)
    best, point = -1.0, outline.mean(axis=0)
    for x in xs:
        for y in ys:
            if footprint.contains(x, y):
                distance = footprint.boundary_distance(x, y)
                if distance > best:
                    best, point = distance, (x, y)
    return point


def geometry(city, out, theme):
    """Plan of the city: ground types, buildings colored by height, and the source."""
    t = style(theme)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    cmap = LinearSegmentedColormap.from_list("ramp", t["ramp"][2:])
    heights = [b.height for b in city.buildings]
    norm = plt.Normalize(0.0, max(heights))
    draw_city(ax, city, t, lambda b: cmap(norm(b.height)))
    for building in city.buildings:
        cx, cy = label_point(building.footprint)
        ax.text(cx, cy, f"{building.height:g} m", ha="center", va="center", fontsize=8,
                color="#ffffff" if norm(building.height) > 0.45 else "#0b0b0b", zorder=4)
    ax.plot(*city.source[:2], marker="*", ms=16, color=t["series"][1], mec=t["text"],
            mew=0.6, zorder=5)
    xmin, xmax, ymin, ymax, _, _ = city.extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (east) [m]")
    ax.set_ylabel("y (north) [m]")
    handles = [Rectangle((0, 0), 1, 1, color=t["ground"][k]) for k in ("road", "plaza", "park")]
    handles.append(plt.Line2D([], [], marker="*", ms=12, ls="none", color=t["series"][1],
                              mec=t["text"], mew=0.6))
    ax.legend(handles, ["Road (asphalt)", "Plaza and lots (concrete)", "Park (soil)",
                        "Cs-137 source, 1 m above the street"],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.8, label="Building height [m]")
    save(fig, out, "geometry", theme)


def cross_sections(library, out, theme):
    """Mass attenuation coefficients of the materials and the dose coefficients."""
    t = style(theme)
    temperature = f"{round(reference.TEMPERATURE)}K"
    with h5py.File(library) as stream:
        edges = np.array(stream.attrs["group structure"]) / 1.0e3  # keV
        materials = [name for name in stream if isinstance(stream[name], h5py.Group)]
        totals = {name: stream[f"{name}/{temperature}/total"][()][::-1] for name in materials}
        dose = stream[f"air/{temperature}/dose_ap"][()][::-1]
    city = citylib.load(Path(__file__).resolve().parent / "city.toml")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for k, name in enumerate(materials):
        density = city.materials[name]["density"]
        ax.stairs(totals[name] / density, edges, color=t["series"][k],
                  label=MATERIAL_LABELS.get(name, name), lw=1.6, baseline=None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Photon energy [keV]")
    ax.set_ylabel(r"Group total cross section / density [cm$^2$/g]")
    ax.legend(fontsize=8)
    bx.stairs(dose, edges, color=t["series"][0], lw=1.6, baseline=None,
              label="Group-averaged (air spectrum)")
    e, d = openmc.data.dose_coefficients("photon", reference.DOSE_GEOMETRY)
    keep = e <= edges[-1] * 1.0e3 * 1.2
    bx.plot(e[keep] / 1.0e3, d[keep], color=t["series"][1], lw=1.0, marker="o", ms=3,
            label="ICRP 116, AP")
    bx.set_xscale("log")
    bx.set_xlabel("Photon energy [keV]")
    bx.set_ylabel(r"Effective dose per fluence [pSv cm$^2$]")
    bx.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out, "cross-sections", theme)


def dose_maps(city, sn, ref, mc, out, theme):
    """Total dose rate at 1 m from OpenSn and OpenMC, and their relative difference."""
    t = style(theme)
    x_edges, y_edges, z_range = city.dose_map_boxes()
    volume = (np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :]
              * (z_range[1] - z_range[0])) * 1.0e6
    rate = city.activity * citylib.CS137_GAMMA_YIELD * 3600.0 * 1.0e-6  # pSv/photon -> uSv/h
    sn_map = np.array(sn["dose_map_total"]) * rate
    mc_map = np.array(mc["total"]["dose_map_integral"][0]) / volume * rate
    mc_se = np.array(mc["total"]["dose_map_integral"][1]) / volume * rate
    # Boxes inside a building are left blank, and the color scale spans the open boxes.
    samples = 9
    covered = city.box_coverage(x_edges, y_edges, 0.5 * (z_range[0] + z_range[1]), samples)
    inside = covered == samples**2
    sn_map, mc_map = np.where(inside, np.nan, sn_map), np.where(inside, np.nan, mc_map)
    # The difference is shown for the boxes that compare_openmc.py compares: in the open, at
    # least 10 m from the source, with an OpenMC standard error below 5%. A building wall
    # makes the box average of OpenMC and the Gauss-point average of OpenSn different
    # quantities, so the boxes that a wall crosses are left blank.
    xc, yc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])
    distance = np.hypot(xc[None, :] - city.source[0], yc[:, None] - city.source[1])
    compared = (covered == 0) & (distance >= 10.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.where(compared & (mc_se / mc_map < 0.05), sn_map / mc_map - 1.0, np.nan)
    cmap = LinearSegmentedColormap.from_list("ramp", t["ramp"][1:])
    open_values = sn_map[(covered == 0) & (sn_map > 0)]
    norm = LogNorm(vmin=np.percentile(open_values, 1), vmax=open_values.max())
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for ax, data, title in ((axes[0], sn_map, "OpenSn"), (axes[1], mc_map, "OpenMC")):
        image = ax.pcolormesh(x_edges, y_edges, data, cmap=cmap, norm=norm)
        draw_city(ax, city, t, outline=True)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.85, label=r"Dose rate at 1 m [$\mu$Sv/h]")
    diverging = LinearSegmentedColormap.from_list("div", t["diverging"])
    image = axes[2].pcolormesh(x_edges, y_edges, 100 * diff, cmap=diverging,
                               norm=TwoSlopeNorm(0.0, -10.0, 10.0))
    draw_city(axes[2], city, t, outline=True)
    axes[2].set_title("OpenSn / OpenMC - 1")
    fig.colorbar(image, ax=axes[2], shrink=0.85, label="Difference [%]")
    for ax in axes:
        ax.plot(*city.source[:2], marker="*", ms=12, color=t["series"][1], mec=t["text"],
                mew=0.6, zorder=6)
        ax.set_aspect("equal")
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    fig.tight_layout()
    save(fig, out, "dose-map", theme)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--layout", type=Path, default=here / "city.toml")
    parser.add_argument("--library", type=Path, default=here / reference.LIBRARY)
    parser.add_argument("--opensn", type=Path)
    parser.add_argument("--uncollided", type=Path)
    parser.add_argument("--openmc", type=Path)
    parser.add_argument("--output", type=Path, default=here / "assets")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    city = citylib.load(args.layout)
    for theme in THEMES:
        geometry(city, args.output, theme)
        cross_sections(args.library, args.output, theme)
        if args.opensn and args.openmc:
            sn = json.loads(args.opensn.read_text())
            mc = json.loads(args.openmc.read_text())
            ref = json.loads(args.uncollided.read_text()) if args.uncollided else None
            dose_maps(city, sn, ref, mc, args.output, theme)
    print(f"Wrote figures to {args.output}")


if __name__ == "__main__":
    main()
