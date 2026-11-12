#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Render the figures of the coil example into ``assets/``.

Reads ``coil_deposition.pvtu`` and ``coil_results.json`` written by ``coil.py``
and the OpenMC reference ``openmc_results_<mesh>.json`` of the same mesh. Each
figure is written twice, as ``<name>-light.png`` on a white background and
``<name>-dark.png`` on a dark background, with the themes and the inferno
color map of the tokamak example. Requires pyvista, matplotlib, and numpy.

Usage::

    python3 make_figures.py [--run-dir DIR]
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm, Normalize  # noqa: E402

INSULATION, COPPER, AIR = 1, 2, 3
# Photon energy (MeV) below which the spectra figure shades the groups.
LOW_ENERGY = 0.02
# Size of the written images relative to the size they are drawn at.
IMAGE_SCALE = 0.75
# Viewing direction of the 3D figures: azimuth from +z toward +x about the
# beam axis, and elevation toward +y, in degrees.
VIEW_AZIMUTH = 60.0
VIEW_ELEVATION = 20.0

# The themes of the tokamak example. Each one trims the end of inferno that
# would vanish into its background.
THEMES = {
    "dark": {
        "style": "dark_background", "background": "#0d1b29", "ink": "#f1f6fb",
        "muted": "#aabcca", "grid": "#2a3b4c",
        "cmap": ListedColormap(colormaps["inferno"](np.linspace(0.30, 1.0, 256))),
        "copper": "#c87846", "insulation": "#5b9bd5", "edges": "#8fb8e3",
        "opensn": "#55c6dc", "openmc": "#f6a04d", "beam": "#f1f6fb",
    },
    "light": {
        "style": "default", "background": "#ffffff", "ink": "#1f2328",
        "muted": "#57606a", "grid": "#e4e3df",
        "cmap": ListedColormap(colormaps["inferno"](np.linspace(0.0, 0.88, 256))),
        "copper": "#b5612f", "insulation": "#2f6db0", "edges": "#6f9bd0",
        "opensn": "#1f78b4", "openmc": "#e8710a", "beam": "#1f2328",
    },
}
T = THEMES["light"]


def set_theme(name):
    """Make the named theme the one that the figure functions use."""
    global T
    T = THEMES[name]
    plt.style.use(T["style"])
    plt.rcParams.update({"figure.facecolor": T["background"], "axes.facecolor": T["background"],
                         "savefig.facecolor": T["background"]})


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["muted"])
    ax.tick_params(colors=T["muted"], labelsize=9)
    ax.grid(True, color=T["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def plotter(window_size=(1600, 1200)):
    plot = pv.Plotter(off_screen=True, window_size=window_size)
    plot.set_background(T["background"])
    return plot


def save_render(plot, out):
    """Write a 3D rendering, cropped to its content, at IMAGE_SCALE of its size.

    Resampling the full-size rendering keeps the text in proportion, which a
    smaller window would not.
    """
    pixels = plot.screenshot(return_img=True)
    # Crop the empty background around the picture, keeping a small margin.
    background = pixels[0, 0]
    rows, cols = np.nonzero(np.any(pixels != background, axis=2))
    margin = round(0.03 * max(pixels.shape[:2]))
    top, bottom = max(rows.min() - margin, 0), min(rows.max() + margin + 1, pixels.shape[0])
    left, right = max(cols.min() - margin, 0), min(cols.max() + margin + 1, pixels.shape[1])
    image = Image.fromarray(pixels[top:bottom, left:right])
    size = (round(image.width * IMAGE_SCALE), round(image.height * IMAGE_SCALE))
    image.resize(size, Image.LANCZOS).save(out)


def beam_from_top_camera(plot):
    """Point the camera so that the beam, which travels along -y, points
    straight down the image. With +y as the view up and a parallel projection,
    the beam's parallel rays stay vertical and parallel in the image."""
    azimuth, elevation = np.radians(VIEW_AZIMUTH), np.radians(VIEW_ELEVATION)
    direction = np.array([np.sin(azimuth) * np.cos(elevation), np.sin(elevation),
                          np.cos(azimuth) * np.cos(elevation)])
    focal = np.array([0.0, 0.35, 0.65])
    plot.enable_parallel_projection()
    plot.camera_position = [tuple(focal + 10.0 * direction), tuple(focal), (0, 1, 0)]
    plot.reset_camera()


def geometry_figure(mesh, out):
    """The coil with the beam entering from the top of the image."""
    blocks = mesh.cell_data["Block"]
    copper = mesh.extract_cells(np.flatnonzero(blocks == COPPER))
    jacket = mesh.extract_cells(np.flatnonzero(blocks == INSULATION))
    # Portrait, because the beam above the coil makes the view tall.
    plot = plotter(window_size=(1200, 1500))
    plot.enable_depth_peeling(number_of_peels=8)
    plot.add_mesh(jacket.extract_surface(algorithm="dataset_surface"),
                  color=T["insulation"], opacity=0.15, show_edges=True,
                  edge_color=T["edges"], line_width=0.3)
    plot.add_mesh(copper.extract_surface(algorithm="dataset_surface"), color=T["copper"])
    for x in np.linspace(-0.9, 0.9, 4):
        for z in np.linspace(-0.25, 1.55, 4):
            plot.add_mesh(pv.Arrow(start=(x, 2.05, z), direction=(0, -1, 0), scale=0.5),
                          color=T["beam"])
    beam_from_top_camera(plot)
    plot.camera.zoom(1.05)
    save_render(plot, out)
    plot.close()


def deposition_3d_figure(mesh, name, out, scale, label):
    """Deposition on the surface of the whole coil, in the view of the geometry
    figure. The open end of the wire shows the copper core."""
    blocks = mesh.cell_data["Block"]
    wire = mesh.extract_cells(np.flatnonzero(blocks != AIR))
    values = np.asarray(wire.point_data[name])
    surface = wire.extract_surface(algorithm="dataset_surface")
    plot = plotter(window_size=(1200, 1500))
    # The bar title would overlap the tick labels of a horizontal bar, so the
    # label goes under the bar as text instead.
    bar = {"title": "", "color": T["ink"], "fmt": "%.0e" if scale == "log" else "%.1e",
           "vertical": False,
           "position_x": 0.15, "position_y": 0.06, "width": 0.7, "height": 0.05,
           "label_font_size": 16, "n_labels": 5}
    # Mostly ambient light keeps the colors true to the scale while the diffuse
    # part still shows the shape of the wire.
    shading = {"ambient": 0.55, "diffuse": 0.5, "specular": 0.0}
    if scale == "log":
        positive = values[values > 0]
        lo, hi = np.percentile(positive, 1), positive.max()
        plot.add_mesh(surface, scalars=np.clip(surface.point_data[name], lo, hi),
                      cmap=T["cmap"], log_scale=True, clim=(lo, hi), scalar_bar_args=bar,
                      **shading)
    else:
        # A linear scale over the range of the values. The color bar labels
        # carry the sign.
        clim = (np.percentile(values, 0.5), np.percentile(values, 99.5))
        plot.add_mesh(surface, scalars=name, cmap=T["cmap"], clim=clim,
                      scalar_bar_args=bar, **shading)
    beam_from_top_camera(plot)
    plot.camera.zoom(1.2)
    plot.add_text(label, position="lower_edge", font_size=14, color=T["ink"])
    save_render(plot, out)
    plot.close()


def cross_section_figure(mesh, out):
    """Energy and charge deposition in the four wire cross sections cut by x = 0."""
    blocks = mesh.cell_data["Block"]
    wire = mesh.extract_cells(np.flatnonzero(blocks != AIR))
    section = wire.slice(normal=(1, 0, 0), origin=(0, 0, 0)).triangulate()
    points = section.points
    tris = section.faces.reshape(-1, 4)[:, 1:]
    energy = np.asarray(section.point_data["energy_deposition"])
    charge = np.asarray(section.point_data["charge_deposition"])

    # Helix crossings of x = 0: (y, z) = (+0.6, 0.1375), (-0.6, 0.4125), ...
    centers = [(0.6, 0.1375), (-0.6, 0.4125), (0.6, 0.6875), (-0.6, 0.9625)]
    titles = ["upstream turn 1", "downstream turn 1", "upstream turn 2", "downstream turn 2"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), constrained_layout=True)
    e_pos = energy[energy > 0]
    e_norm = LogNorm(vmin=np.percentile(e_pos, 2), vmax=e_pos.max())
    c_norm = Normalize(vmin=np.percentile(charge, 0.5), vmax=np.percentile(charge, 99.5))
    tri_center = points[tris].mean(axis=1)
    for col, ((yc, zc), title) in enumerate(zip(centers, titles)):
        keep = np.hypot(tri_center[:, 1] - yc, tri_center[:, 2] - zc) < 0.25
        t = tris[keep]
        y = points[:, 1] - yc
        z = points[:, 2] - zc
        for row, (field, norm) in enumerate(((energy, e_norm), (charge, c_norm))):
            ax = axes[row, col]
            values = np.clip(field, norm.vmin, norm.vmax)
            ax.tripcolor(10 * y, 10 * z, t, values, norm=norm, cmap=T["cmap"],
                         shading="gouraud", rasterized=True)
            ax.set_aspect("equal")
            ax.set_xlim(-2.1, 2.1)
            ax.set_ylim(-2.1, 2.1)
            ax.tick_params(colors=T["muted"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, color=T["ink"], fontsize=10)
                # The beam travels along -y.
                ax.annotate("", xy=(0.9, 1.9), xytext=(2.0, 1.9),
                            arrowprops={"arrowstyle": "->", "color": T["muted"]})
            if col == 0:
                ax.set_ylabel("z offset (mm)", color=T["muted"], fontsize=9)
            if row == 1:
                ax.set_xlabel("y offset (mm)", color=T["muted"], fontsize=9)
    for row, label in enumerate(("energy (MeV/cm³)", "net electrons (1/cm³)")):
        bar = fig.colorbar(axes[row, 3].collections[0], ax=axes[row, :], shrink=0.85)
        bar.set_label(label, color=T["ink"])
        bar.ax.tick_params(colors=T["muted"])
        bar.outline.set_visible(False)
    fig.savefig(out, dpi=170 * IMAGE_SCALE)
    plt.close(fig)


def spectra_figure(groups, opensn, openmc, out):
    bounds = np.array(groups["photon"])
    upper, lower = bounds[:, 0], bounds[:, 1]
    lethargy = np.log(upper / lower)
    centers = np.sqrt(upper * lower)
    num_photon = len(bounds)
    regions = ("copper", "insulation", "air")
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3]}, constrained_layout=True)
    for col, region in enumerate(regions):
        sn = np.array(opensn["region_flux"][region][:num_photon]) / lethargy
        mc = np.array(openmc["region_photon_flux"][region]) / lethargy
        sd = np.array(openmc["region_photon_flux_sd"][region]) / lethargy
        ax = axes[0, col]
        edges = np.append(lower[::-1], upper[0])
        ax.stairs(sn[::-1], edges, color=T["opensn"], lw=2, label="OpenSn, 30 groups")
        ax.errorbar(centers, mc, yerr=2 * sd, fmt="o", ms=4, color=T["openmc"],
                    ecolor=T["openmc"], elinewidth=1, label="OpenMC, 2σ")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(region, color=T["ink"], fontsize=11)
        style_axes(ax)
        if col == 0:
            ax.set_ylabel("∫φ dV per unit lethargy\n(cm per incident photon)",
                          color=T["muted"], fontsize=9)
            ax.legend(frameon=False, fontsize=9, labelcolor=T["ink"])
        ratio = axes[1, col]
        good = mc > 0
        values = sn[good] / mc[good]
        # Below LOW_ENERGY the multigroup and continuous-energy photon data
        # differ most, and these photons carry little of the absorbed energy.
        for panel in (ax, ratio):
            panel.axvspan(lower.min(), LOW_ENERGY, color=T["grid"], alpha=0.6, lw=0, zorder=0)
        ratio.axhline(1.0, color=T["muted"], lw=1)
        ratio.errorbar(centers[good], values,
                       yerr=2 * sn[good] * sd[good] / mc[good]**2,
                       fmt="o", ms=3.5, color=T["opensn"], elinewidth=1)
        span = max(0.1, np.abs(values - 1.0).max() * 1.1)
        ratio.set_ylim(1.0 - span, 1.0 + span)
        if col == 0:
            ratio.text(1.2e-3, 1.0 + 0.8 * span, "below 20 keV", color=T["muted"],
                       fontsize=8, va="top")
        ratio.set_xlabel("photon energy (MeV)", color=T["muted"], fontsize=9)
        style_axes(ratio)
        if col == 0:
            ratio.set_ylabel("OpenSn / OpenMC", color=T["muted"], fontsize=9)
    fig.savefig(out, dpi=170 * IMAGE_SCALE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default=".", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    assets = here / "assets"
    assets.mkdir(exist_ok=True)
    mesh = pv.read(args.run_dir / "coil_deposition.pvtu")
    groups = json.loads((here / "xs" / "groups.json").read_text())
    opensn = json.loads((args.run_dir / "coil_results.json").read_text())
    # The OpenMC reference uses the same faceted geometry as the OpenSn mesh.
    openmc_file = here / f"openmc_results_{opensn['settings']['mesh']}.json"
    openmc = json.loads(openmc_file.read_text()) if openmc_file.exists() else None
    for theme in THEMES:
        set_theme(theme)
        geometry_figure(mesh, assets / f"geometry-{theme}.png")
        deposition_3d_figure(mesh, "energy_deposition", assets / f"energy-deposition-{theme}.png",
                             "log", "MeV/cm³ per photon")
        deposition_3d_figure(mesh, "charge_deposition", assets / f"charge-deposition-{theme}.png",
                             "linear", "electrons/cm³ per photon")
        cross_section_figure(mesh, assets / f"wire-cross-sections-{theme}.png")
        if openmc is not None:
            spectra_figure(groups, opensn, openmc, assets / f"openmc-spectra-{theme}.png")


if __name__ == "__main__":
    main()
