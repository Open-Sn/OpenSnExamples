#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Figures of the ICF target chamber example, in light and dark versions, written to assets/.

Inputs are the outputs of the other scripts: the OpenSn results of chamber.py (prefixes, each
with a label), the OpenMC reference (openmc_reference.py), the semi-analytic uncollided flux
(uncollided_reference.py) and the libraries. For example:

    python3 make_figures.py --openmc openmc_seed20261113/results.json \\
        --opensn chamber_4g_fine:"4 groups, fine mesh" \\
        chamber_64g_coarse:"64 groups, coarse mesh" \\
        --sn-only chamber_4g_fine_sn_only.json

Figures: geometry, cross-sections, profiles (the three line-outs), regions (differences with
OpenMC in each block), spectra, maps (total flux in the half-plane of azimuth 45 degrees) and,
with --sn-only, sn-only (ray effects of a discrete-ordinates point source).
"""

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import chamber_geometry as geometry  # noqa: E402

# Validated default palette of the data-visualization guidelines: categorical slots in a fixed
# order, light and dark variants, and the text and surface colors of each theme.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#1a1a19", "secondary": "#5f5e5a",
              "muted": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "secondary": "#c3c2b7",
             "muted": "#5f5e5a",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]},
}
BLOCKS = list(geometry.BLOCKS.values())
BLOCK_LABELS = {"cavity_and_ports": "cavity and ports", "wall": "wall", "shield": "shield",
                "air": "air"}
LINE_TITLES = {"port": "Along the axis of a 23.5° port", "shield": "Through the shield, away "
               "from the ports", "arc": "Across the beam of the port, r = 650 cm"}


def style(ax, theme):
    ax.set_facecolor(theme["surface"])
    for spine in ax.spines.values():
        spine.set_color(theme["muted"])
    ax.tick_params(colors=theme["secondary"], labelsize=8)
    ax.xaxis.label.set_color(theme["secondary"])
    ax.yaxis.label.set_color(theme["secondary"])
    ax.title.set_color(theme["text"])
    ax.grid(True, color=theme["muted"], linewidth=0.4, alpha=0.5)


def steps(ax, values, edges, **kwargs):
    """Draw group values ordered from high to low energy, with their edges (eV) in any order;
    a zero lowest edge is drawn at 1e-5 eV on the logarithmic axis."""
    edges = np.sort(np.asarray(edges, float))
    edges = np.where(edges <= 0.0, 1.0e-5, edges)
    ax.stairs(np.asarray(values, float)[::-1], edges, baseline=None, **kwargs)


def lethargy_widths(edges):
    """Lethargy width of each group, ordered from high to low energy."""
    edges = np.sort(np.asarray(edges, float))[::-1]
    edges = np.where(edges <= 0.0, 1.0e-5, edges)
    return np.log(edges[:-1] / edges[1:])


def figure(theme, *args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(theme["surface"])
    for ax in np.atleast_1d(axes).ravel():
        style(ax, theme)
    return fig, axes


def legend(ax, theme, **kwargs):
    handle = ax.legend(frameon=False, fontsize=8, **kwargs)
    for text in handle.get_texts():
        text.set_color(theme["text"])
    return handle


def save(fig, out, name, mode):
    fig.tight_layout()
    fig.savefig(out / f"{name}-{mode}.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def geometry_figure(theme, out, mode):
    n = 1200
    h = np.linspace(0.0, 820.0, n)
    hh, zz = np.meshgrid(h, h)
    points = np.stack([hh.ravel() / math.sqrt(2), hh.ravel() / math.sqrt(2), zz.ravel()], axis=1)
    r = np.linalg.norm(points, axis=1)
    exit_radius = geometry.port_exit_radius(points / np.maximum(r, 1e-9)[:, None])
    block = np.select([r < geometry.R_CAVITY, (r < geometry.R_SHIELD) & (r < exit_radius),
                       r < geometry.R_WALL, r < geometry.R_SHIELD, r < geometry.R_AIR],
                      [1, 1, 2, 3, 4], np.nan)
    colors = theme["series"][:4]
    fig, ax = figure(theme, figsize=(6.0, 5.6))
    ax.imshow(block.reshape(n, n), origin="lower", extent=(0, 820, 0, 820),
              cmap=ListedColormap(colors), vmin=0.5, vmax=4.5, interpolation="nearest")
    ax.grid(False)
    labels = ["cavity and ports (vacuum)", "wall, aluminum 5086, 10 cm",
              "shield, borated concrete, 40 cm", "air"]
    legend(ax, theme, handles=[Patch(color=c, label=t) for c, t in zip(colors, labels)],
           loc="upper right")
    ax.set_xlabel("distance from the z axis in the plane x = y (cm)")
    ax.set_ylabel("z (cm)")
    ax.set_title("Target chamber in the half-plane of azimuth 45°", fontsize=10)
    save(fig, out, "geometry", mode)


def cross_section_figure(theme, out, mode, libraries):
    fig, axes = figure(theme, 1, 3, figsize=(10.0, 3.4), sharey=False)
    for ax, material in zip(axes, ("aluminum", "borated_concrete", "air")):
        for k, (label, path) in enumerate(libraries):
            with h5py.File(path, "r") as library:
                edges = np.asarray(library.attrs["group structure"])
                total = library[f"{material}/294K/total"][()]
            steps(ax, total, edges, color=theme["series"][k], linewidth=1.5, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("energy (eV)")
        ax.set_title(material.replace("_", " "), fontsize=10)
    axes[0].set_ylabel("total cross section (1/cm)")
    legend(axes[0], theme, loc="lower left")
    save(fig, out, "cross-sections", mode)


def profiles_figure(theme, out, mode, openmc, runs, exact):
    fig, axes = figure(theme, 1, 3, figsize=(12.0, 3.8))
    abscissae = openmc["line_abscissae"]
    for ax, line in zip(axes, ("port", "shield", "arc")):
        x_points = np.asarray(geometry.line_points()[line][0])
        mean, error = (np.asarray(v) for v in openmc["total"]["lines"][line])
        good = mean > 0
        ax.errorbar(np.asarray(abscissae[line])[good], mean[good], yerr=error[good], fmt="o",
                    markersize=3, color=theme["text"], label="OpenMC, total", zorder=3)
        for k, (label, results, _) in enumerate(runs):
            ax.plot(x_points, np.asarray(results["lines"][line]).sum(axis=1),
                    color=theme["series"][k], linewidth=1.5, label=f"OpenSn, {label}")
        ax.plot(x_points, exact["continuous_energy"]["lines"][line], color=theme["secondary"],
                linewidth=1.0, linestyle="--", label="uncollided, exact")
        ax.set_yscale("log")
        ax.set_title(LINE_TITLES[line], fontsize=9)
        ax.set_xlabel("angle to the port axis (degrees)" if line == "arc" else "radius (cm)")
    axes[0].set_ylabel("scalar flux per source neutron (1/cm²)")
    legend(axes[1], theme, loc="upper right")
    save(fig, out, "profiles", mode)


def regions_figure(theme, out, mode, openmc, runs):
    fig, ax = figure(theme, figsize=(7.0, 3.8))
    quantities = [(b, "flux") for b in BLOCKS] + [("leakage", None)]
    positions = np.arange(len(quantities))
    width = 0.8 / max(1, len(runs))
    for k, (label, results, _) in enumerate(runs):
        values = []
        for block, quantity in quantities:
            if block == "leakage":
                ours, reference = sum(results["leakage_groups"]), openmc["total"]["leakage"][0]
            else:
                ours = sum(results["regions"][block][quantity])
                reference = openmc["total"][block][quantity][0]
            values.append(100.0 * (ours / reference - 1.0))
        ax.bar(positions + (k - (len(runs) - 1) / 2) * width, values, width * 0.9,
               color=theme["series"][k], label=label)
    ax.axhline(0.0, color=theme["secondary"], linewidth=0.8)
    ax.set_xticks(positions, [BLOCK_LABELS.get(b, b) for b, _ in quantities])
    ax.set_ylabel("OpenSn − OpenMC, % of OpenMC")
    ax.set_title("Total flux in each block, and leakage", fontsize=10)
    legend(ax, theme, loc="lower left")
    save(fig, out, "regions", mode)


def spectra_figure(theme, out, mode, openmc, runs):
    edges = openmc["group_edges_eV"]
    lethargy = lethargy_widths(edges)
    fig, axes = figure(theme, 1, 3, figsize=(12.0, 3.6))
    for ax, block in zip(axes, ("wall", "shield", "air")):
        mean = np.asarray(openmc["total"][block]["flux_groups"][0])
        steps(ax, mean / lethargy, edges, color=theme["text"], linewidth=1.2, label="OpenMC")
        for k, (label, results, _) in enumerate(runs):
            groups = np.asarray(results["regions"][block]["flux"])
            run_edges = edges if len(groups) == len(mean) else results["group_edges_eV"]
            steps(ax, groups / lethargy_widths(run_edges), run_edges,
                  color=theme["series"][k], linewidth=1.2, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0e-3, 2.0e7)
        ax.set_xlabel("energy (eV)")
        ax.set_title(block, fontsize=10)
    axes[0].set_ylabel("flux per unit lethargy (cm per source neutron)")
    legend(axes[0], theme, loc="lower left")
    save(fig, out, "spectra", mode)


def maps_figure(theme, out, mode, openmc, runs):
    label, results, _ = runs[0]
    ours = np.asarray(results["map"]).sum(axis=2)  # [z, r], z > 0
    grid = openmc["map_grid"]
    mc = np.asarray(openmc["total"]["map"][0])  # [z, r] over -800..800
    nz = mc.shape[0]
    mc_upper = 0.5 * (mc[nz // 2:] + mc[:nz // 2][::-1])  # the two halves are equivalent
    r_edges = np.asarray(grid["r"])
    z_edges = np.asarray(grid["z"])[nz // 2:]
    positive = np.concatenate([ours[ours > 0], mc_upper[mc_upper > 0]])
    norm = LogNorm(vmin=np.quantile(positive, 0.01), vmax=positive.max())
    fig, axes = figure(theme, 1, 3, figsize=(13.0, 4.2))
    for ax, data, title in ((axes[0], ours, f"OpenSn, {label}"), (axes[1], mc_upper, "OpenMC")):
        mesh = ax.pcolormesh(r_edges, z_edges, np.where(data > 0, data, np.nan), norm=norm,
                             cmap="viridis", shading="flat")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("distance from the z axis (cm)")
    axes[0].set_ylabel("z (cm)")
    colorbar = fig.colorbar(mesh, ax=axes[1], fraction=0.046)
    colorbar.ax.tick_params(colors=theme["secondary"], labelsize=8)
    colorbar.set_label("total flux per source neutron (1/cm²)", color=theme["secondary"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((mc_upper > 0) & (ours > 0), ours / mc_upper, np.nan)
    mesh = axes[2].pcolormesh(r_edges, z_edges, ratio, cmap="RdBu_r", vmin=0.5, vmax=1.5,
                              shading="flat")
    axes[2].set_title("OpenSn / OpenMC", fontsize=10)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("distance from the z axis (cm)")
    colorbar = fig.colorbar(mesh, ax=axes[2], fraction=0.046)
    colorbar.ax.tick_params(colors=theme["secondary"], labelsize=8)
    for ax in axes:
        ax.grid(False)
    save(fig, out, "maps", mode)


def sn_only_figure(theme, out, mode, openmc, runs, sn_only):
    label, results, _ = runs[0]
    fig, ax = figure(theme, figsize=(6.5, 4.0))
    x = np.asarray(geometry.line_points()["arc"][0])
    mean, error = (np.asarray(v) for v in openmc["total"]["lines"]["arc"])
    ax.errorbar(openmc["line_abscissae"]["arc"], mean, yerr=error, fmt="o", markersize=3,
                color=theme["text"], label="OpenMC")
    ax.plot(x, np.asarray(results["lines"]["arc"]).sum(axis=1), color=theme["series"][0],
            linewidth=1.5, label=f"uncollided + collided, {label}")
    ax.plot(x, np.asarray(sn_only["lines"]["arc"]).sum(axis=1), color=theme["series"][1],
            linewidth=1.5, label="discrete ordinates alone")
    ax.set_yscale("log")
    ax.set_xlabel("angle to the port axis (degrees)")
    ax.set_ylabel("total flux per source neutron (1/cm²)")
    ax.set_title("Across the beam of a 23.5° port at r = 650 cm", fontsize=10)
    legend(ax, theme, loc="lower center")
    save(fig, out, "sn-only", mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--openmc", required=True)
    parser.add_argument("--opensn", nargs="+", required=True, metavar="PREFIX:LABEL")
    parser.add_argument("--exact", default="uncollided_reference.json")
    parser.add_argument("--provenance", default="provenance/materials.json")
    parser.add_argument("--sn-only", default=None)
    parser.add_argument("--libraries", nargs="+", default=[
        "64 groups P3:materials_64g_p3.h5",
        "4 groups P0, transport corrected:materials_4g_p0tc.h5"])
    parser.add_argument("--out", default="assets")
    args = parser.parse_args()

    with open(args.openmc) as stream:
        openmc = json.load(stream)
    with open(args.exact) as stream:
        exact = json.load(stream)
    with open(args.provenance) as stream:
        coarse_edges = json.load(stream)["settings"]["few_group_edges_eV"]
    runs = []
    for item in args.opensn:
        prefix, label = item.split(":", 1)
        with open(f"{prefix}_results.json") as stream:
            results = json.load(stream)
        if len(results["regions"][BLOCKS[0]]["flux"]) < 64:
            results["group_edges_eV"] = coarse_edges
        with open(f"{prefix}_uncollided.json") as stream:
            uncollided = json.load(stream)
        runs.append((label, results, uncollided))
    sn_only = None
    if args.sn_only:
        with open(args.sn_only) as stream:
            sn_only = json.load(stream)
    libraries = [tuple(item.split(":", 1)) for item in args.libraries]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for mode, theme in THEMES.items():
        geometry_figure(theme, out, mode)
        cross_section_figure(theme, out, mode, libraries)
        profiles_figure(theme, out, mode, openmc, runs, exact)
        regions_figure(theme, out, mode, openmc, runs)
        spectra_figure(theme, out, mode, openmc, runs)
        if "map" in runs[0][1]:
            maps_figure(theme, out, mode, openmc, runs)
        if sn_only is not None:
            sn_only_figure(theme, out, mode, openmc, runs, sn_only)
    print(f"Wrote the figures to {out}/")


if __name__ == "__main__":
    main()
