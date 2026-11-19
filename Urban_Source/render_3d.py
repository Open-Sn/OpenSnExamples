#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Draw the effective dose rate (AP) of the OpenSn solution on the surfaces of the buildings and
the ground, in light and dark variants, to assets/dose-3d-light.png and assets/dose-3d-dark.png.

    python3 render_3d.py --flux urban_source_flux.pvtu --opensn urban_source_results.json

The dose rate of a cell is the mean of its four PWLD nodal values of the group fluxes times the
group dose coefficients, for the 1 Ci source of city.toml. The air is left out. Needs PyVista,
which renders off screen, and Matplotlib.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402

pv.OFF_SCREEN = True

THEMES = {
    "light": dict(background="#ffffff", title="#1f2328", text="#57606a", ramp=(0.0, 0.92)),
    "dark": dict(background="#0d1b29", title="#f1f6fb", text="#aabcca", ramp=(0.30, 1.0)),
}
DECADES = 5


def colormap(ramp):
    """The part of inferno that shows on the background of the theme."""
    return ListedColormap(plt.get_cmap("inferno")(np.linspace(ramp[0], ramp[1], 256)))


def solid_surface(flux_file, results, description):
    """Surface of the buildings and the ground with the cell-mean dose rate in uSv/h."""
    grid = pv.read(flux_file)
    if isinstance(grid, pv.MultiBlock):
        grid = grid.combine()
    coefficients = np.asarray(results["dose_coefficients"])
    photons_per_second = description["activity_Bq"] * description["photons_per_decay"]
    to_rate = photons_per_second * 3600.0 * 1.0e-6 / results["source_rate"]  # pSv -> uSv/h
    dose = np.zeros(grid.n_points)
    for g, coefficient in enumerate(coefficients):
        dose += np.asarray(grid.point_data[f"phi_g{g:03d}_m00"]) * coefficient
    grid.point_data["dose_rate"] = dose * to_rate
    air = [int(k) for k, v in description["blocks"].items() if v["material"] == "air"]
    solid = grid.extract_cells(np.flatnonzero(~np.isin(grid.cell_data["Block"], air)))
    for name in [n for n in solid.point_data.keys() if n != "dose_rate"]:
        solid.point_data.remove(name)
    # Each cell has its own nodes, so averaging them gives the cell mean of a tetrahedron.
    cells = solid.point_data_to_cell_data()
    return cells.extract_surface(algorithm="dataset_surface"), grid.n_cells


def render(surface, theme, norm, size):
    """Off-screen image of the surface, seen from the south-west, about 55 degrees up."""
    values = np.clip(np.asarray(surface.cell_data["dose_rate"]), norm.vmin, None)
    surface.cell_data["dose_rate_clipped"] = values
    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background(theme["background"])
    plotter.add_mesh(surface, scalars="dose_rate_clipped", preference="cell", log_scale=True,
                     cmap=colormap(theme["ramp"]), clim=[norm.vmin, norm.vmax],
                     show_scalar_bar=False, ambient=0.25, diffuse=0.8, specular=0.0)
    plotter.view_vector((-0.35, -0.55, 1.05), viewup=(0.0, 0.0, 1.0))
    plotter.reset_camera()
    plotter.camera.zoom(1.28)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def compose(image, theme, norm, lines, output):
    """Title, image, description, and color bar, as in the other examples' figures."""
    fig = plt.figure(figsize=(16, 14), dpi=100, facecolor=theme["background"])
    fig.text(0.5, 0.965, "Effective Dose Rate from OpenSn", ha="center", va="center",
             fontsize=30, fontweight="bold", color=theme["title"])
    ax = fig.add_axes([0.03, 0.17, 0.94, 0.75])
    ax.imshow(image)
    ax.set_axis_off()
    for i, line in enumerate(lines):
        fig.text(0.04, 0.135 - 0.024 * i, line, ha="left", va="center",
                 fontsize=16 if i == 0 else 14, color=theme["text"])
    cax = fig.add_axes([0.04, 0.045, 0.625, 0.012])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap(theme["ramp"])), cax=cax,
                       orientation="horizontal")
    bar.outline.set_visible(False)
    bar.set_ticks([norm.vmin, norm.vmax])
    bar.set_ticklabels([f"$10^{{{int(round(np.log10(norm.vmin)))}}}$",
                        f"$10^{{{int(round(np.log10(norm.vmax)))}}}$"])
    cax.tick_params(colors=theme["text"], labelsize=14, length=0, pad=8)
    bar.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.savefig(output, facecolor=theme["background"])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flux", type=Path, default=Path("urban_source_flux.pvtu"))
    parser.add_argument("--opensn", type=Path, default=Path("urban_source_results.json"))
    parser.add_argument("--mesh-json", type=Path, default=Path("urban_source_mesh.json"))
    parser.add_argument("--directions", type=int, default=3168,
                        help="number of directions of the run, for the description")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    args = parser.parse_args()

    results = json.loads(args.opensn.read_text())
    description = json.loads(args.mesh_json.read_text())
    surface, num_cells = solid_surface(args.flux, results, description)
    values = np.asarray(surface.cell_data["dose_rate"])
    top = 10.0 ** np.ceil(np.log10(np.percentile(values[values > 0], 99.95)))
    norm = LogNorm(vmin=top / 10.0 ** DECADES, vmax=top)
    settings = results["settings"]
    lines = [f"208 m × 208 m city · {num_cells:,} cells · {results['num_groups']} groups · "
             f"P{settings['scattering_order']}",
             f"{args.directions:,} directions · uncollided plus collided dose · "
             "1 Ci Cs-137 source",
             "Effective dose rate (AP) in µSv/h on the buildings and the ground · "
             "logarithmic scale"]
    args.assets.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        image = render(surface.copy(), theme, norm, (1600, 1150))
        output = args.assets / f"dose-3d-{name}.png"
        compose(image, theme, norm, lines, output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
