# SPDX-License-Identifier: MIT
"""
Three-dimensional views of the total flux of a chamber.py run, rendered with VisIt.

chamber.py writes the total flux (collided plus uncollided) of every group to
<prefix>_flux.pvtu, one piece per rank, per neutron of the whole source. This script sums the
groups and draws, on a logarithmic color scale, either the flux times 4 pi r^2 (the default,
which is 1 for the uncollided flux in a void, so that what is left shows the attenuation, the
scattering and the beams of the ports) or the flux itself. It writes two views of the octant,
each in a light and a dark version, to the output directory:

    <name>-outside-<theme>.png   the outer sphere (r = 800 cm), where the beams of the ports
                                 arrive, seen from outside the octant
    <name>-planes-<theme>.png    the three symmetry planes, which cut through the cavity, the
                                 wall, the shield, the air and the ports in the planes

Run it with VisIt's command-line interface, without a window:

    visit -nowin -cli -s visit_figures.py --pvtu chamber_4g_fine_flux.pvtu --out assets

With many ranks, a parallel engine reads the pieces faster: add -np <ranks> before -s (it
needs a VisIt built with MPI). The other options are listed by --help.
"""

import argparse
import math
import os
import re
import sys

import visit

THEMES = {
    "light": {"surface": (252, 252, 251), "text": (26, 26, 25)},
    "dark": {"surface": (26, 26, 25), "text": (240, 239, 236)},
}
VIEWS = {
    "outside": ((1.0, 0.8, 0.6), 1.1),
    "planes": ((-1.0, -0.8, -0.6), 0.9),
}
QUANTITY_TITLES = {
    "r2flux": "4 pi r² × total flux per source neutron",
    "flux": "total flux per source neutron (1/cm²)",
}


def parse_arguments():
    parser = argparse.ArgumentParser(prog="visit_figures.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pvtu", required=True, help="<prefix>_flux.pvtu of chamber.py")
    parser.add_argument("--out", default="assets", help="Output directory")
    parser.add_argument("--name", default="flux3d", help="Start of the image file names")
    parser.add_argument("--quantity", choices=sorted(QUANTITY_TITLES), default="r2flux")
    parser.add_argument("--min", type=float, default=None,
                        help="Lower end of the color scale (default: the maximum / --range)")
    parser.add_argument("--max", type=float, default=None,
                        help="Upper end of the color scale (default: the maximum)")
    parser.add_argument("--range", type=float, default=1.0e3,
                        help="Ratio of the maximum to the default lower end")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    # With a parallel engine (-np), VisIt appends its own launch arguments, which are ignored.
    args, _ = parser.parse_known_args(list(visit.Argv()))
    return args


def group_fields(database):
    """Names of the scalar flux of each group, phi_gGGG_m00, in group order."""
    metadata = visit.GetMetaData(database)
    names = [metadata.GetScalars(i).name for i in range(metadata.GetNumScalars())]
    groups = sorted(name for name in names if re.fullmatch(r"phi_g\d{3}_m00", name))
    if not groups:
        raise RuntimeError(f"no phi_gGGG_m00 field in {database}")
    return metadata.GetMeshes(0).name, groups


def define_quantity(mesh, groups, quantity):
    visit.DefineScalarExpression("total_flux", " + ".join(groups))
    if quantity == "flux":
        return "total_flux"
    x, y, z = (f"coord({mesh})[{i}]" for i in range(3))
    visit.DefineScalarExpression(
        "r2flux", f"4.0 * {math.pi!r} * ({x} * {x} + {y} * {y} + {z} * {z}) * total_flux")
    return "r2flux"


def color_range(args):
    """The color scale: the maximum of the quantity, and the maximum / --range below it."""
    visit.Query("Max")
    maximum = visit.GetQueryOutputValue()
    upper = args.max if args.max is not None else maximum
    lower = args.min if args.min is not None else upper / args.range
    if not 0.0 < lower < upper:
        raise RuntimeError(f"invalid color range [{lower}, {upper}] (maximum {maximum})")
    return lower, upper


def set_plot(variable, lower, upper):
    attributes = visit.PseudocolorAttributes()
    attributes.scaling = attributes.Log
    attributes.minFlag, attributes.min = 1, lower
    attributes.maxFlag, attributes.max = 1, upper
    attributes.colorTableName = "viridis"
    attributes.lightingFlag = 0
    visit.SetPlotOptions(attributes)


def set_annotations(theme, title):
    colors = THEMES[theme]
    annotation = visit.AnnotationAttributes()
    annotation.userInfoFlag = 0
    annotation.databaseInfoFlag = 0
    annotation.axes3D.visible = 0
    annotation.axes3D.triadFlag = 1
    annotation.axes3D.bboxFlag = 0
    annotation.backgroundMode = annotation.Solid
    annotation.backgroundColor = colors["surface"] + (255,)
    annotation.foregroundColor = colors["text"] + (255,)
    visit.SetAnnotationAttributes(annotation)
    legend = visit.GetAnnotationObject(visit.GetPlotList().GetPlots(0).plotName)
    legend.drawTitle = 0
    legend.drawMinMax = 0
    legend.numberFormat = "%.0e"
    legend.fontHeight = 0.025
    legend.managePosition = 0
    legend.position = (0.04, 0.92)
    legend.useForegroundForTextColor = 1
    for text in visit.GetAnnotationObjectNames():
        if text.startswith("chamber_title"):
            visit.GetAnnotationObject(text).Delete()
    label = visit.CreateAnnotationObject("Text2D", "chamber_title")
    label.text = title
    label.position = (0.04, 0.95)
    label.height = 0.025
    label.useForegroundForTextColor = 1


def set_view(normal, zoom):
    visit.ResetView()
    view = visit.GetView3D()
    norm = math.sqrt(sum(c * c for c in normal))
    view.viewNormal = tuple(c / norm for c in normal)
    view.viewUp = (0.0, 0.0, 1.0)
    view.imageZoom = zoom
    visit.SetView3D(view)


def save(out, file_name, args):
    window = visit.SaveWindowAttributes()
    window.outputToCurrentDirectory = 0
    window.outputDirectory = out
    window.fileName = file_name
    window.family = 0
    window.format = window.PNG
    window.width, window.height = args.width, args.height
    window.resConstraint = window.NoConstraint
    visit.SetSaveWindowAttributes(window)
    visit.SaveWindow()


def main():
    args = parse_arguments()
    database = os.path.abspath(args.pvtu)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if not visit.OpenDatabase(database):
        raise RuntimeError(f"VisIt cannot open {database}")
    mesh, groups = group_fields(database)
    variable = define_quantity(mesh, groups, args.quantity)
    visit.AddPlot("Pseudocolor", variable)
    visit.DrawPlots()
    lower, upper = color_range(args)
    set_plot(variable, lower, upper)
    visit.DrawPlots()
    print(f"{len(groups)} groups, {args.quantity} from {lower:.3e} to {upper:.3e}")
    for theme in THEMES:
        set_annotations(theme, QUANTITY_TITLES[args.quantity])
        for view_name, (normal, zoom) in VIEWS.items():
            set_view(normal, zoom)
            file_name = f"{args.name}-{view_name}-{theme}"
            save(out, file_name, args)
            print(f"wrote {os.path.join(out, file_name)}.png")


main()
sys.exit(0)
