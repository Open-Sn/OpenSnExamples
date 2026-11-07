"""
Render the scalar-flux frames written by kobayashi_td.py (KOBAYASHI_TD_VTK_INTERVAL > 0) and
assemble them into kobayashi_td_animation.gif. Render the PNG frames with the VisIt
command-line interface, then assemble the GIF with Python and Pillow:

    visit -nowin -cli -s render_animation.py
    python3 render_animation.py

The source and duct are drawn as a solid, with the flux on the z = 0 reflecting plane and on the
y = 100 cm vacuum boundary where the duct exits. Flux below FLUX_MIN, including the small negative
values near the wavefront, is drawn in gray. PNG frames go to kobayashi_td_png/.
"""

import glob
import math
import os
import re
import sys

VTK_DIR = os.environ.get("KOBAYASHI_TD_VTK_DIR", "kobayashi_td_vtk")
PNG_DIR = "kobayashi_td_png"
GIF_FILE = "kobayashi_td_animation.gif"
FLUX_MIN, FLUX_MAX = 1.0e-5, 1.0e1
WIDTH, HEIGHT = 960, 640
FRAME_MS, LAST_FRAME_MS = 120, 2000
SOURCE_END = 50.0
BENCHMARK_END = 200.0


def add_flux_plot(legend):
    AddPlot("Pseudocolor", "log_phi")
    p = PseudocolorAttributes()
    p.minFlag, p.min = 1, math.log10(FLUX_MIN)
    p.maxFlag, p.max = 1, math.log10(FLUX_MAX)
    p.useBelowMinColor = 1
    p.belowMinColor = (52, 52, 60, 255)
    p.colorTableName = "inferno"
    p.legendFlag = legend
    p.lightingFlag = 0
    SetPlotOptions(p)


def add_slice_plot(axis, intercept):
    add_flux_plot(0)
    AddOperator("Slice")
    s = SliceAttributes()
    s.originType = s.Intercept
    s.axisType = axis
    s.originIntercept = intercept
    s.project2d = 0
    SetOperatorOptions(s)


def render():
    pvd = os.path.join(VTK_DIR, "kobayashi_td.pvd")
    with open(pvd, encoding="utf-8") as stream:
        times = [float(t) for t in re.findall(r'timestep="([^"]+)"', stream.read())]
    os.makedirs(PNG_DIR, exist_ok=True)

    OpenDatabase(pvd)
    DefineScalarExpression("log_phi", "log10(max(phi_g000_m00, 1.0e-30))")

    # Source and duct (blocks 1 and 2).
    add_flux_plot(1)
    AddOperator("Threshold")
    t = ThresholdAttributes()
    t.listedVarNames = ("Block",)
    t.lowerBounds = (0.5,)
    t.upperBounds = (2.5,)
    SetOperatorOptions(t)
    add_slice_plot(SliceAttributes().ZAxis, 0.001)
    add_slice_plot(SliceAttributes().YAxis, 99.999)

    a = AnnotationAttributes()
    a.userInfoFlag = 0
    a.databaseInfoFlag = 0
    a.axes3D.visible = 0
    a.axes3D.triadFlag = 0
    a.axes3D.bboxFlag = 1
    a.backgroundMode = a.Solid
    a.backgroundColor = (24, 24, 28, 255)
    a.foregroundColor = (235, 235, 235, 255)
    SetAnnotationAttributes(a)

    title = CreateAnnotationObject("Text2D")
    title.text = "OpenSn: time-dependent Kobayashi dog-leg"
    title.position = (0.03, 0.93)
    title.height = 0.035
    label = CreateAnnotationObject("Text2D")
    label.position = (0.03, 0.87)
    label.height = 0.03

    DrawPlots()
    legend = GetAnnotationObject(GetPlotList().GetPlots(0).plotName)
    legend.drawTitle = 0
    legend.drawMinMax = 0
    legend.controlTicks = 0
    decades = range(int(round(math.log10(FLUX_MIN))), int(round(math.log10(FLUX_MAX))) + 1)
    legend.suppliedValues = tuple(float(k) for k in decades)
    legend.suppliedLabels = tuple("1e%+d" % k for k in decades)
    legend.drawLabels = legend.Labels
    legend.managePosition = 0
    legend.position = (0.86, 0.80)
    legend.xScale, legend.yScale = 1.0, 1.6
    legend.fontHeight = 0.025

    v = GetView3D()
    v.viewNormal = (-0.55, -0.75, 0.65)
    v.viewUp = (0.0, 0.0, 1.0)
    v.imageZoom = 0.95
    SetView3D(v)

    w = SaveWindowAttributes()
    w.format = w.PNG
    w.outputToCurrentDirectory = 0
    w.outputDirectory = PNG_DIR
    w.family = 0
    w.width, w.height = WIDTH, HEIGHT
    w.resConstraint = w.NoConstraint
    w.screenCapture = 0

    for f in glob.glob(os.path.join(PNG_DIR, "frame_*.png")):
        os.remove(f)
    for i, time in enumerate(times):
        if time > BENCHMARK_END + 1.0e-9:
            break
        SetTimeSliderState(i)
        state = "source on" if time <= SOURCE_END else "source off"
        label.text = "t = %3.0f s   (%s)    scalar flux, n/cm^2/s" % (time, state)
        w.fileName = "frame_%04d" % i
        SetSaveWindowAttributes(w)
        SaveWindow()


def make_gif():
    from PIL import Image  # VisIt's bundled Pillow lacks Image, so this runs outside VisIt

    # One palette, from a late frame with the full color range, keeps the colors from flickering.
    files = sorted(glob.glob(os.path.join(PNG_DIR, "frame_*.png")))
    palette = Image.open(files[3 * len(files) // 4]).convert("RGB").quantize(colors=255)
    frames = [
        Image.open(f).convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE)
        for f in files
    ]
    durations = [FRAME_MS] * (len(frames) - 1) + [LAST_FRAME_MS]
    frames[0].save(
        GIF_FILE, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True
    )
    print(f"{GIF_FILE}: {len(frames)} frames")


if "OpenDatabase" in globals():  # running under VisIt
    render()
    sys.exit(0)
make_gif()
