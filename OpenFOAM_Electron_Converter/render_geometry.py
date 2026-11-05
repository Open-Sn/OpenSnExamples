#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Render a cutaway of the mesh shared by OpenSn and OpenFOAM."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import meshio
import numpy as np
import vtk
from vtk.util.numpy_support import (
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
    vtk_to_numpy,
)

plt.switch_backend("Agg")


HERE = Path(__file__).resolve().parent
MATERIALS = {
    101: ("tungsten target", "#555b66", 1.0),
    102: ("copper converter", "#dc5c14", 0.30),
    105: ("steel shield", "#33619e", 0.48),
    106: ("water channel", "#0bacef", 0.92),
}
SECTION_MATERIALS = {
    104: ("air", "#f4f4f0"),
    103: ("aluminum housing", "#c7cbd1"),
    102: ("copper converter", "#dc5c14"),
    105: ("steel shield", "#33619e"),
    106: ("water channel", "#0bacef"),
    101: ("tungsten target", "#555b66"),
}


def surface_triangles(grid, block_id):
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(grid)
    threshold.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Block"
    )
    threshold.SetLowerThreshold(block_id)
    threshold.SetUpperThreshold(block_id)
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputConnection(threshold.GetOutputPort())
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(surface.GetOutputPort())
    triangulate.Update()
    poly = triangulate.GetOutput()
    points = vtk_to_numpy(poly.GetPoints().GetData()) * 1000.0
    cells = vtk_to_numpy(poly.GetPolys().GetData()).reshape((-1, 4))[:, 1:]
    return points[cells]


def section_triangles(grid, block_id):
    """Return triangles cut through the target and off-center coolant channel."""
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(grid)
    threshold.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Block"
    )
    threshold.SetLowerThreshold(block_id)
    threshold.SetUpperThreshold(block_id)
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, -0.006)
    plane.SetNormal(0.0, 0.0, 1.0)
    cutter = vtk.vtkCutter()
    cutter.SetInputConnection(threshold.GetOutputPort())
    cutter.SetCutFunction(plane)
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(cutter.GetOutputPort())
    triangulate.Update()
    poly = triangulate.GetOutput()
    points = vtk_to_numpy(poly.GetPoints().GetData()) * 1000.0
    cells = vtk_to_numpy(poly.GetPolys().GetData()).reshape((-1, 4))[:, 1:]
    return points[cells]


def add_domain_box(axis, bounds):
    x0, x1, y0, y1, z0, z1 = np.asarray(bounds) * 1000.0
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for i, j in edges:
        axis.plot(*zip(corners[i], corners[j]), color="0.4", alpha=0.35, lw=0.8)


def read_gmsh_grid(path):
    """Build a VTK grid from the current Gmsh mesh, independent of results."""
    mesh = meshio.read(path)
    tetrahedra = np.asarray(mesh.cells_dict["tetra"], dtype=np.int64)
    block_ids = np.asarray(
        mesh.cell_data_dict["gmsh:physical"]["tetra"], dtype=np.int32
    )

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.points), deep=True))
    connectivity = np.column_stack(
        (np.full(len(tetrahedra), 4, dtype=np.int64), tetrahedra)
    )
    cells = vtk.vtkCellArray()
    cells.SetCells(
        len(tetrahedra), numpy_to_vtkIdTypeArray(connectivity.ravel(), deep=True)
    )
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(vtk.VTK_TETRA, cells)
    vtk_blocks = numpy_to_vtk(block_ids, deep=True)
    vtk_blocks.SetName("Block")
    grid.GetCellData().AddArray(vtk_blocks)
    return grid


def main():
    grid = read_gmsh_grid(HERE / "waterblock.msh")

    triangles = {
        block_id: surface_triangles(grid, block_id) for block_id in MATERIALS
    }
    figure = plt.figure(figsize=(18, 9), dpi=120)
    full_axis = figure.add_subplot(121, projection="3d")
    detail_axis = figure.add_subplot(122, projection="3d")

    for block_id, (_, color, opacity) in MATERIALS.items():
        collection = Poly3DCollection(
            triangles[block_id],
            facecolor=color,
            edgecolor=(0.10, 0.10, 0.10, min(0.55, opacity)),
            linewidth=0.18,
            alpha=opacity,
        )
        full_axis.add_collection3d(collection)
    add_domain_box(full_axis, grid.GetBounds())
    full_axis.set_xlim(-12, 60)
    full_axis.set_ylim(-25, 25)
    full_axis.set_zlim(-25, 25)
    full_axis.set_box_aspect((72, 50, 50))
    full_axis.view_init(elev=24, azim=-126)
    full_axis.set_title("Full converter cutaway", fontsize=15, pad=18)

    for block_id in (101, 106):
        name, color, _ = MATERIALS[block_id]
        collection = Poly3DCollection(
            triangles[block_id],
            facecolor=color,
            edgecolor="0.12",
            linewidth=0.35,
            alpha=1.0,
        )
        detail_axis.add_collection3d(collection)
    detail_axis.set_xlim(-5, 30)
    detail_axis.set_ylim(-27, 20)
    detail_axis.set_zlim(-15, 15)
    detail_axis.set_box_aspect((35, 47, 30))
    detail_axis.view_init(elev=25, azim=-68)
    detail_axis.set_title("Internal target and U-shaped coolant path", fontsize=15, pad=18)

    for axis in (full_axis, detail_axis):
        axis.set_xlabel("x [mm]", labelpad=10)
        axis.set_ylabel("y [mm]", labelpad=10)
        axis.set_zlabel("z [mm]", labelpad=10)
        axis.grid(False)
    legend = [
        Patch(facecolor=color, edgecolor="0.2", label=name, alpha=max(opacity, 0.45))
        for name, color, opacity in MATERIALS.values()
    ]
    full_axis.legend(handles=legend, loc="upper left", framealpha=0.92, fontsize=10)
    figure.suptitle(
        "Electron–photon converter: shared OpenSn/OpenFOAM tetrahedral mesh",
        fontsize=19,
        y=0.98,
    )
    figure.text(
        0.50,
        0.025,
        f"{grid.GetNumberOfCells()} tetrahedra • air and aluminum housing omitted "
        "for cutaway view\n"
        "beam enters from −x; U-shaped water channel shown in cyan",
        ha="center",
        va="center",
        fontsize=11,
        color="0.25",
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    output = HERE / "waterblock_geometry_mesh.png"
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    print(f"Wrote {output}")

    section_figure, section_axis = plt.subplots(figsize=(13, 8), dpi=140)
    for block_id, (name, color) in SECTION_MATERIALS.items():
        section = section_triangles(grid, block_id)
        section_axis.add_collection(
            PolyCollection(
                section[:, :, :2],
                facecolor=color,
                edgecolor=(0.12, 0.12, 0.12, 0.50),
                linewidth=0.25,
                label=name,
            )
        )
    section_axis.annotate(
        "1 MeV electron beam",
        xy=(-2.5, 0.0),
        xytext=(-11.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#9c1c1c"},
        color="#9c1c1c",
        fontsize=12,
        ha="left",
        va="bottom",
    )
    section_axis.set_xlim(-12, 62)
    section_axis.set_ylim(-27, 27)
    section_axis.set_aspect("equal")
    section_axis.set_xlabel("x [mm]")
    section_axis.set_ylabel("y [mm]")
    section_axis.set_title(
        "Material and mesh section through the electron–photon converter (z = -6 mm)",
        fontsize=15,
        pad=14,
    )
    section_axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="0.25", label=name)
            for name, color in SECTION_MATERIALS.values()
        ],
        loc="upper right",
        framealpha=0.95,
        ncol=2,
    )
    section_axis.grid(color="0.8", linewidth=0.5, alpha=0.45)
    section_figure.tight_layout()
    section_output = HERE / "waterblock_mesh_section.png"
    section_figure.savefig(section_output, bbox_inches="tight", facecolor="white")
    print(f"Wrote {section_output}")


if __name__ == "__main__":
    main()
