# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Shared pin-power postprocessing for the VERA Problem 2 lattice cases (2A-2Q).

lbs::VolumePostprocessor batches every fuel pin of a given material into one call (one
RCCLogicalVolume per pin, block_ids restricted to that material, xs_multiplier="kappa-fission"
for a physically-weighted fission-power integral, summed over groups) instead of one
FieldFunctionInterpolationVolume call per pin per group -- e.g. 264 fuel pins * 361 groups = ~95k
individual calls for a Problem 2 lattice, we use 1-2 VolumePostprocessor calls total.

Pin positions (which lattice_csv position each physical pin is in, its exact (x, y) center,
and its symmetry-cut multiplicity correction) are read directly from pin_positions_<case>.csv, a
companion file generate_lattice_meshes.py writes alongside lattice_<case>.obj at mesh-generation
time (see that script's compute_pin_positions), computed once directly from the mesh's own
polygon/vertex data.
"""

import csv

import numpy as np


def load_pin_positions(pin_positions_csv):
    """Reads a pin_positions_<case>.csv (written by generate_lattice_meshes.py's
    compute_pin_positions) into a dict keyed by lattice label, each value a list of
    (row, col, x_center, y_center, multiplier) tuples.
    """
    positions_by_label = {}
    with open(pin_positions_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            positions_by_label.setdefault(row["label"], []).append(
                (
                    int(row["row"]),
                    int(row["col"]),
                    float(row["x_center"]),
                    float(row["y_center"]),
                    float(row["multiplier"]),
                )
            )
    return positions_by_label


def compute_pin_powers(
    phys, RCCLogicalVolume, VolumePostprocessor, pin_positions_csv, label_to_block_xs,
    lattice_shape, fuel_radius=0.4060,
):
    """Computes an unnormalized, kappa-fission-weighted power value for every fuel pin listed in
    pin_positions_csv for a fuel-bearing label.

    Parameters
    ----------
    phys : pyopensn.solver.DiscreteOrdinatesProblem
        The solved problem. VolumePostprocessor reads its current flux at Execute() time.
    RCCLogicalVolume, VolumePostprocessor
        The pyopensn.logvol.RCCLogicalVolume and pyopensn.post.VolumePostprocessor classes,
        passed in by the caller rather than imported here: under the embedded OpenSn console,
        pyopensn classes are pre-bound in the entrypoint script's own namespace, and a *fresh*
        `from pyopensn... import ...` from this separate module re-triggers pybind11's
        duplicate-type-registration error -- this module has no namespace-independent way to
        tell whether it's safe to import them itself, so it never tries.
    pin_positions_csv : str or Path
        Path to the case's pin_positions_<case>.csv (see generate_lattice_meshes.py's
        compute_pin_positions), giving every real lattice position's exact center and
        symmetry-cut multiplicity correction.
    label_to_block_xs : dict[str, tuple[int, pyopensn.xs.MultiGroupXS]]
        Maps every FUEL-bearing lattice label to its (block_id, xs object). Each xs object must
        have been loaded with extra_xs_names=["kappa-fission"] (only present for fissionable
        materials and absent for clad/gap/moderator/etc). Non-fuel labels (guide tubes,
        instrument tubes, water, poison rods, ...) are simply absent from this dict and are
        skipped.
    lattice_shape : tuple[int, int]
        Shape of the returned table -- pass lattice_csv.shape.
    fuel_radius : float
        Fuel pellet outer radius, cm (VERA Problem 2 spec: 0.4060 cm). Only used to size the
        RCCLogicalVolume used to isolate each pin's own fuel-material cells from its neighbors';
        block_ids already restricts the selection to one material, so this just needs to be big
        enough to contain that one pin's cells and small enough not to reach the next pin's --
        exact centering precision doesn't matter for that purpose.

    Returns
    -------
    np.ndarray
        Shape lattice_shape: the kappa-fission-weighted flux integral (already corrected for
        symmetry-cut multiplicity) at every fuel position listed in pin_positions_csv (zero
        elsewhere). Callers apply their own normalization convention.
    """
    positions_by_label = load_pin_positions(pin_positions_csv)

    val_table = np.zeros(lattice_shape)
    for label, (block_id, _xs_obj) in label_to_block_xs.items():
        entries = positions_by_label.get(label, [])
        if not entries:
            continue

        logical_volumes = [
            RCCLogicalVolume(r=fuel_radius, x0=cx, y0=cy, z0=-1.0, vz=2.0)
            for _row, _col, cx, cy, _multiplier in entries
        ]
        pps = VolumePostprocessor(
            problem=phys,
            value_type="integral",
            block_ids=[block_id],
            logical_volumes=logical_volumes,
            xs_multiplier="kappa-fission",
        )
        pps.Execute()
        multipliers = np.array([multiplier for *_rest, multiplier in entries])
        powers = pps.GetValue().sum(axis=1) * multipliers  # sum over groups

        for (row, col, _cx, _cy, _multiplier), power in zip(entries, powers):
            val_table[row, col] = power

    return val_table
