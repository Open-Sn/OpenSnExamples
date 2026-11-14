# Runbook: the 64-group fine-mesh run of the ICF target chamber on a cluster

This runbook takes you from nothing to the 64-group P3 solution on the fine mesh (1.3 M
tetrahedra) and its comparison with OpenMC. [README.md](README.md) in this directory explains
the memory model and the choices behind the defaults; the example itself is described in
[../README.md](../README.md).

You need two repositories: an OpenSn branch that is not merged yet, and this example.

## A. Build OpenSn from the branch with the new uncollided-flux method

The example places the point source on the corner of three reflecting planes. Only the branch
`uncollided-hull-accel` of the fork https://github.com/ragusa/opensn accepts that.

```
git clone https://github.com/ragusa/opensn.git opensn-chamber
cd opensn-chamber
git checkout uncollided-hull-accel
git log -1 --format='%h %s'
#   584810e0e Document point sources on reflecting planes     (or a later commit)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOPENSN_WITH_PYTHON=ON
cmake --build build -j 32
```

- Dependencies are listed in `doc/source/install/install.rst`: a C++20 compiler, MPI,
  CMake 3.29 or later, PETSc 3.17 or later, Boost 1.86 or later, HDF5 1.14 or later, VTK 9.3
  or later, and Caliper.
- The Python of the `opensn` console must provide `numpy` and `mpi4py`, because `chamber.py`
  imports them.
- The console is `build/python/opensn`.

Run the uncollided regression tests. They must all pass.

```
test/run_tests -d test/python/modules/linear_boltzmann_solvers/transport_uncollided \
    -j 16 --exe build/python/opensn
```

## B. Get the example

```
git clone -b icf-chamber https://github.com/ragusa/OpenSnExamples.git
cd OpenSnExamples/ICF_Target_Chamber
conda env create -f environment.yml        # creates the environment "chamber"
```

- The `chamber` environment provides Gmsh 4.15.2, numpy, h5py, matplotlib and OpenMC 0.16.0.
- Without the optional OpenMC step (section F), any Python works if it has `gmsh==4.15.2`,
  `numpy`, `h5py` and `matplotlib`.
- The repository already holds:
  - the cross-section libraries (`materials_*.h5`);
  - the OpenMC reference (`openmc_reference_results.json`);
  - the semi-analytic uncollided flux (`uncollided_reference.json`).

  Do not regenerate them. Nuclear data are needed only for section F.

## C. Configure

1. In `large_machine/env.sh`, set the lines marked `CHANGE_ME`:
   - `RUN_DIR`: a scratch file system. The 64-group stage-1 file is tens of GB.
   - `OPENSN`: the path to `build/python/opensn` from section A.
   - `LAUNCH`: `srun`, or `mpiexec -n <ranks>`.
   - `CHAMBER_PYTHON`: the Python of section B.
   - `OPENMC_CROSS_SECTIONS`: only for section F.
2. In the `#SBATCH` header of every `*.sbatch` file, set `--account` and `--partition`.

The other variables of `env.sh` set the production run: the `--size 10 --port-size 3` mesh,
the 64-group P3 library, and 6 × 24 = 144 directions.

## D. Acceptance test: reproduce a committed result (about 5 minutes on 32 cores)

This checks the build of section A before you spend allocation on it. From
`ICF_Target_Chamber`, on one node:

```
python generate_mesh.py --size 15 --out chamber_coarse.msh
mpiexec -n 32 /path/to/build/python/opensn -i chamber.py \
    --py mesh_file=\"chamber_coarse.msh\" --py library=\"4g_p0tc\" --py prefix=\"accept_4g_coarse\"
python compare_openmc.py --openmc openmc_reference_results.json --opensn accept_4g_coarse \
    --exact uncollided_reference.json --provenance provenance/materials.json
```

The comparison should print these differences of the total flux with OpenMC:

| Cavity and ports | Wall | Shield | Air | Leakage |
|---|---|---|---|---|
| +8.4% | +7.9% | +2.0% | −23.4% | −21.6% |

- These are the 4-group, coarse-mesh results of the README.
- The number of ranks does not change them beyond about 1e-8.
- The log of the collided stage must give a production rate of 0.125, which is one eighth of
  the source.

## E. Pilot of the 64-group library (recommended)

Stage 2 with 64 groups has been measured only at 144 directions. To use another quadrature,
first run the `--size 20` mesh (61,368 tetrahedra) on one node at that quadrature:

- record the peak memory with `sacct -j <job> --format=JobID,Elapsed,MaxRSS`;
- scale it by the number of tetrahedra.

At 144 directions the pilot uses about 0.9 to 1.0 MB per tetrahedron. It gives these
differences with OpenMC:

| Cavity and ports | Wall | Shield | Air | Leakage |
|---|---|---|---|---|
| +0.3% | −1.2% | +1.1% | −8.0% | −7.8% |

## F. Production

Submit from `large_machine/`, because the scripts read the `env.sh` next to them:

```
cd large_machine
j0=$(sbatch --parsable 00_mesh.sbatch)                                 # 1 core, about 1 h
j1=$(sbatch --parsable --dependency=afterok:$j0 01_uncollided.sbatch)  # 4 nodes x 32 ranks
j2=$(sbatch --parsable --dependency=afterok:$j1 02_collided.sbatch)    # 16 nodes x 32 ranks
```

When `$j2` has finished, run `./04_compare.sh` (comparison and figures, section G), and
optionally `./05_visit.sh` (three-dimensional views with VisIt).

What to expect:

- **Mesh (step 0):** about 1 h, on one thread, because Gmsh's size field along the port beams
  is slow.
- **Stage 1 (uncollided):** short, minutes at most. It holds about 74 kB per tetrahedron.
  Rank 0 gathers the whole file before writing it, so the node of rank 0 needs tens of GB of
  headroom.
- **Stage 2 (collided):** about 1.2 to 1.3 TB in total for 1.3 M tetrahedra at 144
  directions.
  - This is the measured 0.9 to 1.0 MB per tetrahedron, extrapolated.
  - Eight nodes of 256 GB are the minimum; the script asks for 16.
  - The wall time was not measured at this size. It should be well under the 12 h limit; this
    is an estimate.

### The optional OpenMC reference (step 3): read this first

The committed reference (10^9 histories) is enough for the comparison, so step 3 is optional.
It is written for the machine where the example was developed. That machine had OpenMC 0.16.0
built **without MPI**, with OpenMP threads, from conda-forge (`nodagmc_nompi`) or from a
module. The OpenMC on your cluster may be built differently. Check these points:

1. **MPI or not.** `03_openmc.sbatch` passes `--processes $SLURM_NNODES`, and
   `openmc_reference.py` then launches `mpiexec -n <processes> openmc`. This is right only
   for an OpenMC built with MPI.
   - With a non-MPI build, every process runs the whole problem with the same seed. The
     result is several identical runs, not one larger run.
   - With a non-MPI build, use `--nodes=1` and `--processes 1`, and all the cores of the node
     as threads.
   - If your cluster launches MPI with `srun` instead of `mpiexec`, adapt the `mpi_args` line
     of `openmc_reference.py`.
2. **The `openmc` executable.** `openmc_reference.py` runs the `openmc` that it finds on
   `PATH`. Calling `$CHAMBER_PYTHON` does not put the conda environment's `openmc` there:
   activate the environment, or load your cluster's OpenMC module.
3. **The nuclear data.** Use the ENDF/B-VIII.0 HDF5 library distributed by the OpenMC
   project, the one used for the libraries and the committed reference. Another evaluation
   gives a reference that is not comparable.
4. **Time.** 10^9 histories took 3.1 h on 30 threads. The 10^10 of the script would take
   roughly 10 to 15 h on one node of 64 to 128 cores (an estimate), which exceeds the 12 h of
   the script on one node.
   - Either raise the time limit, or lower `--particles` or `--batches`.
   - Each batch must still be a large number of histories, because the standard errors come
     from the batch-to-batch spread.
5. **Already handled in the scripts, for information.** OpenMC 0.16.0 lost tracks in
   tracklength tallies on spherical and cylindrical meshes that cover only part of the
   angles. The tally meshes of `openmc_reference.py` therefore cover the whole sphere. Keep
   them that way if you change the tallies.

To compare with your own reference, give `04_compare.sh` its results file:
`REFERENCE=$RUN_DIR/openmc_seed20261114/results.json ./04_compare.sh`.

## G. Figures

Two scripts make the figures, written in light and dark versions to `$RUN_DIR/assets/`.

**`./04_compare.sh`** prints the comparison with OpenMC, then runs `make_figures.py`
(Matplotlib, no VisIt) on the fine run. It makes the same figures as the README of the
example, with this run in place of the README's runs:

| File | Figure |
|---|---|
| `geometry-*.png` | the regions and the ports |
| `cross-sections-*.png` | the total and absorption cross sections of the libraries |
| `profiles-*.png` | the flux along the three line-outs (port axis, shield, across a beam), OpenSn against OpenMC |
| `regions-*.png` | the difference with OpenMC of the total flux in each block and of the leakage |
| `spectra-*.png` | the flux spectrum in each block, OpenSn against OpenMC |
| `maps-*.png` | the total flux in the half-plane of azimuth 45 degrees, OpenSn, OpenMC and their ratio |

It takes seconds (4 s here for a 64-group run) and reads only the small JSON files of the
run, so a login node is fine.

**`./05_visit.sh`** (optional) renders `<prefix>_flux.pvtu` with VisIt (tested with VisIt
3.5.0), through [../visit_figures.py](../visit_figures.py):

- `<prefix>_flux3d-outside-*.png`: the outer sphere, where the beams of the ports arrive;
- `<prefix>_flux3d-planes-*.png`: the three symmetry planes, through the cavity, the wall, the
  shield, the air and the ports.

The color is 4πr² times the total flux, on a log scale over three decades below its maximum.
This product is 1 for the uncollided flux in a void, so the views show the attenuation and the
port beams. The spots on the outer sphere between the ports are the ray effects of the
collided flux at 144 directions. Before running it:

- set `VISIT` in `env.sh` to the VisIt command of your machine, after any `module load`;
- run it on a compute node: the `.pvtu` of the fine 64-group run is several GB;
- for a parallel engine, set `VISIT_ARGS="-np 32"`, for example.

On a 1.3 M-tetrahedron, 4-group file, a serial engine took 12 s here. The fine 64-group file
was not tried; we expect minutes (an estimate).

`visit_figures.py --help`, run through `visit -nowin -cli -s`, lists the options:
`--quantity flux` plots the flux itself, and `--min`, `--max` and `--range` set the color
scale.

To redraw the README figures with this run next to ours, we need the JSON files listed in
section I.

## H. Checks

- The log of stage 1 prints "Conservation error" at about 1e-15.
- The log of stage 2 prints a relative balance of about 1e-8 and a production rate of 0.125.
- Every groupset converges: the log prints "status = converged".
- `04_compare.sh` prints the uncollided flux in each block against the semi-analytic values.
  On this mesh the differences should be at most about 0.15%; they are the faceting of the
  ports.

## I. What to send back

- `<prefix>.uncollided.log` and `<prefix>.collided.log`
- `<prefix>_results.json`, `<prefix>_uncollided.json` and `<prefix>.comparison.txt`; with the
  two JSON files we can redraw every figure of the README with this run
- the `assets/` directory written by `04_compare.sh` and `05_visit.sh`
- the output of `sacct -j <job> --format=JobID,Elapsed,MaxRSS,NNodes` for the mesh, stage-1
  and stage-2 jobs

Do not send the `*.h5` and `*.pvtu` files: they are large and can be regenerated.
