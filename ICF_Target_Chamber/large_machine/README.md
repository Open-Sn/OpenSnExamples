# Running the target chamber at higher resolution

The results of this example were computed on one 32-core node with 250 GB of memory. There,
the 64-group P3 library fits only on a coarse mesh, and the fine mesh that resolves the port
beams is run with the 4-group library. This directory holds what is needed to run the 64-group
P3 library on a fine mesh with a larger quadrature, and a longer OpenMC reference, on a cluster.

Everything else (the libraries, the semi-analytic uncollided flux, the comparison and figure
scripts) is in the parent directory and does not need to be regenerated.

[RUNBOOK.md](RUNBOOK.md) gives the whole sequence, from building OpenSn to the files to send
back, with the notes on running OpenMC on another cluster.

## Steps

Edit the lines marked `CHANGE_ME` in `env.sh` and in the `#SBATCH` headers, then, from this
directory:

| Step | Command | What it does |
|---|---|---|
| 0 | `sbatch 00_mesh.sbatch` | fine mesh with `generate_mesh.py` (one thread, hours) |
| 1 | `sbatch 01_uncollided.sbatch` | stage 1: uncollided flux, written to `<prefix>_uncollided.h5` |
| 2 | `sbatch 02_collided.sbatch` | stage 2: collided and total flux, `<prefix>_results.json` |
| 3 | `sbatch 03_openmc.sbatch` | optional: OpenMC reference with 10 times the histories |
| 4 | `./04_compare.sh` | comparison with OpenMC and figures |
| 5 | `./05_visit.sh` | optional: three-dimensional views of the total flux with VisIt |

Stage 1 must be finished before stage 2 starts (`sbatch --dependency=afterok:<job id>`). The
OpenMC reference does not depend on the OpenSn runs. Without step 3, step 4 uses the committed
reference, `../openmc_reference_results.json`.

OpenSn must be built from the branch that accepts a point source on reflecting planes
(`uncollided-hull-accel`, commit `584810e0e` or later) until that change is merged. The Python
tools (Gmsh, OpenMC, the comparison) come from `../environment.yml`.

## Choosing the mesh, the quadrature and the number of ranks

The memory of stage 2 was measured on this example, on one 32-core node. For N
tetrahedra and D directions, with groupsets solved by GMRES:

- 4 groups, P0: peak memory = N x (0.43 D + 17) kB, plus about 0.15 GB per rank (measured on
  15k to 1.3 M tetrahedra at 144, 256 and 576 directions).
- 64 groups, P3, 144 directions: about 0.9 to 1.0 MB per tetrahedron (121.6 GB for 125,189
  tetrahedra on 32 ranks, 1,314 s). How it grows with the number of directions was not measured
  with 64 groups: run a pilot (for example the `--size 20` mesh, 61,368 tetrahedra) at the
  quadrature you want before sizing the large job.

GMRES is used in every groupset: source iteration converges by only about 7% per iteration on
this problem (scattering ratios up to 0.99, no diffusion acceleration in the void cavity). The
GMRES vectors are copies of the flux moments of their groupset, and every groupset keeps one more
copy of the source moments, so the layout of the 64 groups matters. Measured on 15,491
tetrahedra, 144 directions, 4 ranks:

| Groupsets (`groupset_size`, `gmres_restart`) | Peak memory | Memory per tetrahedron | Time |
|---|---|---|---|
| groups 0-49 and 50-63 (0, 5), the default of chamber.py | 14.0 GB | 0.87 MB | 805 s |
| groups 0-49 and 50-63 (0, 10) | 16.7 GB | 1.04 MB | 790 s |
| one groupset of 64 groups (restart 10) | 16.1 GB | 1.01 MB | 834 s |
| groupsets of 8 groups (8, 5) | 17.0 GB | 1.07 MB | 848 s |
| groupsets of 8 groups (8, 10) | 19.7 GB | 1.24 MB | 689 s |

Smaller groupsets have smaller GMRES vectors but more copies of the source moments, which costs
more here. An OpenSn change that freed the copy of each groupset after its solve would make
small groupsets cheaper; it is not in the branch.

Stage 1 holds the uncollided moments of every group on every rank for a short time, about 74 kB
per tetrahedron with 64 groups and P3 moments, and rank 0 gathers the whole file before writing
it.

Meshes, with the number of tetrahedra (Gmsh 4.15.2, one thread):

| `generate_mesh.py` options | Tetrahedra | Time to generate |
|---|---|---|
| `--size 20` | 61,368 | 1 s |
| `--size 15` | 125,189 | 2 s |
| `--size 10` | 374,226 | 3 s |
| `--size 10 --port-size 5` | 619,294 | 21 min |
| `--size 10 --port-size 3` | 1,299,969 | 54 min |

Finer port meshes (`--port-size 2`) were not generated on this machine; the size field makes
Gmsh slow (54 minutes for `--port-size 3`).

Example: the 64-group P3 library on the `--size 10 --port-size 3` mesh (1.3 M tetrahedra) with
144 directions needs about 1.2 to 1.3 TB in stage 2, for example 8 nodes of 256 GB with 32 ranks
each, or 16 nodes for room; 256 directions need more, by a pilot.

## What to check

- Both balances close: the log of stage 1 prints "Conservation error" at about 1e-15, and the
  log of stage 2 prints the relative balance (about 1e-8).
- The production rate of stage 2 is 0.125, one eighth of the whole source.
- `04_compare.sh` prints the uncollided flux in each block against the semi-analytic values; the
  differences are the faceting of the mesh and fall with the mesh size.
- The GMRES iterations of each groupset converge (the log prints "status = converged").
