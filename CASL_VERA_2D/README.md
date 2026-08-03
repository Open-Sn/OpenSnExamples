# CASL VERA Problem 2 Benchmarks

This directory contains OpenSn inputs and support files for the VERA Core
Physics Benchmark Progression Problem 2 two-dimensional fuel-lattice cases.

The benchmark assets are included with this example: meshes are in
`assets/mesh/casl`, and cross-section libraries are in `assets/xs/casl`.

The meshes were generated with the included `spydermesh.py` utility.
Use `generate_lattice_meshes.py` to regenerate the full benchmark lattice
meshes:

```bash
python generate_lattice_meshes.py
```

By default, this regenerates cases 2A through 2Q into `assets/mesh/casl`. To
regenerate selected cases or write to another directory, pass case names and
`--output-dir`:

```bash
python generate_lattice_meshes.py 2A 2B --output-dir /tmp/casl_meshes
```

## Testing

Testing is split across two independent tools: the shared `test.py` script
at the repository root checks k-eff via its own generic key/value matching
(the same mechanism every other example uses) and has no concept of pin
powers. Pin-power-specific logic lives entirely in this directory's
`check_pin_powers.py` script, so `test.py` doesn't accumulate per-example
special cases.

### k-eff regression (`test.py`)

```bash
OPENSN=/path/to/build/python/opensn pytest test.py -k CASL_VERA_2D
```

For every case, a 64-rank run of `<case>/<case>.py` has its converged k-eff
compared against a baseline value in `casl_reference_k.csv` (relative
tolerance 1e-6).

### Pin-power regression + benchmark comparison (`check_pin_powers.py`)

```bash
python3 check_pin_powers.py                  # run + check all 17 cases, 64 ranks each
python3 check_pin_powers.py 2A 2B --ranks 8  # just these two, 8 ranks each
python3 check_pin_powers.py --skip-run       # check existing power_raw.txt/keff.txt, e.g. after
                                             # run_all_cases.py or extracting a results.tar.gz
```

For each case, we have two checks (in addition to the keff check in `test.py`):

1. **Regression** -- `power_raw.txt` (the raw, pre-normalization
   kappa-fission-weighted VolumePostprocessor output) is compared against
   a committed snapshot, `<case>/power_raw_baseline.txt` (relative/
   absolute tolerance 1e-6).
2. **Benchmark comparison (informational only, via `compare_casl_pin_power.py`)**
   -- OpenSn's k-eff and pin powers are also compared against the actual VERA
   benchmark reference (`vera_reference_keff.csv` and each case's
   `reference_pin_power.txt`) and the result is printed, but never fails the
   script. Some divergence from the published benchmark is expected: most
   cases model only the distinct sub-region of the 17x17 lattice the mesh
   generator actually tiles, and 2Q's spacer grid is a homogenized
   approximation. The "Results vs. Benchmark" table below is exactly this
   comparison.

## References

- VERA benchmark specification:
  [CASL-U-2012-0131-004, VERA Core Physics Benchmark Progression Problem Specifications](https://corephysics.com/docs/CASL-U-2012-0131-004.pdf)
- Original OpenSn CASL benchmark repository:
  [aly-tamu/VERA-Examples](https://github.com/aly-tamu/VERA-Examples/)

## Results vs. Benchmark

Comparison of an OpenSn run of each case (361-group SHEM-361 cross sections,
P3 scattering, n_polar=4/n_azimuthal=32 quadrature, CMFD acceleration) against
the VERA benchmark's own reference k-effective (Table P2-5) and pin powers
(Appendix B), via `compare_casl_pin_power.py`. This is the same comparison the
test suite prints for each case (see Testing above); it is not a pass/fail test,
since the benchmark's continuous-energy Monte Carlo solution and OpenSn's 2D
multigroup deterministic model are not expected to match exactly.

Most cases' meshes only tile a distinct sub-region of the full 17x17 lattice
(see `generate_lattice_meshes.py`'s `compute_pin_positions` docstring for
why), so "Pins Compared" reports how many of the 264 fuel pins are actually
present to compare against; 2B is the only case whose own post-processing
reconstructs the full 17x17 assembly by symmetry (see `2B/2B.py`), so it's
the only one compared at full resolution.

| Case | Description | OpenSn k-eff | Reference k-eff | Δk (pcm) | Pins Compared | Pin Power RMS Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2A | No Poisons (565 K) | 1.180447 | 1.182175 | -172.8 | 72/264 | 0.183% |
| 2B | No Poisons (600 K) | 1.181016 | 1.183360 | -234.4 | 264/264 | 0.036% |
| 2C | No Poisons (900 K) | 1.171658 | 1.173751 | -209.3 | 72/264 | 0.179% |
| 2D | No Poisons (1200 K) | 1.164479 | 1.165591 | -111.2 | 72/264 | 0.189% |
| 2E | 12 Pyrex | 1.067414 | 1.069627 | -221.3 | 72/264 | 0.092% |
| 2F | 24 Pyrex | 0.973122 | 0.976018 | -289.6 | 72/264 | 0.162% |
| 2G | 24 AIC | 0.844456 | 0.847695 | -323.9 | 72/264 | 0.285% |
| 2H | 24 B4C | 0.784611 | 0.788221 | -361.0 | 72/264 | 0.386% |
| 2I | Instrument Thimble | 1.178731 | 1.179916 | -118.5 | 72/264 | 0.153% |
| 2J | Instrument + 24 Pyrex | 0.972526 | 0.975193 | -266.7 | 72/264 | 0.218% |
| 2K | Zoned + 24 Pyrex | 1.017161 | 1.020063 | -290.2 | 72/264 | 0.207% |
| 2L | 80 IFBA | 1.016637 | 1.018915 | -227.8 | 72/264 | 0.112% |
| 2M | 128 IFBA | 0.936539 | 0.938796 | -225.7 | 72/264 | 0.175% |
| 2N | 104 IFBA + 20 WABA | 0.866051 | 0.869615 | -356.4 | 72/264 | 0.541% |
| 2O | 12 Gadolinia | 1.046382 | 1.047729 | -134.7 | 72/264 | 0.683% |
| 2P | 24 Gadolinia | 0.926445 | 0.927410 | -96.5 | 72/264 | 0.151% |
| 2Q | Zircaloy Spacer Grid | 1.166743 | 1.171940 | -519.7 | 72/264 | 4.474% |
