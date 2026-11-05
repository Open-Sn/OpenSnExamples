# Multi-region OpenFOAM preCICE receiver

`multiRegionPrecicePower` is a generic OpenFOAM Foundation v14 function object
for one-way, steady, cell-average energy-deposition coupling. One preCICE
participant concatenates the locally owned cell centers from any positive
number of configured region meshes, receives extensive power in W/cell, and
registers a `Qtransport` field in W/m3 on every region. The region energy
equations consume those fields through `fvModels`.

The current contract is a static mesh and one transfer from OpenSn to
OpenFOAM. No material names or region count are compiled into the library.

## Build and test

Load an ABI-compatible OpenFOAM v14, preCICE 3.4-or-newer, compiler, and MPI
environment, then run:

```sh
./build.sh
./tests/run.sh
```

`build.sh` uses `wmake` and writes
`lib/libmultiRegionPrecicePower.so`. The manufactured test creates three region
meshes with one, two, and three cells; a sender supplies 1, 2, and 3 W. It
requires exact recovery of the 6 W sum with one and two OpenFOAM ranks and
checks that a deliberately incorrect expected power is rejected.

## Configuration

The function-object dictionary accepts:

- `regions`: a nonempty list of registered `fvMesh` names;
- `configurationFile`, `participant`, `mesh`, and `data`: preCICE identifiers;
- `field`: the registered output field, normally `Qtransport`;
- `expectedPower`: an optional nonnegative power reference in W, or
  `expectedPowerText`, the same reference given as a decimal string parsed
  exactly by `std::stod`; give at most one, and omit both to skip the check;
  and
- `powerTolerance`: the relative tolerance of the expected-power check,
  default `1e-10`.

Every received power must be finite and nonnegative, and every cell volume must
be finite and positive. The global power check is fatal on failure. See
[`../system/preciceFunctionObject`](../system/preciceFunctionObject) for the
six-region WATERBLOCK configuration and run the complete coupled problem with
`../run_coupled_test.sh`.
