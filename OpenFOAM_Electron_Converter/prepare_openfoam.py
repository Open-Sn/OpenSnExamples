#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""Prepare the waterblock conjugate heat-transfer case for preCICE heating."""

import argparse
from pathlib import Path
import math
import re
import shutil


CASE = Path(__file__).resolve().parent / "meshCase"
T_INLET = 293.15
U_INLET = 0.01
U_OUTLET = 0.01
SOLID_PROPERTIES = {
    "tungsten": (183.84, 174.0, 134.0, 19300.0),
    "copper": (63.546, 401.0, 385.0, 8960.0),
    "aluminum": (26.9815, 237.0, 900.0, 2700.0),
    "air": (28.97, 0.0262, 1005.0, 1.205),
    "steel": (55.0, 16.2, 500.0, 8000.0),
}
NEIGHBORS = {
    "tungsten": ("air", "copper"),
    "copper": ("water", "tungsten", "air"),
    "aluminum": ("water", "air"),
    "air": ("steel", "water", "tungsten", "aluminum", "copper"),
    "steel": ("air",),
}


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def set_dictionary_scalar(path, name, value):
    text = path.read_text(encoding="utf-8")
    pattern = rf"(^\s*{re.escape(name)}\s+)[^;]+;"
    updated, count = re.subn(
        pattern, rf"\g<1>{value:.12g};", text, count=1, flags=re.MULTILINE
    )
    if count != 1:
        raise RuntimeError(f"Could not set {name} in {path}")
    path.write_text(updated, encoding="utf-8")


def foam_header(class_name, object_name, location=None):
    location_entry = f'    location "{location}";\n' if location else ""
    return (
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        f"    class {class_name};\n{location_entry}    object {object_name};\n}}\n"
    )


def solid_temperature(region):
    patches = ['    ".*" { type zeroGradient; }']
    for neighbor in NEIGHBORS[region]:
        patches.append(
            f"    {region}_to_{neighbor}\n"
            "    {\n"
            "        type coupledTemperature;\n"
            f"        value uniform {T_INLET};\n"
            "    }"
        )
    return (
        foam_header("volScalarField", "T")
        + "dimensions [0 0 0 1 0 0 0];\n"
        + f"internalField uniform {T_INLET};\n"
        + "boundaryField\n{\n"
        + "\n".join(patches)
        + "\n}\n"
    )


def fluid_pressure():
    return (
        foam_header("volScalarField", "p")
        + "dimensions [1 -1 -2 0 0 0 0];\n"
        + "internalField uniform 0;\n"
        + "boundaryField\n{\n"
        + "    waterInlet { type zeroGradient; }\n"
        + "    waterOutlet { type fixedValue; value uniform 0; }\n"
        + '    ".*" { type zeroGradient; }\n'
        + "}\n"
    )


def solid_thermophysical_properties(values):
    mol_weight, conductivity, heat_capacity, density = values
    return (
        foam_header("dictionary", "physicalProperties")
        + "thermoType\n{\n"
        + "    type heSolidThermo;\n    mixture pureMixture;\n"
        + "    transport constIsoSolid;\n    thermo eConst;\n"
        + "    equationOfState rhoConst;\n    specie specie;\n"
        + "    energy sensibleInternalEnergy;\n}\nmixture\n{\n"
        + f"    specie {{ molWeight {mol_weight}; }}\n"
        + f"    transport {{ kappa {conductivity}; }}\n"
        + f"    thermodynamics {{ hf 0; Cv {heat_capacity}; }}\n"
        + f"    equationOfState {{ rho {density}; }}\n"
        + "}\n"
    )


def fluid_thermophysical_properties():
    return (
        foam_header("dictionary", "physicalProperties")
        + "thermoType\n{\n"
        + "    type heRhoThermo;\n    mixture pureMixture;\n"
        + "    transport const;\n    thermo eConst;\n"
        + "    equationOfState rhoConst;\n    specie specie;\n"
        + "    energy sensibleInternalEnergy;\n}\nmixture\n{\n"
        + "    specie { molWeight 18.015; }\n"
        + "    equationOfState { rho 998; }\n"
        + "    thermodynamics { Cv 4180; hf 0; }\n"
        + "    transport { mu 1.002e-3; Pr 6.98; }\n"
        + "}\n"
    )


def fv_models(region):
    # foamMultiRun's solid and fluid modules both call fvModels().source(rho, e),
    # so codeAddRhoSup is the only hook that runs.  multiRegionPrecicePower
    # registers Qtransport in W/m3 before the first time step.
    return (
        foam_header("dictionary", "fvModels")
        + f"{region}TransportHeating\n{{\n"
        + "    type coded;\n"
        + "    cellZone all;\n"
        + "    field e;\n"
        + "    codeAddRhoSup\n    #{\n"
        + "        const volScalarField& qTransport =\n"
        + '            mesh().lookupObject<volScalarField>("Qtransport");\n'
        + "        eqn.source() -= qTransport.primitiveField()*mesh().V();\n"
        + "    #};\n}\n"
    )


def fluid_temperature():
    coupled = []
    for neighbor in ("air", "aluminum", "copper"):
        coupled.append(
            f"    water_to_{neighbor}\n"
            "    {\n"
            "        type coupledTemperature;\n"
            f"        value uniform {T_INLET};\n"
            "    }"
        )
    return (
        foam_header("volScalarField", "T")
        + "dimensions [0 0 0 1 0 0 0];\n"
        + f"internalField uniform {T_INLET};\n"
        + "boundaryField\n{\n"
        + f"    waterInlet {{ type fixedValue; value uniform {T_INLET}; }}\n"
        + "    waterOutlet\n    {\n        type inletOutlet;\n"
        + f"        inletValue uniform {T_INLET};\n"
        + f"        value uniform {T_INLET};\n    }}\n"
        + "\n".join(coupled)
        + "\n}\n"
    )


def fluid_velocity():
    return (
        foam_header("volVectorField", "U")
        + "dimensions [0 1 -1 0 0 0 0];\n"
        + "internalField uniform (0 0 0);\n"
        + "boundaryField\n{\n"
        + f"    waterInlet {{ type fixedValue; value uniform (0 {U_INLET} 0); }}\n"
        + "    waterOutlet\n    {\n        type pressureInletOutletVelocity;\n"
        + f"        value uniform (0 {-U_OUTLET} 0);\n    }}\n"
        + '    ".*" { type fixedValue; value uniform (0 0 0); }\n'
        + "}\n"
    )


def fluid_pressure_rgh():
    return (
        foam_header("volScalarField", "p_rgh")
        + "dimensions [1 -1 -2 0 0 0 0];\n"
        + "internalField uniform 0;\n"
        + "boundaryField\n{\n"
        + "    waterInlet { type fixedFluxPressure; value uniform 0; }\n"
        + "    waterOutlet { type fixedValue; value uniform 0; }\n"
        + '    ".*" { type fixedFluxPressure; value uniform 0; }\n'
        + "}\n"
    )


SOLID_SCHEMES = """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""

SOLID_SOLUTION = """FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    e
    {
        solver PCG;
        preconditioner DIC;
        tolerance 1e-10;
        maxIter 2000;
    }
    eFinal { $e; }
}
PIMPLE { nNonOrthogonalCorrectors 1; }
"""

FLUID_SCHEMES = """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) Gauss upwind;
    div(phi,e) Gauss upwind;
    div(phi,K) Gauss linear;
    div(phi,(p|rho)) Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
fluxRequired { default no; p_rgh; }
"""

FLUID_SOLUTION = """FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    Phi { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; }
    rho { solver PCG; preconditioner DIC; tolerance 1e-10; relTol 0; }
    rhoFinal { $rho; }
    p_rgh { solver GAMG; tolerance 1e-12; relTol 0; smoother GaussSeidel; }
    p_rghFinal { $p_rgh; relTol 0; }
    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0; }
    UFinal { $U; relTol 0; }
    e
    {
        solver PBiCGStab;
        preconditioner DILU;
        tolerance 1e-10;
        relTol 0;
        maxIter 2000;
    }
    eFinal { $e; }
}
PIMPLE
{
    frozenFlow on;
    momentumPredictor on;
    nCorrectors 3;
    nNonOrthogonalCorrectors 1;
}
relaxationFactors
{
    fields { rho 1; p_rgh 0.1; }
    equations { U 0.1; e 1; }
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end-time",
        type=float,
        help="override the OpenFOAM endTime, for example 0.1 for a coupling smoke test",
    )
    args = parser.parse_args()

    if args.end_time is not None and (
        not math.isfinite(args.end_time) or args.end_time <= 0.0
    ):
        raise ValueError("--end-time must be finite and positive")

    solid_list = " ".join(SOLID_PROPERTIES)
    system_dir = Path(__file__).resolve().parent / "system"
    for name in ("controlDict", "fvSchemes", "fvSolution", "preciceFunctionObject"):
        shutil.copyfile(system_dir / name, CASE / "system" / name)
    if args.end_time is not None:
        set_dictionary_scalar(CASE / "system/controlDict", "endTime", args.end_time)
    write(
        CASE / "constant/regionProperties",
        foam_header("dictionary", "regionProperties")
        + f"regions\n(\n    fluid (water)\n    solid ({solid_list})\n);\n",
    )
    for region, values in SOLID_PROPERTIES.items():
        if not (CASE / f"constant/{region}/polyMesh").is_dir():
            raise RuntimeError(f"Missing split mesh for {region}")
        write(CASE / f"0/{region}/T", solid_temperature(region))
        write(
            CASE / f"constant/{region}/physicalProperties",
            solid_thermophysical_properties(values),
        )
        write(CASE / f"constant/{region}/fvModels", fv_models(region))
        write(CASE / f"system/{region}/fvSchemes", SOLID_SCHEMES)
        write(CASE / f"system/{region}/fvSolution", SOLID_SOLUTION)

    region = "water"
    if not (CASE / f"constant/{region}/polyMesh").is_dir():
        raise RuntimeError(f"Missing split mesh for {region}")
    write(CASE / f"0/{region}/T", fluid_temperature())
    write(CASE / f"0/{region}/U", fluid_velocity())
    write(CASE / f"0/{region}/p_rgh", fluid_pressure_rgh())
    write(CASE / f"0/{region}/p", fluid_pressure())
    write(
        CASE / f"constant/{region}/physicalProperties",
        fluid_thermophysical_properties(),
    )
    write(
        CASE / f"constant/{region}/momentumTransport",
        foam_header("dictionary", "momentumTransport") + "simulationType laminar;\n",
    )
    write(
        CASE / f"constant/{region}/thermophysicalTransport",
        foam_header("dictionary", "thermophysicalTransport")
        + "laminar { model Fourier; }\n",
    )
    write(
        CASE / f"constant/{region}/g",
        foam_header("uniformDimensionedVectorField", "g")
        + "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 0);\n",
    )
    write(
        CASE / f"constant/{region}/pRef",
        foam_header("uniformDimensionedScalarField", "pRef")
        + "dimensions [1 -1 -2 0 0 0 0];\nvalue 0;\n",
    )
    write(CASE / f"constant/{region}/fvModels", fv_models(region))
    write(CASE / f"system/{region}/fvSchemes", FLUID_SCHEMES)
    write(CASE / f"system/{region}/fvSolution", FLUID_SOLUTION)
    print(
        "Prepared five conducting solids and one laminar water region "
        "with preCICE transport heating"
    )


if __name__ == "__main__":
    main()
