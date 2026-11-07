// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "multiRegionPrecicePower.H"

#include "IOdictionary.H"
#include "argList.H"
#include "fvMesh.H"
#include "volFields.H"

using namespace Foam;

int main(int argc, char **argv) {
  argList::addOption("precice-config", "file",
                     "Override adapterDict configurationFile");
  argList::addOption("expected-power", "watts",
                     "Override adapterDict expectedPower");
  // clang-format off
#include "setRootCase.H"
#include "createTime.H"
  // clang-format on

  IOdictionary adapter_dict(IOobject("adapterDict", runTime.system(), runTime,
                                     IOobject::MUST_READ, IOobject::NO_WRITE));
  dictionary coupling_dict(adapter_dict);
  if (args.optionFound("precice-config"))
    coupling_dict.set("configurationFile",
                      args.optionRead<fileName>("precice-config"));
  if (args.optionFound("expected-power"))
    coupling_dict.set("expectedPower",
                      args.optionRead<scalar>("expected-power"));
  const wordList regions(adapter_dict.lookup<wordList>("regions"));
  PtrList<fvMesh> meshes(regions.size());
  forAll(regions, region)
      meshes.set(region, new fvMesh(IOobject(regions[region], runTime.name(),
                                             runTime, IOobject::MUST_READ)));

  functionObjects::multiRegionPrecicePower adapter("multiRegionPower", runTime,
                                                   coupling_dict);

  scalar local_power = 0.0;
  forAll(meshes, region) {
    const volScalarField &source =
        meshes[region].lookupObject<volScalarField>("Qtransport");
    local_power += sum(source.primitiveField() * meshes[region].V());
  }
  scalar global_power = local_power;
  reduce(global_power, sumOp<scalar>());
  if (Pstream::master())
    Info << "MULTIREGION_RECEIVED_POWER_W=" << global_power << nl;

  if (mag(global_power - 6.0) > 1.0e-12)
    FatalErrorInFunction << "expected 6 W but recovered " << global_power
                         << " W" << exit(FatalError);
  return 0;
}
