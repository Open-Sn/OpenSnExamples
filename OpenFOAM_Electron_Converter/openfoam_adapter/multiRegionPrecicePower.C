// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "multiRegionPrecicePower.H"

#include "Pstream.H"
#include "addToRunTimeSelectionTable.H"
#include "dimensions.H"
#include "fvMesh.H"
#include "volFields.H"

#include <precice/precice.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace Foam {
namespace functionObjects {
defineTypeNameAndDebug(multiRegionPrecicePower, 0);
addToRunTimeSelectionTable(functionObject, multiRegionPrecicePower, dictionary);
} // namespace functionObjects
} // namespace Foam

Foam::functionObjects::multiRegionPrecicePower::multiRegionPrecicePower(
    const word &name, const Time &run_time, const dictionary &dict)
    : functionObject(name, run_time, dict), run_time_(run_time) {
  read(dict);
}

Foam::functionObjects::multiRegionPrecicePower::~multiRegionPrecicePower() {
  if (participant_) {
    try {
      participant_->finalize();
    } catch (...) {
      // Destructors must not throw. A normal exchange finalizes explicitly.
    }
  }
}

bool Foam::functionObjects::multiRegionPrecicePower::read(
    const dictionary &dict) {
  if (configured_)
    return true;

  try {
    Configure(dict);
    BuildCouplingMesh();
    ReceivePower();
    configured_ = true;
  } catch (const std::exception &error) {
    FatalErrorInFunction
        << "OpenFOAM preCICE multi-region power coupling failed: "
        << error.what() << abort(FatalError);
  }
  return true;
}

void Foam::functionObjects::multiRegionPrecicePower::Configure(
    const dictionary &dict) {
  region_names_ = dict.lookup<wordList>("regions");
  if (region_names_.empty())
    FatalIOErrorInFunction(dict)
        << "regions must contain at least one region mesh"
        << exit(FatalIOError);

  std::unordered_set<std::string> unique_regions;
  for (const word &region : region_names_)
    if (!unique_regions.insert(region).second)
      FatalIOErrorInFunction(dict)
          << "duplicate region mesh '" << region << "'" << exit(FatalIOError);

  configuration_file_ = dict.lookup<fileName>("configurationFile");
  participant_name_ =
      dict.lookupOrDefault<word>("participant", participant_name_);
  mesh_name_ = dict.lookupOrDefault<word>("mesh", mesh_name_);
  data_name_ = dict.lookupOrDefault<word>("data", data_name_);
  field_name_ = dict.lookupOrDefault<word>("field", field_name_);
  power_tolerance_ =
      dict.lookupOrDefault<scalar>("powerTolerance", power_tolerance_);

  if (dict.found("expectedPower") && dict.found("expectedPowerText"))
    FatalIOErrorInFunction(dict)
        << "give at most one of expectedPower and expectedPowerText"
        << exit(FatalIOError);
  has_expected_power_ = false;
  if (dict.found("expectedPower")) {
    expected_power_ = dict.lookup<scalar>("expectedPower");
    has_expected_power_ = true;
  }
  if (dict.found("expectedPowerText")) {
    const string expected_power_text = dict.lookup<string>("expectedPowerText");
    std::size_t parsed_characters = 0;
    expected_power_ = std::stod(expected_power_text, &parsed_characters);
    if (parsed_characters != expected_power_text.size())
      FatalIOErrorInFunction(dict)
          << "expectedPowerText is not a complete floating-point value"
          << exit(FatalIOError);
    has_expected_power_ = true;
  }

  if (has_expected_power_ &&
      (!std::isfinite(expected_power_) || expected_power_ < 0.0))
    FatalIOErrorInFunction(dict)
        << "expected power must be finite and nonnegative"
        << exit(FatalIOError);
  if (!std::isfinite(power_tolerance_) || power_tolerance_ < 0.0)
    FatalIOErrorInFunction(dict)
        << "powerTolerance must be finite and nonnegative"
        << exit(FatalIOError);
}

void Foam::functionObjects::multiRegionPrecicePower::BuildCouplingMesh() {
  std::vector<double> coordinates;
  region_offsets_.clear();
  region_offsets_.push_back(0);

  for (const word &region_name : region_names_) {
    if (!run_time_.foundObject<fvMesh>(region_name))
      FatalErrorInFunction << "region mesh '" << region_name
                           << "' is not registered" << exit(FatalError);

    const fvMesh &mesh = run_time_.lookupObject<fvMesh>(region_name);
    region_meshes_.push_back(&mesh);

    volScalarField *power_density = nullptr;
    if (mesh.foundObject<volScalarField>(field_name_)) {
      power_density = &const_cast<volScalarField &>(
          mesh.lookupObject<volScalarField>(field_name_));
      if (power_density->dimensions() != dimPower / dimVolume)
        FatalErrorInFunction
            << "field '" << field_name_ << "' in region '" << region_name
            << "' does not have power-density dimensions" << exit(FatalError);
    } else {
      auto *field = new volScalarField(
          IOobject(field_name_, run_time_.constant(), mesh, IOobject::NO_READ,
                   IOobject::NO_WRITE, true),
          mesh, dimensionedScalar(dimPower / dimVolume, Zero));
      const_cast<fvMesh &>(mesh).objectRegistry::store(field);
      power_density = field;
    }
    power_density_fields_.push_back(power_density);

    const vectorField &centers = mesh.cellCentres();
    coordinates.reserve(coordinates.size() + 3 * centers.size());
    for (const vector &center : centers) {
      if (!std::isfinite(center.x()) || !std::isfinite(center.y()) ||
          !std::isfinite(center.z()))
        FatalErrorInFunction << "region '" << region_name
                             << "' has a non-finite cell center"
                             << exit(FatalError);
      coordinates.push_back(center.x());
      coordinates.push_back(center.y());
      coordinates.push_back(center.z());
    }
    region_offsets_.push_back(region_offsets_.back() + centers.size());
  }

  participant_ = std::make_unique<precice::Participant>(
      participant_name_, configuration_file_, Pstream::myProcNo(),
      Pstream::nProcs());
  if (participant_->getMeshDimensions(mesh_name_) != 3)
    FatalErrorInFunction << "preCICE coupling mesh '" << mesh_name_
                         << "' must be three-dimensional" << exit(FatalError);
  if (participant_->getDataDimensions(mesh_name_, data_name_) != 1)
    FatalErrorInFunction << "preCICE power data '" << data_name_
                         << "' must be scalar" << exit(FatalError);

  vertex_ids_.resize(region_offsets_.back());
  participant_->setMeshVertices(mesh_name_, coordinates, vertex_ids_);
}

void Foam::functionObjects::multiRegionPrecicePower::ReceivePower() {
  participant_->initialize();
  std::vector<double> cell_power(vertex_ids_.size(), 0.0);
  while (participant_->isCouplingOngoing()) {
    const double time_step = participant_->getMaxTimeStepSize();
    if (!std::isfinite(time_step) || time_step <= 0.0)
      FatalErrorInFunction
          << "preCICE returned a nonpositive or non-finite time-step size"
          << exit(FatalError);
    participant_->readData(mesh_name_, data_name_, vertex_ids_, time_step,
                           cell_power);
    participant_->advance(time_step);
  }
  participant_->finalize();
  participant_.reset();
  StorePowerDensity(cell_power);
}

void Foam::functionObjects::multiRegionPrecicePower::StorePowerDensity(
    const std::vector<double> &cell_power) {
  if (cell_power.size() != region_offsets_.back())
    FatalErrorInFunction
        << "received power vector does not match the local coupling mesh"
        << exit(FatalError);

  scalar local_power = 0.0;
  for (std::size_t region = 0; region < region_meshes_.size(); ++region) {
    const fvMesh &mesh = *region_meshes_[region];
    volScalarField &power_density = *power_density_fields_[region];
    scalarField &values = power_density.primitiveFieldRef();
    const scalarField &volumes = mesh.V();
    const std::size_t offset = region_offsets_[region];
    if (values.size() != volumes.size() ||
        static_cast<std::size_t>(values.size()) !=
            region_offsets_[region + 1] - offset)
      FatalErrorInFunction << "region '" << region_names_[region]
                           << "' field, volume, and coupling sizes disagree"
                           << exit(FatalError);

    forAll(values, cell) {
      const double power = cell_power[offset + cell];
      if (!std::isfinite(power) || power < 0.0)
        FatalErrorInFunction << "region '" << region_names_[region]
                             << "' received negative or non-finite cell power"
                             << exit(FatalError);
      if (!std::isfinite(volumes[cell]) || volumes[cell] <= 0.0)
        FatalErrorInFunction
            << "region '" << region_names_[region]
            << "' contains a nonpositive or non-finite cell volume"
            << exit(FatalError);
      values[cell] = power / volumes[cell];
      local_power += power;
    }
    power_density.correctBoundaryConditions();
  }

  scalar global_power = local_power;
  reduce(global_power, sumOp<scalar>());
  if (!std::isfinite(global_power) || global_power < 0.0)
    FatalErrorInFunction
        << "global received transport power is negative or non-finite"
        << exit(FatalError);

  if (Pstream::master()) {
    std::ostringstream message;
    message << std::setprecision(17)
            << "OpenFOAM preCICE received cell-integrated power = "
            << global_power << " W";
    Info << message.str().c_str() << nl;
  }

  if (has_expected_power_) {
    const scalar scale = max(mag(expected_power_), scalar(1.0e-300));
    const scalar relative_error = mag(global_power - expected_power_) / scale;
    if (Pstream::master()) {
      std::ostringstream message;
      message << std::setprecision(17)
              << "OpenFOAM preCICE expected power = " << expected_power_
              << " W, relative closure = " << relative_error;
      Info << message.str().c_str() << nl;
    }
    if (relative_error > power_tolerance_)
      FatalErrorInFunction << "received power " << global_power
                           << " W differs from expected power "
                           << expected_power_ << " W by " << relative_error
                           << " relative, exceeding tolerance "
                           << power_tolerance_ << exit(FatalError);
  }
}
