// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#ifndef multiRegionPrecicePower_H
#define multiRegionPrecicePower_H

#include "Time.H"
#include "functionObject.H"
#include "volFieldsFwd.H"
#include "wordList.H"

#include <memory>
#include <string>
#include <vector>

namespace precice {
class Participant;
using VertexID = int;
} // namespace precice

namespace Foam {
class fvMesh;

namespace functionObjects {

/**
 * Receive cell-integrated power on one coupling mesh assembled from any
 * number of OpenFOAM region meshes.
 *
 * The function object performs a one-way, steady preCICE exchange during
 * construction, before the first solver time step. Each locally owned region
 * cell contributes one coupling vertex at its cell center. Received W/cell
 * values are divided by the corresponding cell volume and stored in one
 * registered volScalarField per region.
 */
class multiRegionPrecicePower final : public functionObject {
public:
  static const char *typeName_() { return "multiRegionPrecicePower"; }
  static const word typeName;
  static int debug;
  const word &type() const override { return typeName; }

  multiRegionPrecicePower(const word &name, const Time &run_time,
                          const dictionary &dict);
  ~multiRegionPrecicePower() override;

  bool read(const dictionary &dict) override;
  bool execute() override { return true; }
  bool write() override { return true; }
  bool end() override { return true; }
  wordList fields() const override { return wordList(); }

private:
  multiRegionPrecicePower(const multiRegionPrecicePower &) = delete;
  void operator=(const multiRegionPrecicePower &) = delete;

  void Configure(const dictionary &dict);
  void BuildCouplingMesh();
  void ReceivePower();
  void StorePowerDensity(const std::vector<double> &cell_power);

  const Time &run_time_;
  bool configured_ = false;
  wordList region_names_;
  fileName configuration_file_;
  word participant_name_ = "OpenFOAM";
  word mesh_name_ = "OpenFOAM-Cells";
  word data_name_ = "TransportCellPower";
  word field_name_ = "Qtransport";
  bool has_expected_power_ = false;
  scalar expected_power_ = 0.0;
  scalar power_tolerance_ = 1.0e-10;

  std::vector<const fvMesh *> region_meshes_;
  std::vector<volScalarField *> power_density_fields_;
  std::vector<std::size_t> region_offsets_;
  std::vector<precice::VertexID> vertex_ids_;
  std::unique_ptr<precice::Participant> participant_;
};

} // namespace functionObjects
} // namespace Foam

#endif
