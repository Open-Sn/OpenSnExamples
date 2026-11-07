// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include <mpi.h>
#include <precice/precice.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 2 && argc != 3) {
    if (rank == 0)
      std::cerr << "usage: powerSender PRECICE_CONFIGURATION [POWER_SCALE]\n";
    MPI_Finalize();
    return 2;
  }

  try {
    const double scale = argc == 3 ? std::stod(argv[2]) : 1.0;
    precice::Participant participant("OpenSn", argv[1], rank, size);
    std::vector<double> coordinates;
    std::vector<double> powers;
    constexpr double x[] = {0.25, 1.5, 2.75};
    constexpr double value[] = {1.0, 2.0, 3.0};
    for (std::int64_t point = 0; point < 3; ++point)
      if (point % size == rank) {
        coordinates.insert(coordinates.end(), {x[point], 0.5, 0.5});
        powers.push_back(scale * value[point]);
      }

    std::vector<precice::VertexID> vertex_ids(powers.size());
    participant.setMeshVertices("OpenSn-Cells", coordinates, vertex_ids);
    participant.initialize();
    while (participant.isCouplingOngoing()) {
      participant.writeData("OpenSn-Cells", "TransportCellPower", vertex_ids,
                            powers);
      participant.advance(participant.getMaxTimeStepSize());
    }
    participant.finalize();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
