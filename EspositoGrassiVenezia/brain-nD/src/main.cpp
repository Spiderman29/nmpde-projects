#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>
#include "BRAIN.hpp"
#include <fstream>
#include <filesystem>
#include "params.hpp"

// Main function.
int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int mpi_size =
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if (mpi_rank == 0)
    std::filesystem::create_directories("../csv");

  double start_time = MPI_Wtime();

  // Configuration 1
  {
    Params p{
      "../../mesh/brain_gm_wm.msh", // mesh name
        1,                            // degree
        40.0,                         // T
        1.0/3.0,                      // deltat
        1.0,                          // theta
        {0.6, 0.6},                   // alpha {0.3, 0.6}
           {6.0, 6.0},                   // d_ext
           {24, 24},                       // d_axn {0, 3}
           "radial"             // diffusion type
    };

    Brain problem(p);
    double start_setup_time = MPI_Wtime();
    problem.setup();
    double end_setup_time = MPI_Wtime();
    double setup_time = end_setup_time - start_setup_time;
    double start_solve_time = MPI_Wtime();
    problem.solve();
    double end_solve_time = MPI_Wtime();
    double solve_time = end_solve_time - start_solve_time;
    double total_time = end_solve_time - start_time;
    if (mpi_rank == 0)
    {
      std::ofstream setup_time_file("../csv/setup_time.csv", std::ios::app);
      if (setup_time_file.tellp() == 0)
      {
        setup_time_file << "n,time" << std::endl;
      }
      setup_time_file << mpi_size << "," << setup_time << std::endl;

      std::ofstream solve_time_file("../csv/solve_time.csv", std::ios::app);
      if (solve_time_file.tellp() == 0)
      {
        solve_time_file << "n,time" << std::endl;
      }
      solve_time_file << mpi_size << "," << solve_time << std::endl;

      std::ofstream total_time_file("../csv/total_time.csv", std::ios::app);
      if (total_time_file.tellp() == 0)
      {
        total_time_file << "n,time" << std::endl;
      }
      total_time_file << mpi_size << "," << total_time << std::endl;


      std::cout << "Total time: " << total_time << " seconds." << std::endl;
      std::cout << "Setup time: " << setup_time << " seconds." << std::endl;
      std::cout << "Solve time: " << solve_time << " seconds." << std::endl;
    }
  }

  // // Configuration 2
  // {
  //     const std::vector<std::string> anisotropic_axonal_transport_types={"circumferential", "radial", "axonal"};
  //     std::vector<double> deltats = {0.025, 0.05, 0.1, 0.2, 0.3, 0.4};
  //     double alpha = 2.0;
  //     double d = 0.0002;

  //     for (const auto &deltat : deltats)
  //     {
  //         Brain problem(mesh_file_name, degree, T, deltat, theta, d, alpha);
  //         problem.setup();
  //         problem.solve();
  //     }
  // }

  // Configuration 3
  // {
  //     double deltat = 0.1;
  //     std::vector<double> alphas = {1.0, 2.0, 4.0};
  //     std::vector<double> ds = {0.0001, 0.0002, 0.0004};

  //     for (const auto &alpha : alphas)
  //         for (const auto &d : ds)
  //         {
  //             Brain problem(mesh_file_name, degree, T, deltat, theta, d, alpha);
  //             problem.setup();
  //             problem.solve();
  //         }
  // }

  return 0;
}
