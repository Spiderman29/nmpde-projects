#include <fstream>
#include <iostream>
#include <vector>
#include "BRAIN.hpp"
#include <fstream>
#include <filesystem>
#include "params.hpp"
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>

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

  // default values
  {
  ParameterHandler prm;
  prm.declare_entry("mesh_file_name", "../../mesh/brain_gm_wm.msh");
  prm.declare_entry("degree", "1", Patterns::Integer());
  prm.declare_entry("T", "40.0",Patterns::Double());
  prm.declare_entry("deltat", "0.333333333",Patterns::Double());
  prm.declare_entry("theta", "1.0",Patterns::Double());
  prm.declare_entry("alpha", "0.3,0.6");
  prm.declare_entry("d_ext", "6.0,6.0");
  prm.declare_entry("d_axn", "3.0,3.0");
  prm.declare_entry("diffusion_type", "radial");


  // Open the parameter file

  std::ifstream parameter_file("../../params/parameters.prm");
  if (!parameter_file.is_open()) {
        std::cerr << "Error: Could not open parameter file!" << std::endl;
        return 1;
    }

    // Parse the parameter file
    try {
        prm.parse_input(parameter_file);
    } catch (const std::exception &e) {
        std::cerr << "Error while parsing parameter file: " << e.what() << std::endl;
        return 1; 
    }



  //keep old structure with Params
  Params p{
    prm.get("mesh_file_name"),
    static_cast<unsigned int>(prm.get_integer("degree")),
    prm.get_double("T"),
    prm.get_double("deltat"),
    prm.get_double("theta"),
    Utilities::string_to_double(Utilities::split_string_list(prm.get("alpha"), ',')),
    Utilities::string_to_double(Utilities::split_string_list(prm.get("d_ext"), ',')),
    Utilities::string_to_double(Utilities::split_string_list(prm.get("d_axn"), ',')),
    prm.get("diffusion_type")
  };


  // Create the Brain object with the parameters.
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
      std::ios::openmode mode = (mpi_size == 1 ? std::ios::trunc : std::ios::app);

      std::ofstream setup_time_file("../csv/setup_time.csv", mode);
      if (setup_time_file.tellp() == 0)
      {
        setup_time_file << "n,time,alpha values" << std::endl;
      }
      setup_time_file << mpi_size << "," << setup_time <<","<< p.alpha[0] <<","<<p.alpha[1] << std::endl;

      std::ofstream solve_time_file("../csv/solve_time.csv", mode);
      if (solve_time_file.tellp() == 0)
      {
        solve_time_file << "n,time,alpha values" << std::endl;
      } 
      solve_time_file << mpi_size << "," << solve_time << "," << p.alpha[0] << "," << p.alpha[1] << std::endl;

      std::ofstream total_time_file("../csv/total_time.csv", mode);
      if (total_time_file.tellp() == 0)
      {
        total_time_file << "n,time,alpha values" << std::endl;
      }
      total_time_file << mpi_size << "," << total_time << "," << p.alpha[0] << "," << p.alpha[1] << std::endl;

      std::cout << "Total time: " << total_time << " seconds." << std::endl;
      std::cout << "Setup time: " << setup_time << " seconds." << std::endl;
      std::cout << "Solve time: " << solve_time << " seconds." << std::endl;
    }
  }
  return 0;
}
