#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>
#include "BRAIN.hpp"
#include <fstream>

// Main function.
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int               mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    const std::string mesh_file_name = "../../mesh/brain_gm_wm_cb.msh";
    // const std::string mesh_file_name = "../../mesh/brain-h3.0.msh";

    const unsigned int degree = 1;
    const double T = 12.0;
    const double theta = 1.0;
 

    // Configuration 1
    {
        double deltat = 0.1;
        //WHITE, GREY
        std::vector<double> alpha = {0.3, 0.6};
        std::vector<double> d_ext = {1.5, 1.5};
        std::vector<double> d_axn = {0, 3};
        Brain problem(mesh_file_name, degree, T, deltat, theta, d_ext, d_axn, alpha);
        problem.setup();
        problem.solve();
    }

    // // Configuration 2
    // {
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
