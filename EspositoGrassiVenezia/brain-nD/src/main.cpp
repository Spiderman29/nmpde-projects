#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>
#include "BRAIN.hpp"
#include <fstream>
#include "params.hpp"

// Main function.
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int               mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

 
    // Configuration 1
    {
        Params p{
            "../src/brain_gm_wm_cb.msh", //mesh name
            1, // degree
            20.0, // T
            0.1, // deltat
            1.0, // theta
            {0.3, 0.6}, // alpha
            {1.5, 1.5}, // d_ext
            {0, 3}, // d_axn
            "circumferential" // diffusion type
        };

        Brain problem(p);
        problem.setup();
        problem.solve();
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
