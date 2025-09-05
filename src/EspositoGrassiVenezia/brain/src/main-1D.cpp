#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>
#include "BRAIN-1D.hpp"
#include <fstream>

// Main function.
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    

    unsigned int N = 199;
    const unsigned int degree = 1;
    const double T = 20.0;
    const double theta = 1.0;
    for (int i = 0; i < 500; i++)
    {
        std::string filename = "output-" + std::to_string(i) + ".vtu";
        std::remove(filename.c_str());
    }
 

    // Configuration 1
    {
        double deltat = 0.1;
        double alpha = 1.0;
        double d = 0.0001;
        Brain1D problem(N, degree, T, deltat, theta, d, alpha);
        problem.setup();
        problem.solve();
    }

    // Configuration 2
    {
        std::vector<double> deltats = {0.025, 0.05, 0.1, 0.2, 0.3, 0.4};
        double alpha = 2.0;
        double d = 0.0002;

        for (const auto &deltat : deltats)
        {
            Brain1D problem(N, degree, T, deltat, theta, d, alpha);
            problem.setup();
            problem.solve();
        }
    }

    // Configuration 3
    {
        double deltat = 0.1;
        std::vector<double> alphas = {1.0, 2.0, 4.0};
        std::vector<double> ds = {0.0001, 0.0002, 0.0004};

        for (const auto &alpha : alphas)
            for (const auto &d : ds)
            {
                Brain1D problem(N, degree, T, deltat, theta, d, alpha);
                problem.setup();
                problem.solve();
            }
    }

    return 0;
}
