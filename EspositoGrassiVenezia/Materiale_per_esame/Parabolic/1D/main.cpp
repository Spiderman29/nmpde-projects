#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>
#include "Parabolic.hpp"

// Main function.
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const std::vector<unsigned int> N_s = {4, 9, 19, 39};
    const unsigned int degree = 1;

    const double T = 1.0;
    const std::vector<double> deltat_vector = {0.05, 0.005};
    const double theta = 1.0;
    std::vector<double> errors_L2;
    std::vector<double> errors_H1;
    for (const auto &deltat : deltat_vector)
    {
        for (const auto &N : N_s)
        {
            Parabolic1D problem(N, degree, T, deltat, theta);

            problem.setup();
            problem.solve();
            errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
            errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
        }
    }
    // Print the errors and estimate the convergence order.


    for (unsigned int i = 0; i < deltat_vector.size(); ++i)
    {   
        std::cout << "==============================================="
              << std::endl;
        std::ofstream convergence_file("convergence-" + std::to_string(deltat_vector[i]) + ".csv");
        convergence_file << "N,eL2,eH1" << std::endl;

        for (unsigned int n = 0; n < N_s.size(); ++n)
        {
            unsigned int index=i*N_s.size()+n;
            convergence_file << N_s[n] << "," << errors_L2[index] << ","
                             << errors_H1[index] << std::endl;

            std::cout << std::scientific << "N = " << std::setw(4)
                      << std::setprecision(2) << N_s[n];

            std::cout << std::scientific << " | eL2 = " << errors_L2[index];

            // Estimate the convergence order.
            if (n > 0)
            {
                const double p =
                    std::log(errors_L2[index] / errors_L2[index - 1]) /
                    std::log(N_s[n] / N_s[n - 1]);

                std::cout << " (" << std::fixed << std::setprecision(2)
                          << std::setw(4) << p << ")";
            }
            else
                std::cout << " (  - )";

            std::cout << std::scientific << " | eH1 = " << errors_H1[index];

            // Estimate the convergence order.
            if (n > 0)
            {
                const double p =
                    std::log(errors_H1[index] / errors_H1[index - 1]) /
                    std::log(N_s[n] / N_s[n - 1]);

                std::cout << " (" << std::fixed << std::setprecision(2)
                          << std::setw(4) << p << ")";
            }
            else
                std::cout << " (  - )";

            std::cout << "\n";
        }
    }

    return 0;
}