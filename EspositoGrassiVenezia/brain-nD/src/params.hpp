#ifndef PARAMS_HPP
#define PARAMS_HPP
#include <string>
#include <vector>

struct Params{
    std::string const mesh_file_name;
    const unsigned int degree;
    const double T;
    const double deltat;
    const double theta;
    const std::vector<double> alpha;
    const std::vector<double> d_ext;
    const std::vector<double> d_axn;
    const std::string diffusion;
    
};

#endif