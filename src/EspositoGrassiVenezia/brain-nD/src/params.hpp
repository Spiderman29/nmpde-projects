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
    const std::map<unsigned int, double> alpha;
    const std::map<unsigned int, double> d_ext;
    const std::map<unsigned int, double> d_axn;
    const std::string diffusion;
};

#endif