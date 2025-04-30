#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <string>
#include <unordered_map>

// Gamma parameters (in eV) for CNDO/2
static const std::unordered_map<std::string, double> gamma_parameters = {
    {"H", 12.85},
    {"C", 15.42},
    {"N", 14.45},
    {"O", 13.60}
};

// Ionization potentials (negative) for core Hamiltonian diagonal (in eV)
static const std::unordered_map<std::string, double> ionization_potentials = {
    {"H", -13.60},
    {"C", -11.26},
    {"N", -14.53},
    {"O", -13.62}
};

#endif // PARAMETERS_H
