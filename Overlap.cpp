#include "Overlap.h"
#include <iostream>
#include <cmath>

Matrix compute_overlap_matrix(const std::vector<BasisFunction>& basis_functions) {
    std::cout << "Computing overlap matrix..." << std::endl;

    const int n = basis_functions.size();
    Matrix overlap(n, n, arma::fill::zeros);

    const double beta = 0.5;  // Decay parameter (adjustable)

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double dx = basis_functions[i].center(0) - basis_functions[j].center(0);
            double dy = basis_functions[i].center(1) - basis_functions[j].center(1);
            double dz = basis_functions[i].center(2) - basis_functions[j].center(2);
            double r2 = dx*dx + dy*dy + dz*dz;
            double S = std::exp(-beta * r2);

            // Adjust for orbital types
            if (basis_functions[i].orbital_type == "s" && basis_functions[j].orbital_type == "s") {
                // s-s overlap, S unchanged
            } else if (basis_functions[i].orbital_type[0] == 'p' && basis_functions[j].orbital_type[0] == 'p') {
                // p-p overlap: directional
                double pi_dot = 0.0;
                if (basis_functions[i].orbital_type == basis_functions[j].orbital_type) {
                    if (basis_functions[i].orbital_type == "px") pi_dot = dx;
                    if (basis_functions[i].orbital_type == "py") pi_dot = dy;
                    if (basis_functions[i].orbital_type == "pz") pi_dot = dz;
                    S *= pi_dot * pi_dot;
                } else {
                    S = 0.0; // Different p directions: negligible overlap
                }
            } else {
                // s-p or p-s overlap: linear in distance along p-axis
                std::string p_orbital, s_orbital;
                int p_index = -1;

                if (basis_functions[i].orbital_type == "s" && basis_functions[j].orbital_type[0] == 'p') {
                    p_orbital = basis_functions[j].orbital_type;
                    p_index = j;
                } else if (basis_functions[i].orbital_type[0] == 'p' && basis_functions[j].orbital_type == "s") {
                    p_orbital = basis_functions[i].orbital_type;
                    p_index = i;
                }

                if (p_index != -1) {
                    double dist_component = 0.0;
                    if (p_orbital == "px") dist_component = dx;
                    if (p_orbital == "py") dist_component = dy;
                    if (p_orbital == "pz") dist_component = dz;
                    S *= dist_component;
                } else {
                    S = 0.0; // Should not happen
                }
            }

            overlap(i, j) = S;
            overlap(j, i) = S; // Symmetric
        }
    }

    return overlap;
}
