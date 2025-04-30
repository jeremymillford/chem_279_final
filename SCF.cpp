#include "SCF.h"
#include "Parameters.h"
#include <iostream>
#include <cmath>

void build_core_hamiltonian(Matrix& h_core,
                             const Matrix& overlap_matrix,
                             const std::vector<BasisFunction>& basis_functions) {
    std::cout << "Building core Hamiltonian..." << std::endl;

    int n = overlap_matrix.n_rows;
    h_core.set_size(n, n);
    h_core.zeros();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                auto it = ionization_potentials.find(basis_functions[i].atom_label);
                if (it != ionization_potentials.end()) {
                    h_core(i, i) = it->second;
                } else {
                    std::cerr << "Unknown atom " << basis_functions[i].atom_label
                              << " for ionization potential." << std::endl;
                    h_core(i, i) = -10.0; // Rough guess
                }
            } else {
                // Off-diagonal elements: small overlap-scaled interaction
                h_core(i, j) = -0.5 * overlap_matrix(i, j);
            }
        }
    }
}

void initialize_density_matrix(Matrix& density_matrix, int n) {
    std::cout << "Initializing density matrix..." << std::endl;
    density_matrix.set_size(n, n);
    density_matrix.zeros();
}

void build_fock_matrix(Matrix& fock_matrix,
                       const Matrix& h_core,
                       const Matrix& density_matrix,
                       const std::vector<BasisFunction>& basis_functions) {
    int n = h_core.n_rows;
    fock_matrix.set_size(n, n);
    fock_matrix.zeros();

    fock_matrix = h_core;

    for (int mu = 0; mu < n; ++mu) {
        for (int nu = 0; nu < n; ++nu) {
            double coulomb_sum = 0.0;
            for (int A = 0; A < n; ++A) {
                auto it = gamma_parameters.find(basis_functions[A].atom_label);
                if (it != gamma_parameters.end()) {
                    double gamma_A = it->second;
                    coulomb_sum += density_matrix(A, A) * gamma_A;
                }
            }
            fock_matrix(mu, nu) += coulomb_sum;
        }
    }
}

void run_scf_cycle(Matrix& fock_matrix,
                   const Matrix& overlap_matrix,
                   const Matrix& h_core,
                   Matrix& density_matrix,
                   const std::vector<BasisFunction>& basis_functions,
                   Matrix& coefficient_matrix,
                   Vector& orbital_energies,
                   int num_electrons) {
    const int n = overlap_matrix.n_rows;
    const double convergence_threshold = 1e-5;
    const int max_iterations = 100;

    Matrix old_density;
    old_density.set_size(n, n);
    old_density.zeros();

    for (int iter = 0; iter < max_iterations; ++iter) {
        build_fock_matrix(fock_matrix, h_core, density_matrix, basis_functions);

        arma::mat S_inv_F = arma::solve(overlap_matrix, fock_matrix);
        S_inv_F = 0.5 * (S_inv_F + S_inv_F.t()); // Force symmetric

        arma::eig_sym(orbital_energies, coefficient_matrix, S_inv_F);

        // Build new density matrix
        old_density = density_matrix;
        density_matrix.zeros();

        int occ_orbitals = num_electrons / 2; // closed shell
        for (int mu = 0; mu < n; ++mu) {
            for (int nu = 0; nu < n; ++nu) {
                double sum = 0.0;
                for (int m = 0; m < occ_orbitals; ++m) {
                    sum += 2.0 * coefficient_matrix(mu, m) * coefficient_matrix(nu, m);
                }
                density_matrix(mu, nu) = sum;
            }
        }

        // Check convergence
        double rms_change = arma::accu(arma::square(density_matrix - old_density));
        rms_change = std::sqrt(rms_change / (n * n));
        std::cout << "Iteration " << iter+1 << ", RMS density change: " << rms_change << std::endl;

        if (rms_change < convergence_threshold) {
            std::cout << "SCF converged!" << std::endl;
            break;
        }

        if (iter == max_iterations - 1) {
            std::cerr << "SCF did not converge!" << std::endl;
        }
    }
}
