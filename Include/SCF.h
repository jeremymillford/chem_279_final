#ifndef SCF_H
#define SCF_H

#include "Arma_Utils.h"
#include <vector>
#include <string>
#include "Molecule.h" // <- to get BasisFunction

void build_core_hamiltonian(Matrix& h_core,
                            const Matrix& overlap_matrix,
                            const std::vector<BasisFunction>& basis_functions);

void initialize_density_matrix(Matrix& density_matrix, int num_basis_functions);

void build_fock_matrix(Matrix& fock_matrix,
                       const Matrix& h_core,
                       const Matrix& density_matrix,
                       const std::vector<BasisFunction>& basis_functions);

void run_scf_cycle(Matrix& fock_matrix,
                   const Matrix& overlap_matrix,
                   const Matrix& h_core,
                   Matrix& density_matrix,
                   const std::vector<BasisFunction>& basis_functions,
                   Matrix& coefficient_matrix,
                   Vector& orbital_energies,
                   int num_electrons);

#endif // SCF_H
