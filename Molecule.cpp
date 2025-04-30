#include "Molecule.h"
#include "Overlap.h"
#include "SCF.h"
#include <iostream>
#include <fstream>
#include <sstream>

Molecule::Molecule()
    : num_atoms_(0), num_basis_functions_(0) {}

void Molecule::load_geometry(const std::string& filename) {
    std::cout << "Loading molecule from file: " << filename << std::endl;

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }

    std::string line;
    std::getline(infile, line);
    std::istringstream iss_num(line);
    iss_num >> num_atoms_;

    // Skip comment line
    std::getline(infile, line);

    coordinates_.set_size(num_atoms_, 3);
    atom_labels_.resize(num_atoms_);

    for (int i = 0; i < num_atoms_; ++i) {
        std::getline(infile, line);
        std::istringstream iss_atom(line);
        std::string element;
        double x, y, z;
        iss_atom >> element >> x >> y >> z;

        atom_labels_[i] = element;
        coordinates_(i, 0) = x;
        coordinates_(i, 1) = y;
        coordinates_(i, 2) = z;

        // Add basis functions for this atom
        if (element == "H") {
            basis_functions_.emplace_back(element, "s", coordinates_.row(i));
        } else if (element == "C" || element == "N" || element == "O") {
            basis_functions_.emplace_back(element, "s", coordinates_.row(i));
            basis_functions_.emplace_back(element, "px", coordinates_.row(i));
            basis_functions_.emplace_back(element, "py", coordinates_.row(i));
            basis_functions_.emplace_back(element, "pz", coordinates_.row(i));
        } else {
            std::cerr << "Warning: Unknown element " << element << ", adding only 's' orbital." << std::endl;
            basis_functions_.emplace_back(element, "s", coordinates_.row(i));
        }
    }

    num_basis_functions_ = basis_functions_.size();
    infile.close();

    std::cout << "Loaded " << num_atoms_ << " atoms and " << num_basis_functions_ << " basis functions." << std::endl;
}

void Molecule::build_overlap_matrix() {
    std::cout << "Building overlap matrix..." << std::endl;
    overlap_matrix_ = compute_overlap_matrix(basis_functions_);

    std::cout << "Overlap matrix S:" << std::endl;
    overlap_matrix_.print();
}

void Molecule::run_scf() {
    std::cout << "Running SCF procedure..." << std::endl;

    build_core_hamiltonian(fock_matrix_, overlap_matrix_, basis_functions_);
    initialize_density_matrix(density_matrix_, num_basis_functions_);
    run_scf_cycle(fock_matrix_, overlap_matrix_, fock_matrix_, density_matrix_, basis_functions_,
                  coefficient_matrix_, orbital_energies_, num_atoms_);

    std::cout << "Orbital energies (eigenvalues):" << std::endl;
    orbital_energies_.print();
}
