#ifndef MOLECULE_H
#define MOLECULE_H

#include "Arma_Utils.h"
#include <vector>
#include <string>

// Define a basis function
struct BasisFunction {
    std::string atom_label;
    std::string orbital_type; // "s", "px", "py", "pz"
    arma::rowvec center;       // 1x3 vector for (x,y,z)

    BasisFunction(const std::string& label, const std::string& orb_type, const arma::rowvec& pos)
        : atom_label(label), orbital_type(orb_type), center(pos) {}
};

class Molecule {
public:
    Molecule();

    void load_geometry(const std::string& filename);
    void build_overlap_matrix();
    void run_scf();

private:
    int num_atoms_;
    int num_basis_functions_;
    arma::mat coordinates_; // Atomic coordinates (atom centers)
    std::vector<std::string> atom_labels_; // Atom labels (O, H, etc.)

    std::vector<BasisFunction> basis_functions_; // NEW: list of basis functions!

    Matrix overlap_matrix_;
    Matrix fock_matrix_;
    Matrix coefficient_matrix_;
    Vector orbital_energies_;
    Matrix density_matrix_;
};

#endif // MOLECULE_H
