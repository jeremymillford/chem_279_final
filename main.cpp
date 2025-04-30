#include "Molecule.h"
#include <iostream>

int main() {
    Molecule mol;
    mol.load_geometry("water.xyz");
    mol.build_overlap_matrix();
    mol.run_scf();

    std::cout << "Finished CNDO/2 Overlap Calculation." << std::endl;
    return 0;
}

