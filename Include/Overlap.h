#ifndef OVERLAP_H
#define OVERLAP_H

#include "Arma_Utils.h"
#include <vector>
#include <string>
#include "Molecule.h" // Need BasisFunction

// Compute overlap matrix given list of basis functions
Matrix compute_overlap_matrix(const std::vector<BasisFunction>& basis_functions);

#endif // OVERLAP_H
