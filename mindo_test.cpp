#include <iostream>
#include "austin_impl_MINDO.cpp"
#include <Eigen/Dense> 
#include <iomanip>
#include <numeric>
#include <iostream>
#include <string>

#include "Gaussian.cpp"
#include "Integrator.cpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <molecule_file.txt> <num_alpha_electrons> <num_beta_electrons>" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    int alpha = std::stoi(argv[2]);
    int beta  = std::stoi(argv[3]);

    try {
        std::vector<Atom> atoms = parse_file(filepath, true);
        auto [p_alpha, p_beta] = run_CNDO2(atoms, alpha, beta); // MINDO reuses CNDO2 machinery here
        double energy = E_CNDO2(p_alpha, p_beta, atoms); // again, just a naming thing
        std::cout << "Total MINDO energy: " << energy << " eV" << std::endl;
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }

    return 0;
}
