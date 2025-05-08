#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hw5_utils.cpp"
#include "Gaussian.cpp"
#include "Integrator.cpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        throw std::runtime_error("No input file supplied");
    }

    std::string filepath = argv[1];
    auto atoms = parse_file(filepath, false);

    int p = 0, q = 0;
    if (argc == 4 || argc == 5) {
        p = std::stoi(argv[2]);
        q = std::stoi(argv[3]);
    } else if (argc == 2) {
        int total = get_total_valence_electrons(atoms);
        q = total / 2;
        p = total - q;
    } else {
        throw std::invalid_argument("argc must be 2, 4, or 5");
    }

    bool use_overlap = (argc == 5 && std::string(argv[4]) == "--overlap");

    Eigen::MatrixXd F, P;
    if (use_overlap) {
        std::tie(F, P) = run_CNDO_S(atoms, p, q);
        std::cout << "[CNDO/S] Using overlap matrix (S ≠ I)" << std::endl;
    } else {
        std::tie(F, P) = run_CNDO2(atoms, p, q);
        std::cout << "[CNDO/2] Using orthonormal basis (S = I)" << std::endl;
    }

    // Energies
    double nuclear_energy   = get_nuclear_repulsion_energy(atoms);
    Eigen::MatrixXd H       = get_hamiltonian(atoms);
    double electron_energy = scf_electronic_energy(P, H, F);

    std::cout << "Nuclear Repulsion Energy is "   << nuclear_energy   << " eV." << std::endl;
    std::cout << "Electron Energy is "           << electron_energy << " eV." << std::endl;

    // Fock diagnostics
    std::cout << "x" << std::endl;
    auto X = get_x_matrix(atoms, P);
    std::cout << X << std::endl;

    std::cout << "y" << std::endl;
    auto Y = get_y_matrix(atoms, P, P);
    std::cout << Y << std::endl;

    // Overlap derivative diagnostics
    std::cout << "Suv_RA (A is the center of u)" << std::endl;
    auto vcg      = get_vector_of_contracted_gaussians(atoms);
    auto vec_s    = get_vector_of_s_dir(atoms, vcg);
    auto s_dir    = vec_s(0);
    int dim       = (int)vcg.size();
    for (int u = 0; u < dim; ++u) {
        for (int v = 0; v < dim; ++v) {
            auto d = s_dir(u,v);
            std::cout
                << "s " << u << v << "/dR_A"
                << " x:" << d.x()
                << " y:" << d.y()
                << " z:" << d.z()
                << std::endl;
        }
    }

    std::cout << "gammaAB_RA" << std::endl;
    auto vec_g    = get_vector_of_gamma_dir(atoms, vcg);
    auto g_dir    = vec_g(0);
    int Natom     = (int)atoms.size();
    for (int i = 0; i < Natom; ++i) {
        for (int j = 0; j < Natom; ++j) {
            auto d = g_dir(i,j);
            std::cout
                << "gamma " << i << j << "/dR_A"
                << " x:" << d.x()
                << " y:" << d.y()
                << " z:" << d.z()
                << std::endl;
        }
    }

    // Gradient diagnostics
    std::cout << "gradient (Nuclear part)" << std::endl;
    auto nuc_grad = get_nuclear_energy_dir(atoms);
    for (auto& g : nuc_grad) {
        std::cout
            << "x: " << g.x()
            << " y: " << g.y()
            << " z: " << g.z()
            << std::endl;
    }

    std::cout << "gradient (Electron part)" << std::endl;
    auto ele_grad = get_energy_derivative_vector(atoms, vcg, P, P);
    for (int i = 0; i < (int)ele_grad.size(); ++i) {
        auto total_g = ele_grad[i] - nuc_grad[i];
        std::cout
            << "x: " << total_g.x()
            << " y: " << total_g.y()
            << " z: " << total_g.z()
            << std::endl;
    }

    std::cout << "gradient" << std::endl;
    for (auto& g : ele_grad) {
        std::cout
            << "x: " << g.x()
            << " y: " << g.y()
            << " z: " << g.z()
            << std::endl;
    }

    return 0;
}
