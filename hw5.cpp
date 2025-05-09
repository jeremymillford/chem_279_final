#include <Eigen/Dense> 
#include <iomanip>
#include <numeric>
#include <iostream>
#include <string>

#include "Gaussian.cpp"
#include "Integrator.cpp"
#include "hw5_utils.cpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./hw5 <molecule-file> [--overlap]" << std::endl;
        return 1;
    }

    // Parse command line arguments
    std::string filepath = argv[1];
    bool use_overlap = false;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--overlap") {
            use_overlap = true;
        }
    }
    
    try {
        // Load molecule
        std::vector<Atom> atoms = parse_file(filepath, false);
        auto vcg = get_vector_of_contracted_gaussians(atoms);
        
        // Determine electron counts
        int total_valence_electrons = get_total_valence_electrons(atoms);
        int q = total_valence_electrons / 2;
        int p = total_valence_electrons - q;
        
        std::cout << "Loaded molecule: " << filepath << std::endl;
        std::cout << "Number of atoms: " << atoms.size() << std::endl;
        std::cout << "Alpha electrons: " << p << ", Beta electrons: " << q << std::endl;
        
        // Run calculation
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> res;
        double nuclear_energy, total_energy, electron_energy;
        
        // If using overlap, modify parameters
        if (use_overlap) {
            std::cout << "[CNDO/S] Using overlap matrix (S ≠ I)" << std::endl;
            
            // Save original parameters
            double original_h_beta = PARAMETER_INFO.at(1).at("-Beta");
            double original_h_i = PARAMETER_INFO.at(1).at("1/2(Is + As)");
            
            // Modify parameters
            std::map<std::string, double>& h_params = 
                const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(1));
            
            h_params["-Beta"] = original_h_beta * 1.3;
            h_params["1/2(Is + As)"] = original_h_i * 0.8;
            
            std::cout << "Modified parameters for H:" << std::endl;
            std::cout << "  Beta: " << h_params["-Beta"] << " (original: " << original_h_beta << ")" << std::endl;
            std::cout << "  I+A/2: " << h_params["1/2(Is + As)"] << " (original: " << original_h_i << ")" << std::endl;
            
            // Run CNDO/S
            res = run_CNDO_S(atoms, p, q);
            
            // Calculate energy using E_CNDO_S function
            total_energy = E_CNDO_S(res.first, res.second, atoms);
            nuclear_energy = get_nuclear_repulsion_energy(atoms);
            electron_energy = total_energy - nuclear_energy;
            
            // Restore original parameters
            h_params["-Beta"] = original_h_beta;
            h_params["1/2(Is + As)"] = original_h_i;
        } 
        else {
            // Standard CNDO/2 calculation
            std::cout << "[CNDO/2] Using orthonormal basis (S = I)" << std::endl;
            
            res = run_CNDO2(atoms, p, q);
            nuclear_energy = get_nuclear_repulsion_energy(atoms);
            total_energy = E_CNDO2(res.first, res.second, atoms);
            electron_energy = total_energy - nuclear_energy;
        }
        
        auto p_alpha = res.first;
        auto p_beta = res.second;
        
        // Display energy results
        std::cout << std::fixed << std::setprecision(6) << std::endl;
        std::cout << "Energy Results:" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        std::cout << "Nuclear Repulsion Energy: " << nuclear_energy << " eV" << std::endl;
        std::cout << "Electronic Energy: " << electron_energy << " eV" << std::endl;
        std::cout << "Total Energy: " << total_energy << " eV" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        
        // Display matrices
        std::cout << "x" << std::endl; 
        auto x = get_x_matrix(atoms, p_alpha + p_beta);
        std::cout << x << std::endl; 
        
        std::cout << "y" << std::endl; 
        auto y = get_y_matrix(atoms, p_alpha, p_beta); 
        std::cout << y << std::endl;
        
        // Calculate gradients
        std::cout << "Suv_RA (A is the center of u)" << std::endl; 
        auto vec_s_dir = get_vector_of_s_dir(atoms, vcg); 
        auto s_dir = vec_s_dir(0);
        for (int u = 0; u < vcg.size(); u++) {
            for (int v = 0; v < vcg.size(); v++) {
                std::cout 
                    << std::fixed
                    << std::setprecision(4)
                    << std::showpos
                    << "s " 
                    << std::to_string(u) 
                    << std::to_string(v) 
                    << "/dR_A" 
                    << " is: x:" 
                    << s_dir(u,v).x()  
                    << " y:" 
                    << s_dir(u,v).y() 
                    << " z: " 
                    << s_dir(u,v).z() 
                    << std::endl; 
            }
        }
        
        std::cout << "gammaAB_RA" << std::endl;
        auto vec_of_gamma_dir = get_vector_of_gamma_dir(atoms,vcg); 
        auto gamma_dir = vec_of_gamma_dir(0); 
        for (int i = 0; i < atoms.size(); i++) {
            for (int j = 0; j < atoms.size(); j++) {
                std::cout 
                    << std::fixed
                    << std::setprecision(4)
                    << std::showpos
                    << "gamma " 
                    << std::to_string(i) 
                    << std::to_string(j) 
                    << "/dR_A" 
                    << " is: x:" 
                    << gamma_dir(i,j).x()  
                    << " y:" 
                    << gamma_dir(i,j).y() 
                    << " z: " 
                    << gamma_dir(i,j).z() 
                    << std::endl; 
            }
        }
        
        std::cout << "gradient (Nuclear part)" << std::endl; 
        auto nuclear_energy_dir = get_nuclear_energy_dir(atoms);
        for (auto grad : nuclear_energy_dir) {
            std::cout << "x: " << grad.x() << " y: " << grad.y() << " z: " << grad.z() << std::endl;
        }
        
        auto vec_of_energy_dir = get_energy_derivative_vector(atoms, vcg, p_alpha, p_beta); 
        std::cout << "gradient (Electron part)" << std::endl; 
        for (int i = 0; i < vec_of_energy_dir.size(); i++) {
            auto nuclear_dir = nuclear_energy_dir(i); 
            auto energy_dir = vec_of_energy_dir(i); 
            auto electron_dir = energy_dir - nuclear_dir; 
            std::cout << "x: " << electron_dir.x() << " y: " << electron_dir.y() << " z: " << electron_dir.z() << std::endl;
        }
        
        std::cout << "gradient" << std::endl;
        for (auto energy_dir : vec_of_energy_dir) {
            std::cout << "x: " << energy_dir.x() << " y: " << energy_dir.y() << " z: " << energy_dir.z() << std::endl;  
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}