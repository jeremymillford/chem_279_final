#include <Eigen/Dense> 
#include <iomanip>
#include <iostream>
#include <vector>

#include "Gaussian.cpp"
#include "Integrator.cpp"
#include "hw5_utils.cpp"

// This test program focuses solely on H₂ and includes detailed diagnostics
int main() {
    std::cout << "CNDO/S Diagnostics for H₂" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Create H₂ molecule at experimental bond length
    std::vector<Atom> atoms;
    
    Atom h1;
    h1.z_num = 1;
    h1.pos = Eigen::Vector3d(0, 0, 0);
    atoms.push_back(h1);
    
    Atom h2;
    h2.z_num = 1;
    h2.pos = Eigen::Vector3d(0.74, 0, 0);
    atoms.push_back(h2);
    
    std::cout << "Created H₂ with bond length 0.74 Å" << std::endl;
    
    // Get basis functions and overlap matrix
    std::vector<ContractedGaussian> vcg = get_vector_of_contracted_gaussians(atoms);
    Eigen::MatrixXd S = make_overlap_matrix(vcg);
    
    std::cout << "\nOverlap matrix:" << std::endl;
    std::cout << S << std::endl;
    
    // Calculate gamma matrix
    Eigen::MatrixXd gamma = get_gamma(atoms);
    
    std::cout << "\nGamma matrix:" << std::endl;
    std::cout << gamma << std::endl;
    
    // Print original parameters
    std::cout << "\nOriginal parameters:" << std::endl;
    std::cout << "  H Beta: " << PARAMETER_INFO.at(1).at("-Beta") << " eV" << std::endl;
    std::cout << "  H I+A/2: " << PARAMETER_INFO.at(1).at("1/2(Is + As)") << " eV" << std::endl;
    
    // Run CNDO/2 for comparison
    std::cout << "\nRunning CNDO/2..." << std::endl;
    auto result_cndo2 = run_CNDO2(atoms, 1, 1);
    double energy_cndo2 = E_CNDO2(result_cndo2.first, result_cndo2.second, atoms);
    
    std::cout << "\nCNDO/2 results:" << std::endl;
    std::cout << "  Alpha density matrix:" << std::endl;
    std::cout << result_cndo2.first << std::endl;
    std::cout << "  Beta density matrix:" << std::endl;
    std::cout << result_cndo2.second << std::endl;
    std::cout << "  Total energy: " << energy_cndo2 << " eV" << std::endl;
    
    // Try several parameter combinations for CNDO/S
    std::vector<std::pair<double, double>> param_sets = {
        {1.0, 1.0},  // Original parameters
        {1.3, 0.8},  // Recommended parameters
        {1.5, 0.7},  // Stronger adjustment
        {2.0, 0.5}   // Extreme adjustment
    };
    
    std::cout << "\nRunning CNDO/S with different parameters:" << std::endl;
    std::cout << std::setw(10) << "Beta Scale" 
              << std::setw(10) << "I Scale" 
              << std::setw(15) << "Energy (eV)"
              << std::setw(15) << "O.E. Energy" 
              << std::setw(15) << "T.E. Energy" 
              << std::setw(15) << "Nuclear" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    
    for (const auto& [beta_scale, i_scale] : param_sets) {
        // Save original parameters
        double original_h_beta = PARAMETER_INFO.at(1).at("-Beta");
        double original_h_i = PARAMETER_INFO.at(1).at("1/2(Is + As)");
        
        // Adjust parameters
        std::map<std::string, double>& h_params = 
            const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(1));
        
        h_params["-Beta"] = original_h_beta * beta_scale;
        h_params["1/2(Is + As)"] = original_h_i * i_scale;
        
        try {
            // Run CNDO/S
            auto result = run_CNDO_S(atoms, 1, 1, 1e-6, false);
            
            // Calculate energy components
            Eigen::MatrixXd p_total = result.first + result.second;
            Eigen::MatrixXd h_matrix = get_hamiltonian(atoms);
            Eigen::MatrixXd f_alpha = get_fock(atoms, p_total, result.first);
            Eigen::MatrixXd f_beta = get_fock(atoms, p_total, result.second);
            
            double e_one_electron = 0.0;
            double e_two_electron = 0.0;
            int n = S.rows();
            
            for (int u = 0; u < n; u++) {
                for (int v = 0; v < n; v++) {
                    double p_tot_uv = result.first(u,v) + result.second(u,v);
                    e_one_electron += p_tot_uv * h_matrix(u,v) * S(u,v);
                    
                    double two_e_alpha = (f_alpha(u,v) - h_matrix(u,v)) * S(u,v);
                    double two_e_beta = (f_beta(u,v) - h_matrix(u,v)) * S(u,v);
                    e_two_electron += 0.5 * (result.first(u,v) * two_e_alpha + 
                                            result.second(u,v) * two_e_beta);
                }
            }
            
            double nuclear_energy = 0.0;
            for (size_t A = 0; A < atoms.size(); A++) {
                for (size_t B = A + 1; B < atoms.size(); B++) {
                    nuclear_energy += (get_Z(atoms.at(A)) * get_Z(atoms.at(B))) / 
                                    (atoms.at(A).pos - atoms.at(B).pos).norm();
                }
            }
            nuclear_energy *= CONVERSION_FACTOR;
            
            double total_energy = e_one_electron + e_two_electron + nuclear_energy;
            
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(10) << beta_scale
                      << std::setw(10) << i_scale
                      << std::setw(15) << total_energy
                      << std::setw(15) << e_one_electron
                      << std::setw(15) << e_two_electron
                      << std::setw(15) << nuclear_energy << std::endl;
                      
            // Print detailed matrices for the first parameter set
            if (beta_scale == 1.0 && i_scale == 1.0) {
                std::cout << "\nDetailed diagnostics for original parameters:" << std::endl;
                std::cout << "Alpha density matrix:" << std::endl;
                std::cout << result.first << std::endl;
                std::cout << "Beta density matrix:" << std::endl;
                std::cout << result.second << std::endl;
                std::cout << "Hamiltonian matrix:" << std::endl;
                std::cout << h_matrix << std::endl;
                std::cout << "Alpha Fock matrix:" << std::endl;
                std::cout << f_alpha << std::endl;
                std::cout << "Beta Fock matrix:" << std::endl;
                std::cout << f_beta << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << std::setw(10) << beta_scale
                      << std::setw(10) << i_scale
                      << std::setw(15) << "ERROR"
                      << std::setw(15) << e.what() << std::endl;
        }
        
        // Restore original parameters
        h_params["-Beta"] = original_h_beta;
        h_params["1/2(Is + As)"] = original_h_i;
    }
    
    // Try with even stronger parameter adjustments
    std::cout << "\nTrying more extreme parameter adjustments:" << std::endl;
    
    for (double beta_scale = 2.0; beta_scale <= 5.0; beta_scale += 1.0) {
        for (double i_scale = 0.2; i_scale <= 0.5; i_scale += 0.1) {
            // Save original parameters
            double original_h_beta = PARAMETER_INFO.at(1).at("-Beta");
            double original_h_i = PARAMETER_INFO.at(1).at("1/2(Is + As)");
            
            // Adjust parameters
            std::map<std::string, double>& h_params = 
                const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(1));
            
            h_params["-Beta"] = original_h_beta * beta_scale;
            h_params["1/2(Is + As)"] = original_h_i * i_scale;
            
            try {
                // Run CNDO/S
                auto result = run_CNDO_S(atoms, 1, 1, 1e-6, false);
                double energy = E_CNDO_S(result.first, result.second, atoms);
                
                std::cout << std::fixed << std::setprecision(2)
                          << std::setw(10) << beta_scale
                          << std::setw(10) << i_scale
                          << std::setw(15) << energy << std::endl;
            }
            catch (const std::exception& e) {
                std::cout << std::setw(10) << beta_scale
                          << std::setw(10) << i_scale
                          << std::setw(15) << "ERROR"
                          << std::setw(15) << e.what() << std::endl;
            }
            
            // Restore original parameters
            h_params["-Beta"] = original_h_beta;
            h_params["1/2(Is + As)"] = original_h_i;
        }
    }
    
    std::cout << "\nCNDO/2 energy: " << energy_cndo2 << " eV" << std::endl;
    std::cout << "Experimental binding energy: -4.52 eV" << std::endl;
    
    return 0;
}