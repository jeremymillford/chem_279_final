#include <Eigen/Dense> 
#include <iomanip>
#include <iostream>

#include "Gaussian.cpp"
#include "Integrator.cpp"
#include "hw5_utils.cpp"

// This is a test program that explicitly runs both CNDO/2 and CNDO/S
// on H2 and compares the results

int main() {
    std::cout << "Simple test program for CNDO/S implementation" << std::endl;
    std::cout << "==============================================" << std::endl;
    
    // Create H2 molecule explicitly
    std::vector<Atom> atoms;
    
    Atom h1;
    h1.z_num = 1;
    h1.pos = Eigen::Vector3d(0, 0, 0);
    atoms.push_back(h1);
    
    Atom h2;
    h2.z_num = 1;
    h2.pos = Eigen::Vector3d(0.74, 0, 0);  // Standard H2 bond length
    atoms.push_back(h2);
    
    std::cout << "Created H2 molecule with bond length 0.74 Å" << std::endl;
    
    // Save original parameters so we can restore them later
    double original_h_beta = PARAMETER_INFO.at(1).at("-Beta");
    double original_h_i = PARAMETER_INFO.at(1).at("1/2(Is + As)");
    
    // Run standard CNDO/2
    std::cout << "\nRunning CNDO/2 calculation..." << std::endl;
    auto result_cndo2 = run_CNDO2(atoms, 1, 1);
    double energy_cndo2 = E_CNDO2(result_cndo2.first, result_cndo2.second, atoms);
    
    // Run CNDO/S with adjusted parameters
    std::cout << "\nAdjusting parameters for CNDO/S..." << std::endl;
    std::map<std::string, double>& h_params = 
        const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(1));
    
    h_params["-Beta"] = original_h_beta * 1.3;
    h_params["1/2(Is + As)"] = original_h_i * 0.8;
    
    std::cout << "Original H beta: " << original_h_beta << " eV" << std::endl;
    std::cout << "Modified H beta: " << h_params["-Beta"] << " eV" << std::endl;
    std::cout << "Original H I+A/2: " << original_h_i << " eV" << std::endl;
    std::cout << "Modified H I+A/2: " << h_params["1/2(Is + As)"] << " eV" << std::endl;
    
    std::cout << "\nRunning CNDO/S calculation..." << std::endl;
    auto result_cndos = run_CNDO_S(atoms, 1, 1);
    
    // Now calculate energy with the CNDO/S method
    double energy_cndos = E_CNDO_S(result_cndos.first, result_cndos.second, atoms);
    
    // Calculate nuclear repulsion energy for reference
    double nuclear_energy = get_nuclear_repulsion_energy(atoms);
    
    // Calculate electronic energies
    double elec_energy_cndo2 = energy_cndo2 - nuclear_energy;
    double elec_energy_cndos = energy_cndos - nuclear_energy;
    
    // Print results
    std::cout << "\nResults:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Nuclear repulsion energy: " << nuclear_energy << " eV" << std::endl;
    std::cout << std::endl;
    std::cout << "CNDO/2 electronic energy: " << elec_energy_cndo2 << " eV" << std::endl;
    std::cout << "CNDO/2 total energy:     " << energy_cndo2 << " eV" << std::endl;
    std::cout << std::endl;
    std::cout << "CNDO/S electronic energy: " << elec_energy_cndos << " eV" << std::endl;
    std::cout << "CNDO/S total energy:     " << energy_cndos << " eV" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Restore original parameters
    h_params["-Beta"] = original_h_beta;
    h_params["1/2(Is + As)"] = original_h_i;
    
    return 0;
}