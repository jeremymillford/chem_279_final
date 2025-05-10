/*
 * CNDO/S Analysis Program
 * 
 * This program implements several analyses for comparing CNDO/2 and CNDO/S:
 * 1. Bond length optimization for H₂
 * 2. Parameter optimization for H₂
 * 3. Testing with hydrocarbons
 * 
 * Usage: ./optimize_cndo_s [options]
 *   --bond-length          : Optimize H₂ bond length
 *   --parameters           : Optimize CNDO/S parameters
 *   --hydrocarbons         : Test with hydrocarbons
 *   --all                  : Run all analyses
 *   --use-optimized        : Use optimized parameters (when used with --hydrocarbons)
 *   --set-params h:beta:1.3,h:i:0.8,c:beta:0.9 : Set specific parameter values
 */

#include <Eigen/Dense> 
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <string>
#include <map>
#include <sstream>

// Include your existing files
#include "Gaussian.cpp"
#include "Integrator.cpp"
#include "hw5_utils.cpp"

//==============================================================================
// PARAMETER HANDLING
//==============================================================================

// Configuration structure to maintain parameters and options
struct CNDOConfig {
    // Parameter scaling factors
    double h_beta_scale = 1.0;
    double h_i_scale = 1.0;
    double c_beta_scale = 1.0;
    
    // Original parameter values (backup)
    double original_h_beta;
    double original_h_i;
    double original_c_beta;
    
    // Program options
    bool use_optimized_params = false;
    
    // Constructor to save original parameters
    CNDOConfig() {
        original_h_beta = PARAMETER_INFO.at(1).at("-Beta");
        original_h_i = PARAMETER_INFO.at(1).at("1/2(Is + As)");
        original_c_beta = PARAMETER_INFO.at(6).at("-Beta");
    }
    
    // Apply parameters using scale factors
    void apply_parameters() {
        std::map<std::string, double>& h_params = 
            const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(1));
        std::map<std::string, double>& c_params = 
            const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(6));
        
        h_params["-Beta"] = original_h_beta * h_beta_scale;
        h_params["1/2(Is + As)"] = original_h_i * h_i_scale;
        c_params["-Beta"] = original_c_beta * c_beta_scale;
    }
    
    // Restore original parameters
    void restore_parameters() {
        std::map<std::string, double>& h_params = 
            const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(1));
        std::map<std::string, double>& c_params = 
            const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(6));
        
        h_params["-Beta"] = original_h_beta;
        h_params["1/2(Is + As)"] = original_h_i;
        c_params["-Beta"] = original_c_beta;
    }
    
    // Load best parameters from bond length/parameter analysis
    void load_optimized_parameters() {
        h_beta_scale = 1.3;  // Best value from parameter optimization
        h_i_scale = 0.8;     // Best value from parameter optimization
        c_beta_scale = 0.9;  // Adjusted for carbon
        
        apply_parameters();
    }
    
    // Print current parameter values
    void print_parameters() {
        std::map<std::string, double>& h_params = 
            const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(1));
        std::map<std::string, double>& c_params = 
            const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(6));
        
        std::cout << "Current parameters:" << std::endl;
        std::cout << "  H Beta: " << h_params["-Beta"] 
                  << " (scale: " << h_beta_scale 
                  << ", original: " << original_h_beta << ")" << std::endl;
        std::cout << "  H I+A/2: " << h_params["1/2(Is + As)"] 
                  << " (scale: " << h_i_scale 
                  << ", original: " << original_h_i << ")" << std::endl;
        std::cout << "  C Beta: " << c_params["-Beta"] 
                  << " (scale: " << c_beta_scale 
                  << ", original: " << original_c_beta << ")" << std::endl;
        std::cout << std::endl;
    }
    
    // Parse parameter string in format element:param:value
    bool parse_parameter_string(const std::string& param_str) {
        std::istringstream ss(param_str);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            size_t pos1 = token.find(':');
            size_t pos2 = token.find(':', pos1 + 1);
            
            if (pos1 == std::string::npos || pos2 == std::string::npos) {
                std::cerr << "Invalid parameter format: " << token << std::endl;
                std::cerr << "Expected format: element:param:value (e.g., h:beta:1.3)" << std::endl;
                return false;
            }
            
            std::string element = token.substr(0, pos1);
            std::string param = token.substr(pos1 + 1, pos2 - pos1 - 1);
            std::string value_str = token.substr(pos2 + 1);
            double value;
            
            try {
                value = std::stod(value_str);
            } catch(const std::exception& e) {
                std::cerr << "Invalid number format: " << value_str << std::endl;
                return false;
            }
            
            if (element == "h" || element == "H") {
                if (param == "beta" || param == "Beta") {
                    h_beta_scale = value;
                } else if (param == "i" || param == "I") {
                    h_i_scale = value;
                } else {
                    std::cerr << "Unknown parameter for H: " << param << std::endl;
                    return false;
                }
            } else if (element == "c" || element == "C") {
                if (param == "beta" || param == "Beta") {
                    c_beta_scale = value;
                } else {
                    std::cerr << "Unknown parameter for C: " << param << std::endl;
                    return false;
                }
            } else {
                std::cerr << "Unknown element: " << element << std::endl;
                return false;
            }
        }
        
        return true;
    }
};

//==============================================================================
// ANALYSIS 1: BOND LENGTH OPTIMIZATION
//==============================================================================

// Function to optimize H₂ bond length with CNDO/S
void optimize_h2_bond_length(CNDOConfig& config) {
    std::cout << "H₂ Bond Length Optimization with CNDO/S" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    
    // Output both to console and file
    std::ofstream outfile("h2_bond_optimization.csv");
    outfile << "Distance,CNDO2_Energy,CNDOS_Energy,CNDOS_Gradient" << std::endl;
    
    std::cout << std::setw(12) << "Distance (Å)" 
              << std::setw(15) << "CNDO/2 (eV)" 
              << std::setw(15) << "CNDO/S (eV)" 
              << std::setw(15) << "Gradient" << std::endl;
    
    // Set parameters for CNDO/S (save current config first)
    double saved_h_beta = config.h_beta_scale;
    double saved_h_i = config.h_i_scale;
    
    config.h_beta_scale = 1.3;
    config.h_i_scale = 0.8;
    config.apply_parameters();
    
    double min_energy_cndo2 = std::numeric_limits<double>::max();
    double min_energy_cndos = std::numeric_limits<double>::max();
    double optimal_dist_cndo2 = 0.0;
    double optimal_dist_cndos = 0.0;
    double min_gradient = std::numeric_limits<double>::max();
    
    // Scan range of bond lengths
    for (double dist = 0.4; dist <= 1.2; dist += 0.05) {
        // Create H₂ with this bond length
        std::vector<Atom> atoms;
        
        Atom h1;
        h1.z_num = 1;
        h1.pos = Eigen::Vector3d(0, 0, 0);
        atoms.push_back(h1);
        
        Atom h2;
        h2.z_num = 1;
        h2.pos = Eigen::Vector3d(dist, 0, 0);
        atoms.push_back(h2);
        
        // Run CNDO/2 first for comparison
        auto result_cndo2 = run_CNDO2(atoms, 1, 1);
        double energy_cndo2 = E_CNDO2(result_cndo2.first, result_cndo2.second, atoms);
        
        // Now run CNDO/S
        auto result_cndos = run_CNDO_S(atoms, 1, 1);
        auto p_alpha = result_cndos.first;
        auto p_beta = result_cndos.second;
        
        // Calculate CNDO/S energy
        double energy_cndos = E_CNDO_S(p_alpha, p_beta, atoms);
        
        // Get basis and calculate gradient for CNDO/S
        auto vcg = get_vector_of_contracted_gaussians(atoms);
        auto vec_of_energy_dir = get_energy_derivative_vector(atoms, vcg, p_alpha, p_beta);
        double gradient_mag = vec_of_energy_dir(0).norm();  // Magnitude of gradient vector
        
        // Output to console and file
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(12) << dist
                  << std::setw(15) << energy_cndo2
                  << std::setw(15) << energy_cndos
                  << std::setw(15) << gradient_mag << std::endl;
        
        outfile << dist << "," << energy_cndo2 << "," << energy_cndos << "," << gradient_mag << std::endl;
        
        // Track minimum energies
        if (energy_cndo2 < min_energy_cndo2) {
            min_energy_cndo2 = energy_cndo2;
            optimal_dist_cndo2 = dist;
        }
        
        if (energy_cndos < min_energy_cndos) {
            min_energy_cndos = energy_cndos;
            optimal_dist_cndos = dist;
        }
        
        // Track minimum gradient
        if (gradient_mag < min_gradient) {
            min_gradient = gradient_mag;
        }
    }
    
    std::cout << "\nCNDO/2 optimal bond length: " << optimal_dist_cndo2 << " Å" << std::endl;
    std::cout << "CNDO/2 minimum energy: " << min_energy_cndo2 << " eV" << std::endl;
    
    std::cout << "\nCNDO/S optimal bond length: " << optimal_dist_cndos << " Å" << std::endl;
    std::cout << "CNDO/S minimum energy: " << min_energy_cndos << " eV" << std::endl;
    std::cout << "CNDO/S minimum gradient magnitude: " << min_gradient << " eV/Å" << std::endl;
    
    std::cout << "\nExperimental bond length: 0.74 Å" << std::endl;
    std::cout << "Experimental binding energy: -4.52 eV" << std::endl;
    
    outfile.close();
    std::cout << "Data saved to h2_bond_optimization.csv" << std::endl;
    
    // Restore original config values
    config.h_beta_scale = saved_h_beta;
    config.h_i_scale = saved_h_i;
    config.restore_parameters();
}

//==============================================================================
// ANALYSIS 2: PARAMETER OPTIMIZATION
//==============================================================================

// Function to optimize parameters for H₂
void optimize_parameters_for_h2(CNDOConfig& config) {
    std::cout << "\nParameter Optimization for CNDO/S with H₂" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Output both to console and file
    std::ofstream outfile("h2_parameter_optimization.csv");
    outfile << "BetaScale,IScale,Energy,EnergyDiff" << std::endl;
    
    // Create H₂ at experimental bond length
    std::vector<Atom> atoms;
    Atom h1; h1.z_num = 1; h1.pos = Eigen::Vector3d(0, 0, 0); atoms.push_back(h1);
    Atom h2; h2.z_num = 1; h2.pos = Eigen::Vector3d(0.74, 0, 0); atoms.push_back(h2);
    
    std::cout << std::setw(10) << "Beta Scale" 
              << std::setw(10) << "I Scale" 
              << std::setw(15) << "Energy (eV)"
              << std::setw(15) << "Diff from Exp" << std::endl;
    
    double best_energy_diff = std::numeric_limits<double>::max();
    double best_beta_scale = 0.0;
    double best_i_scale = 0.0;
    double best_energy = 0.0;
    
    // Target experimental binding energy (negative)
    const double experimental_energy = -4.52;  // eV
    
    // Run CNDO/2 for comparison
    auto result_cndo2 = run_CNDO2(atoms, 1, 1);
    double energy_cndo2 = E_CNDO2(result_cndo2.first, result_cndo2.second, atoms);
    std::cout << "CNDO/2 Energy: " << energy_cndo2 << " eV" << std::endl;
    std::cout << "Experimental Energy: " << experimental_energy << " eV" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    
    // Try different parameter combinations
    for (double beta_scale = 1.0; beta_scale <= 1.5; beta_scale += 0.1) {
        for (double i_scale = 0.7; i_scale <= 1.0; i_scale += 0.1) {
            // Set parameters for this iteration
            config.h_beta_scale = beta_scale;
            config.h_i_scale = i_scale;
            config.apply_parameters();
            
            try {
                // Run CNDO/S
                auto result = run_CNDO_S(atoms, 1, 1, 1e-6, false);
                double energy = E_CNDO_S(result.first, result.second, atoms);
                
                // Find parameters closest to experimental value
                double energy_diff = std::abs(energy - experimental_energy);
                
                std::cout << std::fixed << std::setprecision(2)
                          << std::setw(10) << beta_scale
                          << std::setw(10) << i_scale
                          << std::setw(15) << energy
                          << std::setw(15) << energy_diff << std::endl;
                
                outfile << beta_scale << "," << i_scale << "," << energy << "," << energy_diff << std::endl;
                
                if (energy_diff < best_energy_diff) {
                    best_energy_diff = energy_diff;
                    best_beta_scale = beta_scale;
                    best_i_scale = i_scale;
                    best_energy = energy;
                }
            }
            catch (const std::exception& e) {
                std::cout << std::setw(10) << beta_scale
                          << std::setw(10) << i_scale
                          << std::setw(15) << "ERROR"
                          << std::setw(15) << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "\nBest parameters:" << std::endl;
    std::cout << "  Beta scale: " << best_beta_scale << std::endl;
    std::cout << "  I scale: " << best_i_scale << std::endl;
    std::cout << "  Energy: " << best_energy << " eV" << std::endl;
    std::cout << "  Absolute difference from experimental: " << best_energy_diff << " eV" << std::endl;
    std::cout << "  Experimental energy: " << experimental_energy << " eV" << std::endl;
    
    // Save best parameters to config
    config.h_beta_scale = best_beta_scale;
    config.h_i_scale = best_i_scale;
    
    outfile.close();
    std::cout << "Data saved to h2_parameter_optimization.csv" << std::endl;
    
    // Restore original parameters
    config.restore_parameters();
}

//==============================================================================
// ANALYSIS 3: HYDROCARBON TESTING
//==============================================================================

// Create methane molecule programmatically
std::vector<Atom> create_methane() {
    std::vector<Atom> atoms;
    
    // Carbon at center
    Atom c;
    c.z_num = 6;
    c.pos = Eigen::Vector3d(0, 0, 0);
    atoms.push_back(c);
    
    // Four hydrogens in tetrahedral arrangement
    Atom h1;
    h1.z_num = 1;
    h1.pos = Eigen::Vector3d(0, 0, 1.09);
    atoms.push_back(h1);
    
    Atom h2;
    h2.z_num = 1;
    h2.pos = Eigen::Vector3d(1.03, 0, -0.36);
    atoms.push_back(h2);
    
    Atom h3;
    h3.z_num = 1;
    h3.pos = Eigen::Vector3d(-0.51, 0.89, -0.36);
    atoms.push_back(h3);
    
    Atom h4;
    h4.z_num = 1;
    h4.pos = Eigen::Vector3d(-0.51, -0.89, -0.36);
    atoms.push_back(h4);
    
    return atoms;
}

// Create ethane molecule programmatically
std::vector<Atom> create_ethane() {
    std::vector<Atom> atoms;
    
    // First carbon
    Atom c1;
    c1.z_num = 6;
    c1.pos = Eigen::Vector3d(0, 0, 0);
    atoms.push_back(c1);
    
    // Second carbon
    Atom c2;
    c2.z_num = 6;
    c2.pos = Eigen::Vector3d(1.54, 0, 0);  // C-C bond length 1.54 Å
    atoms.push_back(c2);
    
    // Hydrogens on first carbon
    Atom h1;
    h1.z_num = 1;
    h1.pos = Eigen::Vector3d(-0.36, 1.03, 0);
    atoms.push_back(h1);
    
    Atom h2;
    h2.z_num = 1;
    h2.pos = Eigen::Vector3d(-0.36, -0.51, 0.89);
    atoms.push_back(h2);
    
    Atom h3;
    h3.z_num = 1;
    h3.pos = Eigen::Vector3d(-0.36, -0.51, -0.89);
    atoms.push_back(h3);
    
    // Hydrogens on second carbon
    Atom h4;
    h4.z_num = 1;
    h4.pos = Eigen::Vector3d(1.90, 1.03, 0);
    atoms.push_back(h4);
    
    Atom h5;
    h5.z_num = 1;
    h5.pos = Eigen::Vector3d(1.90, -0.51, 0.89);
    atoms.push_back(h5);
    
    Atom h6;
    h6.z_num = 1;
    h6.pos = Eigen::Vector3d(1.90, -0.51, -0.89);
    atoms.push_back(h6);
    
    return atoms;
}

// Create ethylene molecule programmatically
std::vector<Atom> create_ethylene() {
    std::vector<Atom> atoms;
    
    // First carbon
    Atom c1;
    c1.z_num = 6;
    c1.pos = Eigen::Vector3d(0, 0, 0);
    atoms.push_back(c1);
    
    // Second carbon
    Atom c2;
    c2.z_num = 6;
    c2.pos = Eigen::Vector3d(1.33, 0, 0);  // C=C bond length 1.33 Å
    atoms.push_back(c2);
    
    // Hydrogens on first carbon
    Atom h1;
    h1.z_num = 1;
    h1.pos = Eigen::Vector3d(-0.23, 0.94, 0);
    atoms.push_back(h1);
    
    Atom h2;
    h2.z_num = 1;
    h2.pos = Eigen::Vector3d(-0.23, -0.94, 0);
    atoms.push_back(h2);
    
    // Hydrogens on second carbon
    Atom h3;
    h3.z_num = 1;
    h3.pos = Eigen::Vector3d(1.56, 0.94, 0);
    atoms.push_back(h3);
    
    Atom h4;
    h4.z_num = 1;
    h4.pos = Eigen::Vector3d(1.56, -0.94, 0);
    atoms.push_back(h4);
    
    return atoms;
}

// Create butane molecule programmatically (anti conformation)
std::vector<Atom> create_butane() {
    std::vector<Atom> atoms;
    const double C_C = 1.54;
    const double C_H = 1.09;

    // C-backbone
    std::array<Eigen::Vector3d,4> Cpos = {
        Eigen::Vector3d(0,0,0),
        Eigen::Vector3d(C_C,0,0),
        Eigen::Vector3d(2*C_C,0,0),
        Eigen::Vector3d(3*C_C,0,0)
    };
    for (auto &p : Cpos) {
        Atom c; c.z_num=6; c.pos=p; atoms.push_back(c);
    }

    // H's on C1:
    atoms.push_back(Atom{1, Eigen::Vector3d(-C_H,  0,   0)});
    atoms.push_back(Atom{1, Eigen::Vector3d( 0,    C_H, 0)});
    atoms.push_back(Atom{1, Eigen::Vector3d( 0,   -C_H, 0)});

    // H's on C2:
    atoms.push_back(Atom{1, Eigen::Vector3d(C_C,   C_H, 0)});
    atoms.push_back(Atom{1, Eigen::Vector3d(C_C,  -C_H, 0)});

    // H's on C3:
    atoms.push_back(Atom{1, Eigen::Vector3d(2*C_C, C_H, 0)});
    atoms.push_back(Atom{1, Eigen::Vector3d(2*C_C,-C_H, 0)});

    // H's on C4:
    atoms.push_back(Atom{1, Eigen::Vector3d(3*C_C + C_H,  0,   0)});
    atoms.push_back(Atom{1, Eigen::Vector3d(3*C_C,        C_H, 0)});
    atoms.push_back(Atom{1, Eigen::Vector3d(3*C_C,       -C_H, 0)});

    return atoms;
}

// Create benzene molecule programmatically (planar hexagon)
std::vector<Atom> create_benzene() {
    std::vector<Atom> atoms;
    const double C_C = 1.40;
    const double C_H = 1.09;
    const double R = C_C;
    const double R_H = R + C_H;

    // 6 carbons
    for (int i = 0; i < 6; ++i) {
        double θ = i * M_PI/3.0;
        Atom c;
        c.z_num = 6;
        c.pos = Eigen::Vector3d(R * cos(θ), R * sin(θ), 0);
        atoms.push_back(c);
    }
    // 6 hydrogens
    for (int i = 0; i < 6; ++i) {
        double θ = i * M_PI/3.0;
        Atom h;
        h.z_num = 1;
        h.pos = Eigen::Vector3d(R_H * cos(θ), R_H * sin(θ), 0);
        atoms.push_back(h);
    }
    return atoms;
}

// Function to test CNDO/S with hydrocarbons
void test_hydrocarbons(CNDOConfig& config) {
    std::cout << "\nTesting CNDO/S with Hydrocarbons" << std::endl;
    std::cout << "------------------------------" << std::endl;
    
    // Apply parameters based on configuration
    if (config.use_optimized_params) {
        std::cout << "Using optimized parameters from CNDO/S" << std::endl;
        config.load_optimized_parameters();
    } else {
        std::cout << "Using custom parameters:" << std::endl;
        config.apply_parameters();
    }
    
    config.print_parameters();
    
    // Create molecules programmatically
    std::vector<std::pair<std::string, std::vector<Atom>>> molecules = {
        {"Methane (CH₄)",   create_methane()},
        {"Ethane (C₂H₆)",   create_ethane()},
        {"Ethylene (C₂H₄)", create_ethylene()},
        {"Butane (C₄H₁₀)",  create_butane()},
        {"Benzene (C₆H₆)",  create_benzene()}
    };
    
    // Output both to console and file
    std::ofstream outfile("hydrocarbon_results.csv");
    outfile << "Molecule,CNDO2_Energy,CNDOS_Energy,Difference" << std::endl;
    
    std::cout << std::setw(20) << "Molecule" 
              << std::setw(15) << "CNDO/2 (eV)" 
              << std::setw(15) << "CNDO/S (eV)" 
              << std::setw(15) << "Difference" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    for (const auto& [name, atoms] : molecules) {
        try {
            // Get electron counts
            int total_valence_electrons = get_total_valence_electrons(atoms);
            int q = total_valence_electrons / 2;
            int p = total_valence_electrons - q;
            
            // Run CNDO/2
            auto result_cndo2 = run_CNDO2(atoms, p, q);
            double energy_cndo2 = E_CNDO2(result_cndo2.first, result_cndo2.second, atoms);
            
            // Run CNDO/S
            auto result_cndos = run_CNDO_S(atoms, p, q);
            double energy_cndos = E_CNDO_S(result_cndos.first, result_cndos.second, atoms);
            
            // Calculate difference
            double difference = energy_cndos - energy_cndo2;
            
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(20) << name
                      << std::setw(15) << energy_cndo2
                      << std::setw(15) << energy_cndos
                      << std::setw(15) << difference << std::endl;
            
            outfile << name << "," << energy_cndo2 << "," << energy_cndos << "," << difference << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error with " << name << ": " << e.what() << std::endl;
        }
    }
    
    outfile.close();
    std::cout << "Data saved to hydrocarbon_results.csv" << std::endl;
    
    // Restore original parameters
    config.restore_parameters();
}

//==============================================================================
// MAIN FUNCTION
//==============================================================================

// Main function to run all analyses
int main(int argc, char** argv) {
    // Create and initialize config
    CNDOConfig config;
    
    // Parse command line options
    bool run_bond_length = false;
    bool run_parameters = false;
    bool run_hydrocarbons = false;
    bool param_string_provided = false;
    std::string param_string;
    
    // Process command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--bond-length") {
            run_bond_length = true;
        }
        else if (arg == "--parameters") {
            run_parameters = true;
        }
        else if (arg == "--hydrocarbons") {
            run_hydrocarbons = true;
        }
        else if (arg == "--all") {
            run_bond_length = true;
            run_parameters = true;
            run_hydrocarbons = true;
        }
        else if (arg == "--use-optimized") {
            config.use_optimized_params = true;
        }
        else if (arg.substr(0, 12) == "--set-params") {
            if (i + 1 < argc) {
                param_string = argv[++i];
                param_string_provided = true;
            } else {
                std::cerr << "Error: --set-params requires parameter string" << std::endl;
                return 1;
            }
        }
        else {
            std::cout << "Unknown option: " << arg << std::endl;
            std::cout << "Available options:" << std::endl;
            std::cout << "  --bond-length          : Optimize H₂ bond length" << std::endl;
            std::cout << "  --parameters           : Optimize CNDO/S parameters" << std::endl;
            std::cout << "  --hydrocarbons         : Test with hydrocarbons" << std::endl;
            std::cout << "  --all                  : Run all analyses" << std::endl;
            std::cout << "  --use-optimized        : Use optimized parameters (when used with --hydrocarbons)" << std::endl;
            std::cout << "  --set-params [params]  : Set specific parameter values (format: h:beta:1.3,h:i:0.8,c:beta:0.9)" << std::endl;
            return 1;
        }
    }
    
    // Apply custom parameters if provided
    if (param_string_provided) {
        if (!config.parse_parameter_string(param_string)) {
            return 1;  // Error already printed by parse_parameter_string
        }
    }
    
    // If no specific analysis was requested, show usage
    if (!run_bond_length && !run_parameters && !run_hydrocarbons) {
        std::cout << "CNDO/S Analysis Program" << std::endl;
        std::cout << "---------------------" << std::endl;
        std::cout << "Usage: ./optimize_cndo_s [options]" << std::endl;
        std::cout << "Available options:" << std::endl;
        std::cout << "  --bond-length          : Optimize H₂ bond length" << std::endl;
        std::cout << "  --parameters           : Optimize CNDO/S parameters" << std::endl;
        std::cout << "  --hydrocarbons         : Test with hydrocarbons" << std::endl;
        std::cout << "  --all                  : Run all analyses" << std::endl;
        std::cout << "  --use-optimized        : Use optimized parameters (with --hydrocarbons)" << std::endl;
        std::cout << "  --set-params [params]  : Set specific parameter values" << std::endl;
        std::cout << "                           Format: element:param:value,..." << std::endl;
        std::cout << "                           Example: h:beta:1.3,h:i:0.8,c:beta:0.9" << std::endl;
        return 0;
    }
    
    // Run the requested analyses
    if (run_bond_length) {
        optimize_h2_bond_length(config);
    }
    
    if (run_parameters) {
        optimize_parameters_for_h2(config);
    }
    
    if (run_hydrocarbons) {
        // If running all analyses and optimized params is requested
        // but --use-optimized wasn't explicitly specified,
        // print a note about using parameters from the current run
        if (run_parameters && !config.use_optimized_params && param_string_provided) {
            std::cout << "\nNote: Using custom parameters for hydrocarbon analysis." << std::endl;
            std::cout << "Add --use-optimized to use the best parameters from parameter optimization." << std::endl;
        }
        
        test_hydrocarbons(config);
    }
    
    return 0;
}

//usage g++ -std=c++20 -Wall -I/usr/include/eigen3 -o optimize_cndo_s optimize_cndo_s.cpp