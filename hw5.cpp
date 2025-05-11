#include <Eigen/Dense> 
#include <iomanip>
#include <numeric>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include "Gaussian.cpp"
#include "Integrator.cpp"
#include "hw5_utils.cpp"

// Function to run CNDO/S with simplified model
std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
run_cndo_s_simplified(const std::vector<Atom>& atoms, int p, int q, const double tol = 1e-6, const bool verbose = false) {
    // Step 1: Build basis and overlap matrices
    std::vector<ContractedGaussian> vcg = get_vector_of_contracted_gaussians(atoms);
    Eigen::MatrixXd S = make_overlap_matrix(vcg);
    int n = S.rows();

    // Initial density matrices (zero)
    Eigen::MatrixXd p_alpha = Eigen::MatrixXd::Zero(n, n);
    Eigen::MatrixXd p_beta = Eigen::MatrixXd::Zero(n, n);

    // Core Hamiltonian (one-electron part)
    Eigen::MatrixXd h = get_hamiltonian(atoms);
    
    // SCF convergence parameters
    bool converged = false;
    int iter = 0;
    int max_iter = 100;
    
    // Damping factor to help convergence
    double damp_factor = 0.7;  // Increased damping for better stability
    
    if (verbose) {
        std::cout << "Starting SCF iterations with simplified CNDO/S model..." << std::endl;
        std::cout << "Overlap matrix size: " << S.rows() << " x " << S.cols() << std::endl;
        std::cout << "Initial density matrix norm: " << p_alpha.norm() << std::endl;
    }
    
    while (!converged && iter < max_iter) {
        // Get total density matrix
        Eigen::MatrixXd p_total = p_alpha + p_beta;
        
        // Build Fock matrices
        Eigen::MatrixXd F_alpha = h;  // Start with core Hamiltonian
        Eigen::MatrixXd F_beta = h;
        
        // Add electron-electron interactions
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                // Skip off-diagonal elements for simplicity
                if (u == v) {
                    double P_uu = p_total(u, u);
                    F_alpha(u, u) += 0.5 * P_uu * 0.625;  // Simple approximation 
                    F_beta(u, u) += 0.5 * P_uu * 0.625;
                }
            }
        }
        
        if (verbose) {
            std::cout << "Iteration " << iter << " Fock matrix norm: " << F_alpha.norm() << std::endl;
        }
        
        // Solve the generalized eigenvalue problem F C = S C ε
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es_alpha(F_alpha, S);
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es_beta(F_beta, S);
        
        if (es_alpha.info() != Eigen::Success || es_beta.info() != Eigen::Success) {
            std::cout << "WARNING: Eigenvalue solver failed. Trying again with regularized S matrix." << std::endl;
            
            // Regularize S matrix by adding a small diagonal term
            Eigen::MatrixXd S_reg = S + Eigen::MatrixXd::Identity(n, n) * 1e-5;
            
            // Try again
            Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es_alpha_reg(F_alpha, S_reg);
            Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es_beta_reg(F_beta, S_reg);
            
            if (es_alpha_reg.info() != Eigen::Success || es_beta_reg.info() != Eigen::Success) {
                throw std::runtime_error("Eigenvalue solver failed even with regularization");
            }
            
            // Use regularized results
            es_alpha = es_alpha_reg;
            es_beta = es_beta_reg;
        }
        
        Eigen::MatrixXd C_alpha = es_alpha.eigenvectors();
        Eigen::MatrixXd C_beta = es_beta.eigenvectors();
        
        // Build new density matrices
        Eigen::MatrixXd p_alpha_new = build_density_matrix(p, C_alpha);
        Eigen::MatrixXd p_beta_new = build_density_matrix(q, C_beta);
        
        // Apply damping for better convergence
        Eigen::MatrixXd p_alpha_damped = damp_factor * p_alpha_new + (1.0 - damp_factor) * p_alpha;
        Eigen::MatrixXd p_beta_damped = damp_factor * p_beta_new + (1.0 - damp_factor) * p_beta;
        
        // Check for convergence
        double delta_alpha = (p_alpha_damped - p_alpha).norm();
        double delta_beta = (p_beta_damped - p_beta).norm();
        
        if (verbose) {
            std::cout << "Iteration " << iter << " delta: " << delta_alpha + delta_beta 
                      << " (alpha: " << delta_alpha << ", beta: " << delta_beta << ")" << std::endl;
            
            if (iter % 10 == 0) {
                std::cout << "Eigenvalues (alpha): " << es_alpha.eigenvalues().transpose().head(5) << " ..." << std::endl;
            }
        }
        
        if (delta_alpha < tol && delta_beta < tol) {
            converged = true;
            if (verbose) {
                std::cout << "SCF converged in " << iter+1 << " iterations." << std::endl;
            }
        }
        
        // Update density matrices for next iteration
        p_alpha = p_alpha_damped;
        p_beta = p_beta_damped;
        
        iter++;
        
        // Adjust damping factor based on convergence
        if (iter > 10 && (delta_alpha + delta_beta) > 0.1) {
            damp_factor = std::min(0.9, damp_factor + 0.05);  // Increase damping if convergence is slow
            if (verbose) {
                std::cout << "Adjusting damping factor to " << damp_factor << std::endl;
            }
        }
    }
    
    if (!converged) {
        std::cout << "WARNING: SCF did not converge after " << max_iter << " iterations." << std::endl;
    }
    
    return std::make_pair(p_alpha, p_beta);
}

// Simplified energy calculation function for CNDO/S
double E_cndo_s_simplified(const Eigen::MatrixXd &p_alpha, const Eigen::MatrixXd &p_beta, 
                         const std::vector<Atom> &atoms) {
    // Get core Hamiltonian
    Eigen::MatrixXd h = get_hamiltonian(atoms);
    
    // Calculate electronic energy using a simpler formula
    Eigen::MatrixXd p_total = p_alpha + p_beta;
    double elec_energy = 0.0;
    
    // One-electron terms
    for (int i = 0; i < p_total.rows(); i++) {
        for (int j = 0; j < p_total.cols(); j++) {
            elec_energy += p_total(i, j) * h(i, j);
        }
    }
    
    // Simplified two-electron terms
    double two_e_energy = 0.0;
    for (int i = 0; i < p_total.rows(); i++) {
        two_e_energy += 0.625 * p_total(i, i) * p_total(i, i);
    }
    
    // Add the two-electron contribution
    elec_energy += 0.5 * two_e_energy;
    
    // Add nuclear repulsion
    double nuclear_energy = get_nuclear_repulsion_energy(atoms);
    
    return elec_energy + nuclear_energy;
}

// Function to run bond displacement test
void run_displacement_test(const std::string& filename, bool use_overlap) {
    // Define displacement range
    double min_disp = -0.2;  // Å
    double max_disp = 0.2;   // Å
    double step = 0.05;      // Å
    
    std::cout << "Running displacement analysis for " << filename << std::endl;
    std::cout << "Displacement,Total_Energy,Nuclear_Energy,Electronic_Energy" << std::endl;
    
    for (double disp = min_disp; disp <= max_disp; disp += step) {
        // Load molecule
        std::vector<Atom> atoms = parse_file(filename, false);
        
        // Displace first bond (between atoms 0 and 1)
        Eigen::Vector3d orig_vec = atoms[1].pos - atoms[0].pos;
        double orig_length = orig_vec.norm();
        Eigen::Vector3d unit_vec = orig_vec / orig_length;
        atoms[1].pos = atoms[0].pos + unit_vec * (orig_length + disp);
        
        // Calculate energies
        int total_valence_electrons = get_total_valence_electrons(atoms);
        int q = total_valence_electrons / 2;
        int p = total_valence_electrons - q;
        
        double nuclear_energy, total_energy, electron_energy;
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> res;
        
        if (use_overlap) {
            // Save original parameters
            double original_h_beta = PARAMETER_INFO.at(1).at("-Beta");
            double original_h_i = PARAMETER_INFO.at(1).at("1/2(Is + As)");
            
            // Modify parameters for CNDO/S
            std::map<std::string, double>& h_params = 
                const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(1));
            
            h_params["-Beta"] = original_h_beta * 1.3;
            h_params["1/2(Is + As)"] = original_h_i * 0.8;
            
            // Run simplified CNDO/S
            res = run_cndo_s_simplified(atoms, p, q);
            
            // Calculate energy using simplified function
            total_energy = E_cndo_s_simplified(res.first, res.second, atoms);
            
            // Restore original parameters
            h_params["-Beta"] = original_h_beta;
            h_params["1/2(Is + As)"] = original_h_i;
        } else {
            // Standard CNDO/2 calculation
            res = run_CNDO2(atoms, p, q);
            total_energy = E_CNDO2(res.first, res.second, atoms);
        }
        
        nuclear_energy = get_nuclear_repulsion_energy(atoms);
        electron_energy = total_energy - nuclear_energy;
        
        // Output results in CSV format
        std::cout << disp << "," << total_energy << "," 
                  << nuclear_energy << "," << electron_energy << std::endl;
    }
}

// Function to save displacement results to a file
void save_displacement_results(const std::string& filename, bool use_overlap, 
                               const std::string& output_file) {
    // Open output file
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return;
    }
    
    // Define displacement range
    double min_disp = -0.2;  // Å
    double max_disp = 0.2;   // Å
    double step = 0.05;      // Å
    
    out_file << "Displacement,Total_Energy,Nuclear_Energy,Electronic_Energy" << std::endl;
    
    for (double disp = min_disp; disp <= max_disp; disp += step) {
        // Load molecule
        std::vector<Atom> atoms = parse_file(filename, false);
        
        // Displace first bond (between atoms 0 and 1)
        Eigen::Vector3d orig_vec = atoms[1].pos - atoms[0].pos;
        double orig_length = orig_vec.norm();
        Eigen::Vector3d unit_vec = orig_vec / orig_length;
        atoms[1].pos = atoms[0].pos + unit_vec * (orig_length + disp);
        
        // Calculate energies
        int total_valence_electrons = get_total_valence_electrons(atoms);
        int q = total_valence_electrons / 2;
        int p = total_valence_electrons - q;
        
        double nuclear_energy, total_energy, electron_energy;
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> res;
        
        if (use_overlap) {
            // Save original parameters
            double original_h_beta = PARAMETER_INFO.at(1).at("-Beta");
            double original_h_i = PARAMETER_INFO.at(1).at("1/2(Is + As)");
            
            // Modify parameters for CNDO/S
            std::map<std::string, double>& h_params = 
                const_cast<std::map<std::string, double>&>(PARAMETER_INFO.at(1));
            
            h_params["-Beta"] = original_h_beta * 1.3;
            h_params["1/2(Is + As)"] = original_h_i * 0.8;
            
            // Run simplified CNDO/S
            res = run_cndo_s_simplified(atoms, p, q);
            
            // Calculate energy using simplified function
            total_energy = E_cndo_s_simplified(res.first, res.second, atoms);
            
            // Restore original parameters
            h_params["-Beta"] = original_h_beta;
            h_params["1/2(Is + As)"] = original_h_i;
        } else {
            // Standard CNDO/2 calculation
            res = run_CNDO2(atoms, p, q);
            total_energy = E_CNDO2(res.first, res.second, atoms);
        }
        
        nuclear_energy = get_nuclear_repulsion_energy(atoms);
        electron_energy = total_energy - nuclear_energy;
        
        // Output results in CSV format
        out_file << disp << "," << total_energy << "," 
                 << nuclear_energy << "," << electron_energy << std::endl;
    }
    
    out_file.close();
    std::cout << "Displacement results saved to " << output_file << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./hw5 <molecule-file> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --overlap          : Use simplified CNDO/S model" << std::endl;
        std::cout << "  --displacement-test: Run bond displacement test" << std::endl;
        std::cout << "  --save-disp <file> : Save displacement results to file" << std::endl;
        std::cout << "  --displace <value> : Displace bond by specified amount" << std::endl;
        std::cout << "  --verbose          : Show detailed calculation information" << std::endl;
        return 1;
    }

    // Parse command line arguments
    std::string filepath = argv[1];
    bool use_overlap = false;
    bool verbose = false;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--overlap") {
            use_overlap = true;
        }
        else if (arg == "--verbose") {
            verbose = true;
        }
    }
    
    // Check for displacement test
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--displacement-test") {
            run_displacement_test(filepath, use_overlap);
            return 0;
        }
        else if (arg == "--save-disp" && i + 1 < argc) {
            std::string output_file = argv[i+1];
            save_displacement_results(filepath, use_overlap, output_file);
            return 0;
        }
    }
    
    // Check for single displacement
    double displacement = 0.0;
    bool do_displacement = false;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--displace" && i + 1 < argc) {
            displacement = std::stod(argv[i+1]);
            do_displacement = true;
            break;
        }
    }
    
    try {
        // Load molecule
        std::vector<Atom> atoms = parse_file(filepath, false);
        
        // Apply displacement if requested
        if (do_displacement) {
            // Displace the bond between atoms 0 and 1
            Eigen::Vector3d orig_vec = atoms[1].pos - atoms[0].pos;
            double orig_length = orig_vec.norm();
            
            // Get atom symbols for output
            std::string atom1_symbol, atom2_symbol;
            if (atoms[0].z_num == 1) atom1_symbol = "H";
            else if (atoms[0].z_num == 6) atom1_symbol = "C";
            else if (atoms[0].z_num == 7) atom1_symbol = "N";
            else if (atoms[0].z_num == 8) atom1_symbol = "O";
            else if (atoms[0].z_num == 9) atom1_symbol = "F";
            else atom1_symbol = std::to_string(atoms[0].z_num);
            
            if (atoms[1].z_num == 1) atom2_symbol = "H";
            else if (atoms[1].z_num == 6) atom2_symbol = "C";
            else if (atoms[1].z_num == 7) atom2_symbol = "N";
            else if (atoms[1].z_num == 8) atom2_symbol = "O";
            else if (atoms[1].z_num == 9) atom2_symbol = "F";
            else atom2_symbol = std::to_string(atoms[1].z_num);
            
            // Create unit vector along bond
            Eigen::Vector3d unit_vec = orig_vec / orig_length;
            
            // Apply displacement
            atoms[1].pos = atoms[0].pos + unit_vec * (orig_length + displacement);
            
            std::cout << atom1_symbol << "-" << atom2_symbol << " bond displaced by " << displacement 
                      << " Å (new length: " << (orig_length + displacement) << " Å)" << std::endl;
        }
        
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
        
        // If using overlap, use simplified CNDO/S
        if (use_overlap) {
            std::cout << "[CNDO/S Simplified] Using overlap matrix with simplified model" << std::endl;
            
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
            
            // Run simplified CNDO/S
            res = run_cndo_s_simplified(atoms, p, q, 1e-6, verbose);
            
            // Calculate energy using simplified function
            total_energy = E_cndo_s_simplified(res.first, res.second, atoms);
            nuclear_energy = get_nuclear_repulsion_energy(atoms);
            electron_energy = total_energy - nuclear_energy;
            
            // Restore original parameters
            h_params["-Beta"] = original_h_beta;
            h_params["1/2(Is + As)"] = original_h_i;
        } 
        else {
            // Standard CNDO/2 calculation
            std::cout << "[CNDO/2] Using orthonormal basis (S = I)" << std::endl;
            
            res = run_CNDO2(atoms, p, q, 1e-6, verbose);
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
        
        // Display matrices if verbose
        if (verbose) {
            std::cout << "Density matrix (alpha):" << std::endl; 
            std::cout << p_alpha << std::endl;
            
            std::cout << "Density matrix (beta):" << std::endl; 
            std::cout << p_beta << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}