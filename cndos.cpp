#include <Eigen/Dense> 
#include <iomanip>
#include <numeric>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include "Gaussian.cpp"
#include "Integrator.cpp"
#include "cndos_utils.cpp" // Using the renamed utilities file

// Function to run bond displacement test
void run_displacement_test(const std::string& filename, bool use_overlap, bool verbose = false) {
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
            
            // Run CNDO/S
            res = run_CNDO_S(atoms, p, q, 1e-6, verbose);
            
            // Calculate energy using E_CNDO_S function
            total_energy = E_CNDO_S(res.first, res.second, atoms);
            
            // Restore original parameters
            h_params["-Beta"] = original_h_beta;
            h_params["1/2(Is + As)"] = original_h_i;
        } else {
            // Standard CNDO/2 calculation
            res = run_CNDO2(atoms, p, q, 1e-6, verbose);
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
                               const std::string& output_file, bool verbose = false) {
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
            // Run CNDO/S simplified - no parameter modifications needed
            res = run_CNDO_S(atoms, p, q, 1e-6, verbose);
            
            // Calculate energy using simplified function
            total_energy = E_CNDO_S(res.first, res.second, atoms);
        } else {
            // Standard CNDO/2 calculation
            res = run_CNDO2(atoms, p, q, 1e-6, verbose);
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
        std::cout << "  --overlap          : Use CNDO/S model with overlap" << std::endl;
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
    bool displacement_test = false;
    std::string output_file = "";
    double displacement = 0.0;
    bool do_displacement = false;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--overlap") {
            use_overlap = true;
        }
        else if (arg == "--verbose") {
            verbose = true;
        }
        else if (arg == "--displacement-test") {
            displacement_test = true;
        }
        else if (arg == "--save-disp" && i + 1 < argc) {
            output_file = argv[i+1];
            i++; // Skip the next argument
        }
        else if (arg == "--displace" && i + 1 < argc) {
            displacement = std::stod(argv[i+1]);
            do_displacement = true;
            i++; // Skip the next argument
        }
    }
    
    // Run displacement test if requested
    if (displacement_test) {
        run_displacement_test(filepath, use_overlap, verbose);
        return 0;
    }
    
    // Save displacement results if requested
    if (!output_file.empty()) {
        save_displacement_results(filepath, use_overlap, output_file, verbose);
        return 0;
    }
    
    try {
        // Load molecule
        std::vector<Atom> atoms = parse_file(filepath, verbose);
        
        if (verbose) {
            std::cout << "\nLoaded molecule structure:" << std::endl;
            for (size_t i = 0; i < atoms.size(); i++) {
                std::string atom_symbol;
                if (atoms[i].z_num == 1) atom_symbol = "H";
                else if (atoms[i].z_num == 6) atom_symbol = "C";
                else if (atoms[i].z_num == 7) atom_symbol = "N";
                else if (atoms[i].z_num == 8) atom_symbol = "O";
                else if (atoms[i].z_num == 9) atom_symbol = "F";
                else atom_symbol = std::to_string(atoms[i].z_num);
                
                std::cout << "Atom " << i << " (" << atom_symbol << "): " 
                          << atoms[i].pos.x() << " " 
                          << atoms[i].pos.y() << " " 
                          << atoms[i].pos.z() << std::endl;
            }
        }
        
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
            
            if (verbose) {
                std::cout << "\nDisplaced molecule structure:" << std::endl;
                for (size_t i = 0; i < atoms.size(); i++) {
                    std::string atom_symbol;
                    if (atoms[i].z_num == 1) atom_symbol = "H";
                    else if (atoms[i].z_num == 6) atom_symbol = "C";
                    else if (atoms[i].z_num == 7) atom_symbol = "N";
                    else if (atoms[i].z_num == 8) atom_symbol = "O";
                    else if (atoms[i].z_num == 9) atom_symbol = "F";
                    else atom_symbol = std::to_string(atoms[i].z_num);
                    
                    std::cout << "Atom " << i << " (" << atom_symbol << "): " 
                              << atoms[i].pos.x() << " " 
                              << atoms[i].pos.y() << " " 
                              << atoms[i].pos.z() << std::endl;
                }
            }
        }
        
        // Determine electron counts
        int total_valence_electrons = get_total_valence_electrons(atoms);
        int q = total_valence_electrons / 2;
        int p = total_valence_electrons - q;
        
        std::cout << "Loaded molecule: " << filepath << std::endl;
        std::cout << "Number of atoms: " << atoms.size() << std::endl;
        std::cout << "Alpha electrons: " << p << ", Beta electrons: " << q << std::endl;
        
        // Get contracted gaussians
        auto vcg = get_vector_of_contracted_gaussians(atoms);
        
        if (verbose) {
            std::cout << "\nContracted Gaussians:" << std::endl;
            for (size_t i = 0; i < vcg.size(); i++) {
                std::cout << "Contracted Gaussian " << i << ": " << std::endl;
                std::cout << vcg[i] << std::endl;
            }
            
            // Display overlap matrix
            Eigen::MatrixXd S = make_overlap_matrix(vcg);
            std::cout << "\nOverlap Matrix:" << std::endl;
            std::cout << S << std::endl;
            
            // Display gamma matrix
            Eigen::MatrixXd gamma = get_gamma(atoms);
            std::cout << "\nGamma Matrix:" << std::endl;
            std::cout << gamma << std::endl;
        }
        
        // Run calculation
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> res;
        double nuclear_energy, total_energy, electron_energy;
        
        // If using overlap, use CNDO/S simplified
        if (use_overlap) {
            std::cout << "[CNDO/S Simplified] Using overlap matrix with simplified model" << std::endl;
            
            // Run CNDO/S simplified - no parameter modifications needed
            res = run_CNDO_S(atoms, p, q, 1e-6, verbose);
            
            // Calculate energy using simplified function
            total_energy = E_CNDO_S(res.first, res.second, atoms);
            nuclear_energy = get_nuclear_repulsion_energy(atoms);
            electron_energy = total_energy - nuclear_energy;
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
        
        // Display matrices and other diagnostic information if verbose or specifically requested
        auto p_total = p_alpha + p_beta;
            
        std::cout << "x" << std::endl; 
        auto x = get_x_matrix(atoms, p_total);
        std::cout << x << std::endl; 
        
        std::cout << "y" << std::endl; 
        auto y = get_y_matrix(atoms, p_alpha, p_beta); 
        std::cout << y << std::endl;
        
        // Calculate gradients for detailed output
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
        auto vec_of_gamma_dir = get_vector_of_gamma_dir(atoms, vcg); 
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