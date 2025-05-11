#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <filesystem>

#include "Gaussian.cpp"
#include "Integrator.cpp"
#include "hw5_utils.cpp"

// Helper function to run both calculations and save results
void run_and_save_molecule(const std::string& filepath, const std::string& mol_name) {
    std::cout << "\n========================================\n";
    std::cout << "Processing molecule: " << mol_name << std::endl;
    std::cout << "Using file: " << filepath << std::endl;
    std::cout << "========================================\n";
    
    try {
        // Check if file exists
        std::ifstream file_check(filepath);
        if (!file_check.good()) {
            std::cerr << "Error: File not found: " << filepath << std::endl;
            return;
        }
        file_check.close();
        
        // Load molecule
        std::vector<Atom> atoms = parse_file(filepath, false);
        auto vcg = get_vector_of_contracted_gaussians(atoms);
        
        // Determine electron counts
        int total_valence_electrons = get_total_valence_electrons(atoms);
        int q = total_valence_electrons / 2;
        int p = total_valence_electrons - q;
        
        std::cout << "Number of atoms: " << atoms.size() << std::endl;
        std::cout << "Alpha electrons: " << p << ", Beta electrons: " << q << std::endl;
        
        // Create output file
        std::string output_filename = mol_name + "_results.txt";
        std::ofstream outfile(output_filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open output file: " << output_filename << std::endl;
            return;
        }
        
        outfile << "Molecule: " << mol_name << std::endl;
        outfile << "Filepath: " << filepath << std::endl;
        outfile << "Number of atoms: " << atoms.size() << std::endl;
        outfile << "Number of valence electrons: " << total_valence_electrons << std::endl;
        outfile << "Alpha electrons: " << p << ", Beta electrons: " << q << std::endl;
        outfile << "\n========================================\n";
        
        // Run standard CNDO/2 calculation (S = I)
        std::cout << "[CNDO/2] Running with orthonormal basis (S = I)" << std::endl;
        outfile << "[CNDO/2] Results with orthonormal basis (S = I)" << std::endl;
        
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> res_cndo2;
        double nuclear_energy_cndo2, total_energy_cndo2, electron_energy_cndo2;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        res_cndo2 = run_CNDO2(atoms, p, q);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        nuclear_energy_cndo2 = get_nuclear_repulsion_energy(atoms);
        total_energy_cndo2 = E_CNDO2(res_cndo2.first, res_cndo2.second, atoms);
        electron_energy_cndo2 = total_energy_cndo2 - nuclear_energy_cndo2;
        
        // Save CNDO/2 results
        outfile << std::fixed << std::setprecision(6);
        outfile << "Calculation time: " << elapsed.count() << " seconds\n";
        outfile << "Nuclear Repulsion Energy: " << nuclear_energy_cndo2 << " eV\n";
        outfile << "Electronic Energy: " << electron_energy_cndo2 << " eV\n";
        outfile << "Total Energy: " << total_energy_cndo2 << " eV\n";
        outfile << "Energy per atom: " << total_energy_cndo2 / atoms.size() << " eV/atom\n";
        outfile << "Energy per electron: " << total_energy_cndo2 / total_valence_electrons << " eV/electron\n\n";
        
        // Now run CNDO/S calculation (S ≠ I)
        std::cout << "[CNDO/S] Running with overlap matrix (S ≠ I)" << std::endl;
        outfile << "[CNDO/S] Results with overlap matrix (S ≠ I)" << std::endl;
        
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> res_cndos;
        double nuclear_energy_cndos, total_energy_cndos, electron_energy_cndos;
        
        start_time = std::chrono::high_resolution_clock::now();
        res_cndos = run_CNDO_S_simplified(atoms, p, q);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed = end_time - start_time;
        
        nuclear_energy_cndos = get_nuclear_repulsion_energy(atoms);
        total_energy_cndos = E_CNDO_S_simplified(res_cndos.first, res_cndos.second, atoms);
        electron_energy_cndos = total_energy_cndos - nuclear_energy_cndos;
        
        // Save CNDO/S results
        outfile << "Calculation time: " << elapsed.count() << " seconds\n";
        outfile << "Nuclear Repulsion Energy: " << nuclear_energy_cndos << " eV\n";
        outfile << "Electronic Energy: " << electron_energy_cndos << " eV\n";
        outfile << "Total Energy: " << total_energy_cndos << " eV\n";
        outfile << "Energy per atom: " << total_energy_cndos / atoms.size() << " eV/atom\n";
        outfile << "Energy per electron: " << total_energy_cndos / total_valence_electrons << " eV/electron\n\n";
        
        // Calculate and save differences
        outfile << "[COMPARISON] CNDO/S vs CNDO/2 Differences\n";
        outfile << "Nuclear Repulsion Energy Difference: " << nuclear_energy_cndos - nuclear_energy_cndo2 << " eV\n";
        outfile << "Electronic Energy Difference: " << electron_energy_cndos - electron_energy_cndo2 << " eV\n";
        outfile << "Total Energy Difference: " << total_energy_cndos - total_energy_cndo2 << " eV\n";
        outfile << "Relative Energy Difference: " << 
            100.0 * (total_energy_cndos - total_energy_cndo2) / std::abs(total_energy_cndo2) << "%\n";
        
        // Save gradient information
        outfile << "\n[GRADIENTS]\n";
        outfile << "CNDO/2 Gradients:\n";
        auto gradients_cndo2 = get_energy_derivative_vector(atoms, vcg, res_cndo2.first, res_cndo2.second);
        
        double grad_norm_cndo2 = 0.0;
        for (int i = 0; i < atoms.size(); i++) {
            Eigen::Vector3d grad = gradients_cndo2(i);
            outfile << "Atom " << i << " (" << atoms[i].z_num << "): ";
            outfile << "x: " << grad.x() << " y: " << grad.y() << " z: " << grad.z();
            outfile << " magnitude: " << grad.norm() << std::endl;
            grad_norm_cndo2 += grad.squaredNorm();
        }
        grad_norm_cndo2 = std::sqrt(grad_norm_cndo2);
        outfile << "Total gradient norm: " << grad_norm_cndo2 << std::endl << std::endl;
        
        outfile << "CNDO/S Gradients:\n";
        auto gradients_cndos = get_energy_derivative_vector(atoms, vcg, res_cndos.first, res_cndos.second);
        
        double grad_norm_cndos = 0.0;
        for (int i = 0; i < atoms.size(); i++) {
            Eigen::Vector3d grad = gradients_cndos(i);
            outfile << "Atom " << i << " (" << atoms[i].z_num << "): ";
            outfile << "x: " << grad.x() << " y: " << grad.y() << " z: " << grad.z();
            outfile << " magnitude: " << grad.norm() << std::endl;
            grad_norm_cndos += grad.squaredNorm();
        }
        grad_norm_cndos = std::sqrt(grad_norm_cndos);
        outfile << "Total gradient norm: " << grad_norm_cndos << std::endl;
        
        // Save density matrix information
        outfile << "\n[ELECTRON DENSITY]\n";
        Eigen::VectorXd charges_cndo2 = get_p_total_atomwise_diag_vector(atoms, res_cndo2.first + res_cndo2.second);
        Eigen::VectorXd charges_cndos = get_p_total_atomwise_diag_vector(atoms, res_cndos.first + res_cndos.second);
        
        outfile << "Atomic Charges:\n";
        outfile << "Atom\tCNDO/2\tCNDO/S\tDifference\n";
        for (int i = 0; i < atoms.size(); i++) {
            double cndo2_charge = get_Z(atoms[i]) - charges_cndo2(i);
            double cndos_charge = get_Z(atoms[i]) - charges_cndos(i);
            outfile << i << " (" << atoms[i].z_num << ")\t" 
                   << cndo2_charge << "\t" 
                   << cndos_charge << "\t" 
                   << cndos_charge - cndo2_charge << std::endl;
        }
        
        outfile.close();
        std::cout << "Results saved to " << output_filename << std::endl;
        
        // Also save a simple CSV with key results for easy comparison
        std::string csv_filename = "hydrocarbon_results.csv";
        std::ofstream csv_file;
        
        bool file_exists = std::ifstream(csv_filename).good();
        
        if (!file_exists) {
            csv_file.open(csv_filename);
            csv_file << "Molecule,Atoms,Electrons,CNDO2_Total,CNDO2_Electronic,CNDO2_Nuclear,";
            csv_file << "CNDO2_PerAtom,CNDO2_PerElectron,CNDO2_GradNorm,";
            csv_file << "CNDOS_Total,CNDOS_Electronic,CNDOS_Nuclear,";
            csv_file << "CNDOS_PerAtom,CNDOS_PerElectron,CNDOS_GradNorm,";
            csv_file << "Energy_Diff,Energy_Diff_Percent\n";
        } else {
            csv_file.open(csv_filename, std::ios_base::app);
        }
        
        csv_file << std::fixed << std::setprecision(6);
        csv_file << mol_name << "," << atoms.size() << "," << total_valence_electrons << ",";
        csv_file << total_energy_cndo2 << "," << electron_energy_cndo2 << "," << nuclear_energy_cndo2 << ",";
        csv_file << total_energy_cndo2/atoms.size() << "," << total_energy_cndo2/total_valence_electrons << "," << grad_norm_cndo2 << ",";
        csv_file << total_energy_cndos << "," << electron_energy_cndos << "," << nuclear_energy_cndos << ",";
        csv_file << total_energy_cndos/atoms.size() << "," << total_energy_cndos/total_valence_electrons << "," << grad_norm_cndos << ",";
        csv_file << total_energy_cndos - total_energy_cndo2 << ",";
        csv_file << 100.0 * (total_energy_cndos - total_energy_cndo2) / std::abs(total_energy_cndo2) << "\n";
        
        csv_file.close();
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing " << mol_name << ": " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    // Define data directory - where molecule files are stored
    std::string data_dir = "./";  // Default to current directory
    
    // If a directory is provided via command line, use it
    if (argc > 1) {
        data_dir = argv[1];
        // Add trailing slash if needed
        if (data_dir.back() != '/' && data_dir.back() != '\\') {
            data_dir += '/';
        }
    }
    
    std::cout << "Looking for molecule files in: " << data_dir << std::endl;
    
    // List of hydrocarbon molecules to analyze
    std::vector<std::pair<std::string, std::string>> molecules = {
        {data_dir + "H2.txt", "H2"},
        {data_dir + "CH4.txt", "CH4"},
        {data_dir + "C2H4.txt", "C2H4"},
        {data_dir + "C2H6.txt", "C2H6"},
        {data_dir + "C4H10.txt", "C4H10"},
        {data_dir + "C6H6.txt", "C6H6"}
    };
    
    // Process each molecule
    for (const auto& mol : molecules) {
        run_and_save_molecule(mol.first, mol.second);
    }
    
    std::cout << "\nAll calculations completed!\n";
    std::cout << "Full details are in individual *_results.txt files\n";
    std::cout << "Summary data is in hydrocarbon_results.csv\n";
    
    return 0;
}