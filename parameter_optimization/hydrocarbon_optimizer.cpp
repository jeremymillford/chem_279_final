#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <iomanip>
#include <algorithm>

#include "hw5_utils.cpp"
#include "Gaussian.cpp"
#include "Integrator.cpp"
#include "Central_Difference.cpp"

// Define CNDOConfig structure to hold configuration parameters
struct CNDOConfig {
    bool useOverlap = true;          // Use CNDO/S with overlap matrix
    double bondLengthScale = 1.0;    // Scale factor for bond lengths
    double betaScale = 1.0;          // Scale factor for -Beta parameters
    std::string xyzDirectory = "";   // Directory containing XYZ files
    bool verbose = false;            // Enable verbose output
};

// Helper function to fix the .contains() issue in hw5_utils.cpp
// This function checks if a key exists in a map (C++17 compatible)
template <typename Map, typename Key>
bool map_contains(const Map& map, const Key& key) {
    return map.find(key) != map.end();
}

// Function to parse XYZ files with atomic numbers
std::vector<Atom> parseXYZFile(const std::string& filepath) {
    std::vector<Atom> atoms;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    
    std::string line;
    int numAtoms;
    
    // Read number of atoms
    std::getline(file, line);
    std::istringstream(line) >> numAtoms;
    
    // For standard XYZ format, there's a comment line that we should skip
    // But in the simplified format, there might not be one
    std::streampos pos = file.tellg();  // Save current position
    std::getline(file, line);
    std::istringstream test(line);
    
    int possibleAtomicNum;
    double x, y, z;
    bool hasCommentLine = false;
    
    // Test if this line is a data line or a comment line
    if (!(test >> possibleAtomicNum >> x >> y >> z)) {
        // Not a data line, so it's a comment line
        hasCommentLine = true;
    } else {
        // It's a data line, go back to where we were
        file.seekg(pos);
    }
    
    // Read atom data
    for (int i = 0; i < numAtoms; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        
        int z_num;
        
        if (!(iss >> z_num >> x >> y >> z)) {
            throw std::runtime_error("Failed to parse atom data on line: " + line);
        }
        
        // Validate supported elements
        if (z_num != 1 && z_num != 6 && z_num != 7 && z_num != 8 && z_num != 9) {
            throw std::runtime_error("Unsupported element with atomic number: " + std::to_string(z_num));
        }
        
        Atom a;
        a.z_num = z_num;
        a.pos = Eigen::Vector3d(x, y, z);
        atoms.push_back(a);
    }
    
    return atoms;
}

// Structure to hold optimization parameters
struct OptimizationParams {
    double bondLengthScale = 1.0;
    double betaScale = 1.0;  // Scale factor for the -Beta parameters
};

// Function to scale bond lengths
std::vector<Atom> scaleBondLengths(const std::vector<Atom>& originalAtoms, double scale) {
    if (originalAtoms.size() < 2) return originalAtoms;
    
    // First, find the centroid
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& atom : originalAtoms) {
        centroid += atom.pos;
    }
    centroid /= originalAtoms.size();
    
    // Scale positions from centroid
    std::vector<Atom> scaledAtoms = originalAtoms;
    for (auto& atom : scaledAtoms) {
        atom.pos = centroid + (atom.pos - centroid) * scale;
    }
    
    return scaledAtoms;
}

// Modify the CNDO parameters by scaling
void scaleBetaParameters(double scale) {
    // Create a deep copy of the original parameters
    auto originalParams = PARAMETER_INFO;
    
    for (auto& [key, value] : PARAMETER_INFO) {
        // Safely modify the -Beta parameter by the scale factor
        if (map_contains(value, "-Beta")) {
            value["-Beta"] *= scale;
        }
    }
}

// Restore original CNDO parameters
void restoreBetaParameters(const std::map<int, std::map<std::string, double>>& original) {
    // Deep copy the original parameters back
    for (const auto& [key, valueMap] : original) {
        for (const auto& [paramName, paramValue] : valueMap) {
            if (paramName == "-Beta") {
                PARAMETER_INFO.at(key).at(paramName) = paramValue;
            }
        }
    }
}

// Calculate energy with CNDO/S method
double calculateEnergy(const std::vector<Atom>& atoms, const OptimizationParams& params, bool useOverlap = true) {
    // Save original parameters
    auto originalParams = PARAMETER_INFO;
    
    try {
        // Scale atoms for bond length
        std::vector<Atom> scaledAtoms = scaleBondLengths(atoms, params.bondLengthScale);
        
        // Scale beta parameters
        scaleBetaParameters(params.betaScale);
        
        // Calculate the number of electrons
        int totalValenceElectrons = get_total_valence_electrons(scaledAtoms);
        int q = totalValenceElectrons / 2;
        int p = totalValenceElectrons - q;
        
        // Run CNDO calculation
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> result;
        if (useOverlap) {
            result = run_CNDO_S(scaledAtoms, p, q);
        } else {
            result = run_CNDO2(scaledAtoms, p, q);
        }
        
        auto p_alpha = result.first;
        auto p_beta = result.second;
        
        // Calculate total energy
        double energy = E_CNDO2(p_alpha, p_beta, scaledAtoms);
        
        // Restore original parameters
        restoreBetaParameters(originalParams);
        
        return energy;
    } catch (const std::exception& e) {
        std::cerr << "Error in energy calculation: " << e.what() << std::endl;
        
        // Restore original parameters in case of error
        restoreBetaParameters(originalParams);
        
        return std::numeric_limits<double>::max();
    }
}

// Simple gradient descent optimization
OptimizationParams optimizeParameters(const std::vector<Atom>& atoms, bool useOverlap = true, bool verbose = false) {
    OptimizationParams params;
    OptimizationParams bestParams = params;
    
    double bestEnergy = calculateEnergy(atoms, params, useOverlap);
    double learningRate = 0.01;
    double epsilon = 0.0001;
    int maxIterations = 100;
    
    std::cout << "Starting optimization...\n";
    std::cout << "Initial energy: " << bestEnergy << " eV\n";
    
    for (int iter = 0; iter < maxIterations; iter++) {
        // Try changing bond length
        OptimizationParams testParams = bestParams;
        testParams.bondLengthScale += epsilon;
        double energyPlusBond = calculateEnergy(atoms, testParams, useOverlap);
        
        testParams = bestParams;
        testParams.bondLengthScale -= epsilon;
        double energyMinusBond = calculateEnergy(atoms, testParams, useOverlap);
        
        double bondGradient = (energyPlusBond - energyMinusBond) / (2 * epsilon);
        
        // Try changing beta scale
        testParams = bestParams;
        testParams.betaScale += epsilon;
        double energyPlusBeta = calculateEnergy(atoms, testParams, useOverlap);
        
        testParams = bestParams;
        testParams.betaScale -= epsilon;
        double energyMinusBeta = calculateEnergy(atoms, testParams, useOverlap);
        
        double betaGradient = (energyPlusBeta - energyMinusBeta) / (2 * epsilon);
        
        // Update parameters
        params.bondLengthScale = bestParams.bondLengthScale - learningRate * bondGradient;
        params.betaScale = bestParams.betaScale - learningRate * betaGradient;
        
        // Prevent extreme values
        params.bondLengthScale = std::max(0.5, std::min(1.5, params.bondLengthScale));
        params.betaScale = std::max(0.5, std::min(1.5, params.betaScale));
        
        double energy = calculateEnergy(atoms, params, useOverlap);
        
        if (energy < bestEnergy) {
            bestEnergy = energy;
            bestParams = params;
            
            if (verbose || ((iter + 1) % 5 == 0)) {
                std::cout << "Iteration " << iter << ": Energy = " << bestEnergy 
                          << " eV, Bond scale = " << bestParams.bondLengthScale 
                          << ", Beta scale = " << bestParams.betaScale << std::endl;
            }
        }
        
        // Check for convergence
        if (std::abs(bondGradient) < 1e-4 && std::abs(betaGradient) < 1e-4) {
            std::cout << "Converged after " << iter << " iterations.\n";
            break;
        }
    }
    
    std::cout << "Optimization complete. Final energy: " << bestEnergy << " eV\n";
    return bestParams;
}

// Process all XYZ files in a directory
void processDirectory(const CNDOConfig& config) {
    std::cout << "Processing XYZ files in directory: " << config.xyzDirectory << std::endl;
    
    // Check if directory exists
    if (!std::filesystem::exists(config.xyzDirectory)) {
        std::cerr << "Directory does not exist: " << config.xyzDirectory << std::endl;
        return;
    }
    
    // Open a file to write results
    std::ofstream resultsFile("optimization_results.csv");
    resultsFile << "Filename,Original Energy (eV),Optimized Energy (eV),Bond Scale,Beta Scale\n";
    
    // Process each XYZ file
    for (const auto& entry : std::filesystem::directory_iterator(config.xyzDirectory)) {
        if (entry.path().extension() == ".xyz") {
            std::string filename = entry.path().filename().string();
            std::cout << "\nProcessing file: " << filename << std::endl;
            
            try {
                // Parse the XYZ file
                std::vector<Atom> atoms = parseXYZFile(entry.path().string());
                
                // Calculate original energy
                OptimizationParams defaultParams;
                double originalEnergy = calculateEnergy(atoms, defaultParams, config.useOverlap);
                
                std::cout << "Original structure energy: " << originalEnergy << " eV" << std::endl;
                
                // Optimize parameters
                OptimizationParams optimizedParams = optimizeParameters(atoms, config.useOverlap, config.verbose);
                
                // Calculate optimized energy
                double optimizedEnergy = calculateEnergy(atoms, optimizedParams, config.useOverlap);
                
                // Write results to file
                resultsFile << filename << "," 
                           << originalEnergy << "," 
                           << optimizedEnergy << "," 
                           << optimizedParams.bondLengthScale << "," 
                           << optimizedParams.betaScale << std::endl;
                
                // Save optimized structure
                std::string outPath = config.xyzDirectory + "/optimized_" + filename;
                std::ofstream outFile(outPath);
                
                // Scale atoms for output
                std::vector<Atom> optimizedAtoms = scaleBondLengths(atoms, optimizedParams.bondLengthScale);
                
                outFile << optimizedAtoms.size() << "\n";
                outFile << "Optimized structure (bond scale: " << optimizedParams.bondLengthScale 
                        << ", beta scale: " << optimizedParams.betaScale 
                        << ", energy: " << optimizedEnergy << " eV)\n";
                
                for (const auto& atom : optimizedAtoms) {
                    // Output using atomic numbers to match input format
                    outFile << std::setw(2) << atom.z_num << "  " 
                            << std::fixed << std::setprecision(6) 
                            << atom.pos.x() << "  " 
                            << atom.pos.y() << "  " 
                            << atom.pos.z() << "\n";
                }
                outFile.close();
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing file " << filename << ": " << e.what() << std::endl;
            }
        }
    }
    
    resultsFile.close();
    std::cout << "\nResults saved to optimization_results.csv" << std::endl;
}

// Function to optimize a specific molecule (e.g., methane)
void optimize_methane_bond_length(CNDOConfig& config) {
    std::string filePathMethane = config.xyzDirectory + "/methane.xyz";
    
    try {
        std::vector<Atom> atoms = parseXYZFile(filePathMethane);
        
        std::cout << "Optimizing methane bond length...\n";
        
        // Simple optimization: try different bond lengths
        double bestEnergy = std::numeric_limits<double>::max();
        double bestScale = 1.0;
        
        for (double scale = 0.8; scale <= 1.2; scale += 0.02) {
            OptimizationParams params;
            params.bondLengthScale = scale;
            
            double energy = calculateEnergy(atoms, params, config.useOverlap);
            
            if (energy < bestEnergy) {
                bestEnergy = energy;
                bestScale = scale;
                
                std::cout << "Scale: " << scale << ", Energy: " << energy << " eV\n";
            }
        }
        
        std::cout << "Best bond length scale for methane: " << bestScale 
                 << " with energy: " << bestEnergy << " eV\n";
                 
    } catch (const std::exception& e) {
        std::cerr << "Error optimizing methane: " << e.what() << std::endl;
    }
}

// Function to optimize parameters specifically for methane
void optimize_parameters_for_methane(CNDOConfig& config) {
    std::string filePathMethane = config.xyzDirectory + "/methane.xyz";
    
    try {
        std::vector<Atom> atoms = parseXYZFile(filePathMethane);
        
        std::cout << "Optimizing parameters for methane...\n";
        OptimizationParams optimizedParams = optimizeParameters(atoms, config.useOverlap, config.verbose);
        
        double optimizedEnergy = calculateEnergy(atoms, optimizedParams, config.useOverlap);
        
        std::cout << "Optimized parameters for methane:\n"
                 << "  Bond scale: " << optimizedParams.bondLengthScale << "\n"
                 << "  Beta scale: " << optimizedParams.betaScale << "\n"
                 << "  Energy: " << optimizedEnergy << " eV\n";
                 
    } catch (const std::exception& e) {
        std::cerr << "Error optimizing methane parameters: " << e.what() << std::endl;
    }
}

// Calculate energies for a series of hydrocarbons
std::vector<std::pair<std::string, double>> calculate_hydrocarbon_energies(CNDOConfig& config) {
    std::vector<std::pair<std::string, double>> results;
    
    for (const auto& entry : std::filesystem::directory_iterator(config.xyzDirectory)) {
        if (entry.path().extension() == ".xyz") {
            std::string filename = entry.path().filename().string();
            
            try {
                std::vector<Atom> atoms = parseXYZFile(entry.path().string());
                
                OptimizationParams params;
                params.bondLengthScale = config.bondLengthScale;
                params.betaScale = config.betaScale;
                
                double energy = calculateEnergy(atoms, params, config.useOverlap);
                
                results.push_back({filename, energy});
                
                std::cout << "Energy for " << filename << ": " << energy << " eV\n";
                
            } catch (const std::exception& e) {
                std::cerr << "Error calculating energy for " << filename << ": " << e.what() << std::endl;
            }
        }
    }
    
    return results;
}

// Optimize parameters for a series of hydrocarbons
void optimize_parameters_for_hydrocarbons(CNDOConfig& config) {
    // Process all XYZ files - this is the main optimization function
    processDirectory(config);
}

int main(int argc, char** argv) {
    CNDOConfig config;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--no-overlap") {
            config.useOverlap = false;
            std::cout << "Using CNDO/2 (no overlap matrix)\n";
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--bond-scale" && i + 1 < argc) {
            config.bondLengthScale = std::stod(argv[++i]);
        } else if (arg == "--beta-scale" && i + 1 < argc) {
            config.betaScale = std::stod(argv[++i]);
        } else if (arg == "--optimize-methane") {
            config.xyzDirectory = (i + 1 < argc) ? argv[++i] : ".";
            optimize_methane_bond_length(config);
            return 0;
        } else if (arg == "--optimize-methane-params") {
            config.xyzDirectory = (i + 1 < argc) ? argv[++i] : ".";
            optimize_parameters_for_methane(config);
            return 0;
        } else if (arg == "--calculate-energies") {
            config.xyzDirectory = (i + 1 < argc) ? argv[++i] : ".";
            calculate_hydrocarbon_energies(config);
            return 0;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options] <directory_path>\n\n"
                      << "Options:\n"
                      << "  --no-overlap              Use CNDO/2 instead of CNDO/S (no overlap matrix)\n"
                      << "  --verbose                 Enable verbose output\n"
                      << "  --bond-scale <value>      Set the bond length scale factor\n"
                      << "  --beta-scale <value>      Set the beta parameter scale factor\n"
                      << "  --optimize-methane <dir>  Optimize methane bond length\n"
                      << "  --optimize-methane-params <dir> Optimize parameters for methane\n"
                      << "  --calculate-energies <dir> Calculate energies for all hydrocarbons\n"
                      << "  --help                    Display this help message\n";
            return 0;
        } else {
            // Assume this is the directory path
            config.xyzDirectory = arg;
        }
    }
    
    if (config.xyzDirectory.empty()) {
        std::cerr << "No directory specified. Use --help for usage information.\n";
        return 1;
    }
    
    // Run the main optimization process
    optimize_parameters_for_hydrocarbons(config);
    
    return 0;
}