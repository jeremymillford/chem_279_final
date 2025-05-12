#include <Eigen/Dense> 
#include <iomanip>
#include <numeric>
 
#include "hw5_utils.cpp"
#include "Gaussian.cpp"
#include "Integrator.cpp"

int main(int argc, char** argv){
    if (argc < 2){
        throw std::runtime_error("No Filepath was supplied");
    }

    std::string filepath = argv[1];
    std::string model_type = argv[4];
    std::vector<Atom> atoms = parse_file(filepath, false);
    auto bond_distances_start = compute_bond_distances(atoms);
    auto vcg = get_vector_of_contracted_gaussians(atoms); 

    write_distances_to_csv(filepath, 0, bond_distances_start, model_type);

    int p, q = 0; 
    if (argc == 4 || argc == 5){
        p = std::stoi(argv[2]);
        q = std::stoi(argv[3]); 
    } else if (argc == 2){
        int total_valence_electrons = get_total_valence_electrons(atoms);
        q = total_valence_electrons / 2; 
        p = total_valence_electrons - q; 
    } else {
        throw std::invalid_argument("argc must be 2, 4, or 5");
    }

    // === NEW: Overlap flag handling ===
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> res;
    double nuclear_energy = get_nuclear_repulsion_energy(atoms); 
    double total_energy;

    if (argc == 5) {
        // === NEW: Run CNDO or CNDO+S or MINDO+3 based on flag ===
        if (model_type == "CNDOS") {
            res = run_CNDO_S(atoms, p, q);
            std::cout << "[CNDO/S] Using overlap matrix (S ≠ I)" << std::endl;
        } 
        else if (model_type == "CNDO2") {
            res = run_CNDO2(atoms, p, q);
            std::cout << "[CNDO/2] Using orthonormal basis (S = I)" << std::endl;
        }
        else if (model_type == "MINDO") {
            res = run_MINDO3(atoms, p, q);
            std::cout << "[MINDO/3]" << std::endl;
        }
        else {
            std::cerr << "Unknown model type: " << model_type << std::endl;
                std::cerr << "Valid options are: CNDO2, CNDOS, MINDO" << std::endl;
        }
    }
    
    Eigen::MatrixXd p_alpha = res.first;
    Eigen::MatrixXd p_beta = res.second;
    if (model_type == "MINDO") {
        total_energy = E_MINDO3(p_alpha, p_beta, atoms);
    }
    else {
        total_energy = E_CNDO2(p_alpha, p_beta, atoms);
    }
   
    const int n = vcg.size(); 
    const int N = atoms.size();

    std::cout << "Nuclear Repulsion Energy is " << nuclear_energy << " eV." << std::endl;
    std::cout << "Electron Energy is " << total_energy - nuclear_energy << " eV." << std::endl;

    // === Fock matrix diagnostics ===
    std::cout << "x" << std::endl; 
    auto x = get_x_matrix(atoms, p_alpha + p_beta);
    std::cout << x << std::endl; 

    std::cout << "y" << std::endl; 
    auto y = get_y_matrix(atoms, p_alpha, p_beta); 
    std::cout << y << std::endl; 

    std::cout << "Suv_RA (A is the center of u)" << std::endl; 
    auto vec_s_dir = get_vector_of_s_dir(atoms, vcg); 
    auto s_dir = vec_s_dir(0);
    for (int u = 0; u < n; u++){
        for (int v = 0; v < n; v++){
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
    for (int i = 0; i < N; i++){
        for (int j =0; j < N; j++){
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
    for (auto grad : nuclear_energy_dir){
        std::cout << "x: " << grad.x() << " y: " << grad.y() << " z: " << grad.z() << std::endl;
    }
    
    auto vec_of_energy_dir = get_energy_derivative_vector(atoms, vcg, p_alpha, p_beta); 
    std::cout << "gradient (Electron part)" << std::endl; 
    for (int i = 0; i < vec_of_energy_dir.size(); i++){
        auto nuclear_dir = nuclear_energy_dir(i); 
        auto energy_dir = vec_of_energy_dir(i); 
        auto electron_dir = energy_dir - nuclear_dir; 
        std::cout << "x: " << electron_dir.x() << " y: " << electron_dir.y() << " z: " << electron_dir.z() << std::endl;
    }

    std::cout << "gradient" << std::endl;
    for (auto energy_dir : vec_of_energy_dir){
        std::cout << "x: " << energy_dir.x() << " y: " << energy_dir.y() << " z: " << energy_dir.z() << std::endl;  
    }
    
    return 0; 
}
