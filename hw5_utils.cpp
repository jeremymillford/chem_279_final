#pragma once

#include <vector>
#include <map> 
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <stdexcept>
#include <fstream> 
#include <sstream> 
#include <utility> 
#include <exception>
#include <numeric> 
#include <algorithm> 
#include <cmath> 

#include "Gaussian.cpp"
#include "Integrator.cpp"
#include "Central_Difference.cpp"

constexpr int NUM_CONTRACTED = 3; 
constexpr double CONVERSION_FACTOR = 27.211; // eV / atomic unit 

struct Atom {
    int z_num;
    Eigen::Vector3d pos;
};

std::vector<ContractedGaussian> get_vector_of_contracted_gaussians(const std::vector<Atom>& atoms);
std::vector<ContractedGaussian> get_contracted_gaussian_basis(const Atom& a);
double get_gamma_elem(const Atom& A, const Atom& B);
Eigen::MatrixXd get_hamiltonian(const std::vector<Atom>& atoms);
double contracted_gaussian_overlap(const ContractedGaussian& A, const ContractedGaussian& B);


// Semi-Empirical parameters that define the CNDO/2 model (units eV)
// NOTE SOMETHING VERY SKETCHY. I AM STORING INTS IN THIS DOUBLE CONTAINER. I SHOULD BE SCARED OF THIS. 
const std::map<int, std::map<std::string, double>> PARAMETER_INFO {
    {1, {
        {"Valence Orbitals", 1},
        {"Valence Atoms", 1},
        {"1/2(Is + As)", 7.176},
        {"-Beta", 9.0},  
    }},
    {6, {
        {"Valence Orbitals", 4},
        {"Valence Atoms", 4},
        {"1/2(Is + As)", 14.051},
        {"1/2(Ip + Ap)", 5.572},
        {"-Beta", 21},  
    }},
    {7, {
        {"Valence Orbitals", 4},
        {"Valence Atoms", 5},
        {"1/2(Is + As)", 19.316},
        {"1/2(Ip + Ap)", 7.275},
        {"-Beta", 25},  
    }},
    {8, {
        {"Valence Orbitals", 4},
        {"Valence Atoms", 6},
        {"1/2(Is + As)", 25.390},
        {"1/2(Ip + Ap)", 9.111},
        {"-Beta", 31},  
    }},
    {9, {
        {"Valence Orbitals", 4},
        {"Valence Atoms", 7},
        {"1/2(Is + As)", 32.272},
        {"1/2(Ip + Ap)", 11.080},
        {"-Beta", 39},  
    }},
};

const std::map<std::string, Eigen::Vector3d> H_INFO {  
    {"exponent",      Eigen::Vector3d{ 3.42525091, 0.62391373, 0.16885540}},
    {"s contraction", Eigen::Vector3d{ 0.15432897, 0.53532814, 0.44463454}},
}; 

const std::map<std::string, Eigen::Vector3d> C_INFO {
    {"exponent",      Eigen::Vector3d{ 2.94124940, 0.68348310, 0.22228990}},
    {"s contraction", Eigen::Vector3d{-0.09996723, 0.39951283, 0.70011547}},
    {"p contraction", Eigen::Vector3d{ 0.15591627, 0.60768372, 0.39195739}},
};

const std::map<std::string, Eigen::Vector3d> N_INFO {
    {"exponent",      Eigen::Vector3d{ 3.78045590, 0.87849660, 0.28571440}},
    {"s contraction", Eigen::Vector3d{-0.09996723, 0.39951283, 0.70011547}},
    {"p contraction", Eigen::Vector3d{ 0.15591627, 0.60768372, 0.39195739}},
}; 

const std::map<std::string, Eigen::Vector3d> O_INFO {
    {"exponent",      Eigen::Vector3d{ 5.03315130, 1.16959610, 0.38038900}},
    {"s contraction", Eigen::Vector3d{-0.09996723, 0.39951283, 0.70011547}},
    {"p contraction", Eigen::Vector3d{ 0.15591627, 0.60768372, 0.39195739}},
}; 

const std::map<std::string, Eigen::Vector3d> F_INFO {
    {"exponent",      Eigen::Vector3d{ 6.46480320, 1.50228120, 0.48858850}},
    {"s contraction", Eigen::Vector3d{-0.09996723, 0.39951283, 0.70011547}},
    {"p contraction", Eigen::Vector3d{ 0.15591627, 0.60768372, 0.39195739}},
}; 

const std::map<int, std::map<std::string, Eigen::Vector3d>> BASIS_INFO {
    {1, H_INFO}, 
    {6, C_INFO},
    {7, N_INFO},
    {8, O_INFO}, 
    {9, F_INFO},
}; 




// Helper function broken out so that it can be independently called later
ContractedGaussian get_s_contracted_gaussian(const Atom &a){
    if (!BASIS_INFO.contains(a.z_num)) throw std::invalid_argument("no basis information atom");
    auto atom_basis_info = BASIS_INFO.at(a.z_num); 
    std::vector<Gaussian> gaussians; 
    std::vector<double>   cont_coef; 
    for(int i = 0; i < NUM_CONTRACTED; i++){    
        gaussians.push_back(Gaussian (a.pos, atom_basis_info.at("exponent")[i], Eigen::Vector3i {0, 0, 0}));
        cont_coef.push_back(atom_basis_info.at("s contraction")[i]); 
    }
    return ContractedGaussian(a.z_num, gaussians, cont_coef);
}

// This function returns the contracted gaussians for a single atom
std::vector<ContractedGaussian> get_contracted_gaussian_basis(const Atom &a){
    if (!BASIS_INFO.contains(a.z_num)) throw std::invalid_argument("no basis information atom");
    auto atom_basis_info = BASIS_INFO.at(a.z_num); 
    std::vector<ContractedGaussian> cgv; 
    
    // add S shell
    cgv.push_back(get_s_contracted_gaussian(a)); 

    std::vector<Gaussian> gaussians; 
    std::vector<double>   cont_coef; 
    // check if there should be a P shell 
    if (a.z_num > 2){
        // add P shell
        for(const auto &power : {Eigen::Vector3i(1,0,0), Eigen::Vector3i(0,1,0), Eigen::Vector3i(0,0,1)}){
            gaussians.clear(); 
            cont_coef.clear();

            for(int i = 0; i < NUM_CONTRACTED; i++){    
                gaussians.push_back(Gaussian (a.pos, atom_basis_info.at("exponent")[i], power));
                cont_coef.push_back(atom_basis_info.at("p contraction")[i]); 
            }
            cgv.push_back(ContractedGaussian(a.z_num, gaussians, cont_coef));
        }
    }
    return cgv; 
}

/*
This function iterates through a list of atoms and adds basis functions
*/
std::vector<ContractedGaussian> get_vector_of_contracted_gaussians(const std::vector<Atom> &atoms){
    std::vector<ContractedGaussian> cgv;
    for (const Atom& a : atoms){
        auto h_cgv = get_contracted_gaussian_basis(a);
        cgv.insert(cgv.end(), h_cgv.begin(), h_cgv.end());
    }
    return cgv; 
}



// Helper to get Z_?  
double get_Z(const Atom &atom){
    return PARAMETER_INFO.at(atom.z_num).at("Valence Atoms"); 
}

// Helper to get all the valence orbitals, for intializing things to the right size
int get_total_valence_orbitals(const std::vector<Atom> atoms){
    int sum = 0; 
    for (const Atom &atom : atoms) {
        sum += static_cast<int>(PARAMETER_INFO.at(atom.z_num).at("Valence Orbitals")); 
    }
    return sum; 
}

// Helper to get all the valence electrons, for calcuating p and q 
int get_total_valence_electrons(const std::vector<Atom> atoms){
    int sum = 0;
    for (const Atom &atom : atoms){
        sum += static_cast<int>(get_Z(atom)); 
    }
    return sum; 
}

// Helper lookup function to build the density matrix 
Eigen::MatrixXd build_density_matrix(const int count, const Eigen::MatrixXd &c){
    const int n = c.rows(); 
    Eigen::MatrixXd p = Eigen::MatrixXd::Zero(n, n); 
    for (int u = 0; u < n; u++){
        for (int v = 0; v < n; v++){
            double sum = 0; 
            for (int i = 0; i < count; i++){
                sum += c(u,i)*c(v,i); 
            }
            p(u,v) = sum; 
        }
    }
    return p; 
}



// Helper lookup function to get the appropriate data from the header in atomic units! 
double get_half_IuAu(const ContractedGaussian &cg){
    int shell = cg.gaussians.front().power.sum(); 
    if (shell == 0){
        return PARAMETER_INFO.at(cg.z_num).at("1/2(Is + As)"); // / CONVERSION_FACTOR; 
    } else if (shell == 1){
        return PARAMETER_INFO.at(cg.z_num).at("1/2(Ip + Ap)"); // / CONVERSION_FACTOR; 
    }
    throw std::runtime_error("The shell for this contracted gaussian isn't s or p");   
}

// Helper to get B_? in atomic units 
double get_B(const Atom &atom){
    return (-1 * PARAMETER_INFO.at(atom.z_num).at("-Beta")); // / CONVERSION_FACTOR);
}

// Helper to calculate the integral over the product of 4 gaussians 
double get_integral_over_4_gaussians(
    const Gaussian &k, 
    const Gaussian &k_prime, 
    const Gaussian &l, 
    const Gaussian &l_prime
){  
    double sigma_A = (1.0) / (k.alpha + k_prime.alpha); 
    double U_A = pow(sigma_A * M_PI, 1.5); 
    double sigma_B = (1.0) / (l.alpha + l_prime.alpha);
    double U_B = pow(sigma_B * M_PI, 1.5); 
    double V_squared = pow((sigma_A + sigma_B),-1); 
    double T = V_squared * (k.center - l.center).squaredNorm(); 

    if (k.center == l.center){ // Same Atom 
        return U_A * U_B * sqrt(2 * V_squared) * sqrt(2 / M_PI); 
    } else {    // Diff Atoms 
        double temp = sqrt(pow((k.center - l.center).squaredNorm(),-1));
        return U_A * U_B * temp * erf(sqrt(T)); 
    }
}

Eigen::Vector3d get_integral_over_4_gaussians_dir(
    const Gaussian &k, 
    const Gaussian &k_prime, 
    const Gaussian &l, 
    const Gaussian &l_prime
){  
    double sigma_A = (1.0) / (k.alpha + k_prime.alpha); 
    double U_A = pow(sigma_A * M_PI, 1.5); 
    double sigma_B = (1.0) / (l.alpha + l_prime.alpha);
    double U_B = pow(sigma_B * M_PI, 1.5); 
    double V_squared = pow((sigma_A + sigma_B),-1); 
    double T = V_squared * (k.center - l.center).squaredNorm(); 

    if (k.center == l.center){ // Same Atom 
        return Eigen::Vector3d::Zero(); 
    } else {    // Diff Atoms 
        return (
            U_A 
            * U_B 
            * (k.center - l.center) 
            / (k.center - l.center).squaredNorm() 
            * ((- erf(sqrt(T)) / (k.center - l.center).norm()) + (2 * sqrt(V_squared) * exp(-T) / sqrt(M_PI) ))
            ); 
    }
}

double get_gamma_elem(const Atom &A, const Atom &B){
    ContractedGaussian A_s = get_s_contracted_gaussian(A);
    ContractedGaussian B_s = get_s_contracted_gaussian(B); 

    double sum = 0; 
    for (int k = 0; k < NUM_CONTRACTED; k++){
    for (int k_prime = 0; k_prime < NUM_CONTRACTED; k_prime++){
    for (int l = 0; l < NUM_CONTRACTED; l++){
    for (int l_prime = 0; l_prime < NUM_CONTRACTED; l_prime++){
        double temp = 1;
        temp *= A_s.cont_coef.at(k)       * A_s.norm_coef.at(k); 
        temp *= A_s.cont_coef.at(k_prime) * A_s.norm_coef.at(k_prime); 
        temp *= B_s.cont_coef.at(l)       * B_s.norm_coef.at(l); 
        temp *= B_s.cont_coef.at(l_prime) * B_s.norm_coef.at(l_prime); 
        temp *= get_integral_over_4_gaussians(A_s.gaussians.at(k), A_s.gaussians.at(k_prime), B_s.gaussians.at(l), B_s.gaussians.at(l_prime));
        temp *= CONVERSION_FACTOR;
        sum += temp; 
    }
    }
    }
    }
    return sum; 
}

// Helper to get the gamma matrix from a list of atoms
Eigen::MatrixXd get_gamma(std::vector<Atom> atoms){
    const int N = atoms.size(); 
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(N,N);
    for (int A = 0; A < N; A++){
        for (int B = 0; B < N; B++){
            gamma(A,B) = get_gamma_elem(atoms.at(A),atoms.at(B)); 
        }
    }
    return gamma; 
}

// helper function to calculate a derivative of gamma
Eigen::Vector3d get_gamma_dir_elem(const Atom &A, const Atom &B, const std::vector<ContractedGaussian> &vcg){
    ContractedGaussian A_s = get_s_contracted_gaussian(A); 
    ContractedGaussian B_s = get_s_contracted_gaussian(B); 

    Eigen::Vector3d dir = Eigen::Vector3d::Zero();

    for (int k = 0; k < NUM_CONTRACTED; k++){
    for (int k_prime = 0; k_prime < NUM_CONTRACTED; k_prime++){
    for (int l = 0;  l < NUM_CONTRACTED; l++){
    for (int l_prime = 0; l_prime < NUM_CONTRACTED; l_prime++){
        double coeff = 1; 
        coeff *= A_s.norm_coef.at(k); 
        coeff *= A_s.cont_coef.at(k); 
        coeff *= A_s.norm_coef.at(k_prime); 
        coeff *= A_s.cont_coef.at(k_prime);
        coeff *= B_s.norm_coef.at(l); 
        coeff *= B_s.cont_coef.at(l); 
        coeff *= B_s.norm_coef.at(l_prime); 
        coeff *= B_s.cont_coef.at(l_prime);
        coeff *= CONVERSION_FACTOR; 
        dir += coeff * get_integral_over_4_gaussians_dir(
            A_s.gaussians.at(k), 
            A_s.gaussians.at(k_prime), 
            B_s.gaussians.at(l), 
            B_s.gaussians.at(l_prime)
            ); 
    }
    }
    }
    }
    return dir; 
}
// ── 1) Overlap matrix via contracted_gaussian_overlap ──
Eigen::MatrixXd make_overlap_matrix(
    const std::vector<ContractedGaussian>& basis)
{
    int dim = basis.size();
    Eigen::MatrixXd S(dim, dim);
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        S(i,j) = contracted_gaussian_overlap(basis[i], basis[j]);
    return S;
}

Eigen::MatrixXd build_density_matrix_overlap(
    int nocc,
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& S)
{
    Eigen::MatrixXd Cocc = C.leftCols(nocc);
    return 2.0 * (Cocc * (Cocc.transpose() * S));
}


// ── 3) SCF convergence test on max|ΔF| ──
bool close_enough(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    double tol)
{
    return (A - B).cwiseAbs().maxCoeff() < tol;
}
// ── 4)
double scf_electronic_energy(
    const Eigen::MatrixXd& P,
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& F)
{
    // E = 0.5 * Tr[P * (H + F)]
    Eigen::MatrixXd M = H + F;
    return 0.5 * (P.cwiseProduct(M)).sum();
} 

// Vector of Matrices of Derivatives
Eigen::VectorX<Eigen::MatrixX<Eigen::Vector3d>> get_vector_of_gamma_dir(
    const std::vector<Atom> &atoms, 
    const std::vector<ContractedGaussian> &vcg
    )
{
    const int N = atoms.size(); 
    const int n = vcg.size(); 
    Eigen::VectorX<Eigen::MatrixX<Eigen::Vector3d>> vector_of_gamma_dir (N);
    for (int C = 0; C < N; C++){
        Eigen::MatrixX<Eigen::Vector3d> dir (n,n); 
    for (int A = 0; A < N; A++){
    for (int B = 0; B < N; B++){
    if (((A == C) && (C != B)) || ((B == C) && (C != A))){
        dir(A,B) = get_gamma_dir_elem(atoms.at(A), atoms.at(B), vcg);
    }
    }
    }
        vector_of_gamma_dir(C) = dir; 
    }
    return vector_of_gamma_dir; 
}

// This returns an Eigen::VectorXd ov Atomic Density 
Eigen::VectorXd get_p_total_atomwise_diag_vector(const std::vector<Atom> &atoms,  const Eigen::MatrixXd &p_total) {
    Eigen::VectorXd p_total_diag_atomwise_sum = Eigen::VectorXd::Zero(atoms.size());
    int p_total_index = 0; 
    for (int atom_index = 0; atom_index < atoms.size(); atom_index++){
        int atomic_orbitals = PARAMETER_INFO.at(atoms.at(atom_index).z_num).at("Valence Orbitals");
        for (int orbital_index = 0; orbital_index < atomic_orbitals; orbital_index++){
            p_total_diag_atomwise_sum(atom_index) += p_total(p_total_index, p_total_index); 
            p_total_index++; 
        }
    }
    return p_total_diag_atomwise_sum; 
}

// Helper to get the third term of 1.4 
double get_third_term_of_1_4(
    const int A, 
    const std::vector<Atom> &atoms, 
    const Eigen::VectorXd &p_total_atomwise_diag_vector,
    const Eigen::MatrixXd &gamma){
    double sum = 0; 
    for( int B = 0; B < atoms.size(); B++){
        if (B != A){
            sum += (p_total_atomwise_diag_vector(B) - get_Z(atoms.at(B)))*gamma(A,B); 
        }
    } 
    return sum; 
}

// returns the on diagonal fock matrix element for the spin type 
double get_on_diagonal_fock_element(
    const int A,
    const int u,  
    const std::vector<Atom> &atoms, 
    const std::vector<ContractedGaussian> &vcg, 
    const Eigen::MatrixXd &p_spin, 
    const Eigen::MatrixXd &p_total, 
    const Eigen::VectorXd &p_total_atomwise_diag_vector,
    const Eigen::MatrixXd &gamma
    )
{
    double result = 0; 
    result -= get_half_IuAu(vcg.at(u)); 
    result += ((p_total_atomwise_diag_vector(A) - get_Z(atoms.at(A))) - (p_spin(u,u) - (0.5)))*gamma(A,A); 
    result += get_third_term_of_1_4(A, atoms, p_total_atomwise_diag_vector, gamma);
    return result; 
}

// returns the on diagonal fock matrix element for the spin type
double get_off_diagonal_fock_element(
    const int A, 
    const int B, 
    const int u, 
    const int v,
    const std::vector<Atom> &atoms, 
    const Eigen::MatrixXd &s, 
    const Eigen::MatrixXd &p_spin, 
    const Eigen::MatrixXd &gamma
    )
{
    return ((0.5) * (get_B(atoms.at(A)) + get_B(atoms.at(B))) * (s(u,v))) - (p_spin(u,v) * gamma(A,B));  
}

// get the fock matrix 
Eigen::MatrixXd get_fock(
        const std::vector<Atom> &atoms, 
        const Eigen::MatrixXd &p_total, 
        const Eigen::MatrixXd &p_spin)
    {
    const int N = get_total_valence_orbitals(atoms); 
    Eigen::MatrixXd f = Eigen::MatrixXd::Zero(N,N); // Fock Matrix

    std::vector<ContractedGaussian> vcg = get_vector_of_contracted_gaussians(atoms); // Vector of contracted Gaussians
    Eigen::MatrixXd gamma = get_gamma(atoms); 
    Eigen::VectorXd p_total_atomwise_diag_vector = get_p_total_atomwise_diag_vector(atoms, p_total);
    Eigen::MatrixXd s = make_overlap_matrix(vcg); 
    
    int u = 0; 
    for(int A = 0; A < atoms.size(); A++){ 
        int num_A_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(A).z_num).at("Valence Orbitals")); 
        for (int index_in_A = 0; index_in_A < num_A_orbitals; index_in_A++){
            // Iterating through u, first through atoms, then through contracted gaussians 
            int v = 0; 
            for (int B = 0; B < atoms.size(); B++){
                int num_B_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(B).z_num).at("Valence Orbitals"));
                for (int index_in_B = 0; index_in_B < num_B_orbitals; index_in_B++){
                    // Iterating through v, first through atoms, then through contracted gaussians
                    if (u == v) { // On Diagonal 
                        f(u,u) = get_on_diagonal_fock_element(A, u, atoms, vcg, p_spin, p_total, p_total_atomwise_diag_vector, gamma); 
                    } else { // Off Diagonal
                        f(u,v) = get_off_diagonal_fock_element(A, B, u, v, atoms, s, p_spin, gamma); 
                    }
                    v++; 
                }
            }
            u++; 
        }
    }

    return f; 
}

// Helper to get the third term of 1.4 
double get_third_term_of_2_6(
    const int A, 
    const std::vector<Atom> &atoms, 
    const Eigen::MatrixXd &gamma){
    double sum = 0; 
    for( int B = 0; B < atoms.size(); B++){
        if (B != A){
            sum += (get_Z(atoms.at(B))*gamma(A,B)); 
        }
    } 
    return sum; 
}


// Helper function which returns an on diagonal element of the hamiltonian 
double get_on_diagonal_hamiltonian_element(
    const int A, 
    const int u, 
    const std::vector<Atom> &atoms, 
    const std::vector<ContractedGaussian> &vcg, 
    const Eigen::MatrixXd &gamma 
){  
    double result = 0; 
    result -= get_half_IuAu(vcg.at(u)); 
    result -= ((get_Z(atoms.at(A)) - 0.5) * gamma(A,A)); 
    result -= get_third_term_of_2_6(A, atoms, gamma); 
    return result; 
}

// Helper function which returns an off diagonal element of the hamiltonian 
double get_off_diagonal_hamiltonian_element(
    const int A, 
    const int B, 
    const int u, 
    const int v, 
    const std::vector<Atom> &atoms,
    const Eigen::MatrixXd &s)
{
    return ((0.5) * (get_B(atoms.at(A)) + get_B(atoms.at(B))) * (s(u,v))); 
}


// Helper to make the core hamiltonian matrix
Eigen::MatrixXd get_hamiltonian(const std::vector<Atom> &atoms){
    const int N = get_total_valence_orbitals(atoms); 
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(N,N);

    std::vector<ContractedGaussian> vcg = get_vector_of_contracted_gaussians(atoms);  
    Eigen::MatrixXd s = make_overlap_matrix(vcg);
    Eigen::MatrixXd gamma = get_gamma(atoms); 

    int u = 0; 
    for(int A = 0; A < atoms.size(); A++){ 
        int num_A_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(A).z_num).at("Valence Orbitals")); 
        for (int index_in_A = 0; index_in_A < num_A_orbitals; index_in_A++){
            // Iterating through u, first through atoms, then through contracted gaussians 
            int v = 0; 
            for (int B = 0; B < atoms.size(); B++){
                int num_B_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(B).z_num).at("Valence Orbitals"));
                for (int index_in_B = 0; index_in_B < num_B_orbitals; index_in_B++){
                    // Iterating through v, first through atoms, then through contracted gaussians
                    if (u == v) { // On Diagonal 
                        h(u,u) = get_on_diagonal_hamiltonian_element(A, u, atoms, vcg, gamma); 
                    } else { // Off Diagonal
                        h(u,v) = get_off_diagonal_hamiltonian_element(A, B, u, v, atoms, s); 
                    }
                    v++; 
                }
            }
            u++; 
        }
    }
    
    return h; 
}

// ─────────────────────────────────────────────────────────────────────────────
// CNDO/S SCF driver (run_CNDO_S) with overlap‐aware density build
// Make sure you have this helper un-commented above in hw5_utils.cpp:
// 
// Eigen::MatrixXd build_density_matrix_overlap(
//     int nocc,
//     const Eigen::MatrixXd& C,
//     const Eigen::MatrixXd& S)
// {
//     Eigen::MatrixXd Cocc = C.leftCols(nocc);
//     return 2.0 * (Cocc * (Cocc.transpose() * S));
// }
// ─────────────────────────────────────────────────────────────────────────────

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
run_CNDO_S(const std::vector<Atom>& atoms,
           int p, int q)
{
    using namespace Eigen;
    const int Natom = atoms.size();

    // 1) Build AO basis
    auto basis = get_vector_of_contracted_gaussians(atoms);
    const int dim = basis.size();

    // 2) Overlap matrix S_uv
    MatrixXd S(dim, dim);
    for (int u = 0; u < dim; ++u)
      for (int v = 0; v < dim; ++v)
        S(u,v) = contracted_gaussian_overlap(basis[u], basis[v]);

    // 3) One-electron core Hamiltonian H_uv
    MatrixXd H = get_hamiltonian(atoms);

    // 4) Atomic γ-matrix Γ(A,B)
    MatrixXd Gamma(Natom, Natom);
    for (int A = 0; A < Natom; ++A)
      for (int B = 0; B < Natom; ++B)
        Gamma(A,B) = get_gamma_elem(atoms[A], atoms[B]);

    // 5) Map each AO basis function index u → its parent atom A
    std::vector<int> basis_to_atom;
    basis_to_atom.reserve(dim);
    for (int A = 0; A < Natom; ++A) {
        auto cgv = get_contracted_gaussian_basis(atoms[A]);
        for (size_t i = 0; i < cgv.size(); ++i)
            basis_to_atom.push_back(A);
    }

    // 6) Initial Fock = diag(H)
    MatrixXd F = H.diagonal().asDiagonal();

    // 7) SCF parameters
    MatrixXd P_old = MatrixXd::Zero(dim, dim);
    const double DAMP    = 0.3;
    const double TOL     = 1e-8;
    const int    MAX_ITR = 500;

    // 8) SCF loop
    for (int iter = 0; iter < MAX_ITR; ++iter) {
        // 8a) Generalized eigenproblem F C = S C ε
        GeneralizedSelfAdjointEigenSolver<MatrixXd> ges(F, S);
        if (ges.info() != Success)
            throw std::runtime_error("CNDO/S eigensolver failed");
        MatrixXd C = ges.eigenvectors();

        // 8b) Build new density with overlap
        int nocc = p + q;
        MatrixXd P_new = build_density_matrix_overlap(nocc, C, S);

        // 8c) Damped mixing
        MatrixXd P = DAMP * P_new + (1.0 - DAMP) * P_old;
        P_old = P;

        // 8d) Rebuild Fock: H + two-electron terms
        MatrixXd F_new = H;
        for (int u = 0; u < dim; ++u) {
            int A = basis_to_atom[u];
            for (int v = 0; v < dim; ++v) {
                int B = basis_to_atom[v];
                if (u == v) {
                    // diagonal contribution
                    double sum = 0.0;
                    for (int w = 0; w < dim; ++w)
                        sum += P(w,w) * Gamma(A, basis_to_atom[w]);
                    F_new(u,u) += sum;
                } else {
                    // off-diagonal contribution
                    F_new(u,v) += 0.5 * P(u,v)
                                  * (Gamma(A,A) + Gamma(B,B));
                }
            }
        }

        // 8e) Convergence check
        double dF = (F_new - F).cwiseAbs().maxCoeff();
        if (dF < TOL) {
            std::cout << "CNDO/S SCF converged in " << iter+1
                      << " iterations; ΔF_max = " << dF << "\n";
            return {F_new, P};
        }
        F = std::move(F_new);
    }

    throw std::runtime_error(
      "CNDO/S: SCF did not converge after max iterations");
}



// runs CNDO2 till convergence and returns the density matricies 
std::pair<Eigen::MatrixXd,Eigen::MatrixXd> run_CNDO2(
    const std::vector<Atom> &atoms, 
    const int p, 
    const int q, 
    const double tol = 1e-6, 
    const bool verbose = false
    ) 
{
    int total_valence_orbitals = get_total_valence_orbitals(atoms); 

    Eigen::MatrixXd p_alpha = Eigen::MatrixXd::Zero(total_valence_orbitals, total_valence_orbitals); 
    Eigen::MatrixXd p_beta  = Eigen::MatrixXd::Zero(total_valence_orbitals, total_valence_orbitals); 

    bool converged = false; 
    int count = 0; 

    while (!converged && (count < 100)) {
        if (verbose){ 
            std::cout << "Iteration: " << count << std::endl; 
        }

        Eigen::MatrixXd p_total = p_alpha + p_beta; 
        Eigen::MatrixXd fock_alpha = get_fock(atoms, p_total, p_alpha);
        Eigen::MatrixXd fock_beta  = get_fock(atoms, p_total, p_beta); 

        if (verbose){
            std::cout << "Fa" << std::endl; 
            std::cout << fock_alpha << std::endl; 

            std::cout << "Fb" << std::endl;
            std::cout << fock_beta << std::endl; 
        }
       

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Solver_alpha(fock_alpha);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Solver_beta(fock_beta);

        if (verbose){
            std::cout << "after solving eigen equation: " << count << std::endl; 
        }

        Eigen::MatrixXd c_alpha = Solver_alpha.eigenvectors().real(); 
        Eigen::MatrixXd c_beta  = Solver_beta.eigenvectors().real(); 

        if (verbose){
            std::cout << "Ca" << std::endl; 
            std::cout << c_alpha << std::endl; 

            std::cout << "Cb" << std::endl; 
            std::cout << c_beta << std::endl; 
        }

        Eigen::MatrixXd p_alpha_old (p_alpha); 
        Eigen::MatrixXd p_beta_old  (p_beta); 

        if (verbose) {
            std::cout << " p = " << p << ", q = " << q << std::endl;
        }

        p_alpha = build_density_matrix(p, c_alpha); 
        p_beta  = build_density_matrix(q, c_beta); 

        if (verbose) {
            std::cout << "Pa_new" << std::endl; 
            std::cout << p_alpha  << std::endl; 
            std::cout << "Pb_new" << std::endl;
            std::cout << p_beta   << std::endl;  
        }

        if (close_enough(p_alpha, p_alpha_old, tol) && close_enough(p_beta, p_beta_old, tol)) {
            converged = true; 
        } else {
            count++; 
        }
    }
    return std::make_pair(p_alpha, p_beta); 
}

// This calcuates the energy in a spin 
double E_spin(
    const Eigen::MatrixXd p_spin,
    const Eigen::MatrixXd h, 
    const Eigen::MatrixXd f
){
    const int n = p_spin.rows(); 
    double sum = 0; 
    for (int u = 0; u < n; u++){
        for (int v = 0; v < n; v++){
            sum += p_spin(u,v)*(h(u,v) + f(u,v)); 
        }
    }
    return sum; 
}

Eigen::Vector3d get_analytical_s_dir_elem_per_gaussian(const Gaussian &g1, const Gaussian &g2){
    double Ix = analytical_S_ab_dim(g1, g2, 0); 
    double Iy = analytical_S_ab_dim(g1, g2, 1); 
    double Iz = analytical_S_ab_dim(g1, g2, 2);
    double dIxdx = analytical_S_ab_dim_dir(g1, g2, 0);
    double dIydy = analytical_S_ab_dim_dir(g1, g2, 1); 
    double dIzdz = analytical_S_ab_dim_dir(g1, g2, 2);
    return Eigen::Vector3d (dIxdx * Iy * Iz, Ix * dIydy * Iz, Ix * Iy * dIzdz);
}

Eigen::Vector3d get_analytical_s_dir_elem(const ContractedGaussian &cg1, const ContractedGaussian &cg2){
    Eigen::Vector3d dir = Eigen::Vector3d::Zero(); 
    for (int k = 0; k < NUM_CONTRACTED; k++){
    for (int l = 0; l < NUM_CONTRACTED; l++){
        double coeff = cg1.cont_coef.at(k) * cg1.norm_coef.at(k) * cg2.cont_coef.at(l) * cg2.norm_coef.at(l); 
        dir += coeff * get_analytical_s_dir_elem_per_gaussian(cg1.gaussians.at(k), cg2.gaussians.at(l)); 
    }
    }
    return dir; 
}


Eigen::VectorX<Eigen::MatrixX<Eigen::Vector3d>> get_vector_of_s_dir(
    const std::vector<Atom> &atoms,
    const std::vector<ContractedGaussian> &vcg
    )
{
    const int n = vcg.size(); 
    const int N = atoms.size(); 
    Eigen::VectorX<Eigen::MatrixX<Eigen::Vector3d>> s_dir (N);

    for (int C = 0; C < N; C++){
        Eigen::MatrixX<Eigen::Vector3d> dir (n,n);
        int u = 0; 
        for (int A = 0; A < N; A++){ 
            int num_A_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(A).z_num).at("Valence Orbitals")); 
            for (int index_in_A = 0; index_in_A < num_A_orbitals; index_in_A++){
                // Iterating through u, first through atoms, then through contracted gaussians 
                int v = 0; 
                for (int B = 0; B < N; B++){
                    int num_B_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(B).z_num).at("Valence Orbitals"));
                    for (int index_in_B = 0; index_in_B < num_B_orbitals; index_in_B++){
                        // Iterating through v, first through atoms, then through contracted gaussians
                        if(A != B){
                            dir(u,v) = get_analytical_s_dir_elem(vcg.at(u), vcg.at(v)); 
                        }
                        v++; 
                    }
                }
                u++; 
            }
        }
        s_dir(C) = dir; 
    }
    
    return s_dir; 
}

// This helper function calculates the nuclear repulsion energy of a set of atoms
double get_nuclear_repulsion_energy(const std::vector<Atom> &atoms){
    const int N = atoms.size();
    double energy = 0; 
    for (int A = 0; A < N; A++){
        for (int B = A + 1; B < N; B++){
                energy += (get_Z(atoms.at(A)) * get_Z(atoms.at(B))) / (atoms.at(A).pos - atoms.at(B).pos).norm();
        }
    }
    return energy * CONVERSION_FACTOR; 
}


Eigen::VectorX<Eigen::Vector3d> get_nuclear_energy_dir(
    const std::vector<Atom> &atoms
){
    const int N = atoms.size(); 
    Eigen::VectorX<Eigen::Vector3d> nuclear_energy_dir(N);
    for (int A = 0; A < N; A++){
        Eigen::Vector3d dir = Eigen::Vector3d::Zero(); 
        for (int B = 0; B < N; B++){
           if (A != B){
            dir -= (
                get_Z(atoms.at(A)) 
                * get_Z(atoms.at(B)) 
                * (atoms.at(A).pos - atoms.at(B).pos) 
                / pow((atoms.at(A).pos - atoms.at(B).pos).norm(),3)
                * CONVERSION_FACTOR
                ); 
           }
        }
        nuclear_energy_dir(A) = dir; 
    }
    return nuclear_energy_dir; 
}

// This calculates the energy of a converged density matrix in CNDO2 
double E_CNDO2(
    const Eigen::MatrixXd &p_alpha, 
    const Eigen::MatrixXd &p_beta, 
    const std::vector<Atom> &atoms
){
    Eigen::MatrixXd p_total = p_alpha + p_beta; 
    Eigen::MatrixXd h = get_hamiltonian(atoms); 
    Eigen::MatrixXd f_alpha = get_fock(atoms, p_total, p_alpha); 
    Eigen::MatrixXd f_beta  = get_fock(atoms, p_total, p_beta); 
    double energy = 0;
    energy += ((0.5) * E_spin(p_alpha, h, f_alpha)); 
    energy += ((0.5) * E_spin(p_beta , h, f_beta ));
    energy += get_nuclear_repulsion_energy(atoms);  
    return energy; 
}


double get_x_element(int u, int v, int A, int B, const std::vector<Atom> &atoms, const Eigen::MatrixXd &p_total){
    return (get_B(atoms.at(A)) + get_B(atoms.at(B))) * p_total(u,v); 
}

Eigen::MatrixXd get_x_matrix(const std::vector<Atom> &atoms, const Eigen::MatrixXd &p_total){
    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(p_total.rows(), p_total.cols()); 
    int u = 0; 
    for(int A = 0; A < atoms.size(); A++){ 
        int num_A_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(A).z_num).at("Valence Orbitals")); 
        for (int index_in_A = 0; index_in_A < num_A_orbitals; index_in_A++){
            // Iterating through u, first through atoms, then through contracted gaussians 
            int v = 0; 
            for (int B = 0; B < atoms.size(); B++){
                int num_B_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(B).z_num).at("Valence Orbitals"));
                for (int index_in_B = 0; index_in_B < num_B_orbitals; index_in_B++){
                    // Iterating through v, first through atoms, then through contracted gaussians
                    x(u,v) = get_x_element(u, v, A, B, atoms, p_total);
                    v++; 
                }
            }
            u++; 
        }
    }
    return x; 
} 

// THIS FUNCTION MUST BE CALLED WHEN u and V are at their first positions. 
double get_y_elem_last_term(
    const int u, 
    const int v, 
    const int A, 
    const int B,
    const std::vector<Atom> &atoms,
    const Eigen::MatrixXd &p_alpha, 
    const Eigen::MatrixXd &p_beta
    )
{   
    double result = 0;
    int num_A_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(A).z_num).at("Valence Orbitals")); 
    int num_B_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(B).z_num).at("Valence Orbitals")); 
    for (int i = u; i < u + num_A_orbitals; i++){
    for (int j = v; j < v + num_B_orbitals; j++){
        result += (p_alpha(i,j) * p_alpha(i,j)) + (p_beta(i,j) * p_beta(i,j));
    }
    }
    return result; 
}

// THIS FUNCTION MUST BE CALLED WHEN u and v are at their first positions. 
double get_y_element(
    const int u, 
    const int v, 
    const int A, 
    const int B, 
    const std::vector<Atom> &atoms, 
    Eigen::VectorXd p_total_atomwise_diag_vector, 
    const Eigen::MatrixXd &p_alpha, 
    const Eigen::MatrixXd &p_beta
    )
{
    double result = 0; 
    result += p_total_atomwise_diag_vector(A) * p_total_atomwise_diag_vector(B); 
    result -= get_Z(atoms.at(B)) * p_total_atomwise_diag_vector(A);
    result -= get_Z(atoms.at(A)) * p_total_atomwise_diag_vector(B); 
    result -= get_y_elem_last_term(u, v, A, B, atoms, p_alpha, p_beta); 
    return result; 
}   


Eigen::MatrixXd get_y_matrix(
    const std::vector<Atom> &atoms,
    const Eigen::MatrixXd &p_alpha, 
    const Eigen::MatrixXd &p_beta
){
    Eigen::MatrixXd p_total = p_alpha + p_beta; 
    Eigen::VectorXd p_total_atomwise_diag_vector = get_p_total_atomwise_diag_vector(atoms, p_total); 
    Eigen::MatrixXd y = Eigen::MatrixXd::Zero(p_total.rows(), p_total.cols()); 
    const int N = atoms.size(); 
    int u = 0; 
    for(int A = 0; A < N; A++){ 
            // Iterating through u, first through atoms, then through contracted gaussians 
            int v = 0; 
            for (int B = 0; B < N; B++){
                    // Iterating through v, first through atoms, then through contracted gaussians
                    y(u,v) = get_y_element(u, v, A, B, atoms, p_total_atomwise_diag_vector, p_alpha, p_beta);
                    v += static_cast<int>(PARAMETER_INFO.at(atoms.at(B).z_num).at("Valence Orbitals")); 
            }
            u += static_cast<int>(PARAMETER_INFO.at(atoms.at(A).z_num).at("Valence Orbitals")); 
    }
    return y; 
}


// Sum of x_uv * S_dir_uv for u != v 
Eigen::Vector3d get_first_term_of_energy_derivative_of_atom_A(
    const int A_index,
    const std::vector<Atom> &atoms, 
    const std::vector<ContractedGaussian> &vcg, 
    const Eigen::MatrixX<Eigen::Vector3d> &s_dir, 
    const Eigen::MatrixXd &x
    )
{
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    const int N = atoms.size();  

    int u = 0; 
    for(int A = 0; A < N; A++){ 
        int num_A_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(A).z_num).at("Valence Orbitals")); 
        for (int index_in_A = 0; index_in_A < num_A_orbitals; index_in_A++){
            // Iterating through u, first through atoms, then through contracted gaussians 
            int v = 0; 
            for (int B = 0; B < N; B++){
                int num_B_orbitals = static_cast<int>(PARAMETER_INFO.at(atoms.at(B).z_num).at("Valence Orbitals"));
                for (int index_in_B = 0; index_in_B < num_B_orbitals; index_in_B++){
                    // Iterating through v, first through atoms, then through contracted gaussians
                    if ((A == A_index) && (A != B)) {
                        result += (x(u,v) * s_dir(u,v)); 
                    }
                    v++; 
                }
            }
            u++; 
        }
    }
    return result; 
}


Eigen::Vector3d get_second_term_of_energy_derivative_of_atom_A(
    const int A, 
    const Eigen::MatrixX<Eigen::Vector3d> &gamma_dir, 
    const Eigen::MatrixXd &y
    ){
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    const int N = gamma_dir.rows(); 
    
    for (int B = 0; B < N; B++){
        if (A != B){
            result += y(A,B) * gamma_dir(A,B); 
        }
    }

    return result; 
}

Eigen::Vector3d get_energy_derivative_of_atom_A(
    const int A,
    const std::vector<Atom> &atoms, 
    const std::vector<ContractedGaussian> &vcg, 
    const Eigen::MatrixX<Eigen::Vector3d> &s_dir, 
    const Eigen::MatrixX<Eigen::Vector3d> &gamma_dir,
    const Eigen::MatrixX<Eigen::Vector3d> &nuclear_energy_dir, 
    const Eigen::MatrixXd &x, 
    const Eigen::MatrixXd &y
    )
{
    Eigen::Vector3d dirivative = Eigen::Vector3d::Zero(); 
    dirivative += get_first_term_of_energy_derivative_of_atom_A(A, atoms, vcg, s_dir, x);
    dirivative += get_second_term_of_energy_derivative_of_atom_A(A, gamma_dir, y); 
    dirivative += nuclear_energy_dir(A); 
    return dirivative; 
}

Eigen::VectorX<Eigen::Vector3d> get_energy_derivative_vector(
    const std::vector<Atom> &atoms,
    const std::vector<ContractedGaussian> &vcg, 
    const Eigen::MatrixXd p_alpha, 
    const Eigen::MatrixXd p_beta
){  
    const int N = atoms.size(); 
    Eigen::MatrixXd p_total = p_alpha + p_beta; 

    Eigen::VectorX<Eigen::MatrixX<Eigen::Vector3d>> vector_of_s_dir = get_vector_of_s_dir(atoms, vcg); 
    Eigen::VectorX<Eigen::MatrixX<Eigen::Vector3d>> vector_of_gamma_dir = get_vector_of_gamma_dir(atoms, vcg); 
    Eigen::VectorX<Eigen::Vector3d> nuclear_energy_dir = get_nuclear_energy_dir(atoms);
    Eigen::MatrixXd x = get_x_matrix(atoms, p_total); 
    Eigen::MatrixXd y = get_y_matrix(atoms, p_alpha, p_beta); 
    Eigen::VectorX<Eigen::Vector3d> dir_vec (N); 
    for (int A = 0; A < N; A++){
        dir_vec(A) = get_energy_derivative_of_atom_A(A, atoms, vcg, vector_of_s_dir(A), vector_of_gamma_dir(A), nuclear_energy_dir, x, y);
    }
    return dir_vec; 
}

/*
Typical file parsing helper
*/
std::vector<Atom> parse_file(std::string filepath, bool verbose = false){
    if (verbose)
        std::cout << "Attempting to parse: " << filepath << std::endl; 

    std::vector<Atom> atoms; 
    
    std::ifstream file(filepath); 
    
    if(!file.is_open()){
        throw std::runtime_error("File could not be opened");
    }
    

    std::string line; 
    std::getline(file, line); // remove first line, which has the number of atoms
    while (std::getline(file, line)){
        std::istringstream linestream(line); 
        
        Atom a; 
        Eigen::Vector3d p; 

        linestream >> a.z_num; 
        linestream >> p.x(); 
        linestream >> p.y(); 
        linestream >> p.z(); 
        
        if (verbose)
            std::cout << a.z_num 
                        << "(" 
                        << p.x() 
                        << ", "
                        << p.y()
                        << ", " 
                        << p.z() 
                        << ")" 
                        << std::endl; 
        

        a.pos = p; 

        // This is cheating but I would use map.contains if I was allowed to use c++ 20 
        if (!BASIS_INFO.contains(a.z_num)){
            throw std::runtime_error("One of the elements in the input was not as expected");
        }
        else
        {
            atoms.push_back(a); 
        }
    }

    file.close(); 

    return atoms; 
}