#pragma once

#include <stdexcept> 
#include <Eigen/Dense>
#include <functional> 
#include <cmath>
#include <cassert> 


class Gaussian {
    public: 
    Eigen::Vector3d center; 
    double alpha; 
    Eigen::Vector3i power; 
    std::function<double(Eigen::Vector3d)> func;  
    Gaussian(Eigen::Vector3d center, double alpha, Eigen::Vector3i power) : 
        center(center), 
        alpha(alpha),  
        power(power),
        func(
            [center, alpha, power](Eigen::Vector3d p)->double
            {return 
                std::pow((p.x()-center.x()), power.x())
                * std::pow((p.y()-center.y()), power.y())
                * std::pow((p.z()-center.z()), power.z())
                * std::exp(-alpha*std::pow((p-center).norm(),2));
            })
        {}

    double operator()(const Eigen::Vector3d x) const {
        return func(x);  
    } 
    
    friend std::ostream& operator<<(std::ostream& os, const Gaussian& g) {
        os 
            << "center: (" 
            << g.center.x() 
            << ", "
            << g.center.y()
            << ", "
            << g.center.z()
            << ") alpha: "
            << g.alpha
            << " power: ("
            << g.power.x() 
            << ", "
            << g.power.y()
            << ", "
            << g.power.z() 
            << ")"
            ;
        return os;
    }
};

double analytical_gaussian_overlap(const Gaussian g1, const Gaussian g2);

class ContractedGaussian{
    public:
    int z_num; 
    std::vector<Gaussian> gaussians; 
    std::vector<double> cont_coef; 
    std::vector<double> norm_coef;     
    ContractedGaussian(const int z_num, const std::vector<Gaussian> &gaussians, const std::vector<double> cont_coeffs) :
        z_num(z_num), 
        gaussians(gaussians),
        cont_coef(cont_coeffs),
        norm_coef(cont_coeffs.size()) 
    {
        for (size_t i = 0; i < gaussians.size(); i++){
            norm_coef.at(i) =  1.0 / std::sqrt(analytical_gaussian_overlap(gaussians.at(i), gaussians.at(i)));
        }
        assert(gaussians.size() == cont_coef.size()); 
        assert(gaussians.size() == norm_coef.size()); 
    }

    friend std::ostream& operator<<(std::ostream& os, const ContractedGaussian& cg) {
        for (size_t i = 0; i < cg.gaussians.size(); i++){
            os 
                << "Gaussian " 
                << i 
                << " : "
                << cg.gaussians.at(i)
                << std::endl 
                << "contraction coefficient: " 
                << cg.cont_coef.at(i)
                << std::endl
                << "normalization coefficient: "
                << cg.norm_coef.at(i)
                << std::endl;
        }
        return os;
    }
}; 

// helper function for analytical calcuation
int double_factorial(int i){
    int result = 1; 
    while( i >= 1){
        result *= i; 
        i -=2;
    }
    return result; 
}

// helper function for analytical calculation
int factorial(int i){
    int result = 1; 
    while (i >= 1){
        result *= i; 
        i -=1; 
    }
    return result; 
}

// helper function for analytical calculation
double binomial(int m, int n){
    return static_cast<double>(factorial(m)) / static_cast<double>(factorial(n)*factorial(m-n));  
}

// This is a convenience function for the calculation of a single dimension 
double summation_nonsense(const Gaussian g1, const Gaussian g2, Eigen::Vector3d Rp, int dim){
    int la = g1.power[dim];
    int lb = g2.power[dim]; 
    double sum = 0; 
    for (int i = 0; i <= la; i++){
        for(int j = 0; j <= lb; j++){
            if((i +j) % 2 == 0){
                double element = 1; 
                element *= binomial(la, i);
                element *= binomial(lb, j); 
                element *= double_factorial(i + j -1); 
                element *= std::pow(Rp[dim] - g1.center[dim], la - i); 
                element *= std::pow(Rp[dim] - g2.center[dim], lb - j); 
                element /= std::pow(2*(g1.alpha + g2.alpha), (i+j)/2);
                sum += element; 
            }
        }
    }
    return sum;
}

// This function returns the integral between two gaussians over one dimention 
double analytical_S_ab_dim(const Gaussian g1, const Gaussian g2, int dim){
    double result = 1; 
    Eigen::Vector3d Rp = (g1.alpha*g1.center + g2.alpha*g2.center)/(g1.alpha + g2.alpha); 
    result *= std::exp(- g1.alpha * g2.alpha * std::pow(g1.center[dim] - g2.center[dim],2) / (g1.alpha + g2.alpha)); 
    result *= std::sqrt(M_PI/(g1.alpha + g2.alpha)); 
    result *= summation_nonsense(g1, g2, Rp, dim); 
    return result; 
}

double analytical_S_ab_dim_dir(const Gaussian g1, const Gaussian g2, int dim){
    Gaussian g1_1st_eq (g1); 
    g1_1st_eq.power(dim) -= 1; 
    Gaussian g1_2nd_eq (g1); 
    g1_2nd_eq.power(dim) += 1; 
    return (
        - g1.power(dim) * analytical_S_ab_dim(g1_1st_eq, g2, dim)
        + 2 * g1.alpha * analytical_S_ab_dim(g1_2nd_eq, g2, dim)
    ); 
}

// This function returns the analytical product of two gaussians
double analytical_gaussian_overlap(const Gaussian g1, const Gaussian g2){
    double result = 1; 
    for (int i = 0; i < 3; i++) {
        result *= analytical_S_ab_dim(g1, g2, i);
    }
    return result;
}

// This just returns a matrix of contracted gaussian overlaps
double contracted_gaussian_overlap(const ContractedGaussian &cg1, const ContractedGaussian &cg2){
    double sum = 0.0;
    for(size_t i = 0; i < cg1.gaussians.size(); i++){
        for(size_t j = 0; j < cg2.gaussians.size(); j++){
            sum += (
                analytical_gaussian_overlap(cg1.gaussians.at(i),cg2.gaussians.at(j))
                * cg1.cont_coef.at(i)
                * cg2.cont_coef.at(j)
                * cg1.norm_coef.at(i)
                * cg2.norm_coef.at(j)
                );
        }
    } 
    return sum;  
}


