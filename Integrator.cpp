#pragma once

#include <functional> 
#include <exception>
#include <algorithm> 
#include <Eigen/Dense>

#include "Gaussian.cpp"

class Trapazoidal_Integrator {
    
    std::function<double(double)> func;
    const double lower_limit; 
    const double upper_limit; 
    int step; 
    public:
    double integral; 
    Trapazoidal_Integrator(
        std::function<double(double)> func, 
        const double lower_limit,
        const double upper_limit
        ):
        func(func),
        lower_limit(lower_limit), 
        upper_limit(upper_limit),
        step(0), 
        integral(0.0)
        {}
    
    double next() {
        double x, tnm, sum, delta;
        int it, j; 
        step +=1; 
        if(step == 1){
            integral = (0.5 * (upper_limit - lower_limit)*(func(upper_limit) + func(lower_limit)));
        } else {
            for(it = 1, j=1; j < step - 1; j+=1) it <<= 1; 
            tnm = it; 
            delta = (upper_limit - lower_limit)/tnm; 
            x = lower_limit + (0.5 * delta); 
            sum = 0.0; 
            for(j=0; j < it; j+=1){
                sum += func(x);
                x += delta; 
            }
            integral = 0.5*(integral + (upper_limit - lower_limit)*sum/tnm); 
        }
        return integral; 
    } 

    double accurate_estimate(double epsilon=1e-6, int max_steps=20){
        // Returns an estimate which meets 
        double olds = 0.0; 
        for(int j = 0; j < max_steps; j+=1){
            next();
            if (j > 5){
                if (
                    (abs(integral - olds) < epsilon * abs(olds))
                    ||(abs(integral) < epsilon && abs(olds) < epsilon)
                    ){
                    return integral; 
                }
                olds = integral; 
            } 
        }
        throw std::runtime_error("too many integration steps");
    }
};

double Gaussian_Product_Integrator_1d(Gaussian g1, Gaussian g2, double std_dev=4){
    // This assumes that both gaussian's centers have y=0, z=0; 
    // 4 standard deviations by default
    
    std::function<double(double)> func (
        [g1, g2](double x)->double {return g1(Eigen::Vector3d(x,0,0))*g2(Eigen::Vector3d(x,0,0));}
        ); 

    double lower_bound = std::min(g1.center.x() - g1.alpha * std_dev, g2.center.x() - g2.alpha * std_dev); 
    double upper_bound = std::max(g1.center.x() + g1.alpha * std_dev, g2.center.x() + g2.alpha * std_dev); 

    Trapazoidal_Integrator integral(func, lower_bound, upper_bound); 

    return integral.accurate_estimate(1e-10,30);
}
