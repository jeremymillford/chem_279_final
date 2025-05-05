#include <functional> 

double central_difference(std::function<double(double)> f, const double x_0 = 0, const double stepsize=1e-6){
    return -(f(x_0 + stepsize) - f(x_0 - stepsize))*(0.5 * stepsize); 
}
