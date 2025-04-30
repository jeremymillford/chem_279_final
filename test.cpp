#include <armadillo>
#include <iostream>

int main() {
    arma::mat A = arma::randu<arma::mat>(4,4);
    std::cout << "Random 4x4 matrix:" << std::endl;
    std::cout << A << std::endl;

    arma::mat B = arma::randu<arma::mat>(4,4);
    arma::mat C;
    C = A * B;
    std::cout << C << std::endl;
    std::cout << "Matrix multiplication result:" << std::endl;
    return 0;
}
