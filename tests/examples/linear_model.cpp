#include "aoclda.h"
#include <iostream>

int main() {

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Mean squared error model" << std::endl;
    std::cout << "min ||Ax-b||^2; with A an 5x2 matrix" << std::endl << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    // problem data
    da_int m = 5, n = 2;
    double Al[m * n] = {1, 1, 2, 3, 3, 5, 4, 1, 5, 1};
    double bl[m] = {1, 1, 1, 1, 1};
    da_int nx = 2;
    double x[2];

    // Initialize the linear regression
    da_linreg handle = nullptr;
    da_status status;
    status = da_linreg_d_init(&handle);
    status = da_linreg_d_select_model(handle, linreg_model_mse);
    status = da_linreg_d_define_features(handle, n, m, Al, bl);
    // compute regression
    status = da_linreg_d_fit(handle);
    if (status == da_status_success) {
        std::cout << "regression computed successfully!" << std::endl;
        status = da_linreg_d_get_coef(handle, &nx, x);
        std::cout << "Coefficients: " << x[0] << " " << x[1] << std::endl;
        std::cout << "(Expected   : " << 0.199255 << " " << 0.130354 << ")" << std::endl;
    } else {
        std::cout << "Something wrong happened during MSE regression. Terminating"
                  << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    // Solve the same model with single precision
    // problem data
    float As[10] = {1, 1, 2, 3, 3, 5, 4, 1, 5, 1};
    float bs[5] = {1, 1, 1, 1, 1};
    float xs[2];

    std::cout.precision(2);
    // Initialize the linear regression
    da_linreg handle_s = nullptr;
    status = da_linreg_s_init(&handle_s);
    std::cout << "init status " << status << std::endl;
    status = da_linreg_s_select_model(handle_s, linreg_model_mse);
    std::cout << "model status " << status << std::endl;
    status = da_linreg_s_define_features(handle_s, n, m, As, bs);
    // compute regression
    status = da_linreg_s_fit(handle_s);
    if (status == da_status_success) {
        std::cout << "regression computed successfully!" << std::endl;
        status = da_linreg_s_get_coef(handle_s, &nx, xs);
        std::cout << "Coefficients: " << xs[0] << " " << xs[1] << std::endl;
        std::cout << "(Expected   : " << 0.19 << " " << 0.14 << ")" << std::endl;
    } else {
        std::cout << "Something wrong happened during MSE regression. Terminating"
                  << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    da_linreg_destroy(&handle);
    da_linreg_destroy(&handle_s);

    return 0;
}