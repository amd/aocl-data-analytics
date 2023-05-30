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
    double Al[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bl[5] = {1, 1, 1, 1, 1};
    double x[3];
    da_int nx = 2;

    // Initialize the linear regression
    da_handle handle = nullptr;
    da_status status;
    da_handle_init_d(&handle, da_handle_linmod);
    da_linmod_d_select_model(handle, linmod_model_mse);
    da_linmod_d_define_features(handle, n, m, Al, bl);
    if (da_options_set_int(handle, "linmod intercept", 0) != da_status_success) {
        std::cout << "Something wrong happened setting an option. Terminating"
                  << std::endl;
        return 1;
    }
    if (da_options_set_d_real(handle, "linmod norm2 reg", 5) != da_status_success) {
        std::cout << "Something wrong happened setting an option. Terminating"
                  << std::endl;
        return 1;
    }
    if (da_options_set_string(handle, "print options", "yes") != da_status_success) {
        std::cout << "Something wrong happened setting an option. Terminating"
                  << std::endl;
        return 1;
    }
    if (da_options_set_string(handle, "linmod optim method", "lbfgs") !=
        da_status_success) {
        std::cout << "Something wrong happened setting an option. Terminating"
                  << std::endl;
        return 1;
    }
    // compute regression
    status = da_linmod_d_fit(handle);
    if (status == da_status_success) {
        std::cout << "regression computed successfully!" << std::endl;
        da_linmod_d_get_coef(handle, &nx, x);
        if (nx == 3)
            std::cout << "Coefficients: (intercept) " << x[2] << " p1: " << x[0]
                      << " p2: " << x[1] << std::endl;
        else
            std::cout << "Coefficients: " << x[0] << " " << x[1] << std::endl;
        std::cout << "(Expected        : " << 0.199256 << " " << 0.130354 << ")"
                  << std::endl;
        // octave K=5; x = (A'*A + K * eye(2)) \ A'*b
        std::cout << "(Expected Reg L2 = 5.0: " << 0.185375 << " " << 0.12508 << ")"
                  << std::endl;
    } else {
        std::cout << "Something wrong happened during MSE regression. Terminating"
                  << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

#if 0
    // Solve the same model with single precision
    // problem data
    float As[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    float bs[5] = {1, 1, 1, 1, 1};
    float xs[2];

    std::cout.precision(2);
    // Initialize the linear regression
    da_handle handle_s = nullptr;
    da_handle_init_s(&handle_s, da_handle_linmod);
    da_linmod_s_select_model(handle_s, linmod_model_mse);
    da_linmod_s_define_features(handle_s, n, m, As, bs);
    // compute regression
    status = da_linmod_s_fit(handle_s);
    if (status == da_status_success) {
        std::cout << "regression computed successfully!" << std::endl;
        status = da_linmod_s_get_coef(handle_s, &nx, xs);
        std::cout << "Coefficients: " << xs[0] << " " << xs[1] << std::endl;
        std::cout << "(Expected   : " << 0.20 << " " << 0.13 << ")" << std::endl;
    } else {
        std::cout << "Something wrong happened during MSE regression. Terminating"
                  << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    da_handle_destroy(&handle_s);
#endif
    da_handle_destroy(&handle);
    return 0;
}