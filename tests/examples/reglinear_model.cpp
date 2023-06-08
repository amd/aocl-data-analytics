#include "aoclda.h"
#include <cmath>
#include <iostream>

/* Linear least-squares with ridge term regression example
 *
 * This exampls fits a small dataset
 * to a gaussian model with ridge regularization
 */

int main(void) {

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Mean squared error model" << std::endl;
    std::cout << "min ||Ax-b||^2 + ridge(x); with A an 5x2 matrix" << std::endl
              << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    // Problem data
    da_int m{5}, n{2};
    double Al[10]{1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bl[5]{1, 1, 1, 1, 1};
    double x[2];
    da_int nx = 2;
    double tol = 1.0e-6;

    // Expected solution
    // alpha = 1; lambda = 10; x = (A'*A + lambda/2 * eye(2)) \ A'*b
    double xexp[2]{0.185375, 0.12508};

    // Initialize the linear regression
    da_handle handle = nullptr;
    da_status status;
    if (da_handle_init_d(&handle, da_handle_linmod) != da_status_success) {
        da_handle_print_error_message(handle);
        return 1;
    }
    da_linmod_d_select_model(handle, linmod_model_mse);
    da_linmod_d_define_features(handle, n, m, Al, bl);
    da_options_set_int(handle, "linmod intercept", 0);
    da_options_set_d_real(handle, "linmod alpha", 0.0);
    da_options_set_d_real(handle, "linmod lambda", 10.0);
    da_options_set_string(handle, "print options", "yes");
    da_options_set_string(handle, "linmod optim method", "lbfgs");

    int exit_code = 0;

    // Compute Linear Ridge Regression
    status = da_linmod_d_fit(handle);
    if (status == da_status_success) {
        std::cout << "Regression computed successfully" << std::endl;
        if (da_linmod_d_get_coef(handle, &nx, x) != da_status_success) {
            da_handle_print_error_message(handle);
            da_handle_destroy(&handle);
            return 1;
        }
        std::cout << "Coefficients: " << x[0] << " " << x[1] << std::endl;
        std::cout << "Expected    : " << xexp[0] << " " << xexp[1] << std::endl;

        // Check result
        double err = std::max((x[0] - xexp[0]), (x[1] - xexp[1]));
        if (err > tol) {
            std::cout << "Solution is not within the expected tolerance: " << err
                      << std::endl;
            exit_code = 1;
        }
    } else {
        da_handle_print_error_message(handle);
        exit_code = 1;
    }
    std::cout << "----------------------------------------" << std::endl;

    da_handle_destroy(&handle);
    return exit_code;
}