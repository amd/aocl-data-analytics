/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "aoclda.h"
#include <cmath>
#include <iostream>

/* Linear least-squares with ridge term regression example
 *
 * This example fits a small dataset
 * to a gaussian model with ridge regularization
 */

int main(void) {

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Mean squared error model" << std::endl;
    std::cout << "min ||Ax-b||^2 + ridge(x); with A an 5x2 matrix" << std::endl
              << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    double NA = std::numeric_limits<double>::quiet_NaN();
    // Problem data
    da_int m{5}, n{2};
    da_int ldA = 8;
    double A[16]{NA, 1, 2, 3, 4, 5, NA, NA, NA, 1, 3, 5, 1, 1, NA, NA};
    double *Astart = A + 1;
    double b[5]{1, 1, 1, 1, 1};
    double x[2];
    da_int nx = 2;
    double tol = 1.0e-6;

    // Expected solution
    // alpha = 1; lambda = 5; x = (A'*A + lambda * eye(2)) \ A'*b
    double xexp[2]{0.185375, 0.12508};

    // Initialize the linear regression
    da_handle handle = nullptr;
    da_status status;
    if (da_handle_init_d(&handle, da_handle_linmod) != da_status_success) {
        da_handle_print_error_message(handle);
        return 1;
    }
    da_linmod_select_model_d(handle, linmod_model_mse);
    da_linmod_define_features_d(handle, m, n, Astart, ldA, b);
    da_options_set_int(handle, "intercept", 0);
    da_options_set_real_d(handle, "alpha", 0.0);
    da_options_set_real_d(handle, "lambda", 5.0);
    da_options_set_string(handle, "scaling", "none");
    da_options_set_string(handle, "print options", "yes");
    da_options_set_string(handle, "optim method", "lbfgs");

    int exit_code = 0;

    // Compute Linear Ridge Regression
    status = da_linmod_fit_d(handle);
    if (status == da_status_success) {
        std::cout << "Regression computed successfully" << std::endl;
        if (da_handle_get_result_d(handle, da_linmod_coef, &nx, x) != da_status_success) {
            da_handle_print_error_message(handle);
            da_handle_destroy(&handle);
            return 1;
        }
        std::cout << "Coefficients: " << x[0] << " " << x[1] << std::endl;
        std::cout << "Expected    : " << xexp[0] << " " << xexp[1] << std::endl;

        // Check result
        double err = std::max(std::abs(x[0] - xexp[0]), std::abs(x[1] - xexp[1]));
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
