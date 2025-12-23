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
#include <assert.h>
#include <iostream>
#include <limits>

int main() {

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Mean squared error model" << std::endl;
    std::cout << "min ||Ax-b||^2; with A an 5x2 matrix" << std::endl << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    // Problem data
    da_int m = 5, n = 2;
    da_int ldA = 6;
    double NA = std::numeric_limits<double>::quiet_NaN();
    // Define feature matrix with leading dimension 6 and 5 observations
    double Ad[12] = {1, 2, 3, 4, 5, NA, 1, 3, 5, 1, 1, NA};
    double bd[5] = {1, 1, 1, 1, 1};
    da_int nx = 2;
    double x[2];

    // Initialize the linear regression
    da_handle handle = nullptr;
    da_status status;
    bool pass = true;
    pass = pass && da_handle_init_d(&handle, da_handle_linmod) == da_status_success;
    pass =
        pass && da_linmod_select_model_d(handle, linmod_model_mse) == da_status_success;
    pass = pass &&
           da_linmod_define_features_d(handle, m, n, Ad, ldA, bd) == da_status_success;
    if (!pass) {
        std::cout << "Something unexpected happened in the model definition\n";
        da_handle_destroy(&handle);
        return 1;
    }

    // Compute regression
    status = da_linmod_fit_d(handle);
    if (status == da_status_success) {
        std::cout << "regression computed successfully!" << std::endl;
        nx = 0; // Query the correct size
        da_handle_get_result_d(handle, da_linmod_coef, &nx, x);
        assert(nx == 2);
        da_handle_get_result_d(handle, da_linmod_coef, &nx, x);
        std::cout << "Coefficients: " << x[0] << " " << x[1] << std::endl;
        std::cout << "(Expected   : " << 0.199256 << " " << 0.130354 << ")" << std::endl;
    } else {
        std::cout << "Something wrong happened during MSE regression. Terminating"
                  << std::endl;
        da_handle_destroy(&handle);
        return 1;
    }
    std::cout << "----------------------------------------" << std::endl;

    float NAs = std::numeric_limits<float>::quiet_NaN();
    // Solve the same model with single precision
    // Problem data
    ldA = 6;
    float As[12] = {NAs, 1, 2, 3, 4, 5, NAs, 1, 3, 5, 1, 1};
    float bs[5] = {1, 1, 1, 1, 1};
    float xs[2];

    std::cout.precision(2);
    // Initialize the linear regression
    da_handle handle_s = nullptr;
    pass = true;
    pass = pass && da_handle_init_s(&handle_s, da_handle_linmod) == da_status_success;
    pass =
        pass && da_linmod_select_model_s(handle_s, linmod_model_mse) == da_status_success;
    pass = pass && da_linmod_define_features_s(handle_s, m, n, &As[1], ldA, bs) ==
                       da_status_success;
    if (!pass) {
        std::cout << "Something unexpected happened in the model definition\n";
        da_handle_destroy(&handle);
        da_handle_destroy(&handle_s);
        return 1;
    }

    // Compute regression
    status = da_linmod_fit_s(handle_s);
    if (status == da_status_success) {
        std::cout << "regression computed successfully!" << std::endl;
        // status = da_linmod_s_get_coef(handle_s, &nx, xs);
        nx = 0; // Query the correct size
        da_handle_get_result_s(handle_s, da_linmod_coef, &nx, xs);
        assert(nx == 2);
        da_handle_get_result_s(handle_s, da_linmod_coef, &nx, xs);
        std::cout << "Coefficients: " << xs[0] << " " << xs[1] << std::endl;
        std::cout << "(Expected   : " << 0.20 << " " << 0.13 << ")" << std::endl;
    } else {
        std::cout << "Something wrong happened during MSE regression. Terminating"
                  << std::endl;
        da_handle_destroy(&handle);
        da_handle_destroy(&handle_s);
        return 1;
    }
    std::cout << "----------------------------------------" << std::endl;

    da_handle_destroy(&handle);
    da_handle_destroy(&handle_s);

    return 0;
}
