/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iomanip>
#include <iostream>
#include <vector>

/*
 * Cubic Spline Interpolation Example
 *
 * This example demonstrates how to use the AOCL-DA library to perform cubic spline
 * interpolation on a simple set of 1D data points.
 */

int main() {

    // Initialize the handle
    da_handle handle = nullptr;
    da_status status;
    bool pass = true;

    // Initialize the interpolation handle
    pass =
        pass && da_handle_init_d(&handle, da_handle_interpolation) == da_status_success;
    pass = pass && da_interpolation_select_model_d(handle, interpolation_cubic_spline) ==
                       da_status_success;
    if (!pass) {
        std::cout << "Model initialization failed!" << std::endl;
        da_handle_print_error_message(handle);
        return 1;
    }

    // Interpolation data: sin(x) on 10 uniformly spread interpolation points in [0,9]
    da_int n_sites = 10;
    double x_start = 0., x_end = 9.;
    pass = pass && da_interpolation_set_sites_uniform_d(handle, n_sites, x_start,
                                                        x_end) == da_status_success;
    double y[10];
    double step = (x_end - x_start) / (n_sites - 1);
    for (da_int i = 0; i < n_sites; i++) {
        y[i] = std::sin(x_start + i * step);
    }
    pass = pass && da_interpolation_set_values_d(handle, n_sites, 1, y, n_sites, 0) ==
                       da_status_success;
    if (!pass) {
        std::cout << "Setting interpolation points failed!" << std::endl;
        da_handle_print_error_message(handle);
        return 1;
    }

    // Compute the model
    status = da_interpolation_interpolate_d(handle);
    if (status != da_status_success) {
        std::cout << "Interpolation failed!" << std::endl;
        da_handle_print_error_message(handle);
        return 1;
    }

    // Evaluate the model on points in [-0.2,9.1] and print the values
    da_int n_eval = 20;
    double x_eval[20] = {-0.2, 2.5, 5.0, 7.5, 2.9, 1.1, 2.2, 3.3, 4.4, 5.5,
                         6.6,  7.7, 8.8, 9.1, 0.5, 3.5, 6.5, 9.5, 2.0, 8.0};
    double y_eval[20];
    da_int orders = 0;
    status = da_interpolation_evaluate_d(handle, n_eval, x_eval, y_eval, 1, &orders);
    if (status != da_status_success) {
        std::cout << "Evaluation failed! " << std::endl;
        da_handle_print_error_message(handle);
        return 1;
    }

    // print the results along with the expected sin(x) values
    std::cout << "Interpolated values versus true sin(x):" << std::endl;
    std::cout << std::fixed;
    for (da_int i = 0; i < n_eval; i++) {
        std::cout.precision(1);
        std::cout << "x = " << std::setw(4) << x_eval[i];
        std::cout.precision(5);
        std::cout << ", y = " << std::setw(8) << y_eval[i] << ", true = " << std::setw(8)
                  << std::sin(x_eval[i]) << std::endl;
    }

    da_handle_destroy(&handle);

    return 0;
}