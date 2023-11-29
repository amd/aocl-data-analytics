/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

/*
 * Linear model elastic net regresion example
 * using the data set from
 *
 * EFRON, HASTIE, JOHNSTONE, and TIBSHIRANI (2004).
 * Least angle regression (with discussion).
 * Ann. Statist. 32 407–499. MR2060166
 * https://hastie.su.domains/Papers/LARS/data64.txt
 *
 * The "diabetes data set" consists of 441 observations
 * and 10 features, while the model chosen is linear and
 * fitted with both L1 and L2 penalty terms.
 *
 * The example showcases how to use datastore framework to
 * extract data, but it can be directly loaded using
 * dense matrices using e.g., da_read_csv_d API.
 */
#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

int main() {

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Elastic net regression example using diabetes data" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl
              << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    // problem data
    // m: observations, n: features
    da_int m = 442, n = 10;
    da_int rhs_pos = 10;
    std::vector<double> features;
    std::vector<double> rhs;
    std::vector<double> x;
    // n features
    features.resize(m * n);
    rhs.resize(m);
    x.resize(n + 1);
    // initial parameter estimates: n + intercept
    x.assign({0, 0, 700, 200, 100, 80, 170, 0, 300, 0});

    // Reference solution
    std::vector<double> x_ref(n + 1);
    x_ref.assign({0, -76.0416, 510.9010, 234.9119, 0, 0, -170.8971, 0, 450.2841, 1.2482});

    da_status status;

    // load data from file
    da_datastore csv;
    const char filename[](DATA_DIR "/diabetes.csv");
    da_datastore_init(&csv);
    da_datastore_options_set_int(csv, "CSV whitespace delimiter", 1);
    da_datastore_options_set_string(csv, "CSV comment char", "#");
    da_datastore_options_set_int(csv, "CSV use header row", 1);
    status = da_data_load_from_csv(csv, filename);
    if (status != da_status_success) {
        da_datastore_print_error_message(csv);
        return 1;
    }
    da_int nr, nc;
    status = da_data_get_n_rows(csv, &nr);
    status = da_data_get_n_cols(csv, &nc);
    if (nr != 442 || nc != 11) {
        std::cout << "Unexpected size for the loaded data: "
                     "(rows="
                  << nr << ", cols=" << nc << ")" << std::endl;
        return 2;
    }

    // extract the 10 features into a dense matrix
    bool pass = true;
    pass = pass && da_data_select_columns(csv, "features", 0, n - 1) == da_status_success;
    pass = pass && da_data_extract_selection_real_d(csv, "features", features.data(),
                                                    m) == da_status_success;
    // extract response variable
    pass = pass &&
           da_data_select_columns(csv, "response", rhs_pos, rhs_pos) == da_status_success;
    pass = pass && da_data_extract_selection_real_d(csv, "response", rhs.data(), m) ==
                       da_status_success;
    da_datastore_destroy(&csv);
    if (!pass) {
        std::cout
            << "Unexpected error in the feature and response matrices extraction.\n";
        return 1;
    }

    /* Note: The linear model iterative coordinate solver expects data to be
     * of the form:
     * norm(features(:,i)) = 1 for all i.
     * mean(features(:,1)) = 0 for all i.
     * mean(rhs) = 0
     */

    // estimate mean and variance per each feature
    pass = true;
    std::vector<double> means(n), scale(n);
    da_int dof = 0, mode = 0;
    pass = pass && da_variance_d(da_axis_col, m, n, features.data(), m, dof, means.data(),
                                 scale.data()) == da_status_success;
    for (size_t i = 0; i < scale.size(); i++)
        scale[i] = std::sqrt(m * scale[i]); // use variance to scale vector length
    pass = pass && da_standardize_d(da_axis_col, m, n, features.data(), m, dof, mode,
                                    means.data(), scale.data()) == da_status_success;

    double rhs_mean;
    pass = pass &&
           da_mean_d(da_axis_col, m, 1, rhs.data(), m, &rhs_mean) == da_status_success;
    pass = pass && da_standardize_d(da_axis_col, m, 1, rhs.data(), m, dof, mode,
                                    &rhs_mean, nullptr) == da_status_success;
    if (!pass) {
        std::cout << "Unexpected error in the data standardization.\n";
        return 1;
    }

    // initialize the linear regression
    pass = true;
    da_int nx = 0;
    da_handle handle = nullptr;
    pass = pass && da_handle_init_d(&handle, da_handle_linmod) == da_status_success;
    pass =
        pass && da_linmod_select_model_d(handle, linmod_model_mse) == da_status_success;
    pass = pass && da_options_set_real_d(handle, "linmod alpha", 1) == da_status_success;
    pass =
        pass && da_options_set_real_d(handle, "linmod lambda", 88) == da_status_success;
    pass = pass &&
           da_options_set_string(handle, "print options", "yes") == da_status_success;
    pass = pass && da_options_set_int(handle, "linmod intercept", 0) == da_status_success;
    pass = pass && da_options_set_int(handle, "print level", 2) == da_status_success;
    pass = pass && da_options_set_int(handle, "linmod optim iteration limit", 35) ==
                       da_status_success;
    pass = pass && da_options_set_real_d(handle, "linmod optim convergence tol",
                                         1.0e-5) == da_status_success;
    pass = pass && da_options_set_real_d(handle, "linmod optim progress factor", 1.0) ==
                       da_status_success;
    pass = pass && da_linmod_define_features_d(handle, m, n, features.data(),
                                               rhs.data()) == da_status_success;
    if (!pass) {
        std::cout << "Unexpected error in the model definition.\n";
        da_handle_destroy(&handle);
        return 1;
    }

    // compute regression
    status = da_linmod_fit_start_d(handle, n + 1, x.data());
    bool ok = false;
    if (status == da_status_success) {
        std::cout << "Regression computed" << std::endl;
        // Query the amount of coefficient in the model (n+intercept)
        da_handle_get_result_d(handle, da_linmod_coeff, &nx, x.data());
        x.resize(nx);
        da_handle_get_result_d(handle, da_linmod_coeff, &nx, x.data());
        std::cout << "Coefficients: " << std::endl;
        std::cout.precision(3);

        bool oki;
        ok = !ok;
        for (da_int i = 0; i < nx; i++) {
            oki = std::abs(x[i] - x_ref[i]) <= 5.0e-4;
            std::cout << " x[" << std::setw(2) << i << "] = " << std::setw(9) << x[i]
                      << " expecting " << std::setw(9) << x_ref[i]
                      << (oki ? " (OK)" : " [WRONG]") << std::endl;
            ok &= oki;
        }
    } else {
        std::cout << "Unexpected error:" << std::endl;
        da_handle_print_error_message(handle);
    }
    std::cout << "----------------------------------------" << std::endl;

    da_handle_destroy(&handle);

    return ok ? 0 : 7;
}
