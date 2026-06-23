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
#include <assert.h>
#include <iostream>

int main() {
    bool ssc = true;       // set to false to use RSC
    bool intercept = true; // true acc=1.0; false acc=0.89

    std::cout.precision(5);
    // Problem data
    const da_int n_samples{19}, n_features{2}, n_test{3};
    da_int n_class{3}; // number of classes in the data
    // Define feature matrix with leading dimension 19 and 2 features
    // The data represents 3 clusters
    double X[n_samples * n_features] = {
        // 1st col
        0.5112507, -0.8260241, -0.1424458, -0.4039709, 0.3485765, 0.3567505, -0.257416,
        1.432360, 0.9424596, 1.836593, 2.111865, 1.916961, 0.9455847, 1.334214, -2.279365,
        -2.850008, -3.496761, -2.906845, -2.386532,
        // 2nd col
        -0.7558991, -0.1399068, -0.9634618, 0.4722355, -0.2858268, -0.4037604, 0.162241,
        1.123676, 1.1810599, 1.327981, 1.368268, 1.735552, 2.7652231, 1.239765, 2.297571,
        2.633976, 1.875603, 2.045181, 1.610652};
    double labels[n_samples] = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};

    double Xtest[n_test * n_features] = {// 1st col
                                         0.52543, -3.01651, -2.46221,
                                         // 2nd col
                                         -0.72594, 2.12328, 1.54248};
    double labels_test[n_test] = {0, 2, 2};
    double predictions[n_samples];

    const da_int ldX = n_samples;
    const da_int ldXtest = n_test;
    da_int n_coef;
    double coef[9]; // reserve enough

    if (!ssc) {
        --n_class; // RSC constraint uses n_class-1 coefficients
    }

    n_coef = n_class * (n_features + intercept);

    // Initialize the linear regression
    da_handle handle = nullptr;
    da_status status;
    bool pass = true;
    pass = pass && da_handle_init_d(&handle, da_handle_linmod) == da_status_success;
    pass = pass &&
           da_linmod_select_model_d(handle, linmod_model_logistic) == da_status_success;
    pass = pass && da_options_set_string(handle, "storage order", "column-major") ==
                       da_status_success;
    pass = pass && da_options_set_string(handle, "logistic constraint",
                                         ssc ? "ssc" : "rsc") == da_status_success;
    pass = pass && da_linmod_define_features_d(handle, n_samples, n_features, X, ldX,
                                               labels) == da_status_success;
    pass = pass && da_options_set_int(handle, "intercept", (intercept ? 1 : 0)) ==
                       da_status_success;
    if (!pass) {
        std::cout << "Something unexpected happened in the model definition\n";
        da_handle_destroy(&handle);
        return 1;
    }

    // Compute regression
    status = da_linmod_fit_d(handle);
    if (status == da_status_success) {
        double accuracy = 0.0;
        status = da_handle_get_result_d(handle, da_linmod_coef, &n_coef, coef);
        assert(status == da_status_success);
        assert(n_coef == n_class * (n_features + intercept));
        status = da_linmod_evaluate_model_d(handle, n_samples, n_features, X, ldX,
                                            predictions, labels, &accuracy);
        std::cout << "Training accuracy: " << accuracy << std::endl;

        // coeffs are always stored in col-major order
        std::cout << "Coefficients:\n";
        for (da_int c = 0; c < n_class; ++c) {
            std::cout << "  Class " << c << ": ";
            for (da_int r = 0; r < n_features; ++r) {
                std::cout << coef[c * (n_features + intercept) + r] << " ";
            }
            if (intercept)
                std::cout << "  intercept:"
                          << coef[c * (n_features + intercept) + n_features] << " ";
            std::cout << std::endl;
        }
        status = da_linmod_evaluate_model_d(handle, n_test, n_features, Xtest, ldXtest,
                                            predictions, nullptr, nullptr);
        assert(status == da_status_success);
        std::cout << "Predictions:\n";
        for (da_int i = 0; i < n_test; ++i) {
            std::cout << "  prediction[" << i << "]: " << predictions[i]
                      << " expecting: " << labels_test[i] << "\n";
        }
        std::cout << std::endl;
    }

    da_handle_destroy(&handle);
    return status == da_status_success ? 0 : 1;
}
