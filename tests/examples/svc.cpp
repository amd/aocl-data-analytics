/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <vector>

int main() {

    da_handle handle = nullptr;

    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Support Vector Classification (SVC) Example" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << std::fixed;

    int exit_code = 0;
    bool pass = true;

    // A small 8 by 2 training dataset of 2D points in column-major order:
    double X[16] = {-2.99, -0.15, -0.09, 0.45, -1.03, -0.02, 1.59, 0.34,
                    0.04,  2.52,  0.91,  1.12, 0.3,   -0.9,  1.88, -0.15};
    double y[8] = {0, 0, 0, 1, 0, 0, 1, 1};
    // A 5 by 2 dataset in column-major order:
    double X_test[10] = {1.51, 0.83, -1.66, 1.25, -1.01, 1.78, 1.9, 2.89, 1.42, 0.65};
    double y_test[5] = {1, 1, 0, 1, 0};

    da_int n_samples = 8;
    da_int n_samples_test = 5;
    da_int n_features = 2;
    da_int n_class = 2;
    da_int ldx = n_samples;
    da_int ldx_test = n_samples_test;

    // Step 1: Initialize handle as an SVM handle (double precision).
    pass = pass && (da_handle_init_d(&handle, da_handle_svm) == da_status_success);
    pass = pass && (da_svm_select_model_d(handle, svc) == da_status_success);
    if (!pass) {
        std::cout << "Failed to initialize SVC handle" << std::endl;
        return 1;
    }

    // Step 2: Set training data
    pass = pass && (da_svm_set_data_d(handle, n_samples, n_features, X, ldx, y) ==
                    da_status_success);
    if (!pass) {
        std::cout << "Failed to set training data" << std::endl;
        da_handle_destroy(&handle);
        return 1;
    }

    // Step 3: Set relevant options
    pass = pass && (da_options_set_string(handle, "kernel", "rbf") == da_status_success);
    pass = pass && (da_options_set_real_d(handle, "C", 1.0) == da_status_success);
    pass = pass && (da_options_set_real_d(handle, "gamma", 1.0) == da_status_success);

    // Step 4: Fit the model
    pass = pass && (da_svm_compute_d(handle) == da_status_success);
    if (!pass) {
        std::cout << "Model fitting failed" << std::endl;
        da_handle_destroy(&handle);
        return 1;
    }
    std::cout << "SVC: Model fitted successfully.\n" << std::endl;

    // Step 5: Predict on test data
    std::vector<double> predictions(n_samples_test);
    pass = pass && (da_svm_predict_d(handle, n_samples_test, n_features, X_test, ldx_test,
                                     predictions.data()) == da_status_success);
    if (pass) {
        std::cout << "Predictions on test data: " << std::endl;
        for (da_int i = 0; i < n_samples_test; i++) {
            std::cout << predictions[i] << " ";
        }
        std::cout << std::endl << std::endl;
    } else {
        std::cout << "Prediction failed" << std::endl;
        da_handle_destroy(&handle);
        return 1;
    }

    // Step 6: Evaluate model accuracy
    double accuracy = 0.0;
    pass = pass && (da_svm_score_d(handle, n_samples_test, n_features, X_test, ldx_test,
                                   y_test, &accuracy) == da_status_success);
    if (pass) {
        std::cout << "Model accuracy on training set: " << accuracy << std::endl;
    } else {
        std::cout << "Scoring failed" << std::endl;
        exit_code = 1;
    }

    // Step 7: Extract dual coefficients
    da_int n_sv = 0, one = 1;
    da_int n_classifiers = n_class * (n_class - 1) / 2;
    pass = pass && (da_handle_get_result_int(handle, da_svm_n_support_vectors, &one,
                                             &n_sv) == da_status_success);
    da_int size = n_sv * n_classifiers;
    if (pass) {
        std::vector<double> dual_coefficients(size);
        pass = pass &&
               (da_handle_get_result_d(handle, da_svm_dual_coef, &size,
                                       dual_coefficients.data()) == da_status_success);
        if (pass) {
            std::cout << std::endl << "Dual coefficients: " << std::endl;
            for (da_int i = 0; i < size; i++) {
                std::cout << dual_coefficients[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Failed to extract dual coefficients" << std::endl;
            exit_code = 1;
        }
    } else {
        std::cout << "Failed to get number of support vectors" << std::endl;
        exit_code = 1;
    }

    // Step 8: Clean up
    da_handle_destroy(&handle);

    if (pass) {
        std::cout << "\nSVC example completed successfully." << std::endl;
    } else {
        exit_code = 1;
        std::cout << "\nSome SVC operations failed. Check logs above." << std::endl;
    }

    return exit_code;
}