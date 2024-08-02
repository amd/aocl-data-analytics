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

    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Nu-Support Vector Regression (NuSVR) Example" << std::endl;
    std::cout << "--------------------------------------------\n" << std::fixed;

    int exit_code = 0;
    bool pass = true;

    // A small 8 by 2 training dataset of 2D points in row-major order:
    double X[16] = {-0.46, -0.47, 0.5,   -0.14, -1.72, -0.56, 0.07, -1.42,
                    -0.91, -1.41, -1.01, 0.31,  1.58,  0.77,  1.47, -0.23};
    double y[8] = {-36.2, 27.76, -114.51, -20.17, -79.45, -56.15, 109.22, 85.06};

    // A 5 by 2 dataset in row-major order:
    double X_test[10] = {0.24, -1.91, -0.54, 0.11, -0.23, -0.23, -0.47, 0.54, 0.65, 1.52};
    double y_test[5] = {-17.84, -31.11, -18.28, -19.27, 65.29};

    da_int n_samples = 8;
    da_int n_samples_test = 5;
    da_int n_features = 2;
    da_int ldx = n_features;
    da_int ldx_test = n_features;

    // Step 1: Initialize handle as an SVM handle (double precision).
    pass = pass && (da_handle_init_d(&handle, da_handle_svm) == da_status_success);
    pass = pass && (da_svm_select_model_d(handle, nusvr) == da_status_success);
    if (!pass) {
        std::cout << "Failed to initialize NuSVR handle" << std::endl;
        return 1;
    }

    // Step 2: Select the NuSVR model and set training data
    pass = pass && (da_options_set_string(handle, "storage order", "row-major") ==
                    da_status_success);
    pass = pass && (da_svm_set_data_d(handle, n_samples, n_features, X, ldx, y) ==
                    da_status_success);
    if (!pass) {
        std::cout << "Failed to set training data" << std::endl;
        da_handle_destroy(&handle);
        return 1;
    }

    // Step 3: Set relevant options for NuSVR
    pass =
        pass && (da_options_set_string(handle, "kernel", "linear") == da_status_success);
    pass = pass && (da_options_set_real_d(handle, "C", 1.0) == da_status_success);
    pass = pass && (da_options_set_real_d(handle, "nu", 0.5) == da_status_success);

    // Step 4: Fit the model
    // da_svm_compute_d will trigger the underlying solve for NuSVR
    pass = pass && (da_svm_compute_d(handle) == da_status_success);
    if (!pass) {
        std::cout << "Model fitting failed" << std::endl;
        da_handle_destroy(&handle);
        return 1;
    }
    std::cout << "NuSVR: Model fitted successfully.\n" << std::endl;

    // Step 5: Predict on test data
    std::vector<double> predictions(n_samples_test, 0.0);
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

    // Step 6: Evaluate the model (e.g., R^2 score for regression)
    double r2_score = 0.0;
    pass = pass && (da_svm_score_d(handle, n_samples_test, n_features, X_test, ldx_test,
                                   y_test, &r2_score) == da_status_success);
    if (pass) {
        std::cout << "Model R^2 score on test set: " << r2_score << std::endl;
    } else {
        std::cout << "Scoring failed" << std::endl;
        exit_code = 1;
    }

    // Step 7: Extract dual coefficients
    da_int n_sv, one = 1;
    pass = pass && (da_handle_get_result_int(handle, da_svm_n_support_vectors, &one,
                                             &n_sv) == da_status_success);
    if (pass) {
        std::vector<double> dual_coefficients(n_sv);
        pass = pass &&
               (da_handle_get_result_d(handle, da_svm_dual_coef, &n_sv,
                                       dual_coefficients.data()) == da_status_success);
        if (pass) {
            std::cout << std::endl << "Dual coefficients: " << std::endl;
            for (da_int i = 0; i < n_sv; i++) {
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
        std::cout << "\nNuSVR example completed successfully." << std::endl;
    } else {
        exit_code = 1;
        std::cout << "\nSome NuSVR operations failed. Check logs above." << std::endl;
    }

    return exit_code;
}