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
#include <cstdio>
#include <iostream>

/*
 * PCA Model Serialization Example
 *
 * This example demonstrates saving and loading a trained PCA model.
 * The trained model can be saved to disk and later loaded to make
 * predictions without retraining.
 */

int main() {

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    std::cout << "PCA Model Serialization Example" << std::endl;
    std::cout << "Training, saving, and loading a PCA model" << std::endl << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    bool pass = true;

    // Training data: 6 samples with 5 features each
    da_int n_samples = 6, n_features = 5, n_components = 3;
    double X_train[30] = {2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 5.0, 2.0, 8.0,
                          3.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                          4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 4.0, 2.0, 2.0};

    double variance_orig[3];

    // Train and save original model
    {
        std::cout << "Training PCA model..." << std::endl;
        da_handle handle_orig = nullptr;
        pass =
            pass && (da_handle_init_d(&handle_orig, da_handle_pca) == da_status_success);

        pass = pass && (da_pca_set_data_d(handle_orig, n_samples, n_features, X_train,
                                          n_samples) == da_status_success);
        pass = pass && (da_options_set_int(handle_orig, "n_components", n_components) ==
                        da_status_success);
        pass = pass && (da_pca_compute_d(handle_orig) == da_status_success);

        // Get explained variance from original model
        da_int info_size = 3;
        pass = pass && (da_handle_get_result_d(handle_orig, da_pca_variance, &info_size,
                                               variance_orig) == da_status_success);

        std::cout << "Original model trained successfully" << std::endl;
        std::cout << "Explained variance: " << variance_orig[0] << ", "
                  << variance_orig[1] << ", " << variance_orig[2] << std::endl
                  << std::endl;

        // Save the trained model to disk
        std::cout << "Saving model to 'pca_model.bin'..." << std::endl;
        pass = pass &&
               (da_handle_save_model(handle_orig, "pca_model.bin") == da_status_success);

        if (!pass) {
            std::cerr << "Error: Failed to train or save model" << std::endl;
            da_handle_destroy(&handle_orig);
            return 1;
        }

        da_handle_destroy(&handle_orig);
    }

    // Load and verify the saved model
    {
        std::cout << "Loading model from 'pca_model.bin'..." << std::endl;
        da_handle handle_loaded = nullptr;
        pass = pass && (da_handle_load_model(&handle_loaded, "pca_model.bin") ==
                        da_status_success);

        if (!pass) {
            std::cerr << "Error: Failed to load model" << std::endl;
            return 1;
        }

        // Verify the loaded model has the same variance
        double variance_loaded[3];
        da_int info_size = 3;
        pass = pass && (da_handle_get_result_d(handle_loaded, da_pca_variance, &info_size,
                                               variance_loaded) == da_status_success);

        std::cout << "Loaded model variance: " << variance_loaded[0] << ", "
                  << variance_loaded[1] << ", " << variance_loaded[2] << std::endl
                  << std::endl;

        // Verify variance values match
        double tol = 1.0e-14;
        double max_diff = 0.0;
        for (da_int i = 0; i < 3; i++) {
            double diff = std::abs(variance_orig[i] - variance_loaded[i]);
            max_diff = std::max(max_diff, diff);
        }

        // Use the loaded model to transform new data
        double X_test[15] = {7.0, 3.0, 3.0, 4.0, 2.0, 3.0, 2.0, 5.0,
                             2.0, 9.0, 6.0, 4.0, 3.0, 4.0, 1.0};
        da_int n_test = 3;
        double X_transformed[9]; // 3 samples x 3 components
        pass =
            pass && (da_pca_transform_d(handle_loaded, n_test, n_features, X_test, n_test,
                                        X_transformed, n_test) == da_status_success);

        if (!pass || max_diff > tol) {
            std::cerr << "Error: ";
            if (!pass)
                std::cerr << "Failed to get variance or transform test data";
            else
                std::cerr << "Loaded model variance does not match original (max diff: "
                          << max_diff << ")";
            std::cerr << std::endl;
            da_handle_destroy(&handle_loaded);
            return 1;
        }

        std::cout << "Variance verification: PASSED" << std::endl;
        std::cout << std::endl << "Transformed test data:" << std::endl;
        for (da_int i = 0; i < n_test; i++) {
            std::cout << "Sample " << i << ": ";
            for (da_int j = 0; j < n_components; j++) {
                std::cout << X_transformed[i + j * n_test] << "  ";
            }
            std::cout << std::endl;
        }

        da_handle_destroy(&handle_loaded);
    }

    // Clean up created files
    std::remove("pca_model.bin");

    std::cout << std::endl
              << "PCA model serialization example completed successfully" << std::endl;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return 0;
}
