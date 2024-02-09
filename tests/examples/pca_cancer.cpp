/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iomanip>
#include <iostream>

/*
 * Principal Component Analysis example
 * using the data set from
 *
 * Wolberg, William, Mangasarian, Olvi, Street, Nick, and Street,W.. (1995). 
 * Breast Cancer Wisconsin (Diagnostic). 
 * UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.
 *
 * The "breast cancer data set" consists of 569 observations
 * and 30 features.
 *
 * The example additionally showcases how to use 
 * da_read_csv_? API to extract data
 */
#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

int main() {

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Principal Component Analysis for breast cancer data" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl
              << std::endl;
    std::cout << std::fixed;

    std::cout.precision(5);

    // Create variables
    int exit_code = 0;
    da_datastore store;
    da_status status;
    const char filename[](DATA_DIR "/breast_cancer.csv");
    da_int n_components = 10;
    da_int n_samples, n_features, lda;
    double *A = 0;
    char **headers;

    // Load data
    da_datastore_init(&store);
    da_datastore_options_set_int(store, "CSV use header row", 0);
    status = da_read_csv_d(store, filename, &A, &n_samples, &n_features, &headers);
    if (status != da_status_success) {
        da_datastore_print_error_message(store);
        return 1;
    }

    // Print the size of the loaded data
    std::cout << "Size of the loaded data: "
                 "(rows="
              << n_samples << ", cols=" << n_features << ")" << std::endl
              << std::endl;

    // Create the handle and pass it the data matrix
    da_handle handle = nullptr;
    lda = n_samples;
    bool pass = true;
    pass = pass && (da_handle_init_d(&handle, da_handle_pca) == da_status_success);
    pass = pass && (da_pca_set_data_d(handle, n_samples, n_features, A, lda) ==
                    da_status_success);

    // Set options
    pass = pass && (da_options_set_string(handle, "PCA method", "covariance") ==
                    da_status_success);
    pass = pass && (da_options_set_int(handle, "n_components", n_components) ==
                    da_status_success);

    // Compute the PCA
    pass = pass && (da_pca_compute_d(handle) == da_status_success);

    // Extract the results from handle
    da_int principal_components_dim = n_features * n_components;
    da_int variance_dim = n_components;
    da_int total_variance_dim = 1;
    double *principal_components = new double[principal_components_dim];
    double *variance = new double[variance_dim];
    double *total_variance = new double[total_variance_dim];

    pass = pass && (da_handle_get_result_d(handle, da_pca_principal_components,
                                           &principal_components_dim,
                                           principal_components) == da_status_success);
    pass = pass && (da_handle_get_result_d(handle, da_pca_variance, &variance_dim,
                                           variance) == da_status_success);
    pass = pass &&
           (da_handle_get_result_d(handle, da_pca_total_variance, &total_variance_dim,
                                   total_variance) == da_status_success);

    // If succesful, print and check the results
    if (pass) {
        // Print principal components
        std::cout << "Principal components:" << std::endl << std::endl;
        for (da_int j = 0; j < n_components; j++) {
            std::cout << std::left << " PC " << std::setw(6) << j + 1;
        }
        std::cout << std::endl;
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_components; i++) {
                std::cout << std::right << std::setw(8)
                          << principal_components[n_components * i + j] << "  ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Print explained variance ratios
        std::cout << "Explained variance ratios:" << std::endl << std::endl;
        for (da_int j = 0; j < n_components; j++) {
            std::cout << std::left << "PC " << std::setw(7) << j + 1;
        }
        std::cout << std::endl;
        for (da_int j = 0; j < n_components; j++) {
            std::cout << std::left << std::setw(10) << variance[j] / total_variance[0];
        }
        std::cout << std::endl;

        // Load expected principal components and explained variance ratios
        const char expected_components_filename[](DATA_DIR "/breast_cancer_exp_comp.csv");
        da_int expected_components_n_samples, expected_components_n_features;
        double *expected_components = 0;
        char **expected_components_headers;

        da_read_csv_d(store, expected_components_filename, &expected_components,
                      &expected_components_n_samples, &expected_components_n_features,
                      &expected_components_headers);

        double expected_explained_variance_ratio[10] = {
            0.9820446715106617708, 0.0161764898635110461, 0.0015575107450152387,
            0.0001209319635401169, 0.0000882724535846217, 0.0000066488395123941,
            0.0000040171368200848, 0.0000008220171966557, 0.0000003441352786163,
            0.0000001860187214777};

        // Check against expected results
        double tol = 1.0e-8;
        double err = 0.0;
        for (da_int i = 0; i < principal_components_dim; i++)
            err =
                std::max(err, std::abs(principal_components[i] - expected_components[i]));
        for (da_int i = 0; i < n_components; i++)
            err = std::max(err, std::abs(variance[i] / total_variance[0] -
                                         expected_explained_variance_ratio[i]));
        if (err > tol) {
            std::cout << "Solution is not within the expected tolerance: " << err
                      << std::endl;
            exit_code = 1;
        }
        free(expected_components);
    } else {
        exit_code = 1;
    }

    // Clean up
    free(A);
    delete[] principal_components;
    delete[] variance;
    delete[] total_variance;
    da_handle_destroy(&handle);
    da_datastore_destroy(&store);

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return exit_code;
}