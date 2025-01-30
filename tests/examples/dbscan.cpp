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
#include <iostream>

/*
 * Basic DBSCAN example
 *
 * This example computes DBSCAN clustering for a small data matrix.
 */

int main() {

    // Initialize the handle
    da_handle handle = nullptr;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    std::cout << "Basic DBSCAN" << std::endl;
    std::cout << "DBSCAN clustering for a small data matrix" << std::endl << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    int exit_code = 0;
    bool pass = true;

    // Input data
    double A[20] = {2.0, -1.0, 3.0, 2.0, -3.0, -2.0, -2.0, 1.0, 2.0, -2.0,
                    1.0, -2.0, 2.0, 3.0, -2.0, -1.0, -3.0, 2.0, 2.0, -2.0};

    da_int n_samples = 10, n_features = 2, lda = 10, min_samples = 4;
    double eps = 1.1;

    // Create the handle and pass it the data matrix
    pass = pass && (da_handle_init_d(&handle, da_handle_dbscan) == da_status_success);
    pass = pass && (da_dbscan_set_data_d(handle, n_samples, n_features, A, lda) ==
                    da_status_success);

    // Set options
    pass = pass &&
           (da_options_set_int(handle, "min samples", min_samples) == da_status_success);
    pass = pass && (da_options_set_real_d(handle, "eps", eps) == da_status_success);

    // Compute the clusters
    pass = pass && (da_dbscan_compute_d(handle) == da_status_success);

    // Extract results from the handle
    da_int n_clusters = 0, n_core_samples = 0, dim = 1;

    pass = pass && (da_handle_get_result_int(handle, da_dbscan_n_clusters, &dim,
                                             &n_clusters) == da_status_success);
    pass = pass && (da_handle_get_result_int(handle, da_dbscan_n_core_samples, &dim,
                                             &n_core_samples) == da_status_success);

    da_int *labels = new da_int[n_samples];
    da_int *core_sample_indices = new da_int[n_core_samples];

    pass = pass && (da_handle_get_result_int(handle, da_dbscan_labels, &n_samples,
                                             labels) == da_status_success);
    pass = pass && (da_handle_get_result_int(handle, da_dbscan_core_sample_indices,
                                             &n_core_samples,
                                             core_sample_indices) == da_status_success);

    // Check status (we could do this after every function call)
    if (pass) {
        std::cout << "DBSCAN clustering computed successfully" << std::endl << std::endl;

        std::cout << "Labels:" << std::endl;
        for (da_int i = 0; i < n_samples; i++) {
            std::cout << labels[i] << "  ";
        }
        std::cout << std::endl;

        std::cout << "Core samples:" << std::endl;
        for (da_int i = 0; i < n_core_samples; i++) {
            std::cout << core_sample_indices[i] << "  ";
        }
        std::cout << std::endl;

        // Check against expected results
        da_int labels_exp[10] = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1};
        da_int core_sample_indices_exp[2] = {8, 9};

        bool incorrect_results = false;
        for (da_int i = 0; i < n_samples; i++) {
            if (labels[i] != labels_exp[i]) {
                incorrect_results = true;
                break;
            }
        }

        for (da_int i = 0; i < n_core_samples; i++) {
            if (core_sample_indices[i] != core_sample_indices_exp[i]) {
                incorrect_results = true;
                break;
            }
        }

        if (incorrect_results) {
            std::cout << "The expected solution was not obtained." << std::endl;
            exit_code = 1;
        }
    } else {
        exit_code = 1;
    }

    // Clean up
    da_handle_destroy(&handle);
    delete[] labels;
    delete[] core_sample_indices;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return exit_code;
}
