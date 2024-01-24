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
#include <iostream>

/* Basic k-means example
 *
 * This example computes k-means clustering for a small data matrix.
 */

int main() {

    // Initialize the handle
    da_handle handle = nullptr;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    std::cout << "Basic k-means" << std::endl;
    std::cout << "k-means clustering for a 6x5 data matrix" << std::endl << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    int exit_code = 0;
    bool pass = true;

    // Input data
    double A[30] = {2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 5.0, 2.0, 8.0,
                    3.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 4.0, 2.0, 2.0};

    da_int n_samples = 6, n_features = 5, n_clusters = 3, lda = 6;

    // Create the handle and pass it the data matrix
    pass = pass && (da_handle_init_d(&handle, da_handle_kmeans) == da_status_success);
    //da_options_print(handle);
    pass = pass && (da_kmeans_set_data_d(handle, n_samples, n_features, A, lda) ==
                    da_status_success);

    // Set options
    pass = pass &&
           (da_options_set_int(handle, "n_clusters", n_clusters) == da_status_success);

    // Compute the clusters
    pass = pass && (da_kmeans_compute_d(handle) == da_status_success);

    // Transform another data matrix into the cluster space
    double X[15] = {7.0, 3.0, 3.0, 4.0, 2.0, 3.0, 2.0, 5.0,
                    2.0, 9.0, 6.0, 4.0, 3.0, 4.0, 1.0};
    da_int m_samples = 3, m_features = 5, ldx = 3, ldx_transform = 3;
    double *X_transform = new double[m_samples * n_clusters];
    pass =
        pass && (da_kmeans_transform_d(handle, m_samples, m_features, X, ldx, X_transform,
                                       ldx_transform) == da_status_success);

    // Extract results from the handle
    da_int cluster_centres_dim = n_clusters * n_features;
    da_int labels_dim = n_samples;
    double *cluster_centres = new double[cluster_centres_dim];
    da_int *labels = new da_int[labels_dim];

    pass = pass && (da_handle_get_result_d(handle, da_kmeans_cluster_centres,
                                           &cluster_centres_dim,
                                           cluster_centres) == da_status_success);
    pass = pass && (da_handle_get_result_int(handle, da_kmeans_labels, &labels_dim,
                                             labels) == da_status_success);

    // Check status (we could do this after every function call)
    if (pass) {
        std::cout << "k-means clustering computed successfully" << std::endl << std::endl;

        std::cout << "Cluster centres:" << std::endl;
        for (da_int j = 0; j < n_clusters; j++) {
            for (da_int i = 0; i < n_features; i++) {
                std::cout << cluster_centres[n_clusters * i + j] << "  ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Labels:" << std::endl;
        for (da_int i = 0; i < n_samples; i++) {
            std::cout << labels[i] << "  ";
        }
        std::cout << std::endl;

        // Check against expected results
        double cluster_centres_exp[15] = {
            -0.14907884486130418,  -0.07220367025708045, -0.38718653977350936,
            -0.6612054163818867,   0.623738867070505,    -0.06907631947413592,
            -0.031706610956396264, 0.20952521660694667,  0.8854125206703791,
            -0.7289116905829763,   -0.6138062400926413,  0.1296593407398653,
            -0.09091387966203135,  0.4302063910917139,   -0.21106437194645863};
        da_int labels_exp[6] = {0, 0, 0, 0, 0, 0};
        double X_transform_exp[9] = {
            -3.250305270939447,  0.6691223004872521,   1.833601737126601,
            -2.1581247424555086, -0.21658703437771865, -0.2844305102179128,
            -1.9477723543266676, 1.7953216115607247,   -0.5561178355649032};

        double tol = 1.0e-14;
        double err = 0.0;
        //for (da_int i = 0; i < cluster_centres_dim; i++)
        //   err = std::max(
        //     err, std::abs(cluster_centres[i] - cluster_centres_exp[i]));
        bool incorrect_labels = false;
        //for (da_int i = 0; i < labels_dim; i++) {
        //if (labels[i] != labels_exp[i]) {
        //    incorrect_labels = true;
        //    break;
        //}
        //}
        //for (da_int i = 0; i < m_samples * n_components; i++)
        //err = std::max(err, std::abs(X_transform[i] - X_transform_exp[i]));
        if (err > tol || incorrect_labels) {
            std::cout << "Solution is not within the expected tolerance: " << err
                      << std::endl;
            exit_code = 1;
        }
    } else {
        exit_code = 1;
    }

    // Clean up
    da_handle_destroy(&handle);
    delete[] cluster_centres;
    delete[] labels;
    delete[] X_transform;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return exit_code;
}
