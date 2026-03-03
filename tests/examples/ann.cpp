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
#include <iomanip>
#include <iostream>
#include <vector>

/*
 * Approximate Nearest Neighbors example
 *
 * This example demonstrates the approximate nearest neighbors workflow
 * for efficient similarity search.
 */

int main() {
    std::cout << "---------------------------------------------------------------"
              << std::endl;
    std::cout << "Approximate Nearest Neighbors (double precision)" << std::endl;
    std::cout << "---------------------------------------------------------------"
              << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    da_status status;
    bool pass = true;

    // Define problem dimensions
    da_int n_samples = 16;
    da_int n_features = 2;
    da_int n_queries = 3;
    da_int k = 3; // Number of neighbors to find

    // Training data (column-major order)
    // 4 clusters of 4 points each
    std::vector<double> X_train{0.0,  1.1,  0.0,  1.0,  6.0,  7.2,  6.1,  7.0,
                                0.0,  1.0,  0.1,  1.1,  10.0, 11.1, 10.0, 11.0,
                                -0.1, 0.0,  1.1,  1.0,  0.0,  0.1,  1.0,  1.1,
                                10.0, 10.2, 11.0, 11.1, 10.0, 10.0, 11.2, 11.0};
    da_int ldx_train = n_samples;

    // Query points
    std::vector<double> X_test{3.5, 0.4, 5.6, 0.4, 5.0, 5.1};
    da_int ldx_test = n_queries;

    // Initialize the handle
    da_handle handle = nullptr;
    pass = da_handle_init_d(&handle, da_handle_approx_nn) == da_status_success;

    // Set options for the approximate nearest neighbors algorithm
    pass &= da_options_set_string(handle, "algorithm", "ivfflat") == da_status_success;
    pass &= da_options_set_string(handle, "metric", "sqeuclidean") == da_status_success;
    pass &= da_options_set_int(handle, "number of neighbors", k) == da_status_success;
    pass &= da_options_set_int(handle, "n_list", 4) == da_status_success;
    pass &= da_options_set_int(handle, "n_probe", 1) == da_status_success;
    pass &= da_options_set_int(handle, "k-means_iter", 10) == da_status_success;
    pass &= da_options_set_int(handle, "seed", 123) == da_status_success;
    pass &= da_options_set_real_d(handle, "train fraction", 1.0) == da_status_success;

    if (!pass) {
        std::cout << "Error setting up handle or options.\n";
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return 1;
    }

    // Set training data
    status = da_approx_nn_set_training_data_d(handle, n_samples, n_features,
                                              X_train.data(), ldx_train);
    if (status != da_status_success) {
        std::cout << "Error setting training data.\n";
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return 1;
    }

    // Train the index
    status = da_approx_nn_train_d(handle);
    if (status != da_status_success) {
        std::cout << "Error training the index.\n";
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return 1;
    }
    std::cout << "Index trained successfully.\n\n";

    // Add data to the index
    status = da_approx_nn_add_d(handle, n_samples, n_features, X_train.data(), ldx_train);
    if (status != da_status_success) {
        std::cout << "Error adding data to the index.\n";
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return 1;
    }
    std::cout << "Data added to the index.\n\n";

    // Allocate output arrays for neighbor indices and distances
    std::vector<da_int> k_ind(n_queries * k);
    std::vector<double> k_dist(n_queries * k);

    // Query the k-nearest neighbors
    status = da_approx_nn_kneighbors_d(handle, n_queries, n_features, X_test.data(),
                                       ldx_test, k_ind.data(), k_dist.data(), k, 1);
    if (status != da_status_success) {
        std::cout << "Error computing k-nearest neighbors.\n";
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return 1;
    }

    // Print neighbor indices
    std::cout << "Approximate nearest neighbor indices:\n";
    for (da_int i = 0; i < n_queries; i++) {
        std::cout << "  Query " << i << ": ";
        for (da_int j = 0; j < k; j++) {
            std::cout << k_ind[i + j * n_queries] << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Extract cluster centroids from the trained index
    da_int n_list = 4;
    da_int centroids_dim = n_list * n_features;
    std::vector<double> centroids(centroids_dim);

    status = da_handle_get_result_d(handle, da_approx_nn_cluster_centroids,
                                    &centroids_dim, centroids.data());
    if (status == da_status_success) {
        std::cout << "Cluster centroids:\n";
        for (da_int i = 0; i < n_list; i++) {
            std::cout << "  Centroid " << i << ": ";
            for (da_int j = 0; j < n_features; j++) {
                std::cout << centroids[i + j * n_list] << "  ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Extract list sizes (number of vectors in each cluster)
    std::vector<da_int> list_sizes(n_list);
    status = da_handle_get_result_int(handle, da_approx_nn_list_sizes, &n_list,
                                      list_sizes.data());
    if (status == da_status_success) {
        std::cout << "Number of vectors per cluster:\n";
        for (da_int i = 0; i < n_list; i++) {
            std::cout << "  Cluster " << i << ": " << list_sizes[i] << " vectors\n";
        }
        std::cout << std::endl;
    }

    // Demonstrate train_and_add (alternative workflow)
    std::cout << "Demonstrating train_and_add workflow...\n";

    // Reset training data
    status = da_approx_nn_set_training_data_d(handle, n_samples, n_features,
                                              X_train.data(), ldx_train);
    if (status != da_status_success) {
        std::cout << "Error resetting training data.\n";
        da_handle_destroy(&handle);
        return 1;
    }

    // Train and add in one step
    status = da_approx_nn_train_and_add_d(handle);
    if (status != da_status_success) {
        std::cout << "Error in train_and_add.\n";
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return 1;
    }
    std::cout << "train_and_add completed successfully.\n\n";

    // Demonstrate the effect of n_probe on search accuracy
    // Lower n_probe = faster but potentially less accurate
    // Higher n_probe = slower but more accurate (recovers exact search when n_probe = n_list)
    std::cout << "Comparing n_probe values (approximate vs exact search):\n\n";

    // Search with n_probe = 1
    status = da_approx_nn_kneighbors_d(handle, n_queries, n_features, X_test.data(),
                                       ldx_test, k_ind.data(), k_dist.data(), k, 1);
    if (status != da_status_success) {
        std::cout << "Error computing k-nearest neighbors.\n";
        da_handle_destroy(&handle);
        return 1;
    }
    std::cout << "n_probe = 1 (approximate):\n";
    std::cout << "  Indices:\n";
    for (da_int i = 0; i < n_queries; i++) {
        std::cout << "    Query " << i << ": ";
        for (da_int j = 0; j < k; j++)
            std::cout << k_ind[i + j * n_queries] << "  ";
        std::cout << std::endl;
    }
    std::cout << "  Distances:\n";
    for (da_int i = 0; i < n_queries; i++) {
        std::cout << "    Query " << i << ": ";
        for (da_int j = 0; j < k; j++)
            std::cout << k_dist[i + j * n_queries] << "  ";
        std::cout << std::endl;
    }

    // Search with n_probe = n_list (exact search over all clusters)
    da_options_set_int(handle, "n_probe", n_list);
    status = da_approx_nn_kneighbors_d(handle, n_queries, n_features, X_test.data(),
                                       ldx_test, k_ind.data(), k_dist.data(), k, 1);
    if (status != da_status_success) {
        std::cout << "Error computing k-nearest neighbors.\n";
        da_handle_destroy(&handle);
        return 1;
    }
    std::cout << "\nn_probe = " << n_list << " (exact, searches all clusters):\n";
    std::cout << "  Indices:\n";
    for (da_int i = 0; i < n_queries; i++) {
        std::cout << "    Query " << i << ": ";
        for (da_int j = 0; j < k; j++)
            std::cout << k_ind[i + j * n_queries] << "  ";
        std::cout << std::endl;
    }
    std::cout << "  Distances:\n";
    for (da_int i = 0; i < n_queries; i++) {
        std::cout << "    Query " << i << ": ";
        for (da_int j = 0; j < k; j++)
            std::cout << k_dist[i + j * n_queries] << "  ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Clean up
    da_handle_destroy(&handle);

    std::cout << "---------------------------------------------------------------"
              << std::endl;

    return 0;
}
