/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <vector>

/*
 * Basic radius neighbors example
 *
 * This example computes the radius neighbors and the corresponding distances for a small data matrix.
 */

int main() {
    std::cout << "----------------------------------------------------------------------"
              << std::endl;
    std::cout << "Nearest Neighbors model for finding radius neighbors (double precision)"
              << std::endl;
    std::cout << "----------------------------------------------------------------------"
              << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    da_status status;
    bool pass = true;
    int exit_code = 0;

    // Input data
    da_int n_features = 3;
    da_int n_samples = 6;
    da_int n_queries = 3;
    double radius = 3.2;

    std::vector<double> X_train{-1, -2, -3, 1, 2, 3,  -1, -1, -2,
                                3,  5,  -1, 2, 3, -1, 1,  1,  2};
    std::vector<da_int> y_train{1, 2, 0, 1, 2, 2};

    // Set up and train the Nearest Neighbors model
    da_handle rnn_handle = nullptr;
    pass = da_handle_init_d(&rnn_handle, da_handle_nn) == da_status_success;
    // Set options
    pass &= da_options_set_string(rnn_handle, "metric", "euclidean") == da_status_success;
    pass &= da_options_set_string(rnn_handle, "algorithm", "auto") == da_status_success;
    if (!pass) {
        std::cout << "Failure while setting up the optional parameters.\n";
        da_handle_print_error_message(rnn_handle);
        return 1;
    }

    status =
        da_nn_set_data_d(rnn_handle, n_samples, n_features, X_train.data(), n_samples);
    if (status != da_status_success) {
        std::cout << "Failure while setting up the radius neighbors data.\n";
        da_handle_print_error_message(rnn_handle);
        return 1;
    }

    std::vector<double> X_test{-2, -1, 2, 2, -2, 1, 3, -1, -3};

    // Compute the radius neighbors for each query point.
    // The neighbors will be stored internally and will be retrieved later.
    da_int return_distances = 1; // 1 to return distances, 0 otherwise
    da_int sort_results = 1;     // 1 to sort results, 0 otherwise
    status = da_nn_radius_neighbors_d(rnn_handle, n_queries, n_features, X_test.data(),
                                      n_queries, radius, return_distances, sort_results);

    if (status != da_status_success) {
        std::cout << "Failure while computing the radius neighbors.\n";
        da_handle_print_error_message(rnn_handle);
        da_handle_destroy(&rnn_handle);
        return 1;
    }

    // Retrieve the number of neighbors for each query point
    da_int n_count = n_queries + 1;
    std::vector<da_int> neighbors_count(n_count);
    status = da_handle_get_result_int(rnn_handle, da_nn_radius_neighbors_count, &n_count,
                                      neighbors_count.data());
    if (status != da_status_success) {
        std::cout << "Failure while retrieving the number of neighbors.\n";
        da_handle_print_error_message(rnn_handle);
        da_handle_destroy(&rnn_handle);
        return 1;
    }
    std::cout << "The number of radius neighbors for each query point:\n";
    for (da_int i = 0; i < n_queries; i++) {
        std::cout << "Query point " << i << " has " << neighbors_count[i]
                  << " neighbors\n";
    }
    da_int total_neighbors = neighbors_count[n_queries];
    std::cout << "Total number of radius neighbors: " << total_neighbors << std::endl;
    std::cout << std::endl;

    // Retrieve the offsets of neighbors for each query point
    std::vector<da_int> neighbors_offsets(n_count);
    status = da_handle_get_result_int(rnn_handle, da_nn_radius_neighbors_offsets,
                                      &n_count, neighbors_offsets.data());
    if (status != da_status_success) {
        std::cout << "Failure while retrieving the offsets of neighbors.\n";
        da_handle_print_error_message(rnn_handle);
        da_handle_destroy(&rnn_handle);
        return 1;
    }
    std::cout << "The neighbors offsets:\n";
    for (da_int i = 0; i < n_queries; i++) {
        if (neighbors_offsets[i] == -1)
            std::cout << "Query point " << i << " has no neighbors\n";
        else
            std::cout << "Query point " << i
                      << " has neighbors at offset: " << neighbors_offsets[i] << "\n";
    }
    std::cout << std::endl;

    // Retrieve the neighbors indices
    std::vector<da_int> neighbors_indices(total_neighbors);
    status = da_handle_get_result_int(rnn_handle, da_nn_radius_neighbors_indices,
                                      &total_neighbors, neighbors_indices.data());
    if (status != da_status_success) {
        std::cout << "Failure while retrieving the neighbors indices.\n";
        da_handle_print_error_message(rnn_handle);
        da_handle_destroy(&rnn_handle);
        return 1;
    }

    // Retrieve the neighbors distances
    std::vector<double> neighbors_distances(total_neighbors);
    status = da_handle_get_result_d(rnn_handle, da_nn_radius_neighbors_distances,
                                    &total_neighbors, neighbors_distances.data());
    if (status != da_status_success) {
        std::cout << "Failure while retrieving the neighbors distances.\n";
        da_handle_print_error_message(rnn_handle);
        da_handle_destroy(&rnn_handle);
        return 1;
    }

    std::cout << "The indices of neighbors\n";
    for (da_int i = 0; i < n_queries; i++) {
        std::cout << "Neighbors for query point " << i << ": ";
        for (da_int j = 0; j < neighbors_count[i]; j++) {
            std::cout << neighbors_indices[neighbors_offsets[i] + j] << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << "\nThe corresponding distances\n";
    for (da_int i = 0; i < n_queries; i++) {
        std::cout << "Distances for query point " << i << ": ";
        for (da_int j = 0; j < neighbors_count[i]; j++) {
            std::cout << neighbors_distances[neighbors_offsets[i] + j] << "  ";
        }
        std::cout << std::endl;
    }

    std::vector<da_int> neighbors_count_exp{1, 2, 0, 3}; // expected result for validation
    std::vector<da_int> neighbors_offsets_exp{0, 1, -1,
                                              3};       // expected result for validation
    std::vector<da_int> neighbors_indices_exp{1, 2, 0}; // expected result for validation
    std::vector<double> neighbors_distances_exp{
        3.00000, 2.00000, 3.16228}; // expected result for validation

    // Validate results
    bool incorrect_results = false;
    for (da_int i = 0; i < n_count; i++) {
        if (neighbors_count[i] != neighbors_count_exp[i] ||
            neighbors_offsets[i] != neighbors_offsets_exp[i]) {
            incorrect_results = true;
            break;
        }
    }

    for (da_int i = 0; i < total_neighbors; i++) {
        if (neighbors_indices[i] != neighbors_indices_exp[i] ||
            std::abs(neighbors_distances[i] - neighbors_distances_exp[i]) > 1e-4) {
            incorrect_results = true;
            break;
        }
    }

    if (incorrect_results) {
        std::cout << "\nThe expected solution was not obtained.\n";
        exit_code = 1;
    }

    da_handle_destroy(&rnn_handle);

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return exit_code;
}