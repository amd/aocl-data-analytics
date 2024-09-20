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
#include <vector>

/*
 * Basic k-nearest neighbors (kNN) example
 *
 * This example computes k-nearest neighbors classification for a small data matrix.
 */

int main() {
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "k-Nearest Neighbors model (double precision)" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    da_status status;
    bool pass = true;

    // Input data
    da_int n_features = 3;
    da_int n_samples = 6;
    da_int n_queries = 3;
    da_int n_neigh = 3;
    //std::vector<double> X_train{-1., -2., -3., 1., 2., 3.,  -1., -1., -2.,
    //                          3.,  5.,  -1., 2., 3., -1., 1.,  1.,  2.};
    //std::vector<da_int> y_train{1, 2, 0, 1, 2, 2};

    std::vector<double> X_train{-1, -1, 2, -2, -1, 3, -3, -2, -1,
                                1,  3,  1, 2,  5,  1, 3,  -1, 2};
    std::vector<da_int> y_train{1, 2, 0, 1, 2, 2};

    // Set up and train the kNN
    da_handle knn_handle = nullptr;
    pass = da_handle_init_d(&knn_handle, da_handle_knn) == da_status_success;
    da_options_set_string(knn_handle, "storage order", "row-major");
    pass &= da_knn_set_training_data_d(knn_handle, n_samples, n_features, X_train.data(),
                                       n_features, y_train.data()) == da_status_success;
    // Set options
    pass &= da_options_set_int(knn_handle, "number of neighbors", n_neigh) ==
            da_status_success;
    pass &= da_options_set_string(knn_handle, "metric", "euclidean") == da_status_success;
    pass &= da_options_set_string(knn_handle, "weights", "uniform") == da_status_success;

    if (!pass) {
        std::cout << "Something went wrong setting up the knn data and "
                     "optional parameters.\n";
        return 1;
    }

    //    std::vector<double> X_test{-2., -1., 2., 2., -2., 1., 3., -1., -3.};
    std::vector<double> X_test{-2, 2, 3, -1, -2, -1, 2, 1, -3};

    // Compute the k-nearest neighbors and return the distances
    std::vector<double> k_dist(n_neigh * n_queries);
    std::vector<da_int> k_ind(n_neigh * n_queries);
    status = da_knn_kneighbors_d(knn_handle, n_queries, n_features, X_test.data(),
                                 n_queries, k_ind.data(), k_dist.data(), n_neigh, 1);

    if (status != da_status_success) {
        std::cout << "Failure while computing the neighbors\n";
        return 1;
    }
    std::cout << "The indices of neighbors\n";
    for (da_int i = 0; i < n_queries; i++) {
        for (da_int j = 0; j < n_neigh; j++) {
            std::cout << k_ind[i + j * n_queries] << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\n\nThe corresponding distances\n";
    for (da_int i = 0; i < n_queries; i++) {
        for (da_int j = 0; j < n_neigh; j++) {
            std::cout << k_dist[i + j * n_queries] << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    da_int n_classes = 0; // Set n_classes to zero to do query for the required memory
    status = da_knn_classes_d(knn_handle, &n_classes, nullptr);
    std::cout << "\n\nThe number of available classes\n";
    std::cout << n_classes << std::endl;

    // Allocate required memory for classes
    std::vector<da_int> classes(n_classes);
    status = da_knn_classes_d(knn_handle, &n_classes, classes.data());
    std::cout << "\n\nThe corresponding classes, sorted in ascending order\n";
    for (auto &ic : classes)
        std::cout << ic << std::endl;

    // Allocate required memory for classes
    std::vector<double> proba(n_queries * n_classes);
    status = da_knn_predict_proba_d(knn_handle, n_queries, n_features, X_test.data(),
                                    n_queries, proba.data());
    if (status != da_status_success) {
        std::cout << "Failure while computing the probabilities\n";
        return 1;
    }

    std::cout << "\n\nThe probabilities\n";
    for (da_int i = 0; i < n_queries; i++) {
        for (da_int j = 0; j < n_classes; j++) {
            std::cout << proba[i + j * n_queries] << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Allocate memory for predicted labels for test data
    std::vector<da_int> y_test(n_queries);
    status = da_knn_predict_d(knn_handle, n_queries, n_features, X_test.data(), n_queries,
                              y_test.data());
    if (status != da_status_success) {
        std::cout << "Failure while computing the predicted labels\n";
        return 1;
    }
    std::cout << "\n\nThe label estimates\n";
    for (auto &iy : y_test)
        std::cout << iy << std::endl;

    da_handle_destroy(&knn_handle);

    return 0;
}