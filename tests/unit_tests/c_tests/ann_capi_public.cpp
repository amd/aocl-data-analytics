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
#include "gtest/gtest.h"

/*
 * Test the approximate nearest neighbors C API (double precision).
 * Based on tests/examples/ann.cpp
 */
TEST(AnnCAPI, BasicDouble) {
    da_handle handle = nullptr;

    // Training data: 16 samples, 2 features (column-major)
    double X_train[32] = {0.0, 1.1,  0.0,  1.0,  6.0,  7.2,  6.1,  7.0,  0.0,  1.0, 0.1,
                          1.1, 10.0, 11.1, 10.0, 11.0, -0.1, 0.0,  1.1,  1.0,  0.0, 0.1,
                          1.0, 1.1,  10.0, 10.2, 11.0, 11.1, 10.0, 10.0, 11.2, 11.0};
    da_int n_samples = 16, n_features = 2, ldx_train = 16;

    // Query points: 3 queries, 2 features
    double X_test[6] = {3.5, 0.4, 5.6, 0.4, 5.0, 5.1};
    da_int n_queries = 3, ldx_test = 3, k = 3;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_approx_nn), da_status_success);

    // Set options
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "ivfflat"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "sqeuclidean"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", k), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_list", 4), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_probe", 1), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "k-means_iter", 10), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 123), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle, "train fraction", 1.0), da_status_success);

    // Set training data
    EXPECT_EQ(da_approx_nn_set_training_data_d(handle, n_samples, n_features, X_train,
                                               ldx_train),
              da_status_success);

    // Train
    EXPECT_EQ(da_approx_nn_train_d(handle), da_status_success);

    // Add data to index
    EXPECT_EQ(da_approx_nn_add_d(handle, n_samples, n_features, X_train, ldx_train),
              da_status_success);

    // Query k-nearest neighbors
    da_int k_ind[9];
    double k_dist[9];
    EXPECT_EQ(da_approx_nn_kneighbors_d(handle, n_queries, n_features, X_test, ldx_test,
                                        k_ind, k_dist, k, 1),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test train_and_add combined (double precision).
 */
TEST(AnnCAPI, TrainAndAddDouble) {
    da_handle handle = nullptr;

    double X_train[32] = {0.0, 1.1,  0.0,  1.0,  6.0,  7.2,  6.1,  7.0,  0.0,  1.0, 0.1,
                          1.1, 10.0, 11.1, 10.0, 11.0, -0.1, 0.0,  1.1,  1.0,  0.0, 0.1,
                          1.0, 1.1,  10.0, 10.2, 11.0, 11.1, 10.0, 10.0, 11.2, 11.0};
    da_int n_samples = 16, n_features = 2, ldx_train = 16;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_approx_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "ivfflat"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "sqeuclidean"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", 3), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_list", 4), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 123), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle, "train fraction", 1.0), da_status_success);

    EXPECT_EQ(da_approx_nn_set_training_data_d(handle, n_samples, n_features, X_train,
                                               ldx_train),
              da_status_success);

    // Train and add combined
    EXPECT_EQ(da_approx_nn_train_and_add_d(handle), da_status_success);

    // Query
    double X_test[2] = {3.5, 0.4};
    da_int k_ind[3];
    double k_dist[3];
    EXPECT_EQ(
        da_approx_nn_kneighbors_d(handle, 1, n_features, X_test, 1, k_ind, k_dist, 3, 1),
        da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test the approximate nearest neighbors C API (single precision).
 */
TEST(AnnCAPI, BasicFloat) {
    da_handle handle = nullptr;

    float X_train[32] = {0.0f,  1.1f,  0.0f,  1.0f,  6.0f,  7.2f,  6.1f,  7.0f,
                         0.0f,  1.0f,  0.1f,  1.1f,  10.0f, 11.1f, 10.0f, 11.0f,
                         -0.1f, 0.0f,  1.1f,  1.0f,  0.0f,  0.1f,  1.0f,  1.1f,
                         10.0f, 10.2f, 11.0f, 11.1f, 10.0f, 10.0f, 11.2f, 11.0f};
    da_int n_samples = 16, n_features = 2, ldx_train = 16;

    float X_test[6] = {3.5f, 0.4f, 5.6f, 0.4f, 5.0f, 5.1f};
    da_int n_queries = 3, ldx_test = 3, k = 3;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_approx_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "ivfflat"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "sqeuclidean"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", k), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_list", 4), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_probe", 1), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "k-means_iter", 10), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 123), da_status_success);
    EXPECT_EQ(da_options_set_real_s(handle, "train fraction", 1.0f), da_status_success);

    EXPECT_EQ(da_approx_nn_set_training_data_s(handle, n_samples, n_features, X_train,
                                               ldx_train),
              da_status_success);
    EXPECT_EQ(da_approx_nn_train_s(handle), da_status_success);
    EXPECT_EQ(da_approx_nn_add_s(handle, n_samples, n_features, X_train, ldx_train),
              da_status_success);

    da_int k_ind[9];
    float k_dist[9];
    EXPECT_EQ(da_approx_nn_kneighbors_s(handle, n_queries, n_features, X_test, ldx_test,
                                        k_ind, k_dist, k, 1),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test train_and_add combined (single precision).
 */
TEST(AnnCAPI, TrainAndAddFloat) {
    da_handle handle = nullptr;

    float X_train[32] = {0.0f,  1.1f,  0.0f,  1.0f,  6.0f,  7.2f,  6.1f,  7.0f,
                         0.0f,  1.0f,  0.1f,  1.1f,  10.0f, 11.1f, 10.0f, 11.0f,
                         -0.1f, 0.0f,  1.1f,  1.0f,  0.0f,  0.1f,  1.0f,  1.1f,
                         10.0f, 10.2f, 11.0f, 11.1f, 10.0f, 10.0f, 11.2f, 11.0f};
    da_int n_samples = 16, n_features = 2, ldx_train = 16;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_approx_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "ivfflat"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "sqeuclidean"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", 3), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_list", 4), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 123), da_status_success);
    EXPECT_EQ(da_options_set_real_s(handle, "train fraction", 1.0f), da_status_success);

    EXPECT_EQ(da_approx_nn_set_training_data_s(handle, n_samples, n_features, X_train,
                                               ldx_train),
              da_status_success);
    EXPECT_EQ(da_approx_nn_train_and_add_s(handle), da_status_success);

    float X_test[2] = {3.5f, 0.4f};
    da_int k_ind[3];
    float k_dist[3];
    EXPECT_EQ(
        da_approx_nn_kneighbors_s(handle, 1, n_features, X_test, 1, k_ind, k_dist, 3, 1),
        da_status_success);

    da_handle_destroy(&handle);
}
