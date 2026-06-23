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
#include <cmath>

/*
 * Test the k-means C API (double precision).
 * Based on tests/examples/kmeans.cpp
 */
TEST(KmeansCAPI, BasicDouble) {
    da_handle handle = nullptr;

    // Input data: 8 samples, 2 features (column-major)
    double A[16] = {2.0, -1.0, 3.0, 2.0, -3.0, -2.0, -2.0, 1.0,
                    1.0, -2.0, 2.0, 3.0, -2.0, -1.0, -3.0, 2.0};
    double C[4] = {1.0, -3.0, 1.0, -3.0};

    da_int n_samples = 8, n_features = 2, n_clusters = 2, lda = 8, ldc = 2;

    // Initialize handle
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kmeans), da_status_success);

    // Set options
    EXPECT_EQ(da_options_set_int(handle, "n_clusters", n_clusters), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_init", 1), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "initialization method", "supplied"),
              da_status_success);

    // Set data
    EXPECT_EQ(da_kmeans_set_data_d(handle, n_samples, n_features, A, lda),
              da_status_success);

    // Set initial cluster centres
    EXPECT_EQ(da_kmeans_set_init_centres_d(handle, C, ldc), da_status_success);

    // Compute
    EXPECT_EQ(da_kmeans_compute_d(handle), da_status_success);

    // Get results
    da_int cluster_centres_dim = n_clusters * n_features;
    da_int labels_dim = n_samples;
    double cluster_centres[4];
    da_int labels[8];

    EXPECT_EQ(da_handle_get_result_d(handle, da_kmeans_cluster_centres,
                                     &cluster_centres_dim, cluster_centres),
              da_status_success);
    EXPECT_EQ(da_handle_get_result_int(handle, da_kmeans_labels, &labels_dim, labels),
              da_status_success);

    // Verify results
    double cluster_centres_exp[4] = {2.0, -2.0, 2.0, -2.0};
    da_int labels_exp[8] = {0, 1, 0, 0, 1, 1, 1, 0};
    double tol = 1.0e-14;
    for (da_int i = 0; i < cluster_centres_dim; i++)
        EXPECT_NEAR(cluster_centres[i], cluster_centres_exp[i], tol);
    for (da_int i = 0; i < labels_dim; i++)
        EXPECT_EQ(labels[i], labels_exp[i]);

    // Transform
    double X[4] = {0.0, 0.0, 1.0, -1.0};
    da_int m_samples = 2, m_features = 2, ldx = 2, ldx_transform = 2;
    double X_transform[4];
    EXPECT_EQ(da_kmeans_transform_d(handle, m_samples, m_features, X, ldx, X_transform,
                                    ldx_transform),
              da_status_success);

    // Predict
    da_int X_labels[2];
    EXPECT_EQ(da_kmeans_predict_d(handle, m_samples, m_features, X, ldx, X_labels),
              da_status_success);

    da_int X_labels_exp[2] = {0, 1};
    for (da_int i = 0; i < m_samples; i++)
        EXPECT_EQ(X_labels[i], X_labels_exp[i]);

    da_handle_destroy(&handle);
}

/*
 * Test the k-means C API (single precision).
 */
TEST(KmeansCAPI, BasicFloat) {
    da_handle handle = nullptr;

    float A[16] = {2.0f, -1.0f, 3.0f, 2.0f, -3.0f, -2.0f, -2.0f, 1.0f,
                   1.0f, -2.0f, 2.0f, 3.0f, -2.0f, -1.0f, -3.0f, 2.0f};
    float C[4] = {1.0f, -3.0f, 1.0f, -3.0f};

    da_int n_samples = 8, n_features = 2, n_clusters = 2, lda = 8, ldc = 2;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_kmeans), da_status_success);

    EXPECT_EQ(da_options_set_int(handle, "n_clusters", n_clusters), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_init", 1), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "initialization method", "supplied"),
              da_status_success);

    EXPECT_EQ(da_kmeans_set_data_s(handle, n_samples, n_features, A, lda),
              da_status_success);
    EXPECT_EQ(da_kmeans_set_init_centres_s(handle, C, ldc), da_status_success);
    EXPECT_EQ(da_kmeans_compute_s(handle), da_status_success);

    da_int cluster_centres_dim = n_clusters * n_features;
    da_int labels_dim = n_samples;
    float cluster_centres[4];
    da_int labels[8];

    EXPECT_EQ(da_handle_get_result_s(handle, da_kmeans_cluster_centres,
                                     &cluster_centres_dim, cluster_centres),
              da_status_success);
    EXPECT_EQ(da_handle_get_result_int(handle, da_kmeans_labels, &labels_dim, labels),
              da_status_success);

    float cluster_centres_exp[4] = {2.0f, -2.0f, 2.0f, -2.0f};
    da_int labels_exp[8] = {0, 1, 0, 0, 1, 1, 1, 0};
    float tol = 1.0e-5f;
    for (da_int i = 0; i < cluster_centres_dim; i++)
        EXPECT_NEAR(cluster_centres[i], cluster_centres_exp[i], tol);
    for (da_int i = 0; i < labels_dim; i++)
        EXPECT_EQ(labels[i], labels_exp[i]);

    // Transform
    float X[4] = {0.0f, 0.0f, 1.0f, -1.0f};
    da_int m_samples = 2, m_features = 2, ldx = 2, ldx_transform = 2;
    float X_transform[4];
    EXPECT_EQ(da_kmeans_transform_s(handle, m_samples, m_features, X, ldx, X_transform,
                                    ldx_transform),
              da_status_success);

    da_int X_labels[2];
    EXPECT_EQ(da_kmeans_predict_s(handle, m_samples, m_features, X, ldx, X_labels),
              da_status_success);

    da_int X_labels_exp[2] = {0, 1};
    for (da_int i = 0; i < m_samples; i++)
        EXPECT_EQ(X_labels[i], X_labels_exp[i]);

    da_handle_destroy(&handle);
}
