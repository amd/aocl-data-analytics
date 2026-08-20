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
 * Explicit tests for aoclda_result.h APIs:
 *   da_handle_get_result_d
 *   da_handle_get_result_s
 *   da_handle_get_result_int
 *
 * Uses a simple k-means model to produce results that can be queried.
 */

TEST(ResultCAPI, GetResultDouble) {
    da_handle handle = nullptr;

    // 8 samples, 2 features (column-major)
    double A[16] = {2.0, -1.0, 3.0, 2.0, -3.0, -2.0, -2.0, 1.0,
                    1.0, -2.0, 2.0, 3.0, -2.0, -1.0, -3.0, 2.0};
    double C[4] = {1.0, -3.0, 1.0, -3.0};
    da_int n_samples = 8, n_features = 2, n_clusters = 2, lda = 8, ldc = 2;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_clusters", n_clusters), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_init", 1), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "initialization method", "supplied"),
              da_status_success);
    EXPECT_EQ(da_kmeans_set_data_d(handle, n_samples, n_features, A, lda),
              da_status_success);
    EXPECT_EQ(da_kmeans_set_init_centres_d(handle, C, ldc), da_status_success);
    EXPECT_EQ(da_kmeans_compute_d(handle), da_status_success);

    // da_handle_get_result_d: retrieve cluster centres
    da_int dim = n_clusters * n_features;
    double centres[4];
    EXPECT_EQ(da_handle_get_result_d(handle, da_kmeans_cluster_centres, &dim, centres),
              da_status_success);
    EXPECT_EQ(dim, n_clusters * n_features);

    // da_handle_get_result_int: retrieve labels
    da_int labels_dim = n_samples;
    da_int labels[8];
    EXPECT_EQ(da_handle_get_result_int(handle, da_kmeans_labels, &labels_dim, labels),
              da_status_success);
    EXPECT_EQ(labels_dim, n_samples);

    // da_handle_get_result_d: unknown query returns error
    da_int rinfo_dim = 100;
    double rinfo[100];
    da_status status = da_handle_get_result_d(handle, da_pca_scores, &rinfo_dim, rinfo);
    EXPECT_NE(status, da_status_success);

    // da_handle_get_result_d: dimension too small returns error and sets correct dim
    da_int small_dim = 1;
    double small_buf[1];
    status =
        da_handle_get_result_d(handle, da_kmeans_cluster_centres, &small_dim, small_buf);
    EXPECT_EQ(status, da_status_invalid_array_dimension);
    EXPECT_EQ(small_dim, n_clusters * n_features);

    da_handle_destroy(&handle);
}

TEST(ResultCAPI, GetResultFloat) {
    da_handle handle = nullptr;

    // 8 samples, 2 features (column-major)
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

    // da_handle_get_result_s: retrieve cluster centres
    da_int dim = n_clusters * n_features;
    float centres[4];
    EXPECT_EQ(da_handle_get_result_s(handle, da_kmeans_cluster_centres, &dim, centres),
              da_status_success);
    EXPECT_EQ(dim, n_clusters * n_features);

    // da_handle_get_result_int: retrieve labels (works with float handle too)
    da_int labels_dim = n_samples;
    da_int labels[8];
    EXPECT_EQ(da_handle_get_result_int(handle, da_kmeans_labels, &labels_dim, labels),
              da_status_success);
    EXPECT_EQ(labels_dim, n_samples);

    // da_handle_get_result_s: wrong precision (call _d on float handle)
    da_int wrong_dim = n_clusters * n_features;
    double wrong_buf[4];
    da_status status =
        da_handle_get_result_d(handle, da_kmeans_cluster_centres, &wrong_dim, wrong_buf);
    EXPECT_EQ(status, da_status_wrong_type);

    // da_handle_get_result_s: dimension too small
    da_int small_dim = 1;
    float small_buf[1];
    status =
        da_handle_get_result_s(handle, da_kmeans_cluster_centres, &small_dim, small_buf);
    EXPECT_EQ(status, da_status_invalid_array_dimension);
    EXPECT_EQ(small_dim, n_clusters * n_features);

    da_handle_destroy(&handle);
}
