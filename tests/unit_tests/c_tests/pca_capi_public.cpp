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
 * Test the PCA C API (double precision).
 * Based on tests/examples/pca.cpp
 */
TEST(PcaCAPI, BasicDouble) {
    da_handle handle = nullptr;

    // Input data: 6 samples, 5 features (column-major)
    double A[30] = {2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 5.0, 2.0, 8.0,
                    3.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 4.0, 2.0, 2.0};

    da_int n_samples = 6, n_features = 5, n_components = 3, lda = 6;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_pca), da_status_success);
    EXPECT_EQ(da_pca_set_data_d(handle, n_samples, n_features, A, lda),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "PCA method", "covariance"),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "store U", 1), da_status_success);
    EXPECT_EQ(da_pca_compute_d(handle), da_status_success);

    // Get principal components
    da_int pc_dim = n_components * n_features;
    double principal_components[15];
    EXPECT_EQ(da_handle_get_result_d(handle, da_pca_principal_components, &pc_dim,
                                     principal_components),
              da_status_success);

    // Get scores
    da_int scores_dim = n_samples * n_components;
    double scores[18];
    EXPECT_EQ(da_handle_get_result_d(handle, da_pca_scores, &scores_dim, scores),
              da_status_success);

    // Transform new data
    double X[15] = {7.0, 3.0, 3.0, 4.0, 2.0, 3.0, 2.0, 5.0,
                    2.0, 9.0, 6.0, 4.0, 3.0, 4.0, 1.0};
    da_int m_samples = 3, m_features = 5, ldx = 3, ldx_transform = 3;
    double X_transform[9];
    EXPECT_EQ(da_pca_transform_d(handle, m_samples, m_features, X, ldx, X_transform,
                                 ldx_transform),
              da_status_success);

    // Inverse transform
    da_int k_samples = 3, k_features = n_components;
    double Y_inv[15];
    da_int ldy_inv = 3;
    EXPECT_EQ(da_pca_inverse_transform_d(handle, k_samples, k_features, X_transform, 3,
                                         Y_inv, ldy_inv),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test the PCA C API (single precision).
 */
TEST(PcaCAPI, BasicFloat) {
    da_handle handle = nullptr;

    float A[30] = {2.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 5.0f, 2.0f, 8.0f,
                   3.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 4.0f, 2.0f, 2.0f};

    da_int n_samples = 6, n_features = 5, n_components = 3, lda = 6;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_pca), da_status_success);
    EXPECT_EQ(da_pca_set_data_s(handle, n_samples, n_features, A, lda),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "PCA method", "covariance"),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "store U", 1), da_status_success);
    EXPECT_EQ(da_pca_compute_s(handle), da_status_success);

    da_int pc_dim = n_components * n_features;
    float principal_components[15];
    EXPECT_EQ(da_handle_get_result_s(handle, da_pca_principal_components, &pc_dim,
                                     principal_components),
              da_status_success);

    // Transform
    float X[15] = {7.0f, 3.0f, 3.0f, 4.0f, 2.0f, 3.0f, 2.0f, 5.0f,
                   2.0f, 9.0f, 6.0f, 4.0f, 3.0f, 4.0f, 1.0f};
    da_int m_samples = 3, m_features = 5, ldx = 3, ldx_transform = 3;
    float X_transform[9];
    EXPECT_EQ(da_pca_transform_s(handle, m_samples, m_features, X, ldx, X_transform,
                                 ldx_transform),
              da_status_success);

    // Inverse transform
    float Y_inv[15];
    EXPECT_EQ(
        da_pca_inverse_transform_s(handle, 3, n_components, X_transform, 3, Y_inv, 3),
        da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test the Kernel PCA C API (double precision).
 * Based on tests/examples/kernel_pca.cpp
 */
TEST(KernelPcaCAPI, BasicDouble) {
    da_handle handle = nullptr;

    // Small dataset: 6 samples, 3 features
    double A[18] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.1, 1.9, 3.1,
                    4.1, 5.2, 5.8, 0.9, 2.2, 2.9, 4.2, 4.9, 6.2};
    da_int n_samples = 6, n_features = 3, n_components = 2, lda = 6;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kernel_pca), da_status_success);
    EXPECT_EQ(da_kernel_pca_set_data_d(handle, n_samples, n_features, A, lda),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "kernel", "rbf"), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle, "gamma", 0.1), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "fit inverse transform", "yes"),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_compute_d(handle), da_status_success);

    // Transform
    double X[6] = {2.0, 3.0, 2.5, 3.5, 1.5, 2.5};
    da_int m_samples = 2, ldx = 2, ldx_transform = 2;
    double X_transform[4];
    EXPECT_EQ(da_kernel_pca_transform_d(handle, m_samples, n_features, X, ldx,
                                        X_transform, ldx_transform),
              da_status_success);

    // Inverse transform
    double Y_inv[6];
    EXPECT_EQ(da_kernel_pca_inverse_transform_d(handle, m_samples, n_components,
                                                X_transform, ldx, Y_inv, 2),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test the Kernel PCA C API (single precision).
 */
TEST(KernelPcaCAPI, BasicFloat) {
    da_handle handle = nullptr;

    float A[18] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.1f, 1.9f, 3.1f,
                   4.1f, 5.2f, 5.8f, 0.9f, 2.2f, 2.9f, 4.2f, 4.9f, 6.2f};
    da_int n_samples = 6, n_features = 3, n_components = 2, lda = 6;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_kernel_pca), da_status_success);
    EXPECT_EQ(da_kernel_pca_set_data_s(handle, n_samples, n_features, A, lda),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "kernel", "rbf"), da_status_success);
    EXPECT_EQ(da_options_set_real_s(handle, "gamma", 0.1f), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "fit inverse transform", "yes"),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_compute_s(handle), da_status_success);

    float X[6] = {2.0f, 3.0f, 2.5f, 3.5f, 1.5f, 2.5f};
    da_int m_samples = 2, ldx = 2, ldx_transform = 2;
    float X_transform[4];
    EXPECT_EQ(da_kernel_pca_transform_s(handle, m_samples, n_features, X, ldx,
                                        X_transform, ldx_transform),
              da_status_success);

    float Y_inv[6];
    EXPECT_EQ(da_kernel_pca_inverse_transform_s(handle, m_samples, n_components,
                                                X_transform, ldx, Y_inv, 2),
              da_status_success);

    da_handle_destroy(&handle);
}
