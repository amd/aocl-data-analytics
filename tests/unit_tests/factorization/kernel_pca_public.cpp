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

#include <iostream>
#include <limits>
#include <list>
#include <string>
#include <vector>

#include "aoclda.h"
#include "kernel_pca_test_data.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T> class KernelPCATest : public testing::Test {};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(KernelPCATest, FloatTypes);

TYPED_TEST(KernelPCATest, BadHandleTests) {
    da_handle handle = nullptr;
    TypeParam A = 1;
    TypeParam out = 0;

    // Null handle
    EXPECT_EQ(da_kernel_pca_set_data(handle, 1, 1, &A, 1),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_handle_not_initialized);
    EXPECT_EQ(da_kernel_pca_transform(handle, 1, 1, &A, 1, &out, 1),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_kernel_pca_inverse_transform(handle, 1, 1, &A, 1, &out, 1),
              da_status_handle_not_initialized);

    // Wrong handle type
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);
    EXPECT_EQ(da_kernel_pca_set_data(handle, 1, 1, &A, 1), da_status_invalid_handle_type);
    EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_invalid_handle_type);
    EXPECT_EQ(da_kernel_pca_transform(handle, 1, 1, &A, 1, &out, 1),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_kernel_pca_inverse_transform(handle, 1, 1, &A, 1, &out, 1),
              da_status_invalid_handle_type);
    da_handle_destroy(&handle);
}

TYPED_TEST(KernelPCATest, ErrorExits) {

    // Load the linear_tall base dataset (col-major, unpadded variant)
    std::vector<KernelPCAParamType<TypeParam>> params;
    add_linear_tall(params);
    params.resize(1);

    TypeParam *null_ptr = nullptr;
    da_int nc = params[0].expected_n_components;

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kernel_pca),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", params[0].n_components),
              da_status_success);

    // compute before set_data
    EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_no_data);

    // set_data argument validation
    EXPECT_EQ(
        da_kernel_pca_set_data(handle, 0, params[0].p, params[0].A.data(), params[0].lda),
        da_status_invalid_array_dimension);
    EXPECT_EQ(
        da_kernel_pca_set_data(handle, params[0].n, 0, params[0].A.data(), params[0].lda),
        da_status_invalid_array_dimension);
    EXPECT_EQ(
        da_kernel_pca_set_data(handle, params[0].n, params[0].p, null_ptr, params[0].lda),
        da_status_invalid_pointer);
    EXPECT_EQ(da_kernel_pca_set_data(handle, params[0].n, params[0].p, params[0].A.data(),
                                     params[0].n - 1),
              da_status_invalid_leading_dimension);

    // get_result before compute
    da_int dim = 2;
    TypeParam rinfo[2];
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo), da_status_no_data);

    // Integer kernel PCA results are unavailable before compute
    da_int int_result[1];
    da_int dim_int = 1;
    EXPECT_EQ(da_handle_get_result_int(handle, da_kernel_pca_n_components, &dim_int,
                                       int_result),
              da_status_no_data);

    // Set valid data
    EXPECT_EQ(da_kernel_pca_set_data(handle, params[0].n, params[0].p, params[0].A.data(),
                                     params[0].lda),
              da_status_success);

    // transform and inverse_transform before compute
    std::vector<TypeParam> X_out(params[0].m * nc);
    std::vector<TypeParam> Y_out(params[0].k * params[0].p);
    EXPECT_EQ(da_kernel_pca_transform(handle, params[0].m, params[0].p_transform,
                                      params[0].X_transform_in.data(), params[0].ldx,
                                      X_out.data(), params[0].ldx_transform),
              da_status_no_data);
    EXPECT_EQ(da_kernel_pca_inverse_transform(handle, params[0].k, nc,
                                              params[0].Y_inv.data(), params[0].ldy,
                                              Y_out.data(), params[0].ldy_inv_transform),
              da_status_no_data);

    // Compute without fit_inverse_transform
    EXPECT_EQ(da_options_set_string(handle, "fit inverse transform", "no"),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_success);

    // inverse_transform without fit_inverse_transform=1
    EXPECT_EQ(da_kernel_pca_inverse_transform(handle, params[0].k, nc,
                                              params[0].Y_inv.data(), params[0].ldy,
                                              Y_out.data(), params[0].ldy_inv_transform),
              da_status_no_data);

    // Undersized dim - each query should update dim to the required size
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, 2);

    dim_int = 0;
    EXPECT_EQ(da_handle_get_result_int(handle, da_kernel_pca_n_components, &dim_int,
                                       int_result),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim_int, 1);

    std::vector<TypeParam> evals(nc);
    std::vector<TypeParam> evecs(params[0].n * nc);
    std::vector<TypeParam> scores(params[0].n * nc);
    dim = 0;
    EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvalues, &dim, evals.data()),
              da_status_invalid_array_dimension);
    dim = 0;
    EXPECT_EQ(
        da_handle_get_result(handle, da_kernel_pca_eigenvectors, &dim, evecs.data()),
        da_status_invalid_array_dimension);
    dim = 0;
    EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_scores, &dim, scores.data()),
              da_status_invalid_array_dimension);

    // Query for a PCA-specific result that kernel PCA does not provide
    dim = 10;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_scores, &dim, scores.data()),
              da_status_unknown_query);

    // Null result pointer
    dim = 10;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, null_ptr),
              da_status_invalid_input);

    // transform: wrong number of features
    std::vector<TypeParam> X_wrong(params[0].m * (params[0].p + 1));
    std::vector<TypeParam> X_out_wrong(params[0].m * nc);
    EXPECT_EQ(da_kernel_pca_transform(handle, params[0].m, params[0].p + 1,
                                      X_wrong.data(), params[0].m, X_out_wrong.data(),
                                      params[0].m),
              da_status_invalid_input);

    // Recompute with fit_inverse_transform="yes" for remaining tests
    EXPECT_EQ(da_options_set_string(handle, "fit inverse transform", "yes"),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_success);

    // inverse_transform: wrong nc
    EXPECT_EQ(da_kernel_pca_inverse_transform(handle, params[0].k, nc + 1,
                                              params[0].Y_inv.data(), params[0].ldy,
                                              Y_out.data(), params[0].ldy_inv_transform),
              da_status_invalid_input);

    // kernel param change after compute -> incompatible_options
    EXPECT_EQ(da_options_set_string(handle, "kernel", "rbf"), da_status_success);
    EXPECT_EQ(da_kernel_pca_transform(handle, params[0].m, params[0].p_transform,
                                      params[0].X_transform_in.data(), params[0].ldx,
                                      X_out.data(), params[0].ldx_transform),
              da_status_incompatible_options);
    EXPECT_EQ(da_kernel_pca_inverse_transform(handle, params[0].k, nc,
                                              params[0].Y_inv.data(), params[0].ldy,
                                              Y_out.data(), params[0].ldy_inv_transform),
              da_status_incompatible_options);

    da_handle_destroy(&handle);

    // fit_inverse_transform=1 incompatible with kernel="precomputed"
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kernel_pca),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_set_data(handle, params[0].p, params[0].p, params[0].A.data(),
                                     params[0].n),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "kernel", "precomputed"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "fit inverse transform", "yes"),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_invalid_input);
    da_handle_destroy(&handle);

    // precomputed kernel with n_features != n_samples
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kernel_pca),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_set_data(handle, params[0].n, params[0].p, params[0].A.data(),
                                     params[0].lda),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "kernel", "precomputed"), da_status_success);
    EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_invalid_input);
    da_handle_destroy(&handle);
}

// Verify that calling inverse_transform without fitting returns da_status_no_data,
// and that setting the option and recomputing on the same handle then succeeds with
// correct results.
TYPED_TEST(KernelPCATest, InverseTransformFitLate) {
    std::vector<KernelPCAParamType<TypeParam>> params;
    add_linear_tall(params);
    params.resize(1);
    const KernelPCAParamType<TypeParam> &ds = params[0];

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kernel_pca),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", ds.order.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "copy data", ds.copy_data.c_str()),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_set_data(handle, ds.n, ds.p, ds.A.data(), ds.lda),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "kernel", ds.kernel.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "eigensolver", ds.solver.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", ds.n_components),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "gamma", ds.gamma), da_status_success);
    EXPECT_EQ(da_options_set(handle, "degree", ds.degree), da_status_success);
    EXPECT_EQ(da_options_set(handle, "coef0", ds.coef0), da_status_success);
    EXPECT_EQ(
        da_options_set_string(handle, "remove zero eig", ds.remove_zero_eig.c_str()),
        da_status_success);
    EXPECT_EQ(da_options_set(handle, "alpha", ds.alpha), da_status_success);

    // Compute without inverse transform
    EXPECT_EQ(da_options_set_string(handle, "fit inverse transform", "no"),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_success);

    da_int nc = ds.expected_n_components;
    std::vector<TypeParam> Y_out(ds.k * ds.p);
    EXPECT_EQ(da_kernel_pca_inverse_transform(handle, ds.k, nc, ds.Y_inv.data(), ds.ldy,
                                              Y_out.data(), ds.ldy_inv_transform),
              da_status_no_data);

    // Enable inverse transform and recompute on the same handle
    EXPECT_EQ(da_options_set_string(handle, "fit inverse transform", "yes"),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_success);

    // Verify eigenvalues
    da_int size_evals = nc;
    std::vector<TypeParam> evals(size_evals);
    EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvalues, &size_evals,
                                   evals.data()),
              da_status_success);
    EXPECT_ARR_NEAR(size_evals, evals.data(), ds.expected_eigenvalues.data(), ds.epsilon);

    // Verify scores
    da_int size_scores = ds.n * nc;
    std::vector<TypeParam> scores(size_scores);
    EXPECT_EQ(
        da_handle_get_result(handle, da_kernel_pca_scores, &size_scores, scores.data()),
        da_status_success);
    sign_correct_columns(ds.n, nc, scores, ds.expected_scores, ds.order);
    EXPECT_ARR_NEAR(size_scores, scores.data(), ds.expected_scores.data(), ds.epsilon);

    // Verify inverse transform results
    EXPECT_EQ(da_kernel_pca_inverse_transform(handle, ds.k, nc, ds.Y_inv.data(), ds.ldy,
                                              Y_out.data(), ds.ldy_inv_transform),
              da_status_success);
    da_int size_inv = ds.ldy_inv_transform * ds.p;
    EXPECT_ARR_NEAR(size_inv, Y_out.data(), ds.expected_Y_inv_transform.data(),
                    ds.epsilon);

    da_handle_destroy(&handle);
}

TEST(KernelPCATestPrecision, IncorrectHandlePrecision) {
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_kernel_pca), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_kernel_pca), da_status_success);

    double Ad = 1.0;
    float As = 1.0f;
    double out_d = 0.0;
    float out_s = 0.0f;

    EXPECT_EQ(da_kernel_pca_set_data_d(handle_s, 1, 1, &Ad, 1), da_status_wrong_type);
    EXPECT_EQ(da_kernel_pca_set_data_s(handle_d, 1, 1, &As, 1), da_status_wrong_type);

    EXPECT_EQ(da_kernel_pca_compute_d(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_kernel_pca_compute_s(handle_d), da_status_wrong_type);

    EXPECT_EQ(da_kernel_pca_transform_d(handle_s, 1, 1, &Ad, 1, &out_d, 1),
              da_status_wrong_type);
    EXPECT_EQ(da_kernel_pca_transform_s(handle_d, 1, 1, &As, 1, &out_s, 1),
              da_status_wrong_type);

    EXPECT_EQ(da_kernel_pca_inverse_transform_d(handle_s, 1, 1, &Ad, 1, &out_d, 1),
              da_status_wrong_type);
    EXPECT_EQ(da_kernel_pca_inverse_transform_s(handle_d, 1, 1, &As, 1, &out_s, 1),
              da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

/* ============================================================================
 * Verifies the identity: linear kernel PCA on zero-mean data produces the
 * same scores as standard PCA.
 *
 * For zero-mean X, K = X X^T. Kernel PCA eigendecomposes K directly (centering
 * is a no-op), giving eigenvalues lambda_kpca. Standard PCA (covariance method)
 * eigendecomposes X^T X / (n-1), giving variances lambda_pca.
 * The relationship is lambda_kpca = lambda_pca * (n - 1).
 *
 * Scores from both methods are the same up to column sign.
 * * ============================================================================ */
TYPED_TEST(KernelPCATest, LinearZeroMeanMatchesPCA) {
    std::vector<KernelPCAParamType<TypeParam>> datasets;
    add_linear_zero_mean_colmaj(datasets);
    add_linear_zero_mean_rowmaj(datasets);

    for (const KernelPCAParamType<TypeParam> &ds : datasets) {
        std::cout << "LinearZeroMeanMatchesPCA: " << ds.test_name << std::endl;

        da_int n = ds.n;
        da_int p = ds.p;
        da_int nc = ds.expected_n_components;

        // --- Kernel PCA (linear kernel) ---
        da_handle kpca_handle = nullptr;
        EXPECT_EQ(da_handle_init<TypeParam>(&kpca_handle, da_handle_kernel_pca),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(kpca_handle, "storage order", ds.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_kernel_pca_set_data(kpca_handle, n, p, ds.A.data(), ds.lda),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(kpca_handle, "kernel", "linear"),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(kpca_handle, "eigensolver", "syevd"),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(kpca_handle, "remove zero eig", "yes"),
                  da_status_success);
        EXPECT_EQ(da_kernel_pca_compute<TypeParam>(kpca_handle), da_status_success);

        da_int size_n_components = 1;
        da_int kernel_pca_n_components[1];
        EXPECT_EQ(da_handle_get_result_int(kpca_handle, da_kernel_pca_n_components,
                                           &size_n_components, kernel_pca_n_components),
                  da_status_success);
        EXPECT_EQ(kernel_pca_n_components[0], nc);

        da_int size_evals = nc;
        std::vector<TypeParam> kpca_evals(size_evals);
        EXPECT_EQ(da_handle_get_result(kpca_handle, da_kernel_pca_eigenvalues,
                                       &size_evals, kpca_evals.data()),
                  da_status_success);

        da_int size_scores = n * nc;
        std::vector<TypeParam> kpca_scores(size_scores);
        EXPECT_EQ(da_handle_get_result(kpca_handle, da_kernel_pca_scores, &size_scores,
                                       kpca_scores.data()),
                  da_status_success);

        // --- Standard PCA (covariance method, gesdd solver) ---
        da_handle pca_handle = nullptr;
        EXPECT_EQ(da_handle_init<TypeParam>(&pca_handle, da_handle_pca),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(pca_handle, "storage order", ds.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_pca_set_data(pca_handle, n, p, ds.A.data(), ds.lda),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(pca_handle, "PCA method", "covariance"),
                  da_status_success);
        // gesdd is essential so we can set store U = 1
        EXPECT_EQ(da_options_set_string(pca_handle, "svd solver", "gesdd"),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(pca_handle, "store u", 1), da_status_success);
        EXPECT_EQ(da_options_set_int(pca_handle, "n_components", nc), da_status_success);
        EXPECT_EQ(da_pca_compute<TypeParam>(pca_handle), da_status_success);

        da_int size_variance = nc;
        std::vector<TypeParam> pca_variance(size_variance);
        EXPECT_EQ(da_handle_get_result(pca_handle, da_pca_variance, &size_variance,
                                       pca_variance.data()),
                  da_status_success);

        da_int size_pca_scores = n * nc;
        std::vector<TypeParam> pca_scores(size_pca_scores);
        EXPECT_EQ(da_handle_get_result(pca_handle, da_pca_scores, &size_pca_scores,
                                       pca_scores.data()),
                  da_status_success);

        // --- Compare eigenvalues: kpca_evals[j] == pca_variance[j] * (n - 1) ---
        TypeParam epsilon = ds.epsilon;
        for (da_int j = 0; j < nc; j++) {
            EXPECT_NEAR(kpca_evals[j], pca_variance[j] * static_cast<TypeParam>(n - 1),
                        epsilon)
                << "Eigenvalue mismatch at component " << j;
        }

        // --- Compare scores up to column sign ---
        sign_correct_columns(n, nc, kpca_scores, pca_scores, ds.order);
        EXPECT_ARR_NEAR(size_scores, kpca_scores.data(), pca_scores.data(), epsilon);

        da_handle_destroy(&kpca_handle);
        da_handle_destroy(&pca_handle);
    }
}

// HandleReuse: verify a single handle can be reused across multiple
// compute -> transform -> inverse_transform cycles, and that results match expected
TYPED_TEST(KernelPCATest, HandleReuse) {
    std::vector<KernelPCAParamType<TypeParam>> params;
    add_linear_tall(params);
    add_rbf_wide(params);

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kernel_pca),
              da_status_success);

    for (const KernelPCAParamType<TypeParam> &ds : params) {
        std::cout << "HandleReuse: " << ds.test_name << std::endl;

        // Set all options from the data
        EXPECT_EQ(da_options_set_string(handle, "storage order", ds.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_kernel_pca_set_data(handle, ds.n, ds.p, ds.A.data(), ds.lda),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "kernel", ds.kernel.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "eigensolver", ds.solver.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_components", ds.n_components),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "gamma", ds.gamma), da_status_success);
        EXPECT_EQ(da_options_set(handle, "degree", ds.degree), da_status_success);
        EXPECT_EQ(da_options_set(handle, "coef0", ds.coef0), da_status_success);
        EXPECT_EQ(
            da_options_set_string(handle, "remove zero eig", ds.remove_zero_eig.c_str()),
            da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "fit inverse transform",
                                        ds.fit_inverse_transform.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "alpha", ds.alpha), da_status_success);

        // First compute ->scores ->verify vs expected
        EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_success);

        da_int computed_n_components = 0;
        da_int size_n_components = 1;
        EXPECT_EQ(da_handle_get_result_int(handle, da_kernel_pca_n_components,
                                           &size_n_components, &computed_n_components),
                  da_status_success);
        EXPECT_EQ(computed_n_components, ds.expected_n_components);

        da_int nc = computed_n_components;
        da_int size_scores = ds.n * nc;

        std::vector<TypeParam> scores1(size_scores);
        EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_scores, &size_scores,
                                       scores1.data()),
                  da_status_success);
        sign_correct_columns(ds.n, nc, scores1, ds.expected_scores, ds.order);
        EXPECT_ARR_NEAR(size_scores, scores1.data(), ds.expected_scores.data(),
                        ds.epsilon);

        // Second compute ->scores ->verify vs expected
        EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_success);

        std::vector<TypeParam> scores2(size_scores);
        da_int size_scores2 = size_scores;
        EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_scores, &size_scores2,
                                       scores2.data()),
                  da_status_success);
        sign_correct_columns(ds.n, nc, scores2, ds.expected_scores, ds.order);
        EXPECT_ARR_NEAR(size_scores, scores2.data(), ds.expected_scores.data(),
                        ds.epsilon);

        // First transform ->verify vs expected
        da_int size_transform = (ds.order == "column-major") ? ds.ldx_transform * nc
                                                             : ds.ldx_transform * ds.m;
        std::vector<TypeParam> transform1(size_transform);
        EXPECT_EQ(da_kernel_pca_transform(handle, ds.m, ds.p_transform,
                                          ds.X_transform_in.data(), ds.ldx,
                                          transform1.data(), ds.ldx_transform),
                  da_status_success);
        sign_correct_columns(ds.m, nc, transform1, ds.expected_X_transform, ds.order,
                             ds.ldx_transform);
        EXPECT_ARR_NEAR(size_transform, transform1.data(), ds.expected_X_transform.data(),
                        ds.epsilon);

        // Second transform ->verify vs expected
        std::vector<TypeParam> transform2(size_transform);
        EXPECT_EQ(da_kernel_pca_transform(handle, ds.m, ds.p_transform,
                                          ds.X_transform_in.data(), ds.ldx,
                                          transform2.data(), ds.ldx_transform),
                  da_status_success);
        sign_correct_columns(ds.m, nc, transform2, ds.expected_X_transform, ds.order,
                             ds.ldx_transform);
        EXPECT_ARR_NEAR(size_transform, transform2.data(), ds.expected_X_transform.data(),
                        ds.epsilon);

        // First inverse transform ->verify vs expected
        da_int size_inv = (ds.order == "column-major") ? ds.ldy_inv_transform * ds.p
                                                       : ds.ldy_inv_transform * ds.k;
        std::vector<TypeParam> inv1(size_inv);
        EXPECT_EQ(da_kernel_pca_inverse_transform(handle, ds.k, nc, ds.Y_inv.data(),
                                                  ds.ldy, inv1.data(),
                                                  ds.ldy_inv_transform),
                  da_status_success);
        EXPECT_ARR_NEAR(size_inv, inv1.data(), ds.expected_Y_inv_transform.data(),
                        ds.epsilon);

        // Second inverse transform ->verify vs expected
        std::vector<TypeParam> inv2(size_inv);
        EXPECT_EQ(da_kernel_pca_inverse_transform(handle, ds.k, nc, ds.Y_inv.data(),
                                                  ds.ldy, inv2.data(),
                                                  ds.ldy_inv_transform),
                  da_status_success);
        EXPECT_ARR_NEAR(size_inv, inv2.data(), ds.expected_Y_inv_transform.data(),
                        ds.epsilon);
    }

    da_handle_destroy(&handle);
}

// Check that a precomputed kernel with many negative evals exits
// appropriately
TYPED_TEST(KernelPCATest, NegativeEigenvalueExit) {

    TypeParam A[16] = {0.0, -0.5, 1.5, -1.0, -0.5, 0.0, 1.0,  1.5,
                       1.5, -1.0, 0.0, -0.5, -1.0, 1.5, -0.5, 0.0};
    da_int n = 4;
    da_int nc = 3;
    da_int lda = 4;

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kernel_pca),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", "column-major"),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_set_data(handle, n, n, A, lda), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "kernel", "precomputed"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "eigensolver", "syevd"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", nc), da_status_success);

    EXPECT_EQ(da_kernel_pca_compute<TypeParam>(handle), da_status_numerical_difficulties);

    // transform and inverse_transform should fail with da_status_no_data
    TypeParam X[4] = {1.0, 1.0, 1.0, 1.0};
    TypeParam X_out[3] = {0.0, 0.0, 0.0};
    EXPECT_EQ(da_kernel_pca_transform(handle, 1, n, X, 1, X_out, 1), da_status_no_data);
    EXPECT_EQ(da_kernel_pca_inverse_transform(handle, 1, nc, X_out, 1, X, 1),
              da_status_no_data);

    // all get_result calls should fail with da_status_no_data
    TypeParam result[12];
    da_int size = 4;

    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size, result), da_status_no_data);
    // Reset size just to be sure
    size = 12;
    EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvectors, &size, result),
              da_status_no_data);
    EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_scores, &size, result),
              da_status_no_data);
    EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvalues, &size, result),
              da_status_no_data);

    da_handle_destroy(&handle);
}

// Helper: run syevd (reference) and randomized eigensolver on the same dataset and
// compare eigenvalues and eigenvectors. Mirrors check_randomized_vs_gesdd in pca_public.cpp.
template <typename T> void check_randomized_vs_syevd(const KernelPCAParamType<T> &param) {

    da_int n = param.n;
    da_int nc = 2;
    da_int size_evals = nc;
    da_int size_evecs = n * nc;

    // --- syevd reference ---
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_kernel_pca), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "kernel", param.kernel.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "eigensolver", "syevd"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", nc), da_status_success);
    EXPECT_EQ(da_options_set(handle, "gamma", param.gamma), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "degree", param.degree), da_status_success);
    EXPECT_EQ(da_options_set(handle, "coef0", param.coef0), da_status_success);
    EXPECT_EQ(da_kernel_pca_set_data(handle, n, param.p, param.A.data(), param.lda),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_compute<T>(handle), da_status_success);

    std::vector<T> ref_evals(size_evals), ref_evecs(size_evecs);
    EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvalues, &size_evals,
                                   ref_evals.data()),
              da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvectors, &size_evecs,
                                   ref_evecs.data()),
              da_status_success);
    da_handle_destroy(&handle);

    // --- randomized solver ---
    std::vector<std::string> normalization = {"qr", "lu", "none"};
    da_int q;
    T tol;
    for (const auto &norm : normalization) {
        std::cout << "RandomizedSolver test: " << param.test_name << std::endl;
        std::cout << "Normalization: " << norm << std::endl;
        if (norm == "none") {
            // no norm causes numerical instability
            // don't do too many its and just check results are sensible
            q = 2;
            tol = 1e-3;
        } else {
            // otherwise set q to a large value so results match closely to syevd
            q = 10;
            tol = 1e3 * std::numeric_limits<T>::epsilon();
        }

        EXPECT_EQ(da_handle_init<T>(&handle, da_handle_kernel_pca), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "kernel", param.kernel.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "eigensolver", "randomized"),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_components", nc), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_oversamples", 3), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "power iterations", q), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "power normalization", norm.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
        EXPECT_EQ(da_options_set(handle, "gamma", param.gamma), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "degree", param.degree), da_status_success);
        EXPECT_EQ(da_options_set(handle, "coef0", param.coef0), da_status_success);
        EXPECT_EQ(da_kernel_pca_set_data(handle, n, param.p, param.A.data(), param.lda),
                  da_status_success);
        EXPECT_EQ(da_kernel_pca_compute<T>(handle), da_status_success);

        std::vector<T> rand_evals(size_evals), rand_evecs(size_evecs);
        EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvalues, &size_evals,
                                       rand_evals.data()),
                  da_status_success);
        EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvectors, &size_evecs,
                                       rand_evecs.data()),
                  da_status_success);
        da_handle_destroy(&handle);

        // Eigenvalues: direct comparison (no sign ambiguity)
        EXPECT_ARR_NEAR(size_evals, ref_evals.data(), rand_evals.data(), tol);

        // Eigenvectors: compare absolute values (sign per column is ambiguous)
        for (da_int i = 0; i < size_evecs; ++i) {
            ref_evecs[i] = std::abs(ref_evecs[i]);
            rand_evecs[i] = std::abs(rand_evecs[i]);
        }
        EXPECT_ARR_NEAR(size_evecs, ref_evecs.data(), rand_evecs.data(), tol);
    }
}

TYPED_TEST(KernelPCATest, RandomizedSolver) {
    std::vector<KernelPCAParamType<TypeParam>> params;
    add_linear_tall(params);
    add_linear_wide(params);
    add_rbf_tall(params);
    add_rbf_wide(params);
    add_poly_tall(params);
    add_poly_wide(params);
    add_sigmoid_tall(params);
    add_sigmoid_wide(params);

    for (const KernelPCAParamType<TypeParam> &param : params) {
        check_randomized_vs_syevd(param);
    }
}

template <typename T> void test_functionality(const KernelPCAParamType<T> &param) {
    std::cout << "Functionality test: " << param.test_name << std::endl;

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_kernel_pca), da_status_success);

    EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "copy data", param.copy_data.c_str()),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_set_data(handle, param.n, param.p, param.A.data(), param.lda),
              da_status_success);

    // Set kernel options
    EXPECT_EQ(da_options_set_string(handle, "kernel", param.kernel.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "eigensolver", param.solver.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", param.n_components),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "gamma", param.gamma), da_status_success);
    EXPECT_EQ(da_options_set(handle, "degree", param.degree), da_status_success);
    EXPECT_EQ(da_options_set(handle, "coef0", param.coef0), da_status_success);
    EXPECT_EQ(
        da_options_set_string(handle, "remove zero eig", param.remove_zero_eig.c_str()),
        da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "fit inverse transform",
                                    param.fit_inverse_transform.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "alpha", param.alpha), da_status_success);

    // Compute
    EXPECT_EQ(da_kernel_pca_compute<T>(handle), da_status_success);

    da_int n = param.n;
    da_int nc = 0;
    da_int size_n_components = 1;
    EXPECT_EQ(da_handle_get_result_int(handle, da_kernel_pca_n_components,
                                       &size_n_components, &nc),
              da_status_success);
    EXPECT_EQ(nc, param.expected_n_components);

    // Validate get_result outputs
    if (param.expected_rinfo.size() > 0) {
        da_int size_rinfo = 2;
        std::vector<T> rinfo(size_rinfo);
        EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size_rinfo, rinfo.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(size_rinfo, rinfo.data(), param.expected_rinfo.data(),
                        param.epsilon);
    }

    // Validate gamma
    {
        da_int size_gamma = 1;
        T gamma_result;
        EXPECT_EQ(
            da_handle_get_result(handle, da_kernel_pca_gamma, &size_gamma, &gamma_result),
            da_status_success);
        EXPECT_NEAR(gamma_result, param.expected_gamma, param.epsilon);
    }

    if (param.expected_eigenvalues.size() > 0) {
        da_int size_evals = nc;
        std::vector<T> evals(size_evals);
        EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvalues, &size_evals,
                                       evals.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(size_evals, evals.data(), param.expected_eigenvalues.data(),
                        param.epsilon);
    }

    if (param.expected_eigenvectors.size() > 0) {
        da_int size_evecs = n * nc;
        std::vector<T> evecs(size_evecs);
        EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvectors, &size_evecs,
                                       evecs.data()),
                  da_status_success);
        sign_correct_columns(n, nc, evecs, param.expected_eigenvectors, param.order);
        EXPECT_ARR_NEAR(size_evecs, evecs.data(), param.expected_eigenvectors.data(),
                        param.epsilon);
    }

    if (param.expected_scores.size() > 0) {
        da_int size_scores = n * nc;
        std::vector<T> scores(size_scores);
        EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_scores, &size_scores,
                                       scores.data()),
                  da_status_success);
        sign_correct_columns(n, nc, scores, param.expected_scores, param.order);
        EXPECT_ARR_NEAR(size_scores, scores.data(), param.expected_scores.data(),
                        param.epsilon);
    }

    if (param.expected_X_fit.size() > 0) {
        da_int size_X_fit = param.n * param.p;
        std::vector<T> X_fit(size_X_fit);
        EXPECT_EQ(
            da_handle_get_result(handle, da_kernel_pca_X_fit, &size_X_fit, X_fit.data()),
            da_status_success);
        EXPECT_ARR_NEAR(size_X_fit, X_fit.data(), param.expected_X_fit.data(),
                        param.epsilon);
    }

    // Self-transform consistency: transform(training data) must match scores
    {
        da_int size_scores = n * nc;
        std::vector<T> scores(size_scores);
        da_int size_scores_query = size_scores;
        EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_scores, &size_scores_query,
                                       scores.data()),
                  da_status_success);

        da_int p_xform = (param.kernel == "precomputed") ? n : param.p;
        da_int ldx_t = (param.order == "column-major") ? n : nc;
        std::vector<T> self_transform(size_scores);
        EXPECT_EQ(da_kernel_pca_transform(handle, n, p_xform, param.A.data(), param.lda,
                                          self_transform.data(), ldx_t),
                  da_status_success);

        sign_correct_columns(n, nc, self_transform, scores, param.order);
        EXPECT_ARR_NEAR(size_scores, self_transform.data(), scores.data(), param.epsilon);
    }

    // External transform test
    if (param.X_transform_in.size() > 0) {
        da_int size_out;
        if (param.order == "column-major")
            size_out = param.ldx_transform * nc;
        else
            size_out = param.ldx_transform * param.m;
        std::vector<T> X_transform_out(size_out);
        EXPECT_EQ(da_kernel_pca_transform(handle, param.m, param.p_transform,
                                          param.X_transform_in.data(), param.ldx,
                                          X_transform_out.data(), param.ldx_transform),
                  da_status_success);
        sign_correct_columns(param.m, nc, X_transform_out, param.expected_X_transform,
                             param.order, param.ldx_transform);
        EXPECT_ARR_NEAR(size_out, X_transform_out.data(),
                        param.expected_X_transform.data(), param.epsilon);
    }

    // Inverse transform test
    if (param.Y_inv.size() > 0) {
        da_int size_out;
        if (param.order == "column-major")
            size_out = param.ldy_inv_transform * param.p;
        else
            size_out = param.ldy_inv_transform * param.k;
        std::vector<T> Y_inv_out(size_out);
        EXPECT_EQ(da_kernel_pca_inverse_transform(handle, param.k, nc, param.Y_inv.data(),
                                                  param.ldy, Y_inv_out.data(),
                                                  param.ldy_inv_transform),
                  da_status_success);
        EXPECT_ARR_NEAR(size_out, Y_inv_out.data(), param.expected_Y_inv_transform.data(),
                        param.epsilon);
    }

    da_handle_destroy(&handle);
}

// Parameterized test classes
class DoubleKernelPCAFunctionalityTest
    : public testing::TestWithParam<KernelPCAParamType<double>> {};
class FloatKernelPCAFunctionalityTest
    : public testing::TestWithParam<KernelPCAParamType<float>> {};

template <typename T>
void PrintTo(const KernelPCAParamType<T> &param, ::std::ostream *os) {
    *os << param.test_name;
}

TEST_P(DoubleKernelPCAFunctionalityTest, ParameterizedTest) {
    const KernelPCAParamType<double> &param = GetParam();
    test_functionality(param);
}

TEST_P(FloatKernelPCAFunctionalityTest, ParameterizedTest) {
    const KernelPCAParamType<float> &param = GetParam();
    test_functionality(param);
}

template <typename T> std::vector<KernelPCAParamType<T>> getKernelPCAParams() {
    std::vector<KernelPCAParamType<T>> params;
    GetKernelPCAData(params);
    return params;
}

INSTANTIATE_TEST_SUITE_P(KernelPCA_Functionality_Tests_Double,
                         DoubleKernelPCAFunctionalityTest,
                         ::testing::ValuesIn(getKernelPCAParams<double>()));
INSTANTIATE_TEST_SUITE_P(KernelPCA_Functionality_Tests_Float,
                         FloatKernelPCAFunctionalityTest,
                         ::testing::ValuesIn(getKernelPCAParams<float>()));
