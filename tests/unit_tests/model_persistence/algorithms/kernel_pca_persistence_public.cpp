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

#include "../../datests_cblas.hh"
#include "../../factorization/kernel_pca_test_data.hpp"
#include "../../utest_utils.hpp"
#include "../persistence_test_utils.hpp"
#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

// The fixture is parameterized on double. Both double and float test methods receive
// KernelPCAParamType<double> via GetParam(). The helper converts typed fields to T
// internally, matching the same pattern used by kmeans_persistence_public.cpp.
template <typename T>
void kernel_pca_persistence_test(const KernelPCAParamType<double> &param,
                                 const std::string &model_file) {

    // Convert typed fields from double to T
    std::vector<T> A(param.A.begin(), param.A.end());
    std::vector<T> X_transform_in(param.X_transform_in.begin(),
                                  param.X_transform_in.end());
    std::vector<T> Y_inv(param.Y_inv.begin(), param.Y_inv.end());
    T gamma = static_cast<T>(param.gamma);
    T coef0 = static_cast<T>(param.coef0);
    T alpha = static_cast<T>(param.alpha);

    da_int n = param.n;
    da_int nc = param.expected_n_components;

    // Result arrays for original model
    std::vector<T> evals_orig(nc);
    std::vector<T> scores_orig(n * nc);
    std::vector<T> transform_orig;
    std::vector<T> inv_orig;

    // ==================== ORIGINAL MODEL BLOCK ====================
    {
        da_handle handle = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle, da_handle_kernel_pca), da_status_success);

        EXPECT_EQ(da_options_set(handle, "storage order", param.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "copy data", param.copy_data.c_str()),
                  da_status_success);
        EXPECT_EQ(da_kernel_pca_set_data(handle, param.n, param.p, A.data(), param.lda),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "kernel", param.kernel.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "n_components", param.n_components),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "gamma", gamma), da_status_success);
        EXPECT_EQ(da_options_set(handle, "degree", param.degree), da_status_success);
        EXPECT_EQ(da_options_set(handle, "coef0", coef0), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "fit inverse transform",
                                        param.fit_inverse_transform.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "remove zero eig",
                                        param.remove_zero_eig.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "alpha", alpha), da_status_success);

        EXPECT_EQ(da_kernel_pca_compute<T>(handle), da_status_success);

        da_int evals_dim = nc;
        EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_eigenvalues, &evals_dim,
                                       evals_orig.data()),
                  da_status_success);

        da_int scores_dim = n * nc;
        EXPECT_EQ(da_handle_get_result(handle, da_kernel_pca_scores, &scores_dim,
                                       scores_orig.data()),
                  da_status_success);

        if (X_transform_in.size() > 0) {
            transform_orig.resize(param.expected_X_transform.size());
            EXPECT_EQ(da_kernel_pca_transform(handle, param.m, param.p_transform,
                                              X_transform_in.data(), param.ldx,
                                              transform_orig.data(), param.ldx_transform),
                      da_status_success);
        }

        if (Y_inv.size() > 0) {
            inv_orig.resize(param.expected_Y_inv_transform.size());
            EXPECT_EQ(da_kernel_pca_inverse_transform(handle, param.k, nc, Y_inv.data(),
                                                      param.ldy, inv_orig.data(),
                                                      param.ldy_inv_transform),
                      da_status_success);
        }

        EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_success);
        da_handle_destroy(&handle);
    }

    // ==================== LOADED MODEL BLOCK ====================
    {
        da_handle handle_loaded = nullptr;
        EXPECT_EQ(da_handle_load_model(&handle_loaded, model_file.c_str()),
                  da_status_success);

        std::vector<T> evals_loaded(nc);
        std::vector<T> scores_loaded(n * nc);

        da_int evals_dim = nc;
        EXPECT_EQ(da_handle_get_result(handle_loaded, da_kernel_pca_eigenvalues,
                                       &evals_dim, evals_loaded.data()),
                  da_status_success);

        da_int scores_dim = n * nc;
        EXPECT_EQ(da_handle_get_result(handle_loaded, da_kernel_pca_scores, &scores_dim,
                                       scores_loaded.data()),
                  da_status_success);

        // ==================== COMPARE RESULTS ====================

        EXPECT_ARR_EQ(nc, evals_orig.data(), evals_loaded.data(), 1, 1, 0, 0);

        // Sign-correct scores before comparing (eigenvector sign is arbitrary)
        sign_correct_columns(n, nc, scores_loaded, scores_orig, param.order);
        EXPECT_ARR_EQ(n * nc, scores_orig.data(), scores_loaded.data(), 1, 1, 0, 0);

        if (X_transform_in.size() > 0) {
            da_int trans_size = (da_int)param.expected_X_transform.size();
            std::vector<T> transform_loaded(trans_size);
            EXPECT_EQ(da_kernel_pca_transform(handle_loaded, param.m, param.p_transform,
                                              X_transform_in.data(), param.ldx,
                                              transform_loaded.data(),
                                              param.ldx_transform),
                      da_status_success);
            sign_correct_columns(param.m, nc, transform_loaded, transform_orig,
                                 param.order, param.ldx_transform);
            EXPECT_ARR_EQ(trans_size, transform_orig.data(), transform_loaded.data(), 1,
                          1, 0, 0);
        }

        if (Y_inv.size() > 0) {
            da_int inv_size = (da_int)param.expected_Y_inv_transform.size();
            std::vector<T> inv_loaded(inv_size);
            EXPECT_EQ(da_kernel_pca_inverse_transform(
                          handle_loaded, param.k, nc, Y_inv.data(), param.ldy,
                          inv_loaded.data(), param.ldy_inv_transform),
                      da_status_success);
            EXPECT_ARR_EQ(inv_size, inv_orig.data(), inv_loaded.data(), 1, 1, 0, 0);
        }

        // ==================== VERIFY OPTIONS ====================
        char kernel_loaded[64];
        da_int kernel_len = 64;
        EXPECT_EQ(
            da_options_get_string(handle_loaded, "kernel", kernel_loaded, &kernel_len),
            da_status_success);
        EXPECT_STREQ(kernel_loaded, param.kernel.c_str());

        da_int n_components_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "n_components", &n_components_loaded),
                  da_status_success);
        EXPECT_EQ(n_components_loaded, param.n_components);

        da_int degree_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "degree", &degree_loaded),
                  da_status_success);
        EXPECT_EQ(degree_loaded, param.degree);

        char fit_inv_loaded[4];
        da_int fit_inv_len = 4;
        EXPECT_EQ(da_options_get_string(handle_loaded, "fit inverse transform",
                                        fit_inv_loaded, &fit_inv_len),
                  da_status_success);
        EXPECT_STREQ(fit_inv_loaded, param.fit_inverse_transform.c_str());

        char remove_zero_eig_loaded[4];
        da_int remove_zero_eig_len = 4;
        EXPECT_EQ(da_options_get_string(handle_loaded, "remove zero eig",
                                        remove_zero_eig_loaded, &remove_zero_eig_len),
                  da_status_success);
        EXPECT_STREQ(remove_zero_eig_loaded, param.remove_zero_eig.c_str());

        char copy_data_loaded[4];
        da_int copy_data_len = 4;
        EXPECT_EQ(da_options_get_string(handle_loaded, "copy data", copy_data_loaded,
                                        &copy_data_len),
                  da_status_success);
        EXPECT_STREQ(copy_data_loaded, param.copy_data.c_str());

        T gamma_loaded = T(0);
        EXPECT_EQ(da_options_get(handle_loaded, "gamma", &gamma_loaded),
                  da_status_success);
        EXPECT_EQ(gamma_loaded, gamma);

        T coef0_loaded = T(0);
        EXPECT_EQ(da_options_get(handle_loaded, "coef0", &coef0_loaded),
                  da_status_success);
        EXPECT_EQ(coef0_loaded, coef0);

        T alpha_loaded = T(0);
        EXPECT_EQ(da_options_get(handle_loaded, "alpha", &alpha_loaded),
                  da_status_success);
        EXPECT_EQ(alpha_loaded, alpha);

        da_handle_destroy(&handle_loaded);
    }
}

// ==================== PARAMETRIZED TEST CLASSES ====================

template <typename T> std::vector<KernelPCAParamType<T>> getKernelPCAPersistenceParams() {
    std::vector<KernelPCAParamType<T>> params;
    add_linear_tall(params);  // kernel_linear, includes fit_inverse + transform data
    add_poly_tall(params);    // kernel_poly, includes fit_inverse + transform data
    add_rbf_wide(params);     // kernel_rbf, includes fit_inverse + transform data
    add_sigmoid_tall(params); // kernel_sigmoid, includes fit_inverse + transform data
    add_precomputed_square(params); // kernel_precomputed, includes transform data
    // linear_tall and rbf_wide both exercise copy_data=false
    return params;
}

template <typename T>
void PrintTo(const KernelPCAParamType<T> &param, ::std::ostream *os) {
    *os << param.test_name;
}

class KernelPCAPersistenceTest
    : public testing::TestWithParam<KernelPCAParamType<double>> {
  protected:
    std::string model_file;
    void SetUp() override {
        const KernelPCAParamType<double> &param = GetParam();
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_case = test_info->name();
        std::replace(test_case.begin(), test_case.end(), '/', '_');
        model_file = model_persistence_test_utils::get_test_file_dir() + "/kpca_" +
                     param.test_name + "_" + test_case + ".bin";
    }
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_P(KernelPCAPersistenceTest, double) {
    kernel_pca_persistence_test<double>(GetParam(), model_file);
}

TEST_P(KernelPCAPersistenceTest, float) {
    kernel_pca_persistence_test<float>(GetParam(), model_file);
}

INSTANTIATE_TEST_SUITE_P(KernelPCAPersistenceSuite, KernelPCAPersistenceTest,
                         ::testing::ValuesIn(getKernelPCAPersistenceParams<double>()));

// ==================== ERROR HANDLING TESTS ====================

class KernelPCASerializationErrorTest : public testing::Test {
  protected:
    std::string model_file;
    void SetUp() override {
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = test_info->name();
        model_file = model_persistence_test_utils::get_test_file_dir() + "/kpca_error_" +
                     test_name + ".bin";
    }
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_F(KernelPCASerializationErrorTest, SaveBeforeComputeFails) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kernel_pca), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_no_data);
    da_handle_destroy(&handle);
}

TEST_F(KernelPCASerializationErrorTest, ComputeAfterLoadSucceeds) {
    // Unlike PCA, Kernel PCA serializes the training data, so compute() can be
    // called on a loaded model without calling set_data again.
    da_handle handle_train = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_train, da_handle_kernel_pca), da_status_success);
    std::vector<double> X_train = {1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0};
    EXPECT_EQ(da_kernel_pca_set_data_d(handle_train, 3, 3, X_train.data(), 3),
              da_status_success);
    EXPECT_EQ(da_kernel_pca_compute_d(handle_train), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle_train, model_file.c_str()), da_status_success);
    da_handle_destroy(&handle_train);

    // Load model and recompute -- should succeed because training data was serialized
    da_handle handle_load = nullptr;
    EXPECT_EQ(da_handle_load_model(&handle_load, model_file.c_str()), da_status_success);
    EXPECT_EQ(da_kernel_pca_compute_d(handle_load), da_status_success);
    da_handle_destroy(&handle_load);
}
