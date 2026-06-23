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

// Test parameters for PCA serialization tests
struct pca_serial_params {
    std::string test_name;
    std::string method;
    std::string svd_solver;
    std::string degrees_of_freedom;
    da_int n_components;
    da_int store_U;
    da_int whiten;
};

void PrintTo(const pca_serial_params &param, ::std::ostream *os) {
    *os << param.test_name;
}

// Test configurations covering different codepaths
// Defaults: method="covariance", svd_solver="auto", dof="unbiased", n_comp=1, store_U=0, whiten=0
const std::vector<pca_serial_params> pca_serialization_params = {
    {"correlation_3comp_biased", "correlation", "auto", "biased", 3, 1, 1},
    {"covariance_2comp_noU", "covariance", "auto", "unbiased", 2, 0, 0},
    {"svd_gesvd_biased", "svd", "gesvd", "biased", 3, 1, 0},
    {"svd_gesvdx", "svd", "gesvdx", "unbiased", 3, 1, 0},
    {"covariance_whiten", "covariance", "auto", "unbiased", 3, 1, 1},
};

class PCASerializationTest : public testing::TestWithParam<pca_serial_params> {
  protected:
    std::string model_file;

    // Problem dimensions
    da_int n_samples = 6;
    da_int n_features = 5;
    da_int lda = 6;

    // Test dimensions
    da_int m_samples = 3;
    da_int ldx = 3;

    std::vector<double> A;
    std::vector<double> X_test;

    void SetUp() override {
        const pca_serial_params &pr = GetParam();
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_case = test_info->name();
        std::replace(test_case.begin(), test_case.end(), '/', '_');
        model_file = model_persistence_test_utils::get_test_file_dir() + "/pca_" +
                     pr.test_name + "_" + test_case + ".bin";

        // Training data: 6x5 matrix
        A = {2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 5.0, 2.0, 8.0, 3.0, 2.0, 3.0, 4.0, 4.0,
             3.0, 2.0, 1.0, 2.0, 8.0, 4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 4.0, 2.0, 2.0};

        // Test data for transform: 3x5 matrix
        X_test = {7.0, 3.0, 3.0, 4.0, 2.0, 3.0, 2.0, 5.0,
                  2.0, 9.0, 6.0, 4.0, 3.0, 4.0, 1.0};
    }

    void TearDown() override { std::remove(model_file.c_str()); }
};

template <typename T>
void pca_serialization_test(const pca_serial_params &pr, da_int n_samples,
                            da_int n_features, da_int lda, da_int m_samples, da_int ldx,
                            const std::vector<double> &A_d,
                            const std::vector<double> &X_test_d,
                            const std::string &model_file) {

    da_int n_components = pr.n_components;

    // Convert data to correct type
    std::vector<T> A(A_d.begin(), A_d.end());
    std::vector<T> X_test(X_test_d.begin(), X_test_d.end());

    // Result arrays for original model
    std::vector<T> components_orig(n_components * n_features);
    std::vector<T> scores_orig(n_samples * n_components);
    std::vector<T> transform_orig(m_samples * n_components);

    // ==================== ORIGINAL MODEL BLOCK ====================
    {
        da_handle handle_orig = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle_orig, da_handle_pca), da_status_success);

        EXPECT_EQ(da_pca_set_data(handle_orig, n_samples, n_features, A.data(), lda),
                  da_status_success);

        EXPECT_EQ(da_options_set_string(handle_orig, "PCA method", pr.method.c_str()),
                  da_status_success);
        if (pr.svd_solver != "auto") {
            EXPECT_EQ(
                da_options_set_string(handle_orig, "svd solver", pr.svd_solver.c_str()),
                da_status_success);
        }
        EXPECT_EQ(da_options_set_int(handle_orig, "n_components", pr.n_components),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "store U", pr.store_U),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "whiten", pr.whiten),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle_orig, "degrees of freedom",
                                        pr.degrees_of_freedom.c_str()),
                  da_status_success);

        EXPECT_EQ(da_pca_compute<T>(handle_orig), da_status_success);

        // Get principal components
        da_int comp_dim = n_components * n_features;
        EXPECT_EQ(da_handle_get_result(handle_orig, da_pca_principal_components,
                                       &comp_dim, components_orig.data()),
                  da_status_success);

        // Get scores (if store_U is enabled)
        if (pr.store_U) {
            da_int scores_dim = n_samples * n_components;
            EXPECT_EQ(da_handle_get_result(handle_orig, da_pca_scores, &scores_dim,
                                           scores_orig.data()),
                      da_status_success);
        }

        // Transform test data
        EXPECT_EQ(da_pca_transform(handle_orig, m_samples, n_features, X_test.data(), ldx,
                                   transform_orig.data(), m_samples),
                  da_status_success);

        // Save the model
        EXPECT_EQ(da_handle_save_model(handle_orig, model_file.c_str()),
                  da_status_success);

        // Destroy the original handle
        da_handle_destroy(&handle_orig);
    }

    // ==================== LOADED MODEL BLOCK ====================
    {
        da_handle handle_loaded = nullptr;
        EXPECT_EQ(da_handle_load_model(&handle_loaded, model_file.c_str()),
                  da_status_success);

        // Result arrays for loaded model
        std::vector<T> components_loaded(n_components * n_features);
        std::vector<T> scores_loaded(n_samples * n_components);
        std::vector<T> transform_loaded(m_samples * n_components);

        // Get principal components from loaded model
        da_int comp_dim = n_components * n_features;
        EXPECT_EQ(da_handle_get_result(handle_loaded, da_pca_principal_components,
                                       &comp_dim, components_loaded.data()),
                  da_status_success);

        // Get scores from loaded model (if store_U is enabled)
        if (pr.store_U) {
            da_int scores_dim = n_samples * n_components;
            EXPECT_EQ(da_handle_get_result(handle_loaded, da_pca_scores, &scores_dim,
                                           scores_loaded.data()),
                      da_status_success);
        }

        // Transform from loaded model
        EXPECT_EQ(da_pca_transform(handle_loaded, m_samples, n_features, X_test.data(),
                                   ldx, transform_loaded.data(), m_samples),
                  da_status_success);

        // ==================== COMPARE RESULTS ====================

        // Compare principal components
        EXPECT_ARR_EQ(n_components * n_features, components_orig.data(),
                      components_loaded.data(), 1, 1, 0, 0);

        // Compare scores (if store_U is enabled)
        if (pr.store_U) {
            EXPECT_ARR_EQ(n_samples * n_components, scores_orig.data(),
                          scores_loaded.data(), 1, 1, 0, 0);
        }

        // Compare transforms
        EXPECT_ARR_EQ(m_samples * n_components, transform_orig.data(),
                      transform_loaded.data(), 1, 1, 0, 0);

        // Test multiple transforms work after loading (stability check)
        std::vector<T> transform_again(m_samples * n_components);
        EXPECT_EQ(da_pca_transform(handle_loaded, m_samples, n_features, X_test.data(),
                                   ldx, transform_again.data(), m_samples),
                  da_status_success);
        EXPECT_ARR_EQ(m_samples * n_components, transform_orig.data(),
                      transform_again.data(), 1, 1, 0, 0);

        // ==================== VERIFY OPTIONS ====================
        // Verify that options are preserved after serialization
        char method_loaded[64];
        da_int method_len = 64;
        EXPECT_EQ(da_options_get_string(handle_loaded, "PCA method", method_loaded,
                                        &method_len),
                  da_status_success);
        EXPECT_STREQ(method_loaded, pr.method.c_str());

        if (pr.method == "svd") {
            char solver_loaded[64];
            da_int solver_len = 64;
            EXPECT_EQ(da_options_get_string(handle_loaded, "svd solver", solver_loaded,
                                            &solver_len),
                      da_status_success);
            EXPECT_STREQ(solver_loaded, pr.svd_solver.c_str());
        }

        char dof_loaded[64];
        da_int dof_len = 64;
        EXPECT_EQ(da_options_get_string(handle_loaded, "degrees of freedom", dof_loaded,
                                        &dof_len),
                  da_status_success);
        EXPECT_STREQ(dof_loaded, pr.degrees_of_freedom.c_str());

        da_int n_components_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "n_components", &n_components_loaded),
                  da_status_success);
        EXPECT_EQ(n_components_loaded, pr.n_components);

        da_int store_U_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "store U", &store_U_loaded),
                  da_status_success);
        EXPECT_EQ(store_U_loaded, pr.store_U);

        da_int whiten_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "whiten", &whiten_loaded),
                  da_status_success);
        EXPECT_EQ(whiten_loaded, pr.whiten);

        da_handle_destroy(&handle_loaded);
    }
}

TEST_P(PCASerializationTest, double) {
    const pca_serial_params &pr = GetParam();
    pca_serialization_test<double>(pr, n_samples, n_features, lda, m_samples, ldx, A,
                                   X_test, model_file);
}

TEST_P(PCASerializationTest, float) {
    const pca_serial_params &pr = GetParam();
    pca_serialization_test<float>(pr, n_samples, n_features, lda, m_samples, ldx, A,
                                  X_test, model_file);
}

INSTANTIATE_TEST_SUITE_P(PCASerializationSuite, PCASerializationTest,
                         testing::ValuesIn(pca_serialization_params));

// ==================== ERROR HANDLING TESTS ====================

class PCASerializationErrorTest : public testing::Test {
  protected:
    std::string model_file;
    void SetUp() override {
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = test_info->name();
        model_file = model_persistence_test_utils::get_test_file_dir() + "/pca_error_" +
                     test_name + ".bin";
    }
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_F(PCASerializationErrorTest, SaveBeforeComputeFails) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_pca), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_no_data);
    da_handle_destroy(&handle);
}

TEST_F(PCASerializationErrorTest, FitAfterLoadWithoutSetDataFails) {
    // Train and save a model
    da_handle handle_train = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_train, da_handle_pca), da_status_success);
    std::vector<double> X_train = {1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0};
    EXPECT_EQ(da_pca_set_data_d(handle_train, 3, 3, X_train.data(), 3),
              da_status_success);
    EXPECT_EQ(da_pca_compute_d(handle_train), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle_train, model_file.c_str()), da_status_success);
    da_handle_destroy(&handle_train);

    // Load model and try to fit without calling set_data
    da_handle handle_load = nullptr;
    EXPECT_EQ(da_handle_load_model(&handle_load, model_file.c_str()), da_status_success);
    EXPECT_EQ(da_pca_compute_d(handle_load), da_status_no_data);
    da_handle_destroy(&handle_load);
}