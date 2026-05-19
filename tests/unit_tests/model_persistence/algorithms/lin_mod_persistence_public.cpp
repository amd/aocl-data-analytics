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
#include <string>
#include <vector>

// Test parameters for linmod serialization tests
struct linmod_serial_params {
    std::string test_name;
    linmod_model model;
    double alpha;
    double lambda;
    da_int intercept;
    std::string scaling;
};

void PrintTo(const linmod_serial_params &param, ::std::ostream *os) {
    *os << param.test_name;
}

// Test configurations covering different codepaths
const std::vector<linmod_serial_params> linmod_serialization_params = {
    {"mse_ridge_standardize", linmod_model_mse, 0.0, 0.5, 1, "standardise"},
    {"mse_lasso", linmod_model_mse, 1.0, 1.0, 0, "no"},
    {"mse_elastic_net", linmod_model_mse, 0.5, 1.0, 0, "no"},
    {"mse_intercept_scale", linmod_model_mse, 0.0, 0.0, 1, "scale"},
#ifndef NO_FORTRAN
    {"logistic_ridge", linmod_model_logistic, 0.0, 0.5, 0, "no"},
    {"logistic_intercept", linmod_model_logistic, 0.0, 0.2, 1, "standardise"},
#endif
};

class LinModSerializationTest : public testing::TestWithParam<linmod_serial_params> {
  protected:
    // Problem dimensions for MSE
    da_int m_mse = 5;
    da_int n_mse = 2;

    // Problem dimensions for logistic
    da_int m_logistic = 6;
    da_int n_logistic = 2;

    // Test dimensions
    da_int m_test = 2;

    std::vector<double> A_mse;
    std::vector<double> b_mse;
    std::vector<double> A_logistic;
    std::vector<double> b_logistic;
    std::vector<double> X_test_mse;
    std::vector<double> X_test_logistic;

    std::string model_file;

    void SetUp() override {
        const linmod_serial_params &pr = GetParam();
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_case = test_info->name();
        std::replace(test_case.begin(), test_case.end(), '/', '_');
        model_file = model_persistence_test_utils::get_test_file_dir() + "/linmod_" +
                     pr.test_name + "_" + test_case + ".bin";
        // Training data for regression: simple linear relationship
        A_mse = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 3.0, 5.0, 1.0, 1.0};
        b_mse = {2.5, 5.5, 8.5, 5.5, 6.5};

        // Training data for logistic: binary classification
        A_logistic = {0.1, 0.2, 0.8, 0.9, 0.15, 0.85, 0.1, 0.15, 0.9, 0.85, 0.2, 0.8};
        b_logistic = {0.0, 0.0, 1.0, 1.0, 0.0, 1.0};

        // Test data for predictions
        X_test_mse = {2.0, 4.0, 2.0, 4.0};
        X_test_logistic = {0.1, 0.9, 0.15, 0.85};
    }

    void TearDown() override { std::remove(model_file.c_str()); }
};

template <typename T>
void linmod_serialization_test(const linmod_serial_params &pr, da_int m, da_int n,
                               da_int m_test, const std::vector<double> &A_d,
                               const std::vector<double> &b_d,
                               const std::vector<double> &X_test_d,
                               const std::string &model_file) {

    // Convert data to correct type
    std::vector<T> A(A_d.begin(), A_d.end());
    std::vector<T> b(b_d.begin(), b_d.end());
    std::vector<T> X_test(X_test_d.begin(), X_test_d.end());

    // Coefficient size depends on intercept
    da_int n_coef = n + pr.intercept;

    // Result arrays for original model
    std::vector<T> coef_orig(n_coef);
    std::vector<T> pred_orig(m_test);

    // ==================== ORIGINAL MODEL BLOCK ====================
    {
        da_handle handle_orig = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle_orig, da_handle_linmod), da_status_success);

        EXPECT_EQ(da_linmod_select_model<T>(handle_orig, pr.model), da_status_success);
        EXPECT_EQ(da_linmod_define_features(handle_orig, m, n, A.data(), m, b.data()),
                  da_status_success);

        EXPECT_EQ(da_options_set_int(handle_orig, "intercept", pr.intercept),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle_orig, "alpha", (T)pr.alpha), da_status_success);
        EXPECT_EQ(da_options_set(handle_orig, "lambda", (T)pr.lambda), da_status_success);
        EXPECT_EQ(da_options_set_string(handle_orig, "scaling", pr.scaling.c_str()),
                  da_status_success);

        EXPECT_EQ(da_linmod_fit<T>(handle_orig), da_status_success);

        // Get coefficients
        da_int coef_dim = n_coef;
        EXPECT_EQ(da_handle_get_result(handle_orig, da_linmod_coef, &coef_dim,
                                       coef_orig.data()),
                  da_status_success);

        // Get predictions on test data
        EXPECT_EQ(da_linmod_evaluate_model(handle_orig, m_test, n, X_test.data(), m_test,
                                           pred_orig.data()),
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
        std::vector<T> coef_loaded(n_coef);
        std::vector<T> pred_loaded(m_test);

        // Get coefficients from loaded model
        da_int coef_dim = n_coef;
        EXPECT_EQ(da_handle_get_result(handle_loaded, da_linmod_coef, &coef_dim,
                                       coef_loaded.data()),
                  da_status_success);

        // Get predictions from loaded model
        EXPECT_EQ(da_linmod_evaluate_model(handle_loaded, m_test, n, X_test.data(),
                                           m_test, pred_loaded.data()),
                  da_status_success);

        // ==================== COMPARE RESULTS ====================

        // Compare coefficients
        EXPECT_ARR_EQ(n_coef, coef_orig.data(), coef_loaded.data(), 1, 1, 0, 0);

        // Compare predictions
        EXPECT_ARR_EQ(m_test, pred_orig.data(), pred_loaded.data(), 1, 1, 0, 0);

        // Test multiple predictions work after loading (stability check)
        std::vector<T> pred_again(m_test);
        EXPECT_EQ(da_linmod_evaluate_model(handle_loaded, m_test, n, X_test.data(),
                                           m_test, pred_again.data()),
                  da_status_success);
        EXPECT_ARR_EQ(m_test, pred_orig.data(), pred_again.data(), 1, 1, 0, 0);

        // ==================== VERIFY OPTIONS ====================
        // Verify that options are preserved after serialization
        da_int intercept_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "intercept", &intercept_loaded),
                  da_status_success);
        EXPECT_EQ(intercept_loaded, pr.intercept);

        T alpha_loaded = 0;
        EXPECT_EQ(da_options_get(handle_loaded, "alpha", &alpha_loaded),
                  da_status_success);
        EXPECT_EQ(alpha_loaded, (T)pr.alpha);

        T lambda_loaded = 0;
        EXPECT_EQ(da_options_get(handle_loaded, "lambda", &lambda_loaded),
                  da_status_success);
        EXPECT_EQ(lambda_loaded, (T)pr.lambda);

        char scaling_loaded[64];
        da_int scaling_len = 64;
        EXPECT_EQ(
            da_options_get_string(handle_loaded, "scaling", scaling_loaded, &scaling_len),
            da_status_success);
        EXPECT_STREQ(scaling_loaded, pr.scaling.c_str());

        da_handle_destroy(&handle_loaded);
    }
}

TEST_P(LinModSerializationTest, double) {
    const linmod_serial_params &pr = GetParam();
    if (pr.model == linmod_model_mse) {
        linmod_serialization_test<double>(pr, m_mse, n_mse, m_test, A_mse, b_mse,
                                          X_test_mse, model_file);
    } else {
        linmod_serialization_test<double>(pr, m_logistic, n_logistic, m_test, A_logistic,
                                          b_logistic, X_test_logistic, model_file);
    }
}

TEST_P(LinModSerializationTest, float) {
    const linmod_serial_params &pr = GetParam();
    if (pr.model == linmod_model_mse) {
        linmod_serialization_test<float>(pr, m_mse, n_mse, m_test, A_mse, b_mse,
                                         X_test_mse, model_file);
    } else {
        linmod_serialization_test<float>(pr, m_logistic, n_logistic, m_test, A_logistic,
                                         b_logistic, X_test_logistic, model_file);
    }
}

INSTANTIATE_TEST_SUITE_P(LinModSerializationSuite, LinModSerializationTest,
                         testing::ValuesIn(linmod_serialization_params));

// ==================== ERROR HANDLING TESTS ====================

class LinModSerializationErrorTest : public testing::Test {
  protected:
    std::string model_file;
    void SetUp() override {
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = test_info->name();
        model_file = model_persistence_test_utils::get_test_file_dir() +
                     "/linmod_error_" + test_name + ".bin";
    }
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_F(LinModSerializationErrorTest, SaveBeforeFitFails) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_no_data);
    da_handle_destroy(&handle);
}

TEST_F(LinModSerializationErrorTest, FitAfterLoadWithoutSetDataFails) {
    // Train and save a model
    da_handle handle_train = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_train, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle_train, linmod_model_mse),
              da_status_success);
    std::vector<double> X_train = {1.0, 1.0, 2.0, 3.0, 3.0, 5.0};
    std::vector<double> y_train = {3.0, 6.5, 10.0};
    EXPECT_EQ(da_linmod_define_features_d(handle_train, 3, 2, X_train.data(), 3,
                                          y_train.data()),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_train), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle_train, model_file.c_str()), da_status_success);
    da_handle_destroy(&handle_train);

    // Load model and try to fit without calling define_features
    da_handle handle_load = nullptr;
    EXPECT_EQ(da_handle_load_model(&handle_load, model_file.c_str()), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_load), da_status_no_data);
    da_handle_destroy(&handle_load);
}
