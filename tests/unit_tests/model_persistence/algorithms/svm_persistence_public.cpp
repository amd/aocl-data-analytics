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

// Test parameters for SVM serialization tests.
struct svm_serial_params {
    std::string test_name;
    da_svm_model model;
    std::string kernel;
    double C;
    double nu;
    double epsilon;
    double tolerance;
    da_int degree;
    double coef0;
};

void PrintTo(const svm_serial_params &param, ::std::ostream *os) {
    *os << param.test_name;
}

const std::vector<svm_serial_params> svm_serialization_params = {
    {"svc_rbf", svc, "rbf", 2.0, 0.5, 0.1, 1e-5, 3, 0.0},
    {"svr_linear", svr, "linear", 0.5, 0.5, 0.2, 1e-4, 3, 0.0},
    {"nusvc_sigmoid", nusvc, "sigmoid", 1.0, 0.3, 0.1, 1e-4, 3, 0.5},
    {"nusvr_polynomial", nusvr, "polynomial", 1.0, 0.4, 0.1, 1e-3, 2, 1.0},
};

class SVMSerializationTest : public testing::TestWithParam<svm_serial_params> {
  protected:
    std::string model_file;

    void SetUp() override {
        const svm_serial_params &pr = GetParam();
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_case = test_info->name();
        std::replace(test_case.begin(), test_case.end(), '/', '_');
        model_file = model_persistence_test_utils::get_test_file_dir() + "/svm_" +
                     pr.test_name + "_" + test_case + ".bin";
    }

    void TearDown() override { std::remove(model_file.c_str()); }
};

// All datasets: 5 training samples, 2 features, 2 test samples.
// Data is stored column-major with ldx = n_samples (tight packing).
// Each model uses its own independent data to avoid shared-variable aliasing.
template <typename T>
void svm_serialization_test(const svm_serial_params &pr, const std::string &model_file) {

    da_int n_train = 5, n_feat = 2, n_test = 2;
    da_int n_class = 0;

    // Per-model training data (column-major)
    std::vector<T> X_train, y_train, X_test, y_test;

    if (pr.model == svc) {
        // 3-class problem; rbf kernel; 5 support vectors expected
        X_train = {(T)1.92,  (T)1.76, (T)-1.02, (T)0.79, (T)2.86,
                   (T)-0.52, (T)0.84, (T)0.94,  (T)1.34, (T)-0.43};
        y_train = {(T)0, (T)0, (T)2, (T)1, (T)0};
        X_test = {(T)-0.15, (T)-1.14, (T)2.33, (T)1.11};
        y_test = {(T)1, (T)2};
        n_class = 3;
    } else if (pr.model == svr) {
        // Regression; linear kernel; all 5 samples are support vectors
        X_train = {(T)1.29,  (T)2.3,   (T)-1.73, (T)0.11,  (T)-1.03,
                   (T)-0.73, (T)-0.47, (T)1.66,  (T)-0.38, (T)0.48};
        y_train = {(T)-5.66, (T)45.94, (T)53.25, (T)-21.96, (T)-2.46};
        X_test = {(T)-0.59, (T)0.6, (T)0.91, (T)-0.07};
        y_test = {(T)20.36, (T)38.93};
    } else if (pr.model == nusvc) {
        // 3-class problem; sigmoid kernel; 5 support vectors expected
        X_train = {(T)-0.41, (T)0.62,  (T)1.39, (T)-0.02, (T)1.2,
                   (T)-0.23, (T)-0.49, (T)1.06, (T)0.57,  (T)0.92};
        y_train = {(T)2, (T)1, (T)0, (T)1, (T)0};
        X_test = {(T)-0.73, (T)1.21, (T)-1.14, (T)1.05};
        y_test = {(T)2, (T)0};
        n_class = 3;
    } else { // nusvr
        // Regression; polynomial kernel; 4 support vectors expected
        X_train = {(T)2.07, (T)-0.43, (T)-0.04, (T)0.46, (T)-1.38,
                   (T)0.43, (T)0.3,   (T)0.39,  (T)0.09, (T)-0.54};
        y_train = {(T)125.09, (T)-1.89, (T)22.09, (T)27.03, (T)-98.74};
        X_test = {(T)-0.87, (T)-1.1, (T)1.73, (T)-0.82};
        y_test = {(T)65.67, (T)-103.03};
    }

    bool is_classifier = (pr.model == svc || pr.model == nusvc);

    // Arrays to capture original-model results
    std::vector<T> pred_orig(n_test);
    std::vector<T> decision_orig(is_classifier ? n_test * n_class : 0);
    T score_orig = 0;

    // ==================== ORIGINAL MODEL BLOCK ====================
    {
        da_handle handle_orig = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle_orig, da_handle_svm), da_status_success);
        EXPECT_EQ(da_svm_select_model<T>(handle_orig, pr.model), da_status_success);

        EXPECT_EQ(da_options_set_string(handle_orig, "kernel", pr.kernel.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle_orig, "tolerance", (T)pr.tolerance),
                  da_status_success);

        if (pr.model == svc || pr.model == svr) {
            EXPECT_EQ(da_options_set(handle_orig, "c", (T)pr.C), da_status_success);
        } else {
            EXPECT_EQ(da_options_set(handle_orig, "nu", (T)pr.nu), da_status_success);
        }

        if (pr.model == svr) {
            EXPECT_EQ(da_options_set(handle_orig, "epsilon", (T)pr.epsilon),
                      da_status_success);
        }

        // poly and sigmoid both use coef0; poly also has degree
        if (pr.kernel == "polynomial") {
            EXPECT_EQ(da_options_set_int(handle_orig, "degree", pr.degree),
                      da_status_success);
            EXPECT_EQ(da_options_set(handle_orig, "coef0", (T)pr.coef0),
                      da_status_success);
        } else if (pr.kernel == "sigmoid") {
            EXPECT_EQ(da_options_set(handle_orig, "coef0", (T)pr.coef0),
                      da_status_success);
        }

        EXPECT_EQ(da_svm_set_data(handle_orig, n_train, n_feat, X_train.data(), n_train,
                                  y_train.data()),
                  da_status_success);
        EXPECT_EQ(da_svm_compute<T>(handle_orig), da_status_success);

        // Predictions
        EXPECT_EQ(da_svm_predict(handle_orig, n_test, n_feat, X_test.data(), n_test,
                                 pred_orig.data()),
                  da_status_success);

        // Decision function (classifiers only): result is n_class columns × n_test rows
        if (is_classifier) {
            EXPECT_EQ(da_svm_decision_function(handle_orig, n_test, n_feat, X_test.data(),
                                               n_test, ovr, decision_orig.data(), n_test),
                      da_status_success);
        }

        // Score
        EXPECT_EQ(da_svm_score(handle_orig, n_test, n_feat, X_test.data(), n_test,
                               y_test.data(), &score_orig),
                  da_status_success);

        // Save the model
        EXPECT_EQ(da_handle_save_model(handle_orig, model_file.c_str()),
                  da_status_success);

        da_handle_destroy(&handle_orig);
    }

    // ==================== LOADED MODEL BLOCK ====================
    {
        da_handle handle_loaded = nullptr;
        EXPECT_EQ(da_handle_load_model(&handle_loaded, model_file.c_str()),
                  da_status_success);

        // Predictions from loaded model
        std::vector<T> pred_loaded(n_test);
        EXPECT_EQ(da_svm_predict(handle_loaded, n_test, n_feat, X_test.data(), n_test,
                                 pred_loaded.data()),
                  da_status_success);

        // Decision function from loaded model
        std::vector<T> decision_loaded(is_classifier ? n_test * n_class : 0);
        if (is_classifier) {
            EXPECT_EQ(da_svm_decision_function(handle_loaded, n_test, n_feat,
                                               X_test.data(), n_test, ovr,
                                               decision_loaded.data(), n_test),
                      da_status_success);
        }

        // Score from loaded model
        T score_loaded = 0;
        EXPECT_EQ(da_svm_score(handle_loaded, n_test, n_feat, X_test.data(), n_test,
                               y_test.data(), &score_loaded),
                  da_status_success);

        // ==================== COMPARE RESULTS ====================
        EXPECT_ARR_EQ(n_test, pred_orig.data(), pred_loaded.data(), 1, 1, 0, 0);

        if (is_classifier) {
            EXPECT_ARR_EQ(n_test * n_class, decision_orig.data(), decision_loaded.data(),
                          1, 1, 0, 0);
        }

        EXPECT_EQ(score_orig, score_loaded);

        // Stability: a second predict call should return identical results
        std::vector<T> pred_again(n_test);
        EXPECT_EQ(da_svm_predict(handle_loaded, n_test, n_feat, X_test.data(), n_test,
                                 pred_again.data()),
                  da_status_success);
        EXPECT_ARR_EQ(n_test, pred_loaded.data(), pred_again.data(), 1, 1, 0, 0);

        // ==================== VERIFY OPTIONS ====================
        char kernel_loaded[64];
        da_int kernel_len = 64;
        EXPECT_EQ(
            da_options_get_string(handle_loaded, "kernel", kernel_loaded, &kernel_len),
            da_status_success);
        EXPECT_STREQ(kernel_loaded, pr.kernel.c_str());

        T tolerance_loaded = 0;
        EXPECT_EQ(da_options_get(handle_loaded, "tolerance", &tolerance_loaded),
                  da_status_success);
        EXPECT_EQ(tolerance_loaded, (T)pr.tolerance);

        if (pr.model == svc || pr.model == svr) {
            T c_loaded = 0;
            EXPECT_EQ(da_options_get(handle_loaded, "c", &c_loaded), da_status_success);
            EXPECT_EQ(c_loaded, (T)pr.C);
        } else {
            T nu_loaded = 0;
            EXPECT_EQ(da_options_get(handle_loaded, "nu", &nu_loaded), da_status_success);
            EXPECT_EQ(nu_loaded, (T)pr.nu);
        }

        if (pr.model == svr) {
            T epsilon_loaded = 0;
            EXPECT_EQ(da_options_get(handle_loaded, "epsilon", &epsilon_loaded),
                      da_status_success);
            EXPECT_EQ(epsilon_loaded, (T)pr.epsilon);
        }

        if (pr.kernel == "polynomial") {
            da_int degree_loaded = 0;
            EXPECT_EQ(da_options_get_int(handle_loaded, "degree", &degree_loaded),
                      da_status_success);
            EXPECT_EQ(degree_loaded, pr.degree);

            T coef0_loaded = 0;
            EXPECT_EQ(da_options_get(handle_loaded, "coef0", &coef0_loaded),
                      da_status_success);
            EXPECT_EQ(coef0_loaded, (T)pr.coef0);
        } else if (pr.kernel == "sigmoid") {
            T coef0_loaded = 0;
            EXPECT_EQ(da_options_get(handle_loaded, "coef0", &coef0_loaded),
                      da_status_success);
            EXPECT_EQ(coef0_loaded, (T)pr.coef0);
        }

        da_handle_destroy(&handle_loaded);
    }
}

TEST_P(SVMSerializationTest, double) {
    const svm_serial_params &pr = GetParam();
    svm_serialization_test<double>(pr, model_file);
}

TEST_P(SVMSerializationTest, float) {
    const svm_serial_params &pr = GetParam();
    svm_serialization_test<float>(pr, model_file);
}

INSTANTIATE_TEST_SUITE_P(SVMSerializationSuite, SVMSerializationTest,
                         testing::ValuesIn(svm_serialization_params));

// ==================== ERROR HANDLING TESTS ====================

class SVMSerializationErrorTest : public testing::Test {
  protected:
    std::string model_file;
    void SetUp() override {
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = test_info->name();
        model_file = model_persistence_test_utils::get_test_file_dir() + "/svm_error_" +
                     test_name + ".bin";
    }
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_F(SVMSerializationErrorTest, SaveBeforeComputeFails) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_svm), da_status_success);
    EXPECT_EQ(da_svm_select_model_d(handle, svc), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_no_data);
    da_handle_destroy(&handle);
}

TEST_F(SVMSerializationErrorTest, FitAfterLoadWithoutSetDataFails) {
    // Train and save a model
    da_handle handle_train = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_train, da_handle_svm), da_status_success);
    EXPECT_EQ(da_svm_select_model_d(handle_train, svc), da_status_success);
    std::vector<double> X_train = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0};
    std::vector<double> y_train = {-1, 1, 1, -1};
    EXPECT_EQ(da_svm_set_data(handle_train, 4, 2, X_train.data(), 4, y_train.data()),
              da_status_success);
    EXPECT_EQ(da_svm_compute_d(handle_train), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle_train, model_file.c_str()), da_status_success);
    da_handle_destroy(&handle_train);

    // Load model and try to fit without calling set_data
    da_handle handle_load = nullptr;
    EXPECT_EQ(da_handle_load_model(&handle_load, model_file.c_str()), da_status_success);
    EXPECT_EQ(da_svm_compute_d(handle_load), da_status_no_data);
    da_handle_destroy(&handle_load);
}