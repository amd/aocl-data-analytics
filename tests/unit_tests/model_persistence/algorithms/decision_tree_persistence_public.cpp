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

// Test parameters for decision tree serialization tests
struct dtree_serial_params {
    std::string test_name;
    std::string scoring_function;
    da_int max_depth;
    da_int node_min_samples;
    da_int seed;
};

void PrintTo(const dtree_serial_params &param, ::std::ostream *os) {
    *os << param.test_name;
}

// Test configurations covering different codepaths
// Defaults: scoring="gini", max_depth=29, node_min_samples=2, seed=-1
const std::vector<dtree_serial_params> dtree_serialization_params = {
    {"entropy_depth8_minsamp3", "cross-entropy", 8, 3, 77},
    {"gini_depth10_minsamp4", "gini", 10, 4, 42},
    {"misclass_depth5_minsamp5", "misclass", 5, 5, 123},
    {"entropy_depth12_minsamp3", "cross-entropy", 12, 3, 456},
};

class DTreeSerializationTest : public testing::TestWithParam<dtree_serial_params> {
  protected:
    // Problem dimensions
    da_int n_samples_train = 20;
    da_int n_samples_test = 6;
    da_int n_features = 5;
    da_int n_class = 3;

    std::vector<double> X_train;
    std::vector<da_int> y_train;
    std::vector<double> X_test;
    std::vector<da_int> y_test;

    std::string model_file;

    void SetUp() override {
        const dtree_serial_params &pr = GetParam();
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_case = test_info->name();
        std::replace(test_case.begin(), test_case.end(), '/', '_');
        model_file = model_persistence_test_utils::get_test_file_dir() + "/dtree_" +
                     pr.test_name + "_" + test_case + ".bin";

        // Training data: 20 samples, 5 features, 3 classes
        // Pattern: class depends on feature combinations to ensure multi-node trees
        // clang-format off
        X_train = {
            0.10, 0.20, 0.30, 0.15, 0.25,
            0.15, 0.25, 0.35, 0.18, 0.28,
            0.12, 0.22, 0.32, 0.16, 0.26,
            0.18, 0.28, 0.38, 0.19, 0.29,
            0.11, 0.21, 0.31, 0.14, 0.24,
            0.13, 0.23, 0.33, 0.17, 0.27,
            0.14, 0.24, 0.34, 0.13, 0.23,
            0.50, 0.60, 0.30, 0.55, 0.65,
            0.55, 0.65, 0.35, 0.58, 0.68,
            0.52, 0.62, 0.32, 0.56, 0.66,
            0.58, 0.68, 0.38, 0.59, 0.69,
            0.51, 0.61, 0.31, 0.54, 0.64,
            0.53, 0.63, 0.33, 0.57, 0.67,
            0.54, 0.64, 0.34, 0.53, 0.63,
            0.80, 0.20, 0.80, 0.85, 0.90,
            0.85, 0.25, 0.85, 0.88, 0.93,
            0.82, 0.22, 0.82, 0.86, 0.91,
            0.88, 0.28, 0.88, 0.89, 0.94,
            0.81, 0.21, 0.81, 0.84, 0.89,
            0.83, 0.23, 0.83, 0.87, 0.92
        };
        y_train = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};

        X_test = {
            0.10, 0.20, 0.30, 0.15, 0.25,
            0.50, 0.60, 0.30, 0.55, 0.65,
            0.80, 0.20, 0.80, 0.85, 0.90,
            0.12, 0.22, 0.32, 0.16, 0.26,
            0.55, 0.65, 0.35, 0.58, 0.68,
            0.85, 0.25, 0.85, 0.88, 0.93
        };
        // clang-format on
        y_test = {0, 1, 2, 0, 1, 2};
    }

    void TearDown() override { std::remove(model_file.c_str()); }
};

template <typename T>
void dtree_serialization_test(const dtree_serial_params &pr, da_int n_samples_train,
                              da_int n_samples_test, da_int n_features, da_int n_class,
                              const std::vector<double> &X_train_d,
                              const std::vector<da_int> &y_train,
                              const std::vector<double> &X_test_d,
                              const std::vector<da_int> &y_test,
                              const std::string &model_file) {

    // Convert data to correct type
    std::vector<T> X_train(X_train_d.begin(), X_train_d.end());
    std::vector<T> X_test(X_test_d.begin(), X_test_d.end());

    // Result arrays for original model
    std::vector<da_int> y_pred_orig(n_samples_test);
    std::vector<T> y_proba_orig(n_samples_test * n_class);
    T score_orig;

    // ==================== ORIGINAL MODEL BLOCK ====================
    {
        da_handle handle_orig = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle_orig, da_handle_decision_tree),
                  da_status_success);

        EXPECT_EQ(da_options_set_string(handle_orig, "scoring function",
                                        pr.scoring_function.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "maximum depth", pr.max_depth),
                  da_status_success);
        EXPECT_EQ(
            da_options_set_int(handle_orig, "node minimum samples", pr.node_min_samples),
            da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "seed", pr.seed), da_status_success);

        EXPECT_EQ(da_tree_set_training_data(handle_orig, n_samples_train, n_features,
                                            n_class, X_train.data(), n_samples_train,
                                            y_train.data(), nullptr),
                  da_status_success);
        EXPECT_EQ(da_tree_fit<T>(handle_orig), da_status_success);

        // Make predictions
        EXPECT_EQ(da_tree_predict(handle_orig, n_samples_test, n_features, X_test.data(),
                                  n_samples_test, y_pred_orig.data()),
                  da_status_success);

        // Get probabilities
        EXPECT_EQ(da_tree_predict_proba(handle_orig, n_samples_test, n_features,
                                        X_test.data(), n_samples_test,
                                        y_proba_orig.data(), n_class, n_samples_test),
                  da_status_success);

        // Get score
        EXPECT_EQ(da_tree_score(handle_orig, n_samples_test, n_features, X_test.data(),
                                n_samples_test, y_test.data(), &score_orig),
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
        std::vector<da_int> y_pred_loaded(n_samples_test);
        std::vector<T> y_proba_loaded(n_samples_test * n_class);
        T score_loaded;

        // Make predictions from loaded model
        EXPECT_EQ(da_tree_predict(handle_loaded, n_samples_test, n_features,
                                  X_test.data(), n_samples_test, y_pred_loaded.data()),
                  da_status_success);

        // Get probabilities from loaded model
        EXPECT_EQ(da_tree_predict_proba(handle_loaded, n_samples_test, n_features,
                                        X_test.data(), n_samples_test,
                                        y_proba_loaded.data(), n_class, n_samples_test),
                  da_status_success);

        // Get score from loaded model
        EXPECT_EQ(da_tree_score(handle_loaded, n_samples_test, n_features, X_test.data(),
                                n_samples_test, y_test.data(), &score_loaded),
                  da_status_success);

        // ==================== COMPARE RESULTS ====================

        // Compare predictions
        EXPECT_ARR_EQ(n_samples_test, y_pred_orig.data(), y_pred_loaded.data(), 1, 1, 0,
                      0);

        // Compare probabilities
        EXPECT_ARR_EQ(n_samples_test * n_class, y_proba_orig.data(),
                      y_proba_loaded.data(), 1, 1, 0, 0);

        // Compare scores
        EXPECT_EQ(score_orig, score_loaded);

        // Test multiple predictions work after loading (stability check)
        std::vector<da_int> y_pred_again(n_samples_test);
        EXPECT_EQ(da_tree_predict(handle_loaded, n_samples_test, n_features,
                                  X_test.data(), n_samples_test, y_pred_again.data()),
                  da_status_success);
        EXPECT_ARR_EQ(n_samples_test, y_pred_orig.data(), y_pred_again.data(), 1, 1, 0,
                      0);

        // ==================== VERIFY OPTIONS ====================
        // Verify that options are preserved after serialization
        char scoring_loaded[64];
        da_int scoring_len = 64;
        EXPECT_EQ(da_options_get_string(handle_loaded, "scoring function", scoring_loaded,
                                        &scoring_len),
                  da_status_success);
        EXPECT_STREQ(scoring_loaded, pr.scoring_function.c_str());

        da_int max_depth_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "maximum depth", &max_depth_loaded),
                  da_status_success);
        EXPECT_EQ(max_depth_loaded, pr.max_depth);

        da_int node_min_samples_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "node minimum samples",
                                     &node_min_samples_loaded),
                  da_status_success);
        EXPECT_EQ(node_min_samples_loaded, pr.node_min_samples);

        da_int seed_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "seed", &seed_loaded),
                  da_status_success);
        EXPECT_EQ(seed_loaded, pr.seed);

        da_handle_destroy(&handle_loaded);
    }
}

TEST_P(DTreeSerializationTest, double) {
    const dtree_serial_params &pr = GetParam();
    dtree_serialization_test<double>(pr, n_samples_train, n_samples_test, n_features,
                                     n_class, X_train, y_train, X_test, y_test,
                                     model_file);
}

TEST_P(DTreeSerializationTest, float) {
    const dtree_serial_params &pr = GetParam();
    dtree_serialization_test<float>(pr, n_samples_train, n_samples_test, n_features,
                                    n_class, X_train, y_train, X_test, y_test,
                                    model_file);
}

INSTANTIATE_TEST_SUITE_P(DTreeSerializationSuite, DTreeSerializationTest,
                         testing::ValuesIn(dtree_serialization_params));

// ==================== ERROR HANDLING TESTS ====================

class DTreeSerializationErrorTest : public testing::Test {
  protected:
    std::string model_file;
    void SetUp() override {
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = test_info->name();
        model_file = model_persistence_test_utils::get_test_file_dir() + "/dtree_error_" +
                     test_name + ".bin";
    }
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_F(DTreeSerializationErrorTest, SaveBeforeFitFails) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_decision_tree), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_no_data);
    da_handle_destroy(&handle);
}

TEST_F(DTreeSerializationErrorTest, FitAfterLoadWithoutSetDataFails) {
    // Train and save a model
    da_handle handle_train = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_train, da_handle_decision_tree),
              da_status_success);
    std::vector<double> X_train = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0};
    std::vector<da_int> y_train = {0, 1, 1, 0};
    EXPECT_EQ(da_tree_set_training_data_d(handle_train, 4, 2, 2, X_train.data(), 4,
                                          y_train.data(), nullptr),
              da_status_success);
    EXPECT_EQ(da_tree_fit_d(handle_train), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle_train, model_file.c_str()), da_status_success);
    da_handle_destroy(&handle_train);

    // Load model and try to fit without calling set_training_data
    da_handle handle_load = nullptr;
    EXPECT_EQ(da_handle_load_model(&handle_load, model_file.c_str()), da_status_success);
    EXPECT_EQ(da_tree_fit_d(handle_load), da_status_no_data);
    da_handle_destroy(&handle_load);
}