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
 * Test the decision tree C API (double precision).
 * Uses inline data instead of CSV files.
 */
TEST(DecisionTreeCAPI, BasicDouble) {
    da_handle handle = nullptr;

    // Simple classification: 8 samples, 2 features, 2 classes (column-major)
    double X[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                    1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0};
    da_int y[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    da_int n_samples = 8, n_features = 2, n_class = 2, ldx = 8;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_decision_tree), da_status_success);
    EXPECT_EQ(da_tree_set_training_data_d(handle, n_samples, n_features, n_class, X, ldx,
                                          y, nullptr),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "maximum depth", 5), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 77), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "scoring function", "gini"),
              da_status_success);

    EXPECT_EQ(da_tree_fit_d(handle), da_status_success);

    // Predict on training data
    da_int y_pred[8];
    EXPECT_EQ(da_tree_predict_d(handle, n_samples, n_features, X, ldx, y_pred),
              da_status_success);

    // Predict probabilities
    double y_proba[16]; // n_samples * n_class
    EXPECT_EQ(da_tree_predict_proba_d(handle, n_samples, n_features, X, ldx, y_proba,
                                      n_class, n_samples),
              da_status_success);

    // Predict log probabilities
    double y_log_proba[16];
    EXPECT_EQ(da_tree_predict_log_proba_d(handle, n_samples, n_features, X, ldx,
                                          y_log_proba, n_class, n_samples),
              da_status_success);

    // Score
    double mean_accuracy = 0.0;
    EXPECT_EQ(da_tree_score_d(handle, n_samples, n_features, X, ldx, y, &mean_accuracy),
              da_status_success);
    EXPECT_GE(mean_accuracy, 0.5);

    da_handle_destroy(&handle);
}

/*
 * Test the decision tree C API (single precision).
 */
TEST(DecisionTreeCAPI, BasicFloat) {
    da_handle handle = nullptr;

    float X[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                   1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    da_int y[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    da_int n_samples = 8, n_features = 2, n_class = 2, ldx = 8;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_decision_tree), da_status_success);
    EXPECT_EQ(da_tree_set_training_data_s(handle, n_samples, n_features, n_class, X, ldx,
                                          y, nullptr),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "maximum depth", 5), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 77), da_status_success);

    EXPECT_EQ(da_tree_fit_s(handle), da_status_success);

    da_int y_pred[8];
    EXPECT_EQ(da_tree_predict_s(handle, n_samples, n_features, X, ldx, y_pred),
              da_status_success);

    float y_proba[16];
    EXPECT_EQ(da_tree_predict_proba_s(handle, n_samples, n_features, X, ldx, y_proba,
                                      n_class, n_samples),
              da_status_success);

    float y_log_proba[16];
    EXPECT_EQ(da_tree_predict_log_proba_s(handle, n_samples, n_features, X, ldx,
                                          y_log_proba, n_class, n_samples),
              da_status_success);

    float mean_accuracy = 0.0f;
    EXPECT_EQ(da_tree_score_s(handle, n_samples, n_features, X, ldx, y, &mean_accuracy),
              da_status_success);
    EXPECT_GE(mean_accuracy, 0.5f);

    da_handle_destroy(&handle);
}
