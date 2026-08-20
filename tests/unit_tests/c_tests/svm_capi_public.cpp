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
 * Test the SVM C API (double precision).
 * Based on tests/examples/svc.cpp
 */
TEST(SvmCAPI, BasicDouble) {
    da_handle handle = nullptr;

    // Training data: 8 samples, 2 features (column-major)
    double X[16] = {-2.99, -0.15, -0.09, 0.45, -1.03, -0.02, 1.59, 0.34,
                    0.04,  2.52,  0.91,  1.12, 0.3,   -0.9,  1.88, -0.15};
    double y[8] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0};

    // Test data: 5 samples, 2 features
    double X_test[10] = {1.51, 0.83, -1.66, 1.25, -1.01, 1.78, 1.9, 2.89, 1.42, 0.65};
    double y_test[5] = {1.0, 1.0, 0.0, 1.0, 0.0};

    da_int n_samples = 8, n_samples_test = 5, n_features = 2;
    da_int ldx = n_samples, ldx_test = n_samples_test;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_svm), da_status_success);
    EXPECT_EQ(da_svm_select_model_d(handle, svc), da_status_success);
    EXPECT_EQ(da_svm_set_data_d(handle, n_samples, n_features, X, ldx, y),
              da_status_success);

    EXPECT_EQ(da_options_set_string(handle, "kernel", "rbf"), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle, "C", 1.0), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle, "gamma", 1.0), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "predict probabilities", 1), da_status_success);

    EXPECT_EQ(da_svm_compute_d(handle), da_status_success);

    // Predict
    double predictions[5];
    EXPECT_EQ(da_svm_predict_d(handle, n_samples_test, n_features, X_test, ldx_test,
                               predictions),
              da_status_success);

    // Score
    double accuracy = 0.0;
    EXPECT_EQ(da_svm_score_d(handle, n_samples_test, n_features, X_test, ldx_test, y_test,
                             &accuracy),
              da_status_success);

    // Decision function
    double decision_values[5];
    da_int ldd = n_samples_test;
    EXPECT_EQ(da_svm_decision_function_d(handle, n_samples_test, n_features, X_test,
                                         ldx_test, ovo, decision_values, ldd),
              da_status_success);

    // Get number of support vectors
    da_int n_sv = 0, one = 1;
    EXPECT_EQ(da_handle_get_result_int(handle, da_svm_n_support_vectors, &one, &n_sv),
              da_status_success);

    // Predict probabilities
    double proba[10]; // n_samples_test * n_class
    EXPECT_EQ(da_svm_predict_proba_d(handle, n_samples_test, n_features, X_test, ldx_test,
                                     proba, n_samples_test),
              da_status_success);

    // Predict log probabilities
    double log_proba[10];
    EXPECT_EQ(da_svm_predict_log_proba_d(handle, n_samples_test, n_features, X_test,
                                         ldx_test, log_proba, n_samples_test),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test the SVM C API (single precision).
 */
TEST(SvmCAPI, BasicFloat) {
    da_handle handle = nullptr;

    float X[16] = {-2.99f, -0.15f, -0.09f, 0.45f, -1.03f, -0.02f, 1.59f, 0.34f,
                   0.04f,  2.52f,  0.91f,  1.12f, 0.3f,   -0.9f,  1.88f, -0.15f};
    float y[8] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f};

    float X_test[10] = {1.51f, 0.83f, -1.66f, 1.25f, -1.01f,
                        1.78f, 1.9f,  2.89f,  1.42f, 0.65f};
    float y_test[5] = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f};

    da_int n_samples = 8, n_samples_test = 5, n_features = 2;
    da_int ldx = n_samples, ldx_test = n_samples_test;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_svm), da_status_success);
    EXPECT_EQ(da_svm_select_model_s(handle, svc), da_status_success);
    EXPECT_EQ(da_svm_set_data_s(handle, n_samples, n_features, X, ldx, y),
              da_status_success);

    EXPECT_EQ(da_options_set_string(handle, "kernel", "rbf"), da_status_success);
    EXPECT_EQ(da_options_set_real_s(handle, "C", 1.0f), da_status_success);
    EXPECT_EQ(da_options_set_real_s(handle, "gamma", 1.0f), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "predict probabilities", 1), da_status_success);

    EXPECT_EQ(da_svm_compute_s(handle), da_status_success);

    float predictions[5];
    EXPECT_EQ(da_svm_predict_s(handle, n_samples_test, n_features, X_test, ldx_test,
                               predictions),
              da_status_success);

    float accuracy = 0.0f;
    EXPECT_EQ(da_svm_score_s(handle, n_samples_test, n_features, X_test, ldx_test, y_test,
                             &accuracy),
              da_status_success);

    float decision_values[5];
    EXPECT_EQ(da_svm_decision_function_s(handle, n_samples_test, n_features, X_test,
                                         ldx_test, ovo, decision_values, n_samples_test),
              da_status_success);

    // Predict probabilities
    float proba[10]; // n_samples_test * n_class
    EXPECT_EQ(da_svm_predict_proba_s(handle, n_samples_test, n_features, X_test, ldx_test,
                                     proba, n_samples_test),
              da_status_success);

    // Predict log probabilities
    float log_proba[10];
    EXPECT_EQ(da_svm_predict_log_proba_s(handle, n_samples_test, n_features, X_test,
                                         ldx_test, log_proba, n_samples_test),
              da_status_success);

    da_handle_destroy(&handle);
}
