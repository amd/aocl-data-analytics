/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "da_handle.hpp"
#include "linmod_functions.hpp"
#include "options.hpp"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

/* simple errors tests */
TEST(linmod, badHandle) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_linmod_select_model<double>(handle, linmod_model_mse),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_linmod_select_model<float>(handle, linmod_model_logistic),
              da_status_handle_not_initialized);

    da_int n = 1, m = 1;
    float *af = 0, *bf = 0;
    double *ad = 0, *bd = 0;
    EXPECT_EQ(da_linmod_define_features(handle, m, n, af, bf),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_linmod_define_features(handle, m, n, ad, bd),
              da_status_handle_not_initialized);

    EXPECT_EQ(da_linmod_fit_d(handle), da_status_handle_not_initialized);
    EXPECT_EQ(da_linmod_fit_s(handle), da_status_handle_not_initialized);

    da_int nc = 1;
    float *xf = 0;
    double *xd = 0;
    EXPECT_EQ(da_linmod_get_coef(handle, &nc, xf), da_status_handle_not_initialized);
    EXPECT_EQ(da_linmod_get_coef(handle, &nc, xd), da_status_handle_not_initialized);

    float *predf = 0;
    double *predd = 0;
    EXPECT_EQ(da_linmod_evaluate_model(handle, m, n, xf, predf),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_linmod_evaluate_model(handle, m, n, xd, predd),
              da_status_handle_not_initialized);
}

TEST(linmod, wrongType) {
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_handle_init<float>(&handle_s, da_handle_linmod), da_status_success);

    EXPECT_EQ(da_linmod_select_model<double>(handle_s, linmod_model_mse),
              da_status_wrong_type);
    EXPECT_EQ(da_linmod_select_model<float>(handle_d, linmod_model_logistic),
              da_status_wrong_type);

    da_int n = 1, m = 1;
    float *af = 0, *bf = 0;
    double *ad = 0, *bd = 0;
    EXPECT_EQ(da_linmod_define_features(handle_d, m, n, af, bf), da_status_wrong_type);
    EXPECT_EQ(da_linmod_define_features(handle_s, m, n, ad, bd), da_status_wrong_type);

    EXPECT_EQ(da_linmod_fit_d(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_linmod_fit_s(handle_d), da_status_wrong_type);

    da_int nc = 1;
    float *xf = 0;
    double *xd = 0;
    EXPECT_EQ(da_linmod_get_coef(handle_d, &nc, xf), da_status_wrong_type);
    EXPECT_EQ(da_linmod_get_coef(handle_s, &nc, xd), da_status_wrong_type);

    float *predf = 0;
    double *predd = 0;
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, m, n, xf, predf), da_status_wrong_type);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, m, n, xd, predd), da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

TEST(linmod, invalidInput) {
    // problem data
    da_int m = 5, n = 2;
    double Ad[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bd[5] = {1, 1, 1, 1, 1};
    da_int nx = 2;
    double xd[2];
    float As[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    float bs[5] = {1, 1, 1, 1, 1};
    float xs[2];

    // Initialize and compute the linear regression
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle_d, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_select_model_s(handle_s, linmod_model_mse), da_status_success);

    // define features
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, 0, Ad, bd),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, 0, n, Ad, bd),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, nullptr, bd),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, Ad, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, Ad, bd), da_status_success);

    EXPECT_EQ(da_linmod_define_features_s(handle_s, m, 0, As, bs),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_define_features_s(handle_s, 0, n, As, bs),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_define_features_s(handle_s, m, n, nullptr, bs),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_define_features_s(handle_s, m, n, As, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_define_features_s(handle_s, m, n, As, bs), da_status_success);

    // compute regression
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_success);
    EXPECT_EQ(da_linmod_fit_s(handle_s), da_status_success);

    // get coefficients
    nx = -1;
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coef, &nx, xd),
              da_status_invalid_array_dimension);
    nx = -1;
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_linmod_coef, &nx, xs),
              da_status_invalid_array_dimension);
    nx = 2;
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coef, &nx, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coef, &nx, xd),
              da_status_success);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_linmod_coef, &nx, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_linmod_coef, &nx, xs),
              da_status_success);

    // evaluate models
    double X[2] = {1., 2.};
    double pred[1];
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, 1, 3, X, pred), da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, 1, n, nullptr, pred),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, 1, n, X, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, 0, n, X, pred), da_status_invalid_input);
    float Xs[2] = {1., 2.};
    float preds[1];
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, 1, 3, Xs, preds),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, 1, n, nullptr, preds),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, 1, n, Xs, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, 0, n, Xs, preds),
              da_status_invalid_input);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

TEST(linmod, modOutOfDate) {
    // problem data
    da_int m = 5, n = 2;
    double Ad[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bd[5] = {1, 1, 1, 1, 1};
    da_int nx = 2;
    double xd[2];
    float As[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    float bs[5] = {1, 1, 1, 1, 1};
    float xs[2];

    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_handle_init<float>(&handle_s, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, Ad, bd), da_status_success);
    EXPECT_EQ(da_linmod_define_features_s(handle_s, m, n, As, bs), da_status_success);

    // Model was not yet fitted or out-of-date request of coefficients
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coef, &nx, xd),
              da_status_unknown_query);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_linmod_coef, &nx, xs),
              da_status_unknown_query);

    // Out of date request of model
    double X[2] = {1., 2.};
    double pred[1];
    float Xs[2] = {1., 2.};
    float preds[1];
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, 1, n, X, pred), da_status_out_of_date);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, 1, n, Xs, preds), da_status_out_of_date);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

TEST(linmod, incompatibleOptions) {
    // problem data
    da_int m = 5, n = 2;
    double Ad[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bd[5] = {1, 1, 1, 1, 1};
    da_handle handle_d = nullptr;

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, n, m, Ad, bd), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "QR"), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle_d, linmod_model_logistic),
              da_status_success);

    // QR factorization should not be compatible with logistic regression
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    // lbfgs  with 1-norm term
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "lbfgsb"),
              da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle_d, "lambda", 1.0), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle_d, "alpha", 1.0), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    da_handle_destroy(&handle_d);
}

TEST(linmod, GetResultNegative) {
    // Test public interfaces (under linmod context)
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;
    double dv[2];
    float sv[2];
    da_int iv[2];
    da_int dim;

    // handle is null
    EXPECT_EQ(da_handle_get_result_d(nullptr, da_result::da_rinfo, &dim, dv),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_handle_get_result_s(nullptr, da_result::da_rinfo, &dim, sv),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_handle_get_result_int(nullptr, da_result::da_rinfo, &dim, iv),
              da_status_handle_not_initialized);

    // handle valid but not initialized with any solver
    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_type::da_handle_uninitialized),
              da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_type::da_handle_uninitialized),
              da_status_success);

    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, &dim, dv),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_rinfo, &dim, sv),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_handle_get_result_int(handle_d, da_result::da_rinfo, &dim, iv),
              da_status_handle_not_initialized);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);

    // handle valid but no problem solved -> empty data
    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_type::da_handle_linmod),
              da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_type::da_handle_linmod),
              da_status_success);

    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, &dim, dv),
              da_status_unknown_query);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_rinfo, &dim, sv),
              da_status_unknown_query);
    // enable this test once a handle provides integer data
    // EXPECT_EQ(da_handle_get_result_int(handle_s, da_result::da_integer_type, &dim, iv),
    //           da_status_unknown_query);

    // handle valid but get_result is of different precision than handle
    EXPECT_EQ(da_handle_get_result_d(handle_s, da_result::da_rinfo, &dim, dv),
              da_status_wrong_type);
    EXPECT_EQ(da_handle_get_result_s(handle_d, da_result::da_rinfo, &dim, sv),
              da_status_wrong_type);
    // No need to test get_result_int since it cannot fail on this test.

    // handle valid but query is for a different handle group (linmod vs. pca)
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_pca_scores, &dim, dv),
              da_status_unknown_query);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_pca_scores, &dim, sv),
              da_status_unknown_query);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

} // namespace
