/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../utest_utils.hpp"
#include "aoclda.h"
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
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, 0, n, Ad, bd),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, nullptr, bd),
              da_status_invalid_pointer);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, Ad, nullptr),
              da_status_invalid_pointer);

    // Check we can handle NaN data correctly
    EXPECT_EQ(da_options_set(handle_d, "check data", "yes"), da_status_success);
    bd[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, Ad, bd),
              da_status_invalid_input);
    bd[0] = 1;

    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, Ad, bd), da_status_success);

    EXPECT_EQ(da_linmod_define_features_s(handle_s, m, 0, As, bs),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_linmod_define_features_s(handle_s, 0, n, As, bs),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_linmod_define_features_s(handle_s, m, n, nullptr, bs),
              da_status_invalid_pointer);
    EXPECT_EQ(da_linmod_define_features_s(handle_s, m, n, As, nullptr),
              da_status_invalid_pointer);
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
              da_status_invalid_pointer);
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, 1, n, X, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, 0, n, X, pred),
              da_status_invalid_array_dimension);
    float Xs[2] = {1., 2.};
    float preds[1];
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, 1, 3, Xs, preds),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, 1, n, nullptr, preds),
              da_status_invalid_pointer);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, 1, n, Xs, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, 0, n, Xs, preds),
              da_status_invalid_array_dimension);

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

    // model not yet fitted
    da_int linfo{100};
    double info[100];
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, &linfo, info),
              da_status_unknown_query);

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
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, Ad, bd), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle_d, linmod_model_logistic),
              da_status_success);

    // Solvers that should not be compatible with logistic regression
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "QR"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "cholesky"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "svd"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "coord"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "sparse_cg"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    // lbfgs with logistic 1-norm term
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "lbfgsb"),
              da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle_d, "lambda", 1.0), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle_d, "alpha", 1.0), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    // solvers incompatible with L1 linear regression
    EXPECT_EQ(da_linmod_select_model_d(handle_d, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "QR"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "cholesky"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "svd"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "lbfgs"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "sparse_cg"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    // solvers incompatible with Elastic Net linear regression
    EXPECT_EQ(da_options_set_real_d(handle_d, "alpha", 0.5), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "QR"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "cholesky"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "svd"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "lbfgs"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "sparse_cg"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    // SVD/qr intercept without scaling
    EXPECT_EQ(da_options_set_real_d(handle_d, "alpha", 0.0), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle_d, "lambda", 0.0), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "scaling", "none"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle_d, "intercept", 1), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "svd"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "qr"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    // QR solver with regularization
    EXPECT_EQ(da_options_set_real_d(handle_d, "lambda", 1.0), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "qr"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    da_handle_destroy(&handle_d);
}

TEST(linmod, wideMatrixProblems) {
    // problem data
    da_int m = 2, n = 5;
    double Ad[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double bd[2] = {1, 0};
    da_handle handle_d = nullptr;

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, Ad, bd), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle_d, linmod_model_mse), da_status_success);

    // Can't fit QR solver on underdetermined problem with intercept
    EXPECT_EQ(da_options_set_string(handle_d, "scaling", "auto"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle_d, "intercept", 1), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "qr"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    // Can't fit QR solver on underdetermined problem with standardization
    EXPECT_EQ(da_options_set_string(handle_d, "scaling", "standardize"),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle_d, "intercept", 0), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    // Can't fit svd/chol/cg/qr solver on underdetermined problem without scaling with intercept
    EXPECT_EQ(da_options_set_string(handle_d, "scaling", "none"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle_d, "intercept", 1), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "svd"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "cholesky"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "sparse_cg"),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "qr"), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_incompatible_options);

    da_handle_destroy(&handle_d);
}

TEST(linmod, singularTallMatrix) {
    // problem data
    da_int m = 5, n = 2;
    double Ad[10] = {1, 1, 1, 4, 5, 1, 1, 1, 4, 5};
    double bd[5] = {1, 1, 0, 1, 0};
    da_handle handle_d = nullptr;

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, Ad, bd), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "cholesky"),
              da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle_d, linmod_model_mse), da_status_success);

    // Cholesky factorization should not be able to compute on singular matrix
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_numerical_difficulties);

    da_handle_destroy(&handle_d);
}

TEST(linmod, singularWideMatrix) {
    // problem data
    da_int m = 2, n = 5;
    double Ad[10] = {1, 2, 2, 4, 3, 6, 4, 8, 5, 10};
    double bd[2] = {1, 0};
    da_handle handle_d = nullptr;

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, m, n, Ad, bd), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "cholesky"),
              da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle_d, linmod_model_mse), da_status_success);

    // Cholesky factorization should not be able to compute on singular matrix
    EXPECT_EQ(da_linmod_fit_d(handle_d), da_status_numerical_difficulties);

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

// Check info array copy logic
TEST(linmod, CheckGetInfo) {
    // problem data
    da_int nsamples = 5, nfeat = 2;
    double Ad[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bd[5] = {1, 1, 1, 1, 1};
    da_handle handle_d = nullptr;
    da_int linfo{100};
    double info[100];

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, nsamples, nfeat, Ad, bd),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "QR"), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle_d, linmod_model_mse), da_status_success);
    // model not trained
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, &linfo, info),
              da_status_unknown_query);
    // QR does not provide info (except loss function value and compute time)[*]
    EXPECT_EQ(da_linmod_fit<double>(handle_d), da_status_success);
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, &linfo, info),
              da_status_success);
    // so check that all (except index 0 and 3) are -1.
    for (da_int i = 1; i < 100; ++i) {
        if (i == 3)
            continue;
        EXPECT_EQ(info[i], -1.0);
    }
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "bfgs"), da_status_success);
    EXPECT_EQ(da_linmod_fit<double>(handle_d), da_status_success);

    // model fitted but wrong size of info array
    linfo = 1;
    EXPECT_EQ(da_handle_get_result_d(nullptr, da_result::da_rinfo, &linfo, nullptr),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, nullptr, info),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, &linfo, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, &linfo, info),
              da_status_invalid_array_dimension);

    da_handle_destroy(&handle_d);
}

const double safe_tol{da_numeric::tolerance<double>::safe_tol()};

TEST(linmod, ReturnLastSol) {
    // problem data
    const da_int nsamples = 5, nfeat = 2;
    double Ad[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bd[5] = {1, 1.5, 1.25, 2, 2.15};
    da_handle handle_d = nullptr;
    da_int ncoef{nfeat + 1};
    double coef[nfeat + 1];
    const double coef_exp[nfeat + 1] = {0.265625, -0.07412109375, 0.94619140625};
    const double tol{1.e4 * safe_tol};

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, nsamples, nfeat, Ad, bd),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", "Coord"),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle_d, "optim iteration limit", 2),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle_d, "print level", 0), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "scaling", "scale only"),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle_d, "intercept", 1), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle_d, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_linmod_fit<double>(handle_d), da_status_success);

    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coef, &ncoef, coef),
              da_status_success);
    // check with expected suboptimal solution
    EXPECT_ARR_NEAR(ncoef, coef, coef_exp, tol);

    da_handle_destroy(&handle_d);
}

typedef struct warmstart_params_t {
    std::string test_name;
    std::string solver;
    std::string scaling;
    double alpha;
    double lambda;
    double tol;
    da_int iter;
} warmstart_params;

const warmstart_params warmstart_values[]{
    {"Coord+Z", "coord", "standardize", 0.5, 0.05, 10.0 * safe_tol, 1},
    {"Coord+S", "coord", "scale only", 0.5, 0.05, 10.0 * safe_tol, 1},
    {"BFGS+N", "bfgs", "none", 0.0, 1.0, 1000.0 * safe_tol, 0},
    {"BFGS+C", "bfgs", "centering", 0.0, 1.0, 10.0 * safe_tol, 0},
    {"BFGS+Z", "bfgs", "standardize", 0.0, 1.0, 10000.0 * safe_tol, 0},
    {"BFGS+S", "bfgs", "scale only", 0.0, 1.0, 1000.0 * safe_tol, 0},
    {"CG+N", "sparse_cg", "none", 0.0, 1.0, 1000.0 * safe_tol, 0},
    {"CG+C", "sparse_cg", "centering", 0.0, 1.0, 10.0 * safe_tol, 0},
    {"CG+Z", "sparse_cg", "standardize", 0.0, 1.0, 10000.0 * safe_tol, 0},
    {"CG+S", "sparse_cg", "scale only", 0.0, 1.0, 1000.0 * safe_tol, 0},
};

class linmodWarmStart : public testing::TestWithParam<warmstart_params> {};

void test_linmod_warmstart(const warmstart_params);

TEST_P(linmodWarmStart, WarmStart) {
    const warmstart_params &pr = GetParam();
    test_linmod_warmstart(pr);
}

// Warm start (test scaling for initial iterate)
void test_linmod_warmstart(const warmstart_params pr) {
    // problem data
    const da_int nsamples = 5, nfeat = 4;
    double Ad[20] = {1, 2, 3, 4, 5, 2, 3, 1, 1, 3, 5, 1, 1, 2, 2, 3, 2, 3, 3, 4};
    double bd[5] = {1, 1.5, 1.25, 2, 2.15};
    da_handle handle_d = nullptr;
    da_int ncoef{nfeat + 1};
    double coef[nfeat + 1], wcoef[nfeat + 1];
    double info[100];
    da_int linfo{100};

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle_d, nsamples, nfeat, Ad, bd),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "optim method", pr.solver.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "scaling", pr.scaling.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle_d, "optim convergence tol", safe_tol),
              da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle_d, "optim progress factor", 10.0),
              da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle_d, "alpha", pr.alpha), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle_d, "lambda", pr.lambda), da_status_success);
    EXPECT_EQ(da_options_set_int(handle_d, "print level", 0), da_status_success);
    EXPECT_EQ(da_options_set_int(handle_d, "intercept", 1), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle_d, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_linmod_fit<double>(handle_d), da_status_success);
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coef, &ncoef, coef),
              da_status_success);
    // Reset model and train again from solution (set any option)
    EXPECT_EQ(da_options_set_int(handle_d, "print level", 0), da_status_success);
    EXPECT_EQ(da_linmod_fit_start<double>(handle_d, ncoef, coef), da_status_success);
    // info -> iter == 1 or O depending on method
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, &linfo, info),
              da_status_success);
    std::cout << info[0] << " " << info[1] << " " << info[2] << " " << info[3] << " "
              << info[4] << " \n";
    std::cout << da_optim_info_t::info_iter << std::endl;
    ;
    EXPECT_EQ(info[da_optim_info_t::info_iter], pr.iter);
    // compare with warm start run
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coef, &ncoef, wcoef),
              da_status_success);
    // check with expected suboptimal solution
    EXPECT_ARR_NEAR(ncoef, coef, wcoef, pr.tol);

    da_handle_destroy(&handle_d);
}

TEST(linmod, RowMajor) { // Problem data
    da_int m = 6, n = 2;
    double Al[12] = {1, 2, 3, 4, 5, 6, 1, 3, 5, 8, 7, 9};
    double bl[6] = {3., 6.5, 10., 12., 13., 19.};
    da_int nx = 2;
    double x[2];

    // Initialize the linear regression
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle, m, n, Al, bl), da_status_success);
    // Compute regression
    EXPECT_EQ(da_linmod_fit_d(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result_d(handle, da_linmod_coef, &nx, x), da_status_success);
    da_handle_destroy(&handle);

    // Now repeat in row-major
    double Al_row[12] = {1, 1, 2, 3, 3, 5, 4, 8, 5, 7, 6, 9};
    double x_row[2];
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_options_set(handle, "storage order", "row-major"), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle, m, n, Al_row, bl), da_status_success);
    // Compute regression
    EXPECT_EQ(da_linmod_fit_d(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result_d(handle, da_linmod_coef, &nx, x_row),
              da_status_success);

    EXPECT_ARR_NEAR(2, x, x_row, 10 * std::numeric_limits<double>::epsilon());

    da_handle_destroy(&handle);
}
// Teach GTest how to print the param type
// in this case use only user's unique testname
// It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
void PrintTo(const warmstart_params &param, ::std::ostream *os) {
    *os << param.test_name;
}

INSTANTIATE_TEST_SUITE_P(WarmStartSuite, linmodWarmStart,
                         testing::ValuesIn(warmstart_values));

} // namespace
