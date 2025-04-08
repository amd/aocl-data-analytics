/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../datests_cblas.hh"
#include "../utest_utils.hpp"
#include "aoclda.h"
#include "svm_positive.hpp"
#include "svm_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <list>
#include <string>

template <typename T> class svm_public_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(svm_public_test, FloatTypes);

// In all of these tests we use default C, epsilon, nu, gamma, coef0, degree
TYPED_TEST(svm_public_test, ldx_test) {
    std::function<void(test_ldx_type<TypeParam> & data)> set_test_data[] = {
        set_ldx_test_data_7x2_rbf_svc<TypeParam>,
        set_ldx_test_data_7x2_linear_svr<TypeParam>,
        set_ldx_test_data_7x2_sigmoid_nusvc<TypeParam>,
        set_ldx_test_data_7x2_poly_nusvr<TypeParam>};
    test_ldx_type<TypeParam> data;

    da_int i = 0;
    TypeParam tol = da_numeric::tolerance<TypeParam>::safe_tol();
    for (auto &data_fun : set_test_data) {
        std::cout << "Testing function: " << i << std::endl;
        std::cout << "Column major test: " << std::endl;
        data_fun(data);
        da_handle svm_handle = nullptr;
        EXPECT_EQ(da_handle_init<TypeParam>(&svm_handle, da_handle_svm),
                  da_status_success);
        EXPECT_EQ(da_options_set(svm_handle, "tolerance", (TypeParam)1e-5),
                  da_status_success);
        EXPECT_EQ(da_options_set(svm_handle, "kernel", data.kernel.c_str()),
                  da_status_success);
        EXPECT_EQ(da_svm_select_model<TypeParam>(svm_handle, data.model),
                  da_status_success);
        EXPECT_EQ(da_svm_set_data(svm_handle, data.n_samples_train, data.n_feat,
                                  data.X_train.data(), data.ldx_train,
                                  data.y_train.data()),
                  da_status_success);
        EXPECT_EQ(da_svm_compute<TypeParam>(svm_handle), da_status_success);
        if (data.model == svc || data.model == nusvc) {
            std::vector<TypeParam> decision_values_pred(data.n_class *
                                                        data.lddecision_values);
            EXPECT_EQ(da_svm_decision_function(
                          svm_handle, data.n_samples_test, data.n_feat,
                          data.X_test.data(), data.ldx_test, ovr,
                          decision_values_pred.data(), data.lddecision_values),
                      da_status_success);
            EXPECT_ARR_NEAR(data.n_samples_test * data.lddecision_values,
                            decision_values_pred, data.decision_values, tol);
        }
        TypeParam score_pred;
        EXPECT_EQ(da_svm_score(svm_handle, data.n_samples_test, data.n_feat,
                               data.X_test.data(), data.ldx_test, data.y_test.data(),
                               &score_pred),
                  da_status_success);
        EXPECT_NEAR(score_pred, data.score, tol);
        std::vector<TypeParam> y_pred(data.n_samples_test);
        EXPECT_EQ(da_svm_predict(svm_handle, data.n_samples_test, data.n_feat,
                                 data.X_test.data(), data.ldx_test, y_pred.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(data.n_samples_test, y_pred, data.y_pred, tol);

        // Check the same with row-major order
        std::cout << "Row major test: " << std::endl;
        EXPECT_EQ(da_options_set(svm_handle, "storage order", "row-major"),
                  da_status_success);

        EXPECT_EQ(da_svm_set_data(svm_handle, data.n_samples_train, data.n_feat,
                                  data.X_train_row.data(), data.ldx_train_row,
                                  data.y_train.data()),
                  da_status_success);
        EXPECT_EQ(da_svm_compute<TypeParam>(svm_handle), da_status_success);
        if (data.model == svc || data.model == nusvc) {
            std::vector<TypeParam> decision_values_pred(data.n_samples_test *
                                                        data.lddecision_values_row);
            EXPECT_EQ(da_svm_decision_function(
                          svm_handle, data.n_samples_test, data.n_feat,
                          data.X_test_row.data(), data.ldx_test_row, ovr,
                          decision_values_pred.data(), data.lddecision_values_row),
                      da_status_success);
            EXPECT_ARR_NEAR(data.n_feat * data.lddecision_values_row,
                            decision_values_pred, data.decision_values_row, tol);
        }
        EXPECT_EQ(da_svm_score(svm_handle, data.n_samples_test, data.n_feat,
                               data.X_test_row.data(), data.ldx_test_row,
                               data.y_test.data(), &score_pred),
                  da_status_success);
        EXPECT_NEAR(score_pred, data.score, tol);
        EXPECT_EQ(da_svm_predict(svm_handle, data.n_samples_test, data.n_feat,
                                 data.X_test_row.data(), data.ldx_test_row,
                                 y_pred.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(data.n_samples_test, y_pred, data.y_pred, tol);

        da_handle_destroy(&svm_handle);
        i++;
    }
}

TYPED_TEST(svm_public_test, get_results_test) {

    std::function<void(test_get_results_type<TypeParam> & data)> set_test_data[] = {
        set_get_results_test_data_7x2_rbf_svc<TypeParam>,
        set_get_results_test_data_7x2_linear_svr<TypeParam>,
        set_get_results_test_data_7x2_sigmoid_nusvc<TypeParam>,
        set_get_results_test_data_7x2_poly_nusvr<TypeParam>};
    test_get_results_type<TypeParam> data;

    TypeParam tol = 3e-5;
    for (auto &data_fun : set_test_data) {
        data_fun(data);
        da_handle svm_handle = nullptr;
        EXPECT_EQ(da_handle_init<TypeParam>(&svm_handle, da_handle_svm),
                  da_status_success);
        EXPECT_EQ(da_svm_select_model<TypeParam>(svm_handle, data.model),
                  da_status_success);
        EXPECT_EQ(da_svm_set_data(svm_handle, data.n_samples, data.n_feat,
                                  data.X_train.data(), data.n_samples,
                                  data.y_train.data()),
                  da_status_success);
        EXPECT_EQ(da_options_set(svm_handle, "kernel", data.kernel.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set(svm_handle, "tolerance", (TypeParam)1e-5),
                  da_status_success);
        EXPECT_EQ(da_svm_compute<TypeParam>(svm_handle), da_status_success);

        ////////// COLUMN MAJOR
        // Get the rinfo results and check the values
        da_int dim = 100;
        std::vector<TypeParam> rinfo(dim);
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_rinfo, &dim, rinfo.data()),
            da_status_success);
        std::vector<TypeParam> rinfo_exp{
            (TypeParam)data.n_samples,
            (TypeParam)data.n_feat,
            (TypeParam)data.n_class,
        };
        EXPECT_ARR_NEAR(3, rinfo, rinfo_exp, 1.0e-10);

        // Get the n_sv_per_class and check the values
        dim = data.n_support_per_class_expected.size();
        std::vector<da_int> n_sv_per_class(dim);
        EXPECT_EQ(da_handle_get_result(svm_handle,
                                       da_result::da_svm_n_support_vectors_per_class,
                                       &dim, n_sv_per_class.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, n_sv_per_class, data.n_support_per_class_expected, 1.0e-10);

        // Get the n_sv and check the values
        da_int n_sv;
        dim = 1;
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_n_support_vectors,
                                       &dim, &n_sv),
                  da_status_success);
        EXPECT_EQ(n_sv, data.n_support_expected);

        // Get the support indexes and check the values
        dim = data.support_indexes_expected.size();
        std::vector<da_int> support_indexes(dim);
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_idx_support_vectors,
                                       &dim, support_indexes.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, support_indexes, data.support_indexes_expected, 1e-10);

        // Get the bias and check the values
        dim = data.bias_expected.size();
        std::vector<TypeParam> bias(dim);
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_svm_bias, &dim, bias.data()),
            da_status_success);
        EXPECT_ARR_NEAR(dim, bias, data.bias_expected, tol);

        // Get the n_iterations and check the values
        dim = data.bias_expected.size();
        std::vector<da_int> n_iterations(dim);
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_n_iterations, &dim,
                                       n_iterations.data()),
                  da_status_success);
        for (auto iter : n_iterations) {
            EXPECT_GT(iter, 4);
        }

        // Get the support vectors and check the values
        dim = data.support_vectors_expected.size();
        std::vector<TypeParam> support_vectors(dim);
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_support_vectors,
                                       &dim, support_vectors.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, support_vectors, data.support_vectors_expected, 1e-10);

        // Get the dual coefs and check the values
        dim = data.support_coefficients_expected.size();
        std::vector<TypeParam> support_coeff(dim);
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_dual_coef, &dim,
                                       support_coeff.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, support_coeff, data.support_coefficients_expected, tol);

        ////////// ROW MAJOR
        EXPECT_EQ(da_options_set(svm_handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_svm_set_data(svm_handle, data.n_samples, data.n_feat,
                                  data.X_train_row.data(), data.n_feat,
                                  data.y_train.data()),
                  da_status_success);
        EXPECT_EQ(da_svm_compute<TypeParam>(svm_handle), da_status_success);
        // Get the rinfo results and check the values
        dim = 100;
        std::vector<TypeParam> rinfo_row(dim);
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_rinfo, &dim, rinfo_row.data()),
            da_status_success);
        EXPECT_ARR_NEAR(3, rinfo_row, rinfo_exp, 1.0e-10);

        // Get the n_sv_per_class and check the values
        dim = data.n_support_per_class_expected.size();
        EXPECT_EQ(da_handle_get_result(svm_handle,
                                       da_result::da_svm_n_support_vectors_per_class,
                                       &dim, n_sv_per_class.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, n_sv_per_class, data.n_support_per_class_expected, 1.0e-10);

        // Get the n_sv and check the values
        dim = 1;
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_n_support_vectors,
                                       &dim, &n_sv),
                  da_status_success);
        EXPECT_EQ(n_sv, data.n_support_expected);

        // Get the support indexes and check the values
        dim = data.support_indexes_expected.size();
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_idx_support_vectors,
                                       &dim, support_indexes.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, support_indexes, data.support_indexes_expected, 1e-10);

        // Get the bias and check the values
        dim = data.bias_expected.size();
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_svm_bias, &dim, bias.data()),
            da_status_success);
        EXPECT_ARR_NEAR(dim, bias, data.bias_expected, tol);

        // Get the n_iterations and check the values
        dim = data.bias_expected.size();
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_n_iterations, &dim,
                                       n_iterations.data()),
                  da_status_success);
        for (auto iter : n_iterations) {
            EXPECT_GT(iter, 4);
        }

        // Get the support vectors and check the values
        dim = data.support_vectors_row_expected.size();
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_support_vectors,
                                       &dim, support_vectors.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, support_vectors, data.support_vectors_row_expected, 1e-10);

        // Get the dual coefs and check the values
        dim = data.support_coefficients_row_expected.size();
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_dual_coef, &dim,
                                       support_coeff.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, support_coeff, data.support_coefficients_row_expected, tol);

        ////////// FAIL EXITS
        // Check that querying other algorithm fails
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_linmod_coef, &dim,
                                       rinfo.data()),
                  da_status_unknown_query);
        // int variant
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_linmod_coef, &dim, &n_sv),
            da_status_unknown_query);
        // Check the wrong dimension
        dim = 0;
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_rinfo, &dim, rinfo.data()),
            da_status_invalid_array_dimension);
        EXPECT_EQ(dim, 100);
        dim = 0;
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_dual_coef, &dim,
                                       rinfo.data()),
                  da_status_invalid_array_dimension);
        EXPECT_EQ(dim, data.support_coefficients_expected.size());
        dim = 0;
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_support_vectors,
                                       &dim, rinfo.data()),
                  da_status_invalid_array_dimension);
        EXPECT_EQ(dim, data.support_vectors_expected.size());
        dim = 0;
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_svm_bias, &dim, rinfo.data()),
            da_status_invalid_array_dimension);
        EXPECT_EQ(dim, data.bias_expected.size());
        // int variants
        dim = 0;
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_n_support_vectors,
                                       &dim, &n_sv),
                  da_status_invalid_array_dimension);
        EXPECT_EQ(dim, 1);
        dim = 0;
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_svm_n_iterations, &dim, &n_sv),
            da_status_invalid_array_dimension);
        EXPECT_EQ(
            dim,
            data.bias_expected.size()); // n_iterations array is same length as bias array
        dim = 0;
        EXPECT_EQ(da_handle_get_result(svm_handle,
                                       da_result::da_svm_n_support_vectors_per_class,
                                       &dim, &n_sv),
                  da_status_invalid_array_dimension);
        EXPECT_EQ(dim, data.n_support_per_class_expected.size());
        dim = 0;
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_idx_support_vectors,
                                       &dim, &n_sv),
                  da_status_invalid_array_dimension);
        EXPECT_EQ(dim, data.support_indexes_expected.size());

        // Change an option and check that results are no longer available
        EXPECT_EQ(da_options_set(svm_handle, "epsilon", TypeParam(0.2)),
                  da_status_success);
        dim = 100;
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_rinfo, &dim, rinfo.data()),
            da_status_unknown_query);
        // int variant
        dim = 1;
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_n_support_vectors,
                                       &dim, &n_sv),
                  da_status_unknown_query);

        da_handle_destroy(&svm_handle);
    }
}

TYPED_TEST(svm_public_test, row_major_test) {
    std::function<void(test_row_major_type<TypeParam> & data)> set_test_data[] = {
        set_row_major_test_data_15x2_poly_svc<TypeParam>,
        set_row_major_test_data_15x2_sigmoid_svr<TypeParam>,
        set_row_major_test_data_15x2_rbf_nusvc<TypeParam>,
        set_row_major_test_data_15x2_linear_nusvr<TypeParam>};
    test_row_major_type<TypeParam> data;

    TypeParam tol = 5e-3;
    for (auto &data_fun : set_test_data) {
        data_fun(data);
        da_handle svm_handle = nullptr;
        EXPECT_EQ(da_handle_init<TypeParam>(&svm_handle, da_handle_svm),
                  da_status_success);
        EXPECT_EQ(da_svm_select_model<TypeParam>(svm_handle, data.model),
                  da_status_success);
        // This needs to be set before set_data()
        EXPECT_EQ(da_options_set(svm_handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_svm_set_data(svm_handle, data.n_samples, data.n_feat,
                                  data.X_train.data(), data.n_feat, data.y_train.data()),
                  da_status_success);
        EXPECT_EQ(da_options_set(svm_handle, "kernel", data.kernel.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set(svm_handle, "tolerance", TypeParam(1e-5)),
                  da_status_success);
        EXPECT_EQ(da_svm_compute<TypeParam>(svm_handle), da_status_success);
        std::vector<TypeParam> pred(data.n_samples_test);
        EXPECT_EQ(da_svm_predict(svm_handle, data.n_samples_test, data.n_feat_test,
                                 data.X_test.data(), data.n_feat_test, pred.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(data.n_samples_test, pred, data.y_pred, tol);
        TypeParam score;
        EXPECT_EQ(da_svm_score(svm_handle, data.n_samples_test, data.n_feat_test,
                               data.X_test.data(), data.n_feat_test, data.y_test.data(),
                               &score),
                  da_status_success);
        EXPECT_NEAR(score, data.score, tol);
        // For classification, also check decision function
        if (data.model == svc || data.model == nusvc) {
            std::vector<TypeParam> decision_function_ovr(data.decision_values_ovr.size());
            EXPECT_EQ(da_svm_decision_function(
                          svm_handle, data.n_samples_test, data.n_feat_test,
                          data.X_test.data(), data.n_feat_test, ovr,
                          decision_function_ovr.data(), data.n_class),
                      da_status_success);
            EXPECT_ARR_NEAR((da_int)decision_function_ovr.size(), decision_function_ovr,
                            data.decision_values_ovr, tol);
            da_int n_classifiers = data.n_class * (data.n_class - 1) / 2;
            std::vector<TypeParam> decision_function_ovo(data.decision_values_ovo.size());
            EXPECT_EQ(da_svm_decision_function(
                          svm_handle, data.n_samples_test, data.n_feat_test,
                          data.X_test.data(), data.n_feat_test, ovo,
                          decision_function_ovo.data(), n_classifiers),
                      da_status_success);
            EXPECT_ARR_NEAR((da_int)decision_function_ovo.size(), decision_function_ovo,
                            data.decision_values_ovo, tol);
        }

        da_handle_destroy(&svm_handle);
    }
}

TYPED_TEST(svm_public_test, multiple_calls) {
    // Check we can repeatedly call compute etc with the same single handle

    // Get some data to use
    std::function<void(test_row_major_type<TypeParam> & data)> set_test_data[] = {
        set_row_major_test_data_15x2_poly_svc<TypeParam>,
        set_row_major_test_data_15x2_sigmoid_svr<TypeParam>,
        set_row_major_test_data_15x2_rbf_nusvc<TypeParam>,
        set_row_major_test_data_15x2_linear_nusvr<TypeParam>};
    test_row_major_type<TypeParam> data;

    TypeParam tol = 5e-3;
    da_handle svm_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&svm_handle, da_handle_svm), da_status_success);
    EXPECT_EQ(da_options_set(svm_handle, "storage order", "row-major"),
              da_status_success);

    for (auto &data_fun : set_test_data) {
        data_fun(data);
        EXPECT_EQ(da_svm_select_model<TypeParam>(svm_handle, data.model),
                  da_status_success);
        EXPECT_EQ(da_svm_set_data(svm_handle, data.n_samples, data.n_feat,
                                  data.X_train.data(), data.n_feat, data.y_train.data()),
                  da_status_success);

        EXPECT_EQ(da_options_set(svm_handle, "kernel", data.kernel.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set(svm_handle, "tolerance", TypeParam(1e-5)),
                  da_status_success);

        EXPECT_EQ(da_svm_compute<TypeParam>(svm_handle), da_status_success);

        std::vector<TypeParam> pred(data.n_samples_test);
        EXPECT_EQ(da_svm_predict(svm_handle, data.n_samples_test, data.n_feat_test,
                                 data.X_test.data(), data.n_feat_test, pred.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(data.n_samples_test, pred, data.y_pred, tol);
        TypeParam score;
        EXPECT_EQ(da_svm_score(svm_handle, data.n_samples_test, data.n_feat_test,
                               data.X_test.data(), data.n_feat_test, data.y_test.data(),
                               &score),
                  da_status_success);
        EXPECT_NEAR(score, data.score, tol);
        // For classification, also check decision function
        if (data.model == svc || data.model == nusvc) {
            std::vector<TypeParam> decision_function_ovr(data.decision_values_ovr.size());
            EXPECT_EQ(da_svm_decision_function(
                          svm_handle, data.n_samples_test, data.n_feat_test,
                          data.X_test.data(), data.n_feat_test, ovr,
                          decision_function_ovr.data(), data.n_class),
                      da_status_success);
            EXPECT_ARR_NEAR((da_int)data.decision_values_ovr.size(),
                            decision_function_ovr, data.decision_values_ovr, tol);
        }

        da_int dim = 100;
        std::vector<TypeParam> rinfo(dim);
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_rinfo, &dim, rinfo.data()),
            da_status_success);
        std::vector<TypeParam> rinfo_exp{
            (TypeParam)data.n_samples,
            (TypeParam)data.n_feat,
            (TypeParam)data.n_class,
        };
        EXPECT_ARR_NEAR(3, rinfo, rinfo_exp, 1.0e-10);

        // Get the n_sv_per_class and check the values
        dim = data.n_support_per_class_expected.size();
        std::vector<da_int> n_sv_per_class(dim);
        EXPECT_EQ(da_handle_get_result(svm_handle,
                                       da_result::da_svm_n_support_vectors_per_class,
                                       &dim, n_sv_per_class.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, n_sv_per_class, data.n_support_per_class_expected, 1.0e-10);

        // Get the n_sv and check the values
        da_int n_sv;
        dim = 1;
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_n_support_vectors,
                                       &dim, &n_sv),
                  da_status_success);
        EXPECT_EQ(n_sv, data.n_support_expected);

        // Get the support indexes and check the values
        dim = data.support_indexes_expected.size();
        std::vector<da_int> support_indexes(dim);
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_idx_support_vectors,
                                       &dim, support_indexes.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, support_indexes, data.support_indexes_expected, 1e-10);

        // Get the bias and check the values
        dim = data.bias_expected.size();
        std::vector<TypeParam> bias(dim);
        EXPECT_EQ(
            da_handle_get_result(svm_handle, da_result::da_svm_bias, &dim, bias.data()),
            da_status_success);
        EXPECT_ARR_NEAR(dim, bias, data.bias_expected, tol);

        // Get the support vectors and check the values
        dim = data.support_vectors_expected.size();
        std::vector<TypeParam> support_vectors(dim);
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_support_vectors,
                                       &dim, support_vectors.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, support_vectors, data.support_vectors_expected, 1e-10);

        // Get the dual coefs and check the values
        dim = data.support_coefficients_expected.size();
        std::vector<TypeParam> support_coeff(dim);
        EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_dual_coef, &dim,
                                       support_coeff.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(dim, support_coeff, data.support_coefficients_expected, tol);
    }

    da_handle_destroy(&svm_handle);
}

TYPED_TEST(svm_public_test, invalid_input) {

    std::vector<TypeParam> X{0.0, 1.0, 0.0, 2.0};
    std::vector<TypeParam> y{0.0, 1.0};
    da_int n_samples = 2, n_features = 2;
    TypeParam score;

    da_handle svm_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&svm_handle, da_handle_svm), da_status_success);

    // select_model
    EXPECT_EQ(da_svm_select_model<TypeParam>(nullptr, svc),
              da_status_handle_not_initialized);
    // EXPECT_EQ(da_svm_select_model<TypeParam>(svm_handle, 5), da_status_unknown_query); // Won't compile. Why?

    // set_data
    // Correct input but trying to set data before picking a model
    EXPECT_EQ(
        da_svm_set_data(svm_handle, n_samples, n_features, X.data(), n_samples, y.data()),
        da_status_unknown_query);
    EXPECT_EQ(da_svm_select_model<TypeParam>(svm_handle, svc), da_status_success);
    // Invalid pointers
    TypeParam *X_invalid = nullptr;
    TypeParam *y_invalid = nullptr;
    EXPECT_EQ(da_svm_set_data(svm_handle, n_samples, n_features, X_invalid, n_samples,
                              y.data()),
              da_status_invalid_pointer);
    EXPECT_EQ(da_svm_set_data(svm_handle, n_samples, n_features, X.data(), n_samples,
                              y_invalid),
              da_status_invalid_pointer);
    EXPECT_EQ(
        da_svm_set_data(nullptr, n_samples, n_features, X_invalid, n_samples, y.data()),
        da_status_handle_not_initialized);
    // wrong dimensions
    EXPECT_EQ(da_svm_set_data(svm_handle, 0, n_features, X_invalid, n_samples, y.data()),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_svm_set_data(svm_handle, n_samples, 0, X_invalid, n_samples, y.data()),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_svm_set_data(svm_handle, n_samples, n_features, X_invalid, 1, y.data()),
              da_status_invalid_pointer);

    // Model out of date for evaluation
    EXPECT_EQ(da_svm_compute<TypeParam>(svm_handle),
              da_status_no_data); // Compute without succesful set_data()
    EXPECT_EQ(
        da_svm_set_data(svm_handle, n_samples, n_features, X.data(), n_samples, y.data()),
        da_status_success);
    EXPECT_EQ(
        da_svm_predict(svm_handle, n_samples, n_features, X.data(), n_samples, y.data()),
        da_status_out_of_date); // Predict without succesful compute()
    EXPECT_EQ(da_svm_score(svm_handle, n_samples, n_features, X.data(), n_samples,
                           y.data(), &score),
              da_status_out_of_date); // Score without succesful compute()
    EXPECT_EQ(da_svm_decision_function(svm_handle, n_samples, n_features, X.data(),
                                       n_samples, ovr, y.data(), n_samples),
              da_status_out_of_date); // Dec func without succesful compute()
    EXPECT_EQ(da_svm_compute<TypeParam>(nullptr), da_status_handle_not_initialized);
    EXPECT_EQ(da_svm_compute<TypeParam>(svm_handle), da_status_success);

    // Predict
    // Invalid pointers
    EXPECT_EQ(
        da_svm_predict(svm_handle, n_samples, n_features, X_invalid, n_samples, y.data()),
        da_status_invalid_pointer);
    EXPECT_EQ(
        da_svm_predict(svm_handle, n_samples, n_features, X.data(), n_samples, y_invalid),
        da_status_invalid_pointer);
    EXPECT_EQ(
        da_svm_predict(nullptr, n_samples, n_features, X.data(), n_samples, y.data()),
        da_status_handle_not_initialized);
    // Wrong dimensions
    EXPECT_EQ(da_svm_predict(svm_handle, 0, n_features, X.data(), n_samples, y.data()),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_svm_predict(svm_handle, n_samples, 0, X.data(), n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_svm_predict(svm_handle, n_samples, 4, X.data(), n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_svm_predict(svm_handle, n_samples, n_features, X.data(), 1, y.data()),
              da_status_invalid_leading_dimension);

    // Score
    // Invalid pointers
    EXPECT_EQ(da_svm_score(svm_handle, n_samples, n_features, X_invalid, n_samples,
                           y.data(), &score),
              da_status_invalid_pointer);
    EXPECT_EQ(da_svm_score(svm_handle, n_samples, n_features, X.data(), n_samples,
                           y_invalid, &score),
              da_status_invalid_pointer);
    EXPECT_EQ(da_svm_score(svm_handle, n_samples, n_features, X.data(), n_samples,
                           y.data(), nullptr),
              da_status_invalid_pointer);
    EXPECT_EQ(da_svm_score(nullptr, n_samples, n_features, X.data(), n_samples, y.data(),
                           &score),
              da_status_handle_not_initialized);
    // Wrong dimensions
    EXPECT_EQ(
        da_svm_score(svm_handle, 0, n_features, X.data(), n_samples, y.data(), &score),
        da_status_invalid_array_dimension);
    EXPECT_EQ(
        da_svm_score(svm_handle, n_samples, 0, X.data(), n_samples, y.data(), &score),
        da_status_invalid_input);
    EXPECT_EQ(
        da_svm_score(svm_handle, n_samples, 4, X.data(), n_samples, y.data(), &score),
        da_status_invalid_input);
    EXPECT_EQ(
        da_svm_score(svm_handle, n_samples, n_features, X.data(), 1, y.data(), &score),
        da_status_invalid_leading_dimension);

    // Decision function
    // Invalid pointers
    EXPECT_EQ(da_svm_decision_function(svm_handle, n_samples, n_features, X_invalid,
                                       n_samples, ovr, y.data(), n_samples),
              da_status_invalid_pointer);
    EXPECT_EQ(da_svm_decision_function(svm_handle, n_samples, n_features, X.data(),
                                       n_samples, ovr, y_invalid, n_samples),
              da_status_invalid_pointer);
    EXPECT_EQ(da_svm_decision_function(nullptr, n_samples, n_features, X.data(),
                                       n_samples, ovr, y.data(), n_samples),
              da_status_handle_not_initialized);
    // Wrong dimensions
    EXPECT_EQ(da_svm_decision_function(svm_handle, 0, n_features, X.data(), n_samples,
                                       ovr, y.data(), n_samples),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_svm_decision_function(svm_handle, n_samples, 0, X.data(), n_samples, ovr,
                                       y.data(), n_samples),
              da_status_invalid_input);
    EXPECT_EQ(da_svm_decision_function(svm_handle, n_samples, 4, X.data(), n_samples, ovr,
                                       y.data(), n_samples),
              da_status_invalid_input);
    EXPECT_EQ(da_svm_decision_function(svm_handle, n_samples, n_features, X.data(), 1,
                                       ovr, y.data(), n_samples),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_svm_decision_function(svm_handle, n_samples, n_features, X.data(),
                                       n_samples, ovr, y.data(), 1),
              da_status_invalid_leading_dimension);

    da_handle_destroy(&svm_handle);
}

TYPED_TEST(svm_public_test, invalid_data) {
    // Get some data to use
    std::function<void(test_invalid_data_type<TypeParam> & data)> set_test_data[] = {
        set_invalid_data_y_zeros<TypeParam>,
        set_invalid_data_y_twos<TypeParam>,
        set_invalid_data_y_twos_regr<TypeParam>,
        set_invalid_data_y_missing_class<TypeParam>,
        set_invalid_data_y_negative<TypeParam>,
        set_invalid_data_y_not_whole<TypeParam>,
        set_invalid_data_X_small<TypeParam>,
        set_invalid_data_X_small_regr<TypeParam>,
        set_invalid_data_X_zeros<TypeParam>};
    test_invalid_data_type<TypeParam> data;

    for (auto &data_fun : set_test_data) {
        data_fun(data);
        for (auto &model : data.model) {
            da_handle svm_handle = nullptr;
            EXPECT_EQ(da_handle_init<TypeParam>(&svm_handle, da_handle_svm),
                      da_status_success);
            EXPECT_EQ(da_options_set(svm_handle, "kernel", data.kernel.c_str()),
                      da_status_success);
            EXPECT_EQ(da_svm_select_model<TypeParam>(svm_handle, model),
                      da_status_success);
            EXPECT_EQ(da_svm_set_data(svm_handle, data.n_samples, data.n_feat,
                                      data.X_train.data(), data.n_samples,
                                      data.y_train.data()),
                      data.set_data_expected_status);
            EXPECT_EQ(da_svm_compute<TypeParam>(svm_handle),
                      data.compute_expected_status);
            EXPECT_EQ(da_svm_predict(svm_handle, data.n_samples, data.n_feat,
                                     data.X_train.data(), data.n_samples,
                                     data.y_train.data()),
                      data.predict_expected_status);
            da_handle_destroy(&svm_handle);
        }
    }
}

TYPED_TEST(svm_public_test, bad_handle_tests) {

    // handle not initialized
    da_handle handle = nullptr;
    TypeParam A = 1;
    TypeParam labels = 1;

    EXPECT_EQ(da_svm_select_model<TypeParam>(handle, svc),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_svm_set_data(handle, 1, 1, &A, 1, &labels),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_svm_compute<TypeParam>(handle), da_status_handle_not_initialized);
    EXPECT_EQ(da_svm_predict(handle, 1, 1, &A, 1, &labels),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_svm_decision_function(handle, 1, 1, &A, 1, ovr, &labels, 1),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_svm_score(handle, 1, 1, &A, 1, &labels, &labels),
              da_status_handle_not_initialized);

    // Incorrect handle type
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_linmod), da_status_success);

    EXPECT_EQ(da_svm_select_model<TypeParam>(handle, svc), da_status_invalid_handle_type);
    EXPECT_EQ(da_svm_set_data(handle, 1, 1, &A, 1, &labels),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_svm_compute<TypeParam>(handle), da_status_invalid_handle_type);
    EXPECT_EQ(da_svm_predict(handle, 1, 1, &A, 1, &labels),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_svm_decision_function(handle, 1, 1, &A, 1, ovr, &labels, 1),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_svm_score(handle, 1, 1, &A, 1, &labels, &labels),
              da_status_invalid_handle_type);

    da_handle_destroy(&handle);
}

TEST(svm_public_test, incorrect_handle_precision) {

    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_svm), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_svm), da_status_success);

    da_int n_samples = 0, n_features = 0;
    std::vector<double> X_d{0.0};
    std::vector<float> X_s{0.0};
    std::vector<double> y_d{0.0};
    std::vector<float> y_s{0.0};
    double accuracy_d = 0.0;
    float accuracy_s = 0.0;

    // incorrect handle precision
    EXPECT_EQ(da_svm_set_data_s(handle_d, n_samples, n_features, X_s.data(), n_samples,
                                y_s.data()),
              da_status_wrong_type);
    EXPECT_EQ(da_svm_set_data_d(handle_s, n_samples, n_features, X_d.data(), n_samples,
                                y_d.data()),
              da_status_wrong_type);

    EXPECT_EQ(da_svm_compute_s(handle_d), da_status_wrong_type);
    EXPECT_EQ(da_svm_compute_d(handle_s), da_status_wrong_type);

    EXPECT_EQ(da_svm_predict_s(handle_d, n_samples, n_features, X_s.data(), n_samples,
                               y_s.data()),
              da_status_wrong_type);
    EXPECT_EQ(da_svm_predict_d(handle_s, n_samples, n_features, X_d.data(), n_samples,
                               y_d.data()),
              da_status_wrong_type);

    EXPECT_EQ(da_svm_decision_function_s(handle_d, n_samples, n_features, X_s.data(),
                                         n_samples, ovr, y_s.data(), n_samples),
              da_status_wrong_type);
    EXPECT_EQ(da_svm_decision_function_d(handle_s, n_samples, n_features, X_d.data(),
                                         n_samples, ovr, y_d.data(), n_samples),
              da_status_wrong_type);

    EXPECT_EQ(da_svm_score_s(handle_d, n_samples, n_features, X_s.data(), n_samples,
                             y_s.data(), &accuracy_s),
              da_status_wrong_type);
    EXPECT_EQ(da_svm_score_d(handle_s, n_samples, n_features, X_d.data(), n_samples,
                             y_d.data(), &accuracy_d),
              da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

/***********************************
 ********* Positive tests***********
 ***********************************/
typedef struct svm_param_t {
    std::string test_name; // name of the ctest test
    std::string data_name; // name of the files to read in
    da_svm_model model;    // SVM problem to solve
    std::vector<option_t<da_int>> iopts;
    std::vector<option_t<std::string>> sopts;
    std::vector<option_t<float>> fopts;
    std::vector<option_t<double>> dopts;
    float target_score;
    // scale to pass to expected_precision<T>(T scale=1.0)
    float check_tol_scale{1.0f};
} svm_param_t;

// clang-format off
// Testing dual coefficients, decision function values (in ovr shape) (only classification), predictions and score 
const svm_param_t svm_param_pos[] = {
    // CLASSIFICATION
    // SVC
    {"svc_binary_random_tall_rbf", "binary_random_tall", da_svm_model::svc, {}, {}, {{"tolerance", 1e-4f}, {"C", 0.5f}, {"gamma", -1.0f}}, {{"tolerance", 1e-8}, {"C", 0.5}, {"gamma", -1.0}}, 0.86},
    {"svc_binary_random_tall_linear", "binary_random_tall", da_svm_model::svc, {}, {{"kernel", "linear"}}, {{"tolerance", 1e-4f}, {"C", 0.5f}}, {{"tolerance", 1e-8}, {"C", 0.5}}, 0.8},
    {"svc_binary_random_tall_polynomial", "binary_random_tall", da_svm_model::svc, {{"degree", 2}}, {{"kernel", "poly"}}, {{"tolerance", 1e-4f}, {"C", 0.5f}, {"gamma", -1.0f}, {"coef0", 0.78f}}, {{"tolerance", 1e-8}, {"C", 0.5}, {"gamma", -1.0}, {"coef0", 0.78}}, 0.86},
    {"svc_binary_random_tall_sigmoid", "binary_random_tall", da_svm_model::svc, {}, {{"kernel", "sigmoid"}}, {{"tolerance", 1e-4f}, {"C", 0.5f}, {"gamma", -1.0f}, {"coef0", 0.78f}}, {{"tolerance", 1e-8}, {"C", 0.5}, {"gamma", -1.0}, {"coef0", 0.78}}, 0.73},
    
    {"svc_binary_random_wide_rbf", "binary_random_wide", da_svm_model::svc, {}, {{"kernel", "rbf"}}, {{"tolerance", 1e-4f}, {"C", 1.5f}, {"gamma", 0.5f}}, {{"tolerance", 1e-8}, {"C", 1.5}, {"gamma", 0.5}}, 0.416},
    {"svc_binary_random_wide_linear", "binary_random_wide", da_svm_model::svc, {}, {{"kernel", "linear"}}, {{"tolerance", 1e-4f}, {"C", 1.5f}}, {{"tolerance", 1e-8}, {"C", 1.5}}, 0.416},
    {"svc_binary_random_wide_polynomial", "binary_random_wide", da_svm_model::svc, {{"degree", 3}}, {{"kernel", "poly"}}, {{"tolerance", 1e-4f}, {"C", 1.5f}, {"gamma", 0.5f}, {"coef0", 1.78f}}, {{"tolerance", 1e-8}, {"C", 1.5}, {"gamma", 0.5}, {"coef0", 1.78}}, 0.33},
    {"svc_binary_random_wide_sigmoid", "binary_random_wide", da_svm_model::svc, {}, {{"kernel", "sigmoid"}}, {{"tolerance", 1e-4f}, {"C", 1.5f}, {"gamma", 0.5f}, {"coef0", 1.78f}}, {{"tolerance", 1e-8}, {"C", 1.5}, {"gamma", 0.5}, {"coef0", 1.78}}, 0.5},
    
    {"svc_multiclass_random_tall_rbf", "multiclass_random_tall", da_svm_model::svc, {}, {}, {{"tolerance", 1e-4f}, {"C", 0.5f}, {"gamma", 0.9f}}, {{"tolerance", 1e-8}, {"C", 0.5}, {"gamma", 0.9}}, 0.133},
    {"svc_multiclass_random_tall_linear", "multiclass_random_tall", da_svm_model::svc, {}, {{"kernel", "linear"}}, {{"tolerance", 1e-4f}, {"C", 0.5f}}, {{"tolerance", 1e-8}, {"C", 0.5}}, 0.533, 2},
    {"svc_multiclass_random_tall_polynomial", "multiclass_random_tall", da_svm_model::svc, {{"degree", 2}}, {{"kernel", "poly"}}, {{"tolerance", 1e-4f}, {"C", 0.5f}, {"gamma", 0.9f}, {"coef0", 2.0f}}, {{"tolerance", 1e-8}, {"C", 0.5}, {"gamma", 0.9}, {"coef0", 2.0}}, 0.433},
    {"svc_multiclass_random_tall_sigmoid", "multiclass_random_tall", da_svm_model::svc, {}, {{"kernel", "sigmoid"}}, {{"tolerance", 1e-4f}, {"C", 0.5f}, {"gamma", 0.9f}, {"coef0", 2.0f}}, {{"tolerance", 1e-8}, {"C", 0.5}, {"gamma", 0.9}, {"coef0", 2.0}}, 0.4},
    
    // nuSVC
    {"nusvc_binary_random_tall_rbf", "binary_random_tall", da_svm_model::nusvc, {}, {{"kernel", "rbf"}}, {{"tolerance", 1e-4f}, {"nu", 0.4f}, {"gamma", 1.0f}}, {{"tolerance", 1e-8}, {"nu", 0.4}, {"gamma", 1.0}}, 0.66, 2},
    {"nusvc_binary_random_tall_linear", "binary_random_tall", da_svm_model::nusvc, {}, {{"kernel", "linear"}}, {{"tolerance", 1e-4f}, {"nu", 0.4f}}, {{"tolerance", 1e-8}, {"nu", 0.4}}, 0.8},
    {"nusvc_binary_random_tall_polynomial", "binary_random_tall", da_svm_model::nusvc, {{"degree", 4}}, {{"kernel", "poly"}}, {{"tolerance", 1e-2f}, {"nu", 0.4f}, {"gamma", 1.0f}, {"coef0", 2.0f}}, {{"tolerance", 1e-8}, {"nu", 0.4}, {"gamma", 1.0}, {"coef0", 2.0}}, 0.86},
    {"nusvc_binary_random_tall_sigmoid", "binary_random_tall", da_svm_model::nusvc, {}, {{"kernel", "sigmoid"}}, {{"tolerance", 1e-4f}, {"nu", 0.4f}, {"gamma", 1.0f}, {"coef0", 2.0f}}, {{"tolerance", 1e-8}, {"nu", 0.4}, {"gamma", 1.0}, {"coef0", 2.0}}, 0.86},
    
    {"nusvc_binary_random_wide_rbf", "binary_random_wide", da_svm_model::nusvc, {}, {}, {{"tolerance", 1e-4f}, {"nu", 0.75f}, {"gamma", -1.0f}}, {{"tolerance", 1e-8}, {"nu", 0.75}, {"gamma", -1.0}}, 0.416},
    {"nusvc_binary_random_wide_linear", "binary_random_wide", da_svm_model::nusvc, {}, {{"kernel", "linear"}}, {{"tolerance", 1e-4f}, {"nu", 0.75f}}, {{"tolerance", 1e-8}, {"nu", 0.75}}, 0.416},
    {"nusvc_binary_random_wide_polynomial", "binary_random_wide", da_svm_model::nusvc, {{"degree", 3}}, {{"kernel", "poly"}}, {{"tolerance", 1e-4f}, {"nu", 0.75f}, {"gamma", -1.0f}, {"coef0", 0.2f}}, {{"tolerance", 1e-8}, {"nu", 0.75}, {"gamma", -1.0}, {"coef0", 0.2}}, 0.416},
    {"nusvc_binary_random_wide_sigmoid", "binary_random_wide", da_svm_model::nusvc, {}, {{"kernel", "sigmoid"}}, {{"tolerance", 1e-4f}, {"nu", 0.75f}, {"gamma", -1.0f}, {"coef0", 0.2f}}, {{"tolerance", 1e-8}, {"nu", 0.75}, {"gamma", -1.0}, {"coef0", 0.2}}, 0.416},

    {"nusvc_multiclass_random_tall_rbf", "multiclass_random_tall", da_svm_model::nusvc, {}, {{"kernel", "rbf"}}, {{"tolerance", 1e-4f}, {"nu", 0.5f}}, {{"tolerance", 1e-8}, {"nu", 0.5}}, 0.6, 10},
    {"nusvc_multiclass_random_tall_linear", "multiclass_random_tall", da_svm_model::nusvc, {}, {{"kernel", "linear"}}, {{"tolerance", 1e-4f}, {"nu", 0.5f}}, {{"tolerance", 1e-8}, {"nu", 0.5}}, 0.566},
    {"nusvc_multiclass_random_tall_polynomial", "multiclass_random_tall", da_svm_model::nusvc, {{"degree", 3}}, {{"kernel", "poly"}}, {{"tolerance", 1e-4f}, {"nu", 0.5f}, {"coef0", 0.0f}}, {{"tolerance", 1e-8}, {"nu", 0.5}, {"coef0", 0.0}}, 0.566, 20},
    {"nusvc_multiclass_random_tall_sigmoid", "multiclass_random_tall", da_svm_model::nusvc, {}, {{"kernel", "sigmoid"}}, {{"tolerance", 1e-4f}, {"nu", 0.5f}, {"coef0", 0.0f}}, {{"tolerance", 1e-8}, {"nu", 0.5}, {"coef0", 0.0}}, 0.4, 20},
    
    // REGRESSION
    // SVR
    {"svr_regression_random_tall_rbf", "regression_random_tall", da_svm_model::svr, {}, {}, {{"tolerance", 1e-4f}, {"C", 10.0f}, {"epsilon", 0.3f}}, {{"tolerance", 1e-8}, {"C", 10.0}, {"epsilon", 0.3}}, 0.080},
    {"svr_regression_random_tall_linear", "regression_random_tall", da_svm_model::svr, {}, {{"kernel", "linear"}}, {{"tolerance", 1e-4f}, {"C", 10.0f}, {"epsilon", 0.3f}}, {{"tolerance", 1e-8}, {"C", 10.0}, {"epsilon", 0.3}}, 0.999},
    {"svr_regression_random_tall_polynomial", "regression_random_tall", da_svm_model::svr, {{"degree", 2}}, {{"kernel", "poly"}}, {{"tolerance", 1e-4f}, {"C", 10.0f}, {"epsilon", 0.3f}, {"gamma", -1.0f}, {"coef0", 0.78f}}, {{"tolerance", 1e-8}, {"C", 10.0}, {"epsilon", 0.3}, {"gamma", -1.0}, {"coef0", 0.78}}, 0.487},
    {"svr_regression_random_tall_sigmoid", "regression_random_tall", da_svm_model::svr, {}, {{"kernel", "sigmoid"}}, {{"tolerance", 1e-4f}, {"C", 10.0f}, {"epsilon", 0.3f}, {"gamma", -1.0f}, {"coef0", 0.78f}}, {{"tolerance", 1e-8}, {"C", 10.0}, {"epsilon", 0.3}, {"gamma", -1.0}, {"coef0", 0.78}}, 0.21},
    
    {"svr_regression_random_wide_rbf", "regression_random_wide", da_svm_model::svr, {}, {{"kernel", "rbf"}}, {{"tolerance", 1e-4f}, {"C", 0.2f}, {"epsilon", 0.6f}, {"gamma", 2.0f}}, {{"tolerance", 1e-8}, {"C", 0.2}, {"epsilon", 0.6}, {"gamma", 2.0}}, -0.403},
    {"svr_regression_random_wide_linear", "regression_random_wide", da_svm_model::svr, {}, {{"kernel", "linear"}}, {{"tolerance", 1e-4f}, {"C", 0.2f}, {"epsilon", 0.6f}}, {{"tolerance", 1e-8}, {"C", 0.2}, {"epsilon", 0.6}}, -0.395},
    {"svr_regression_random_wide_polynomial", "regression_random_wide", da_svm_model::svr, {{"degree", 2}}, {{"kernel", "poly"}}, {{"tolerance", 1e-4f}, {"C", 0.2f}, {"epsilon", 0.6f}, {"gamma", 2.0f}, {"coef0", 3.0f}}, {{"tolerance", 1e-8}, {"C", 0.2}, {"epsilon", 0.6}, {"gamma", 2.0}, {"coef0", 3.0}}, -0.475},
    {"svr_regression_random_wide_sigmoid", "regression_random_wide", da_svm_model::svr, {}, {{"kernel", "sigmoid"}}, {{"tolerance", 1e-4f}, {"C", 0.2f}, {"epsilon", 0.6f}, {"gamma", 2.0f}, {"coef0", 3.0f}}, {{"tolerance", 1e-8}, {"C", 0.2}, {"epsilon", 0.6}, {"gamma", 2.0}, {"coef0", 3.0}}, -0.407},
    
    // nuSVR
    {"nusvr_regression_random_tall_rbf", "regression_random_tall", da_svm_model::nusvr, {}, {}, {{"tolerance", 1e-4f}, {"C", 1.2f}, {"nu", 0.5f}, {"gamma", 1.0f}}, {{"tolerance", 1e-8}, {"C", 1.2}, {"nu", 0.5}, {"gamma", 1.0}}, -0.051},
    {"nusvr_regression_random_tall_linear", "regression_random_tall", da_svm_model::nusvr, {}, {{"kernel", "linear"}}, {{"tolerance", 1e-4f}, {"C", 1.2f}, {"nu", 0.5f}}, {{"tolerance", 1e-8}, {"C", 1.2}, {"nu", 0.5}}, 0.33},
    {"nusvr_regression_random_tall_polynomial", "regression_random_tall", da_svm_model::nusvr, {{"degree", 4}}, {{"kernel", "poly"}}, {{"tolerance", 1e-4f}, {"C", 1.2f}, {"nu", 0.5f}, {"gamma", 1.0f}, {"coef0", 0.2f}}, {{"tolerance", 1e-8}, {"C", 1.2}, {"nu", 0.5}, {"gamma", 1.0}, {"coef0", 0.2}}, -0.723, 3},
    {"nusvr_regression_random_tall_sigmoid", "regression_random_tall", da_svm_model::nusvr, {}, {{"kernel", "sigmoid"}}, {{"tolerance", 1e-4f}, {"C", 1.2f}, {"nu", 0.5f}, {"gamma", 1.0f}, {"coef0", 0.2f}}, {{"tolerance", 1e-8}, {"C", 1.2}, {"nu", 0.5}, {"gamma", 1.0}, {"coef0", 0.2}}, 0.050},
    
    {"nusvr_regression_random_wide_rbf", "regression_random_wide", da_svm_model::nusvr, {}, {{"kernel", "rbf"}}, {{"tolerance", 1e-4f}, {"C", 1.0f}, {"nu", 0.2f}, {"gamma", 4.0f}}, {{"tolerance", 1e-8}, {"C", 1.0}, {"nu", 0.2}, {"gamma", 4.0}}, -0.392},
    {"nusvr_regression_random_wide_linear", "regression_random_wide", da_svm_model::nusvr, {}, {{"kernel", "linear"}}, {{"tolerance", 1e-4f}, {"C", 1.0f}, {"nu", 0.2f}}, {{"tolerance", 1e-8}, {"C", 1.0}, {"nu", 0.2}}, -0.158},
    {"nusvr_regression_random_wide_polynomial", "regression_random_wide", da_svm_model::nusvr, {{"degree", 2}}, {{"kernel", "poly"}}, {{"tolerance", 1e-4f}, {"C", 1.0f}, {"nu", 0.2f}, {"gamma", 4.0f}, {"coef0", 0.25f}}, {{"tolerance", 1e-8}, {"C", 1.0}, {"nu", 0.2}, {"gamma", 4.0}, {"coef0", 0.25}}, -0.509, 20},
    {"nusvr_regression_random_wide_sigmoid", "regression_random_wide", da_svm_model::nusvr, {}, {{"kernel", "sigmoid"}}, {{"tolerance", 1e-4f}, {"C", 1.0f}, {"nu", 0.2f}, {"gamma", 4.0f}, {"coef0", 0.25f}}, {{"tolerance", 1e-8}, {"C", 1.0}, {"nu", 0.2}, {"gamma", 4.0}, {"coef0", 0.25}}, -0.37},
    
};
// clang-format on

class svm_positive : public testing::TestWithParam<svm_param_t> {};
// Teach GTest how to print the param type
// in this case use only user's unique testname
// It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
void PrintTo(const svm_param_t &param, ::std::ostream *os) { *os << param.test_name; }

// Positive tests with double and single type
TEST_P(svm_positive, Double) {
    const svm_param_t &param = GetParam();
    test_svm_positive<double>(param.data_name, param.model, param.iopts, param.sopts,
                              param.dopts, (double)param.target_score,
                              (double)param.check_tol_scale);
}
TEST_P(svm_positive, Single) {
    const svm_param_t &param = GetParam();
    test_svm_positive<float>(param.data_name, param.model, param.iopts, param.sopts,
                             param.fopts, (float)param.target_score,
                             (float)param.check_tol_scale);
}

INSTANTIATE_TEST_SUITE_P(svm_pos_suite, svm_positive, testing::ValuesIn(svm_param_pos));
