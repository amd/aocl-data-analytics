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

#include <iostream>
#include <limits>
#include <list>
#include <stdio.h>
#include <string.h>

#include "aoclda.h"
#include "pca_test_data.hpp"

template <typename T> class PCATest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(PCATest, FloatTypes);

TYPED_TEST(PCATest, PCAFunctionality) {

    std::vector<PCAParamType<TypeParam>> params;
    GetPCAData(params);
    da_handle handle = nullptr;
    da_int count = 0;

    for (auto &param : params) {

        count++;
        std::cout << "Test " << std::to_string(count) << ": " << param.test_name
                  << std::endl;

        EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);

        EXPECT_EQ(da_pca_set_data(handle, param.n, param.p, param.A.data(), param.lda),
                  da_status_success);

        if (param.method.size() != 0) {
            EXPECT_EQ(da_options_set_string(handle, "PCA method", param.method.c_str()),
                      da_status_success);
        }
        if (param.degrees_of_freedom.size() != 0) {
            EXPECT_EQ(da_options_set_string(handle, "degrees of freedom",
                                            param.degrees_of_freedom.c_str()),
                      da_status_success);
        }

        EXPECT_EQ(da_options_set_int(handle, "n_components", param.components_required),
                  da_status_success);

        if (param.svd_solver.size() != 0) {
            EXPECT_EQ(
                da_options_set_string(handle, "svd solver", param.svd_solver.c_str()),
                da_status_success);
        }

        EXPECT_EQ(da_pca_compute<TypeParam>(handle), param.expected_status);

        if (param.X.size() > 0) {
            da_int size_X_transform = param.ldx_transform * param.expected_n_components;
            std::vector<TypeParam> X_transform(size_X_transform);
            EXPECT_EQ(da_pca_transform(handle, param.m, param.p, param.X.data(),
                                       param.ldx, X_transform.data(),
                                       param.ldx_transform),
                      da_status_success);
            EXPECT_ARR_NEAR(size_X_transform, X_transform.data(),
                            param.expected_X_transform.data(), param.epsilon);
        }

        if (param.Xinv.size() > 0) {
            da_int size_Xinv_transform = param.ldxinv_transform * param.p;
            std::vector<TypeParam> Xinv_transform(size_Xinv_transform);
            EXPECT_EQ(da_pca_inverse_transform(
                          handle, param.k, param.expected_n_components, param.Xinv.data(),
                          param.ldxinv, Xinv_transform.data(), param.ldxinv_transform),
                      da_status_success);
            EXPECT_ARR_NEAR(size_Xinv_transform, Xinv_transform.data(),
                            param.expected_Xinv_transform.data(), param.epsilon);
        }

        da_int size_one = 1;

        da_int size_scores = param.n * param.expected_n_components;
        da_int size_principal_components = param.p * param.expected_n_components;
        da_int size_variance = param.expected_n_components;
        da_int size_u = param.n * param.expected_n_components;
        da_int size_vt = param.p * param.expected_n_components;
        da_int size_sigma = param.expected_n_components;
        std::vector<TypeParam> scores(size_scores);
        std::vector<TypeParam> components(size_principal_components);
        std::vector<TypeParam> variance(size_variance);
        std::vector<TypeParam> u(size_u);
        std::vector<TypeParam> vt(size_vt);
        std::vector<TypeParam> sigma(size_sigma);
        TypeParam total_variance;

        if (param.expected_scores.size() != 0) {
            EXPECT_EQ(
                da_handle_get_result(handle, da_pca_scores, &size_scores, scores.data()),
                da_status_success);
            EXPECT_ARR_NEAR(size_scores, scores.data(), param.expected_scores.data(),
                            param.epsilon);
        }
        if (param.expected_components.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_principal_components,
                                           &size_principal_components, components.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_principal_components, components.data(),
                            param.expected_components.data(), param.epsilon);
        }
        if (param.expected_variance.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_variance, &size_variance,
                                           variance.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_variance, variance.data(),
                            param.expected_variance.data(), param.epsilon);
        }
        if (param.expected_u.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_u, &size_u, u.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_u, u.data(), param.expected_u.data(), param.epsilon);
        }
        if (param.expected_vt.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &size_vt, vt.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_vt, vt.data(), param.expected_vt.data(), param.epsilon);
        }
        if (param.expected_sigma.size() != 0) {
            EXPECT_EQ(
                da_handle_get_result(handle, da_pca_sigma, &size_sigma, sigma.data()),
                da_status_success);
            EXPECT_ARR_NEAR(size_sigma, sigma.data(), param.expected_sigma.data(),
                            param.epsilon);
        }
        EXPECT_EQ(da_handle_get_result(handle, da_pca_total_variance, &size_one,
                                       &total_variance),
                  da_status_success);
        EXPECT_NEAR(total_variance, param.expected_total_variance, param.epsilon);

        if (param.expected_means.size() != 0) {
            da_int size_means = param.p;
            std::vector<TypeParam> means(size_means);
            EXPECT_EQ(da_handle_get_result(handle, da_pca_column_means, &size_means,
                                           means.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_means, means.data(), param.expected_means.data(),
                            param.epsilon);
        }

        if (param.expected_sdevs.size() != 0) {
            da_int size_sdevs = param.p;
            std::vector<TypeParam> sdevs(size_sdevs);
            EXPECT_EQ(da_handle_get_result(handle, da_pca_column_sdevs, &size_sdevs,
                                           sdevs.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_sdevs, sdevs.data(), param.expected_sdevs.data(),
                            param.epsilon);
        }

        if (param.expected_rinfo.size() > 0) {
            da_int size_rinfo = 3;
            std::vector<TypeParam> rinfo(size_rinfo);
            EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size_rinfo, rinfo.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_rinfo, rinfo.data(), param.expected_rinfo.data(),
                            param.epsilon);
        }

        da_handle_destroy(&handle);
    }
}

TYPED_TEST(PCATest, MultipleCalls) {
    // Check we can repeatedly call compute etc with the same single handle

    // Get some data to use
    std::vector<PCAParamType<TypeParam>> params;
    GetSquareData1(params);
    GetTallThinData1(params);
    GetShortFatData(params);

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);

    for (auto &param : params) {
        EXPECT_EQ(da_pca_set_data(handle, param.n, param.p, param.A.data(), param.lda),
                  da_status_success);

        if (param.method.size() != 0) {
            EXPECT_EQ(da_options_set_string(handle, "PCA method", param.method.c_str()),
                      da_status_success);
        }
        if (param.components_required != 0) {
            EXPECT_EQ(
                da_options_set_int(handle, "n_components", param.components_required),
                da_status_success);
        }
        if (param.svd_solver.size() != 0) {
            EXPECT_EQ(
                da_options_set_string(handle, "svd solver", param.svd_solver.c_str()),
                da_status_success);
        }

        EXPECT_EQ(da_pca_compute<TypeParam>(handle), param.expected_status);

        if (param.X.size() > 0) {
            da_int size_X_transform = param.ldx_transform * param.expected_n_components;
            std::vector<TypeParam> X_transform(size_X_transform);
            EXPECT_EQ(da_pca_transform(handle, param.m, param.p, param.X.data(),
                                       param.ldx, X_transform.data(),
                                       param.ldx_transform),
                      da_status_success);
            EXPECT_ARR_NEAR(size_X_transform, X_transform.data(),
                            param.expected_X_transform.data(), param.epsilon);
        }

        if (param.Xinv.size() > 0) {
            da_int size_Xinv_transform = param.ldxinv_transform * param.p;
            std::vector<TypeParam> Xinv_transform(size_Xinv_transform);
            EXPECT_EQ(da_pca_inverse_transform(
                          handle, param.k, param.expected_n_components, param.Xinv.data(),
                          param.ldxinv, Xinv_transform.data(), param.ldxinv_transform),
                      da_status_success);
            EXPECT_ARR_NEAR(size_Xinv_transform, Xinv_transform.data(),
                            param.expected_Xinv_transform.data(), param.epsilon);
        }

        da_int size_scores = param.n * param.expected_n_components;
        da_int size_principal_components = param.p * param.expected_n_components;
        da_int size_variance = param.expected_n_components;
        da_int size_u = param.n * param.expected_n_components;
        da_int size_vt = param.p * param.expected_n_components;
        da_int size_sigma = param.expected_n_components;
        da_int size_one = 1;
        std::vector<TypeParam> scores(size_scores);
        std::vector<TypeParam> components(size_principal_components);
        std::vector<TypeParam> variance(size_variance);
        std::vector<TypeParam> u(size_u);
        std::vector<TypeParam> vt(size_vt);
        std::vector<TypeParam> sigma(size_sigma);
        TypeParam total_variance;

        if (param.expected_scores.size() != 0) {
            EXPECT_EQ(
                da_handle_get_result(handle, da_pca_scores, &size_scores, scores.data()),
                da_status_success);
            EXPECT_ARR_NEAR(size_scores, scores.data(), param.expected_scores.data(),
                            param.epsilon);
        }
        if (param.expected_components.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_principal_components,
                                           &size_principal_components, components.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_principal_components, components.data(),
                            param.expected_components.data(), param.epsilon);
        }
        if (param.expected_variance.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_variance, &size_variance,
                                           variance.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_variance, variance.data(),
                            param.expected_variance.data(), param.epsilon);
        }
        if (param.expected_u.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_u, &size_u, u.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_u, u.data(), param.expected_u.data(), param.epsilon);
        }
        if (param.expected_vt.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &size_vt, vt.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_vt, vt.data(), param.expected_vt.data(), param.epsilon);
        }
        if (param.expected_sigma.size() != 0) {
            EXPECT_EQ(
                da_handle_get_result(handle, da_pca_sigma, &size_sigma, sigma.data()),
                da_status_success);
            EXPECT_ARR_NEAR(size_sigma, sigma.data(), param.expected_sigma.data(),
                            param.epsilon);
        }
        EXPECT_EQ(da_handle_get_result(handle, da_pca_total_variance, &size_one,
                                       &total_variance),
                  da_status_success);
        EXPECT_NEAR(total_variance, param.expected_total_variance, param.epsilon);

        if (param.expected_rinfo.size() > 0) {
            da_int size_rinfo = 3;
            std::vector<TypeParam> rinfo(size_rinfo);
            EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size_rinfo, rinfo.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_rinfo, rinfo.data(), param.expected_rinfo.data(),
                            param.epsilon);
        }
    }

    da_handle_destroy(&handle);
}

TYPED_TEST(PCATest, ErrorExits) {

    // Get some data to use
    std::vector<PCAParamType<TypeParam>> params;
    GetSquareData1(params);
    TypeParam results_arr[1];
    TypeParam *null_arr = nullptr;
    da_int *null_arr_int = nullptr;
    da_int results_arr_int[1];
    da_int dim = 1;

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);

    // Check da_pca_set_data error exits
    EXPECT_EQ(da_pca_set_data(handle, params[0].n, params[0].p, params[0].A.data(),
                              params[0].n - 1),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_set_data(handle, 0, params[0].p, params[0].A.data(), params[0].lda),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_set_data(handle, params[0].n, 0, params[0].A.data(), params[0].lda),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_set_data(handle, params[0].n, params[0].p, null_arr, params[0].n),
              da_status_invalid_pointer);

    // Check error exits to catch incorrect order of routine calls
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_no_data);
    EXPECT_EQ(da_pca_transform(handle, params[0].m, params[0].p, params[0].X.data(),
                               params[0].ldx, results_arr, 1),
              da_status_no_data);
    EXPECT_EQ(
        da_pca_inverse_transform(handle, params[0].k, params[0].expected_n_components,
                                 params[0].Xinv.data(), params[0].ldxinv, results_arr, 1),
        da_status_no_data);
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, results_arr),
              da_status_no_data);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, &dim, results_arr_int),
              da_status_unknown_query);

    EXPECT_EQ(da_options_set_int(handle, "n_components", params[0].components_required),
              da_status_success);
    EXPECT_EQ(da_pca_set_data(handle, params[0].n, params[0].p, params[0].A.data(),
                              params[0].lda),
              da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);

    // Check da_pca_transform and da_pca_inverse_transform error exits
    EXPECT_EQ(da_pca_transform(handle, params[0].m, params[0].p, params[0].X.data(),
                               params[0].m - 1, results_arr, params[0].ldx_transform),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_transform(handle, 0, params[0].p, params[0].X.data(), params[0].ldx,
                               results_arr, params[0].ldx_transform),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_transform(handle, params[0].m, params[0].p + 1, params[0].X.data(),
                               params[0].ldx, results_arr, params[0].ldx_transform),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_transform(handle, params[0].m, params[0].p, params[0].X.data(),
                               params[0].ldx, results_arr, params[0].m - 1),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_transform(handle, params[0].m, params[0].p, params[0].X.data(),
                               params[0].ldx, null_arr, params[0].m),
              da_status_invalid_pointer);
    EXPECT_EQ(da_pca_inverse_transform(handle, params[0].k,
                                       params[0].expected_n_components,
                                       params[0].Xinv.data(), params[0].k - 1,
                                       results_arr, params[0].ldxinv_transform),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_inverse_transform(handle, 0, params[0].expected_n_components,
                                       params[0].Xinv.data(), params[0].ldxinv,
                                       results_arr, params[0].ldxinv_transform),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_inverse_transform(handle, params[0].k,
                                       params[0].expected_n_components + 1,
                                       params[0].Xinv.data(), params[0].ldxinv,
                                       results_arr, params[0].ldxinv_transform),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_inverse_transform(
                  handle, params[0].k, params[0].expected_n_components + 1,
                  params[0].Xinv.data(), params[0].ldxinv, results_arr, params[0].k - 1),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_inverse_transform(
                  handle, params[0].k, params[0].expected_n_components,
                  params[0].Xinv.data(), params[0].ldxinv, null_arr, params[0].k),
              da_status_invalid_pointer);

    // Check da_handle_get_results error exits for 'standard' results
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, null_arr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, null_arr_int, results_arr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, &dim, null_arr_int),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, null_arr_int, null_arr_int),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_int(handle, da_linmod_coef, &dim, results_arr_int),
              da_status_unknown_query);
    dim = 0;
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, &dim, results_arr_int),
              da_status_unknown_query);
    dim = 0;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, results_arr),
              da_status_invalid_array_dimension);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, results_arr),
              da_status_invalid_array_dimension);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_u, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].n * params[0].components_required);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_scores, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].n * params[0].components_required);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_variance, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].components_required);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].p * params[0].components_required);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_sigma, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].components_required);
    dim = 1;
    EXPECT_EQ(
        da_handle_get_result(handle, da_pca_principal_components, &dim, results_arr),
        da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].n * params[0].components_required);

    // da_handle_results error exits for columns means and column sdevs
    EXPECT_EQ(da_options_set_string(handle, "PCA method", "svd"), da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_column_means, &dim, results_arr),
              da_status_unknown_query);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_column_sdevs, &dim, results_arr),
              da_status_unknown_query);
    EXPECT_EQ(da_options_set_string(handle, "PCA method", "covariance"),
              da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_column_means, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_options_set_string(handle, "PCA method", "correlation"),
              da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_column_sdevs, &dim, results_arr),
              da_status_invalid_array_dimension);

    da_handle_destroy(&handle);
}

TYPED_TEST(PCATest, BadHandleTests) {

    // handle not initialized
    da_handle handle = nullptr;
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_handle_not_initialized);

    TypeParam A = 1;
    EXPECT_EQ(da_pca_set_data(handle, 1, 1, &A, 1), da_status_handle_not_initialized);

    EXPECT_EQ(da_pca_transform(handle, 1, 1, &A, 1, &A, 1),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_pca_inverse_transform(handle, 1, 1, &A, 1, &A, 1),
              da_status_handle_not_initialized);

    // incorrect handle type
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_linmod), da_status_success);

    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_invalid_handle_type);

    EXPECT_EQ(da_pca_set_data(handle, 1, 1, &A, 1), da_status_invalid_handle_type);

    EXPECT_EQ(da_pca_transform(handle, 1, 1, &A, 1, &A, 1),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_pca_inverse_transform(handle, 1, 1, &A, 1, &A, 1),
              da_status_invalid_handle_type);

    da_handle_destroy(&handle);
}

TEST(PCATest, IncorrectHandlePrecision) {

    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_pca), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_pca), da_status_success);

    double Ad;
    float As;

    EXPECT_EQ(da_pca_set_data_d(handle_s, 1, 1, &Ad, 1), da_status_wrong_type);
    EXPECT_EQ(da_pca_set_data_s(handle_d, 1, 1, &As, 1), da_status_wrong_type);

    EXPECT_EQ(da_pca_compute_d(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_pca_compute_s(handle_d), da_status_wrong_type);

    EXPECT_EQ(da_pca_transform_d(handle_s, 1, 1, &Ad, 1, &Ad, 1), da_status_wrong_type);
    EXPECT_EQ(da_pca_transform_s(handle_d, 1, 1, &As, 1, &As, 1), da_status_wrong_type);

    EXPECT_EQ(da_pca_inverse_transform_d(handle_s, 1, 1, &Ad, 1, &Ad, 1),
              da_status_wrong_type);
    EXPECT_EQ(da_pca_inverse_transform_s(handle_d, 1, 1, &As, 1, &As, 1),
              da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}