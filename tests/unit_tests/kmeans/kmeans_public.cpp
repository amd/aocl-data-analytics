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

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "kmeans_test_data.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T> class KMeansTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(KMeansTest, FloatTypes);

TYPED_TEST(KMeansTest, KMeansFunctionality) {
    std::vector<KMeansParamType<TypeParam>> params;
    GetKMeansData(params);
    da_handle handle = nullptr;
    da_int count = 0;

    for (auto &param : params) {

        count++;

        std::cout << "Functionality test " << std::to_string(count) << ": "
                  << param.test_name << std::endl;

        EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "initialization method",
                                        param.initialization_method.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_clusters", param.n_clusters),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "max_iter", param.max_iter),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_init", param.n_init), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", param.seed), da_status_success);
        EXPECT_EQ(
            da_options_set(handle, "convergence tolerance", param.convergence_tolerance),
            da_status_success);

        EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, param.n_features,
                                     param.A.data(), param.lda),
                  da_status_success);

        if (param.initialization_method == "supplied") {
            EXPECT_EQ(da_kmeans_set_init_centres(handle, param.C.data(), param.ldc),
                      da_status_success);
        }

        EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), param.expected_status);

        da_int size_rinfo = 5;
        std::vector<TypeParam> rinfo(size_rinfo);
        EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size_rinfo, rinfo.data()),
                  da_status_success);

        da_int size_centres = param.n_clusters * param.n_features;
        std::vector<TypeParam> centres(size_centres);
        EXPECT_EQ(da_handle_get_result(handle, da_kmeans_cluster_centres, &size_centres,
                                       centres.data()),
                  da_status_success);

        da_int size_labels = param.n_samples;
        std::vector<da_int> labels(size_labels);
        EXPECT_EQ(da_handle_get_result_int(handle, da_kmeans_labels, &size_labels,
                                           labels.data()),
                  da_status_success);

        if (param.is_random == false) {

            // This test is sufficiently deterministic to check values explicitly
            EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                          param.X.data(), param.ldx,
                                          param.X_transform.data(), param.ldx_transform),
                      da_status_success);

            EXPECT_EQ(da_kmeans_predict(handle, param.k_samples, param.k_features,
                                        param.Y.data(), param.ldy, param.Y_labels.data()),
                      da_status_success);

            EXPECT_ARR_NEAR(size_rinfo, rinfo.data(), param.expected_rinfo.data(),
                            param.tol);

            EXPECT_ARR_NEAR(size_centres, centres.data(), param.expected_centres.data(),
                            param.tol);

            EXPECT_ARR_EQ(size_labels, labels.data(), param.expected_labels.data(), 1, 1,
                          0, 0);

            EXPECT_ARR_NEAR(param.ldx_transform * param.m_features,
                            param.X_transform.data(), param.expected_X_transform.data(),
                            param.tol);

            EXPECT_ARR_EQ(param.k_samples, param.Y_labels.data(),
                          param.expected_Y_labels.data(), 1, 1, 0, 0);
        } else {
            // Randomness in this test so just check the final inertia is sufficiently small

            EXPECT_LE(rinfo[4], param.max_allowed_inertia + param.tol);
        }

        da_handle_destroy(&handle);
    }
}

TYPED_TEST(KMeansTest, MultipleCalls) {
    // Check we can repeatedly call compute etc with the same single handle

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans), da_status_success);

    std::vector<KMeansParamType<TypeParam>> params;
    KMeansParamType<TypeParam> param1, param2, param3;
    Get1by1BaseData(param1);
    params.push_back(param1);
    Get3ClustersBaseData(param2);
    params.push_back(param2);
    GetRowMajorBaseData(param3);
    params.push_back(param3);
    param2.algorithm = "lloyd";
    param2.expected_rinfo[3] = 1.0;
    params.push_back(param2);
    param2.algorithm = "macqueen";
    param2.expected_rinfo[3] = 0.0;
    params.push_back(param2);
    param2.algorithm = "elkan";
    param2.expected_rinfo[3] = 1.0;
    params.push_back(param2);

    da_int count = 0;

    for (auto &param : params) {

        count++;

        std::cout << "Multiple call test " << std::to_string(count) << ": "
                  << param.test_name << std::endl;

        EXPECT_EQ(da_options_set_string(handle, "initialization method",
                                        param.initialization_method.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_clusters", param.n_clusters),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "max_iter", param.max_iter),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_init", param.n_init), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", param.seed), da_status_success);
        EXPECT_EQ(
            da_options_set(handle, "convergence tolerance", param.convergence_tolerance),
            da_status_success);

        EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, param.n_features,
                                     param.A.data(), param.lda),
                  da_status_success);

        if (param.initialization_method == "supplied") {
            EXPECT_EQ(da_kmeans_set_init_centres(handle, param.C.data(), param.ldc),
                      da_status_success);
        }

        EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), param.expected_status);

        da_int size_rinfo = 5;
        std::vector<TypeParam> rinfo(size_rinfo);
        EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size_rinfo, rinfo.data()),
                  da_status_success);

        da_int size_centres = param.n_clusters * param.n_features;
        std::vector<TypeParam> centres(size_centres);
        EXPECT_EQ(da_handle_get_result(handle, da_kmeans_cluster_centres, &size_centres,
                                       centres.data()),
                  da_status_success);

        da_int size_labels = param.n_samples;
        std::vector<da_int> labels(size_labels);
        EXPECT_EQ(da_handle_get_result_int(handle, da_kmeans_labels, &size_labels,
                                           labels.data()),
                  da_status_success);

        EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                      param.X.data(), param.ldx, param.X_transform.data(),
                                      param.ldx_transform),
                  da_status_success);

        EXPECT_EQ(da_kmeans_predict(handle, param.k_samples, param.k_features,
                                    param.Y.data(), param.ldy, param.Y_labels.data()),
                  da_status_success);

        EXPECT_ARR_NEAR(size_rinfo, rinfo.data(), param.expected_rinfo.data(), param.tol);

        EXPECT_ARR_NEAR(size_centres, centres.data(), param.expected_centres.data(),
                        param.tol);

        EXPECT_ARR_EQ(size_labels, labels.data(), param.expected_labels.data(), 1, 1, 0,
                      0);

        EXPECT_ARR_NEAR(param.ldx_transform * param.m_features, param.X_transform.data(),
                        param.expected_X_transform.data(), param.tol);

        EXPECT_ARR_EQ(param.k_samples, param.Y_labels.data(),
                      param.expected_Y_labels.data(), 1, 1, 0, 0);

        if (count == 1) {
            // Triggers the code path where the user re-uses a handle, meaning an illegal value of n_clusters hasn't been caught
            EXPECT_EQ(da_options_set_int(handle, "n_clusters", 56), da_status_success);
            EXPECT_EQ(da_kmeans_compute<TypeParam>(handle),
                      da_status_incompatible_options);
        }
    }

    da_handle_destroy(&handle);
}

TYPED_TEST(KMeansTest, ErrorExits) {
    // Get some data to use
    KMeansParamType<TypeParam> param;
    Get1by1BaseData(param);
    TypeParam results_arr[1];
    TypeParam *null_arr = nullptr;
    da_int *null_arr_int = nullptr;
    da_int results_arr_int[1];
    da_int dim = 1;

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans), da_status_success);

    // set_data error exits
    EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, param.n_features, null_arr,
                                 param.lda),
              da_status_invalid_pointer);
    EXPECT_EQ(da_kmeans_set_data(handle, 0, param.n_features, param.A.data(), param.lda),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, 0, param.A.data(), param.lda),
              da_status_invalid_array_dimension);
    EXPECT_EQ(
        da_kmeans_set_data(handle, param.n_samples, param.n_features, param.A.data(), 0),
        da_status_invalid_leading_dimension);

    // error exits to do with routines called in the wrong order
    EXPECT_EQ(da_kmeans_set_init_centres(handle, param.C.data(), param.ldc),
              da_status_no_data);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_no_data);
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                  param.X.data(), param.ldx, param.X_transform.data(),
                                  param.ldx_transform),
              da_status_no_data);
    EXPECT_EQ(da_kmeans_predict(handle, param.k_samples, param.k_features, param.Y.data(),
                                param.ldy, param.Y_labels.data()),
              da_status_no_data);
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, results_arr),
              da_status_no_data);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, &dim, results_arr_int),
              da_status_no_data);

    // Subsequent tests require us to actually provide some data, but use this to test the n_clusters > n_samples warning
    EXPECT_EQ(da_options_set_int(handle, "n_clusters", 10), da_status_success);
    EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, param.n_features,
                                 param.A.data(), param.lda),
              da_status_incompatible_options);

    // init_centres error exits
    EXPECT_EQ(da_kmeans_set_init_centres(handle, null_arr, param.ldc),
              da_status_invalid_pointer);
    EXPECT_EQ(da_kmeans_set_init_centres(handle, param.C.data(), 0),
              da_status_invalid_leading_dimension);

    // compute error exits
    EXPECT_EQ(da_options_set_int(handle, "n_init", 10), da_status_success);
    std::string s = "supplied";
    EXPECT_EQ(da_options_set_string(handle, "initialization method", s.c_str()),
              da_status_success);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_no_data);

    std::string a = "hartigan-wong";
    EXPECT_EQ(da_options_set_string(handle, "algorithm", a.c_str()), da_status_success);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_incompatible_options);

    // Test that check_data works - could do this in any handle type really, so we will do it here
    std::string y = "yes";
    EXPECT_EQ(da_options_set(handle, "check data", y.c_str()), da_status_success);
    TypeParam tmp = param.C.data()[0];
    param.C.data()[0] = std::numeric_limits<TypeParam>::quiet_NaN();
    EXPECT_EQ(da_kmeans_set_init_centres(handle, param.C.data(), param.ldc),
              da_status_invalid_input);
    param.C.data()[0] = tmp;

    // Subsequent tests require compute to be done
    EXPECT_EQ(da_kmeans_set_init_centres(handle, param.C.data(), param.ldc),
              da_status_success);
    std::string a2 = "lloyd";
    EXPECT_EQ(da_options_set_string(handle, "algorithm", a2.c_str()), da_status_success);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_success);

    // transform error exits
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features, null_arr,
                                  param.ldx, param.X_transform.data(),
                                  param.ldx_transform),
              da_status_invalid_pointer);
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                  param.X.data(), param.ldx, null_arr,
                                  param.ldx_transform),
              da_status_invalid_pointer);
    EXPECT_EQ(da_kmeans_transform(handle, 0, param.m_features, param.X.data(), param.ldx,
                                  param.X_transform.data(), param.ldx_transform),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, 0, param.X.data(), param.ldx,
                                  param.X_transform.data(), param.ldx_transform),
              da_status_invalid_input);
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                  param.X.data(), 0, param.X_transform.data(),
                                  param.ldx_transform),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                  param.X.data(), param.ldx, param.X_transform.data(), 0),
              da_status_invalid_leading_dimension);

    // predict error exits
    EXPECT_EQ(da_kmeans_predict(handle, param.k_features, param.k_samples, null_arr,
                                param.ldy, param.Y_labels.data()),
              da_status_invalid_pointer);
    EXPECT_EQ(da_kmeans_predict(handle, param.k_features, param.k_samples, param.Y.data(),
                                param.ldy, null_arr_int),
              da_status_invalid_pointer);
    EXPECT_EQ(da_kmeans_predict(handle, 0, param.k_samples, param.Y.data(), param.ldy,
                                param.Y_labels.data()),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_kmeans_predict(handle, param.k_features, 0, param.Y.data(), param.ldy,
                                param.Y_labels.data()),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_kmeans_predict(handle, param.k_features, param.k_samples, param.Y.data(),
                                0, param.Y_labels.data()),
              da_status_invalid_leading_dimension);

    // get results error exits
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
    EXPECT_EQ(da_handle_get_result(handle, da_linmod_coef, &dim, results_arr),
              da_status_unknown_query);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, &dim, results_arr_int),
              da_status_unknown_query);
    dim = 0;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, results_arr),
              da_status_invalid_array_dimension);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, 5);
    dim = 0;
    EXPECT_EQ(da_handle_get_result(handle, da_kmeans_cluster_centres, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, 1);
    dim = 0;
    EXPECT_EQ(da_handle_get_result_int(handle, da_kmeans_labels, &dim, results_arr_int),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, 1);

    da_handle_destroy(&handle);

    // Final check we can trigger the maximum iteration warning
    KMeansParamType<TypeParam> param2;
    Get3ClustersBaseData(param2);

    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_init", 10), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_clusters", 2), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 1), da_status_success);
    EXPECT_EQ(da_kmeans_set_data(handle, param2.n_samples, param2.n_features,
                                 param2.A.data(), param2.lda),
              da_status_success);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_maxit);

    da_handle_destroy(&handle);
}

TYPED_TEST(KMeansTest, BadHandleTests) {

    // handle not initialized
    da_handle handle = nullptr;
    TypeParam A = 1;
    da_int labels = 1;

    EXPECT_EQ(da_kmeans_set_data(handle, 1, 1, &A, 1), da_status_handle_not_initialized);
    EXPECT_EQ(da_kmeans_set_init_centres(handle, &A, 1),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_handle_not_initialized);
    EXPECT_EQ(da_kmeans_transform(handle, 1, 1, &A, 1, &A, 1),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_kmeans_predict(handle, 1, 1, &A, 1, &labels),
              da_status_handle_not_initialized);

    // Incorrect handle type
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_linmod), da_status_success);

    EXPECT_EQ(da_kmeans_set_data(handle, 1, 1, &A, 1), da_status_invalid_handle_type);
    EXPECT_EQ(da_kmeans_set_init_centres(handle, &A, 1), da_status_invalid_handle_type);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_invalid_handle_type);
    EXPECT_EQ(da_kmeans_transform(handle, 1, 1, &A, 1, &A, 1),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_kmeans_predict(handle, 1, 1, &A, 1, &labels),
              da_status_invalid_handle_type);

    da_handle_destroy(&handle);
}

TEST(KMeansTest, IncorrectHandlePrecision) {
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_kmeans), da_status_success);

    double Ad = 0.0;
    float As = 0.0f;
    da_int labels = 1;

    EXPECT_EQ(da_kmeans_set_data_d(handle_s, 1, 1, &Ad, 1), da_status_wrong_type);
    EXPECT_EQ(da_kmeans_set_data_s(handle_d, 1, 1, &As, 1), da_status_wrong_type);

    EXPECT_EQ(da_kmeans_set_init_centres_d(handle_s, &Ad, 1), da_status_wrong_type);
    EXPECT_EQ(da_kmeans_set_init_centres_s(handle_d, &As, 1), da_status_wrong_type);

    EXPECT_EQ(da_kmeans_compute_d(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_kmeans_compute_s(handle_d), da_status_wrong_type);

    EXPECT_EQ(da_kmeans_transform_d(handle_s, 1, 1, &Ad, 1, &Ad, 1),
              da_status_wrong_type);
    EXPECT_EQ(da_kmeans_transform_s(handle_d, 1, 1, &As, 1, &As, 1),
              da_status_wrong_type);

    EXPECT_EQ(da_kmeans_predict_d(handle_s, 1, 1, &Ad, 1, &labels), da_status_wrong_type);
    EXPECT_EQ(da_kmeans_predict_s(handle_d, 1, 1, &As, 1, &labels), da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}