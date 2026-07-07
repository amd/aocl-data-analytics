/*
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// taken from  "da_kernel_utils.hpp"
enum vectorization_type : da_int {
    undefined = -1,
    scalar = 1,
    avx = 2,
    avx2 = 3,
    avx512 = 4,
    count = 5
};

// Test kernel overrides
TEST(KMeansKernelOverride, SetAndGet) {
    using T = float;

    da_int n_samples = 10;
    da_int n_features = 2;
    const std::vector<T> A{1.0, 1.1, 0.5,  0.49, -2.0, -2.0, 0.53, 0.9,  1.2, -1.8,
                           1.0, 1.2, -2.0, -1.9, 0.5,  0.51, -2.1, 0.95, 0.8, 0.6};
    da_int lda = 10;

    // setup a micro problem
    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "lloyd"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_clusters", 2), da_status_success);
    EXPECT_EQ(da_kmeans_set_data(handle, n_samples, n_features, A.data(), lda),
              da_status_success);

    char answer[100];

    EXPECT_EQ(da_kmeans_compute<T>(handle), da_status_success);
    // Run kmean and expect telemetry data
    EXPECT_EQ(da_debug_get("kmeans.setup", 100, answer), da_status_success);
    // no need to parse...
    // solve again with invalid ISA
    // Rerun with invalid ISA value and expect "scalar" kernel type
    EXPECT_EQ(da_debug_set("kmeans.ISA", "invalid"), da_status_success);
    EXPECT_EQ(da_kmeans_compute<T>(handle), da_status_success);
    // Run kmean and expect telemetry data
    EXPECT_EQ(da_debug_get("kmeans.setup", 100, answer), da_status_success);
    EXPECT_THAT(std::string(answer),
                ::testing::HasSubstr(
                    ("kernel.type=" + std::to_string(vectorization_type::scalar))));

    // solve again Lloyd - AVX2
    EXPECT_EQ(da_debug_set("kmeans.isa", "avx2"), da_status_success);
    EXPECT_EQ(da_kmeans_compute<T>(handle), da_status_success);
    // Check telemetry
    EXPECT_EQ(da_debug_get("kmeans.setup", 100, answer), da_status_success);
    EXPECT_EQ(da_debug_get(nullptr, 100, answer), da_status_success);
    // "kmeans.settings" =
    // "kernel=lloyd,kernel.type=" + std::to_string(kernel_type) + ",kernel.padding=" +
    // std::to_string(padding));
    EXPECT_THAT(std::string(answer),
                ::testing::HasSubstr(
                    ("kernel.type=" + std::to_string(vectorization_type::avx2))));

    // solve again Elkan
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "elkan"), da_status_success);
    EXPECT_EQ(da_kmeans_compute<T>(handle), da_status_success);
    EXPECT_EQ(da_debug_get("kmeans.setup", 100, answer), da_status_success);
    // "kernel=elkan,kernel.update_kernel_type=" + std::to_string(update_kernel_type) +
    // ",kernel.reduce_kernel.type=" + std::to_string(reduce_kernel_type) +
    // ",kernel.padding=" + std::to_string(padding));
    EXPECT_THAT(std::string(answer),
                ::testing::HasSubstr(("kernel.update_kernel.type=" +
                                      std::to_string(vectorization_type::avx2))));
    EXPECT_THAT(std::string(answer),
                ::testing::HasSubstr(("kernel.reduce_kernel.type=" +
                                      std::to_string(vectorization_type::avx2))));
    da_handle_destroy(&handle);
}

template <typename T> class KMeansTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> std::vector<KMeansParamType<T>> getParams() {
    std::vector<KMeansParamType<T>> params;
    GetKMeansData(params);
    return params;
}

template <typename T> void test_functionality(const KMeansParamType<T> &param) {
    da_handle handle = nullptr;
    std::unordered_map<std::string, vectorization_type> isa_list{
        {"none", vectorization_type::scalar}};

    // These kernels have multiple implementations, so we need to set the ISA
    if (param.algorithm == "elkan" || param.algorithm == "lloyd") {
        isa_list = {{"scalar", vectorization_type::scalar},
                    {"avx", vectorization_type::avx},
                    {"avx2", vectorization_type::avx2},
                    // will trickle down to AVX2 where is AVX512 not available
                    {"avx512", vectorization_type::avx512}};
    } else {
        isa_list = {{"none", vectorization_type::scalar}};
    }

    for (const auto &isa : isa_list) {
        if (isa.first == "none") {
            // remove isa requirement
            EXPECT_EQ(da_debug_set("kmeans.isa", ""), da_status_success);
        } else {
            EXPECT_EQ(da_debug_set("kmeans.isa", (isa.first).c_str()), da_status_success);
        }

        std::cout << "Functionality test: " << param.test_name
                  << ": param.algorithm=" << param.algorithm
                  << "  [kmeans.isa=" << isa.first << " (" << std::to_string(isa.second)
                  << ")]" << std::endl;

        EXPECT_EQ(da_handle_init<T>(&handle, da_handle_kmeans), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "initialization method",
                                        param.initialization_method.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
                  da_status_success);
        EXPECT_EQ(
            da_options_set_string(handle, "empty clusters", param.empty_clusters.c_str()),
            da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_clusters", param.n_clusters),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "max_iter", param.max_iter),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_init", param.n_init), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", param.seed), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "afk-mc2 samples", param.afk_mcmc_samples),
                  da_status_success);
        EXPECT_EQ(
            da_options_set(handle, "convergence tolerance", param.convergence_tolerance),
            da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "low precision max_iter", param.lp_max_iter),
                  da_status_success);
        EXPECT_EQ(
            da_options_set(handle, "low precision convergence tolerance", param.lp_tol),
            da_status_success);
        // If typename is double, set mixed precision option
        if (std::is_same<T, double>::value) {
            EXPECT_EQ(da_options_set_string(handle, "mixed precision",
                                            param.mixed_precision.c_str()),
                      da_status_success);
        }

        EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, param.n_features,
                                     param.A.data(), param.lda),
                  da_status_success);

        if (param.initialization_method == "supplied") {
            EXPECT_EQ(da_kmeans_set_init_centres(handle, param.C.data(), param.ldc),
                      da_status_success);
        }

        // Check da_trained before compute
        da_int tr_dim = 1, tr_val = -1;
        EXPECT_EQ(da_handle_get_result(handle, da_result::da_trained, &tr_dim, &tr_val),
                  da_status_success);
        EXPECT_EQ(tr_val, 0);

        EXPECT_EQ(da_kmeans_compute<T>(handle), param.expected_status);

        // If an error was expected, skip result validation and clean up
        if (param.expected_status != da_status_success &&
            param.expected_status != da_status_maxit) {
            da_handle_destroy(&handle);
            continue;
        }

        // Check da_trained after compute
        EXPECT_EQ(da_handle_get_result(handle, da_result::da_trained, &tr_dim, &tr_val),
                  da_status_success);
        EXPECT_EQ(tr_val, 1);

        if (isa.first != "none") {
            // Check that the kernel isa path is correct
            char answer[100];
            EXPECT_EQ(da_debug_get("kmeans.setup", 100, answer), da_status_success);
            auto expect =
                ::testing::HasSubstr(("kernel.type=" + std::to_string(isa.second)));
            if (isa.second == vectorization_type::avx512) {
                // take care of the AVX2/AVX512 fallback
                auto fallback = ::testing::HasSubstr(
                    ("kernel.type=" + std::to_string(vectorization_type::avx2)));
                EXPECT_THAT(std::string(answer), ::testing::AnyOf(expect, fallback));
            } else {
                EXPECT_THAT(std::string(answer), expect);
            }
        }

        da_int size_rinfo = 6;
        std::vector<T> rinfo(size_rinfo);
        EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size_rinfo, rinfo.data()),
                  da_status_success);

        da_int size_centres = param.n_clusters * param.n_features;
        std::vector<T> centres(size_centres);
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
            std::vector<T> X_transform = param.X_transform;
            EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                          param.X.data(), param.ldx, X_transform.data(),
                                          param.ldx_transform),
                      da_status_success);

            std::vector<da_int> Y_labels = param.Y_labels;
            EXPECT_EQ(da_kmeans_predict(handle, param.k_samples, param.k_features,
                                        param.Y.data(), param.ldy, Y_labels.data()),
                      da_status_success);

            EXPECT_ARR_NEAR(size_rinfo, rinfo.data(), param.expected_rinfo.data(),
                            param.tol);

            EXPECT_ARR_NEAR(size_centres, centres.data(), param.expected_centres.data(),
                            param.tol);

            EXPECT_ARR_EQ(size_labels, labels.data(), param.expected_labels.data(), 1, 1,
                          0, 0);

            EXPECT_ARR_NEAR(param.ldx_transform * param.n_clusters, X_transform.data(),
                            param.expected_X_transform.data(), param.tol);

            EXPECT_ARR_EQ(param.k_samples, Y_labels.data(),
                          param.expected_Y_labels.data(), 1, 1, 0, 0);
        } else {
            // Randomness in this test so just check the final inertia is sufficiently small

            EXPECT_LE(rinfo[4], param.max_allowed_inertia + param.tol);
        }

        da_handle_destroy(&handle);
    }
}

class DoubleFunctionalityTest : public testing::TestWithParam<KMeansParamType<double>> {};
class FloatFunctionalityTest : public testing::TestWithParam<KMeansParamType<float>> {};

template <typename T> void PrintTo(const KMeansParamType<T> &param, ::std::ostream *os) {
    *os << param.test_name;
}

TEST_P(DoubleFunctionalityTest, ParameterizedTest) {
    const KMeansParamType<double> &p = GetParam();
    test_functionality(p);
}

TEST_P(FloatFunctionalityTest, ParameterizedTest) {
    const KMeansParamType<float> &p = GetParam();
    test_functionality(p);
}

INSTANTIATE_TEST_SUITE_P(KMeans_Functionality_Tests_Double, DoubleFunctionalityTest,
                         ::testing::ValuesIn(getParams<double>()));
INSTANTIATE_TEST_SUITE_P(KMeans_Functionality_Tests_Float, FloatFunctionalityTest,
                         ::testing::ValuesIn(getParams<float>()));

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(KMeansTest, FloatTypes);

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

        da_int size_rinfo = 6;
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

        std::vector<TypeParam> X_transform = param.X_transform;
        EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                      param.X.data(), param.ldx, X_transform.data(),
                                      param.ldx_transform),
                  da_status_success);

        std::vector<da_int> Y_labels = param.Y_labels;
        EXPECT_EQ(da_kmeans_predict(handle, param.k_samples, param.k_features,
                                    param.Y.data(), param.ldy, Y_labels.data()),
                  da_status_success);

        EXPECT_ARR_NEAR(size_rinfo, rinfo.data(), param.expected_rinfo.data(), param.tol);

        EXPECT_ARR_NEAR(size_centres, centres.data(), param.expected_centres.data(),
                        param.tol);

        EXPECT_ARR_EQ(size_labels, labels.data(), param.expected_labels.data(), 1, 1, 0,
                      0);

        EXPECT_ARR_NEAR(param.ldx_transform * param.m_features, X_transform.data(),
                        param.expected_X_transform.data(), param.tol);

        EXPECT_ARR_EQ(param.k_samples, Y_labels.data(), param.expected_Y_labels.data(), 1,
                      1, 0, 0);

        if (count == 1) {
            // Triggers the code path where the user re-uses a handle, meaning an illegal value of n_clusters hasn't been caught
            EXPECT_EQ(da_options_set_int(handle, "n_clusters", 56), da_status_success);
            if (param.initialization_method == "supplied") {
                EXPECT_EQ(da_kmeans_set_init_centres(handle, param.C.data(), param.ldc),
                          da_status_success);
            }
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
    std::vector<TypeParam> X_transform = param.X_transform;
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                  param.X.data(), param.ldx, X_transform.data(),
                                  param.ldx_transform),
              da_status_no_data);
    std::vector<da_int> Y_labels = param.Y_labels;
    EXPECT_EQ(da_kmeans_predict(handle, param.k_samples, param.k_features, param.Y.data(),
                                param.ldy, Y_labels.data()),
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

    da_handle_destroy(&handle);
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, param.n_features,
                                 param.A.data(), param.lda),
              da_status_success);
    std::string a = "hartigan-wong";
    EXPECT_EQ(da_options_set_string(handle, "algorithm", a.c_str()), da_status_success);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_incompatible_options);

    da_handle_destroy(&handle);
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", a.c_str()), da_status_success);
    EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, param.n_features,
                                 param.A.data(), param.lda),
              da_status_success);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_incompatible_options);

    // Test that cosine distance is incompatible with Hartigan-Wong
    da_handle_destroy(&handle);
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", a.c_str()), da_status_success);
    std::string cos_str = "cosine";
    EXPECT_EQ(da_options_set_string(handle, "distance", cos_str.c_str()),
              da_status_success);
    EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, param.n_features,
                                 param.A.data(), param.lda),
              da_status_success);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_incompatible_options);

    // Test that check_data works - could do this in any handle type really, so we will do it here
    da_handle_destroy(&handle);
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans), da_status_success);
    std::string y = "yes";
    EXPECT_EQ(da_options_set(handle, "check data", y.c_str()), da_status_success);
    TypeParam tmp = param.C.data()[0];
    param.C.data()[0] = std::numeric_limits<TypeParam>::quiet_NaN();
    EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, param.n_features,
                                 param.A.data(), param.lda),
              da_status_success);
    EXPECT_EQ(da_kmeans_set_init_centres(handle, param.C.data(), param.ldc),
              da_status_invalid_input);
    param.C.data()[0] = tmp;

    // Subsequent tests require compute to be done
    da_handle_destroy(&handle);
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans), da_status_success);
    std::string a2 = "lloyd";
    EXPECT_EQ(da_options_set_string(handle, "algorithm", a2.c_str()), da_status_success);
    EXPECT_EQ(da_kmeans_set_data(handle, param.n_samples, param.n_features,
                                 param.A.data(), param.lda),
              da_status_success);
    EXPECT_EQ(da_kmeans_set_init_centres(handle, param.C.data(), param.ldc),
              da_status_success);
    EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_success);

    // transform error exits
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features, null_arr,
                                  param.ldx, X_transform.data(), param.ldx_transform),
              da_status_invalid_pointer);
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                  param.X.data(), param.ldx, null_arr,
                                  param.ldx_transform),
              da_status_invalid_pointer);
    EXPECT_EQ(da_kmeans_transform(handle, 0, param.m_features, param.X.data(), param.ldx,
                                  X_transform.data(), param.ldx_transform),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, 0, param.X.data(), param.ldx,
                                  X_transform.data(), param.ldx_transform),
              da_status_invalid_input);
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                  param.X.data(), 0, X_transform.data(),
                                  param.ldx_transform),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_kmeans_transform(handle, param.m_samples, param.m_features,
                                  param.X.data(), param.ldx, X_transform.data(), 0),
              da_status_invalid_leading_dimension);

    // predict error exits
    EXPECT_EQ(da_kmeans_predict(handle, param.k_features, param.k_samples, null_arr,
                                param.ldy, Y_labels.data()),
              da_status_invalid_pointer);
    EXPECT_EQ(da_kmeans_predict(handle, param.k_features, param.k_samples, param.Y.data(),
                                param.ldy, null_arr_int),
              da_status_invalid_pointer);
    EXPECT_EQ(da_kmeans_predict(handle, 0, param.k_samples, param.Y.data(), param.ldy,
                                Y_labels.data()),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_kmeans_predict(handle, param.k_features, 2, param.Y.data(), param.ldy,
                                Y_labels.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_kmeans_predict(handle, param.k_features, param.k_samples, param.Y.data(),
                                0, Y_labels.data()),
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
    EXPECT_EQ(dim, 6);
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

    // If typename is double, check we can trigger the error when low precision tolerance is smaller than convergence tolerance
    if (std::is_same<TypeParam, double>::value) {
        EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans),
                  da_status_success);
        std::string mp = "yes";
        EXPECT_EQ(da_options_set_string(handle, "mixed precision", mp.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "convergence tolerance", 1e-5),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "low precision convergence tolerance", 1e-6),
                  da_status_success);
        EXPECT_EQ(da_kmeans_set_data(handle, param2.n_samples, param2.n_features,
                                     param2.A.data(), param2.lda),
                  da_status_success);
        EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_incompatible_options);
        da_handle_destroy(&handle);
    }
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

TYPED_TEST(KMeansTest, SphericalKMeans) {
    // Test spherical k-means: centres should be unit-normalized and invariant to data scaling
    // Tests all combinations of algorithms, initializations, and storage orders

    da_int n_samples = 20;
    da_int n_features = 3;
    da_int n_clusters = 3;
    da_int seed = 42;

    // Data in column-major layout (n_samples x n_features)
    std::vector<TypeParam> A_col = {
        1.2,  -0.5, 0.3,  2.1,  -1.8, 0.7,  0.4, -0.2, 1.5,  -0.9, 1.7,  0.6,
        -1.1, 0.8,  -0.3, 2.0,  -1.4, 0.1,  0.9, -0.7, 0.5,  1.3,  -0.8, 0.2,
        1.6,  -1.0, 0.4,  -0.6, 1.1,  -1.3, 0.7, -0.4, 1.8,  -0.1, 0.3,  -1.5,
        0.6,  1.0,  -0.2, 0.8,  -0.3, 0.9,  1.4, -0.7, 0.1,  0.5,  -1.2, 0.3,
        1.6,  -0.4, 0.8,  -0.1, 0.6,  -1.0, 1.3, 0.2,  -0.5, 0.7,  -0.8, 1.1};

    // Data in row-major layout (transpose of column-major)
    std::vector<TypeParam> A_row(n_samples * n_features);
    for (da_int i = 0; i < n_samples; i++)
        for (da_int j = 0; j < n_features; j++)
            A_row[i * n_features + j] = A_col[i + j * n_samples];

    std::vector<std::string> algorithms = {"lloyd", "elkan", "macqueen"};
    std::vector<std::string> inits = {"random", "k-means++", "afk-mc2"};
    std::vector<std::string> orders = {"column-major", "row-major"};

    TypeParam tol =
        std::is_same<TypeParam, float>::value ? (TypeParam)1.0e-6 : (TypeParam)1.0e-12;

    // Helper to index into a centres array respecting storage order
    auto c_idx = [&](da_int i, da_int j, bool row_major) -> da_int {
        return row_major ? i * n_features + j : i + j * n_clusters;
    };

    for (const auto &order : orders) {

        const TypeParam *A_ptr;
        da_int lda;
        if (order == "column-major") {
            A_ptr = A_col.data();
            lda = n_samples;
        } else {
            A_ptr = A_row.data();
            lda = n_features;
        }

        // Create scaled data (A * 2)
        std::vector<TypeParam> A_scaled(n_samples * n_features);
        for (da_int i = 0; i < n_samples * n_features; i++)
            A_scaled[i] = A_ptr[i] * (TypeParam)2.0;

        for (const auto &alg : algorithms) {
            for (const auto &init : inits) {

                std::string ctx =
                    "algorithm=" + alg + ", init=" + init + ", order=" + order;

                // --- Run on original data ---
                da_handle handle = nullptr;
                EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans),
                          da_status_success);
                EXPECT_EQ(da_options_set_int(handle, "n_clusters", n_clusters),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle, "algorithm", alg.c_str()),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle, "distance", "cosine"),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle, "storage order", order.c_str()),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle, "normalize data", "yes"),
                          da_status_success);
                EXPECT_EQ(
                    da_options_set_string(handle, "initialization method", init.c_str()),
                    da_status_success);
                EXPECT_EQ(da_options_set_int(handle, "seed", seed), da_status_success);
                EXPECT_EQ(da_options_set_int(handle, "n_init", 1), da_status_success);
                EXPECT_EQ(da_kmeans_set_data(handle, n_samples, n_features, A_ptr, lda),
                          da_status_success);
                EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_success);

                // Get centres
                da_int dim = n_clusters * n_features;
                std::vector<TypeParam> centres(dim);
                EXPECT_EQ(da_handle_get_result(handle, da_kmeans_cluster_centres, &dim,
                                               centres.data()),
                          da_status_success);

                // Check centres are unit-normalized
                bool is_row_major = (order == "row-major");
                for (da_int i = 0; i < n_clusters; i++) {
                    TypeParam norm_sq = (TypeParam)0.0;
                    for (da_int j = 0; j < n_features; j++) {
                        TypeParam val = centres[c_idx(i, j, is_row_major)];
                        norm_sq += val * val;
                    }
                    EXPECT_NEAR(norm_sq, (TypeParam)1.0, tol)
                        << "Centre " << i << " not unit-normalized for " << ctx;
                }

                da_handle_destroy(&handle);

                // --- Run on scaled data ---
                da_handle handle2 = nullptr;
                EXPECT_EQ(da_handle_init<TypeParam>(&handle2, da_handle_kmeans),
                          da_status_success);
                EXPECT_EQ(da_options_set_int(handle2, "n_clusters", n_clusters),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle2, "algorithm", alg.c_str()),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle2, "distance", "cosine"),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle2, "storage order", order.c_str()),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle2, "normalize data", "yes"),
                          da_status_success);
                EXPECT_EQ(
                    da_options_set_string(handle2, "initialization method", init.c_str()),
                    da_status_success);
                EXPECT_EQ(da_options_set_int(handle2, "seed", seed), da_status_success);
                EXPECT_EQ(da_options_set_int(handle2, "n_init", 1), da_status_success);
                EXPECT_EQ(da_kmeans_set_data(handle2, n_samples, n_features,
                                             A_scaled.data(), lda),
                          da_status_success);
                EXPECT_EQ(da_kmeans_compute<TypeParam>(handle2), da_status_success);

                // Get centres from scaled data
                da_int dim2 = n_clusters * n_features;
                std::vector<TypeParam> centres_scaled(dim2);
                EXPECT_EQ(da_handle_get_result(handle2, da_kmeans_cluster_centres, &dim2,
                                               centres_scaled.data()),
                          da_status_success);

                // Check centres are unit-normalized
                for (da_int i = 0; i < n_clusters; i++) {
                    TypeParam norm_sq = (TypeParam)0.0;
                    for (da_int j = 0; j < n_features; j++) {
                        TypeParam val = centres_scaled[c_idx(i, j, is_row_major)];
                        norm_sq += val * val;
                    }
                    EXPECT_NEAR(norm_sq, (TypeParam)1.0, tol)
                        << "Scaled centre " << i << " not unit-normalized for " << ctx;
                }

                // Check centres from original and scaled data are identical
                for (da_int i = 0; i < n_clusters; i++) {
                    for (da_int j = 0; j < n_features; j++) {
                        da_int idx = c_idx(i, j, is_row_major);
                        EXPECT_NEAR(centres[idx], centres_scaled[idx], tol)
                            << "Centre mismatch at (" << i << "," << j << ") for " << ctx;
                    }
                }

                // --- Test predict and transform with X = 2 * centres ---
                // Since cosine distance is scale-invariant, each row of X is
                // collinear with the corresponding centre, so:
                //   predict should assign row i to cluster i
                //   transform should give distance 0 on the diagonal
                std::vector<TypeParam> X_test(n_clusters * n_features);
                for (da_int i = 0; i < n_clusters * n_features; i++)
                    X_test[i] = centres_scaled[i] * (TypeParam)2.0;

                da_int ldx = is_row_major ? n_features : n_clusters;
                da_int ldx_transform = ldx;

                // Predict
                std::vector<da_int> pred_labels(n_clusters);
                EXPECT_EQ(da_kmeans_predict(handle2, n_clusters, n_features,
                                            X_test.data(), ldx, pred_labels.data()),
                          da_status_success);
                for (da_int i = 0; i < n_clusters; i++) {
                    EXPECT_EQ(pred_labels[i], i)
                        << "Predict mismatch at row " << i << " for " << ctx;
                }

                // Transform
                std::vector<TypeParam> X_transform(n_clusters * n_clusters,
                                                   (TypeParam)99.0);
                EXPECT_EQ(da_kmeans_transform(handle2, n_clusters, n_features,
                                              X_test.data(), ldx, X_transform.data(),
                                              ldx_transform),
                          da_status_success);
                // Diagonal entries (distance to own centre) should be 0
                for (da_int i = 0; i < n_clusters; i++) {
                    da_int diag_idx = i * n_clusters + i;
                    EXPECT_NEAR(X_transform[diag_idx], (TypeParam)0.0, tol)
                        << "Transform diagonal non-zero at (" << i << "," << i << ") for "
                        << ctx;
                }

                da_handle_destroy(&handle2);
            }
        }

        // --- Test with user-supplied (unnormalized) initial centres ---
        for (const auto &alg : algorithms) {

            std::string ctx = "algorithm=" + alg + ", init=supplied, order=" + order;

            // Unnormalized initial centres (deliberately large magnitudes)
            std::vector<TypeParam> C = {10.0, -5.0, 3.0, 7.0, -2.0, 8.0, -4.0, 6.0, 1.0};
            da_int ldc;
            // C is always passed in user's storage order
            std::vector<TypeParam> C_row;
            if (order == "row-major") {
                // Transpose C from col-major (n_clusters x n_features, ld=n_clusters)
                // to row-major (n_clusters x n_features, ld=n_features)
                C_row.resize(n_clusters * n_features);
                for (da_int i = 0; i < n_clusters; i++)
                    for (da_int j = 0; j < n_features; j++)
                        C_row[i * n_features + j] = C[i + j * n_clusters];
                ldc = n_features;
            } else {
                ldc = n_clusters;
            }
            const TypeParam *C_ptr = (order == "row-major") ? C_row.data() : C.data();

            da_handle handle = nullptr;
            EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans),
                      da_status_success);
            EXPECT_EQ(da_options_set_int(handle, "n_clusters", n_clusters),
                      da_status_success);
            EXPECT_EQ(da_options_set_string(handle, "algorithm", alg.c_str()),
                      da_status_success);
            EXPECT_EQ(da_options_set_string(handle, "distance", "cosine"),
                      da_status_success);
            EXPECT_EQ(da_options_set_string(handle, "storage order", order.c_str()),
                      da_status_success);
            EXPECT_EQ(da_options_set_string(handle, "initialization method", "supplied"),
                      da_status_success);
            EXPECT_EQ(da_kmeans_set_data(handle, n_samples, n_features, A_ptr, lda),
                      da_status_success);
            EXPECT_EQ(da_kmeans_set_init_centres(handle, C_ptr, ldc), da_status_success);
            EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_success);

            // Get centres
            da_int dim = n_clusters * n_features;
            std::vector<TypeParam> centres(dim);
            EXPECT_EQ(da_handle_get_result(handle, da_kmeans_cluster_centres, &dim,
                                           centres.data()),
                      da_status_success);

            // Check centres are unit-normalized
            bool is_row_major = (order == "row-major");
            for (da_int i = 0; i < n_clusters; i++) {
                TypeParam norm_sq = (TypeParam)0.0;
                for (da_int j = 0; j < n_features; j++) {
                    TypeParam val = centres[c_idx(i, j, is_row_major)];
                    norm_sq += val * val;
                }
                EXPECT_NEAR(norm_sq, (TypeParam)1.0, tol)
                    << "Centre " << i << " not unit-normalized for " << ctx;
            }

            da_handle_destroy(&handle);
        }
    }

    // --- Test with pre-normalized data: normalize_data on vs off should give same results ---
    {
        // Build a row-normalized version of A_col
        std::vector<TypeParam> A_norm_col(n_samples * n_features);
        for (da_int i = 0; i < n_samples; i++) {
            TypeParam norm_sq = (TypeParam)0.0;
            for (da_int j = 0; j < n_features; j++) {
                TypeParam val = A_col[i + j * n_samples];
                norm_sq += val * val;
            }
            TypeParam inv_norm = (norm_sq > (TypeParam)0.0)
                                     ? (TypeParam)1.0 / std::sqrt(norm_sq)
                                     : (TypeParam)0.0;
            for (da_int j = 0; j < n_features; j++) {
                A_norm_col[i + j * n_samples] = A_col[i + j * n_samples] * inv_norm;
            }
        }

        // Row-major version
        std::vector<TypeParam> A_norm_row(n_samples * n_features);
        for (da_int i = 0; i < n_samples; i++)
            for (da_int j = 0; j < n_features; j++)
                A_norm_row[i * n_features + j] = A_norm_col[i + j * n_samples];

        for (const auto &order : orders) {
            const TypeParam *A_ptr;
            da_int lda;
            if (order == "column-major") {
                A_ptr = A_norm_col.data();
                lda = n_samples;
            } else {
                A_ptr = A_norm_row.data();
                lda = n_features;
            }

            for (const auto &alg : algorithms) {
                for (const auto &init : inits) {

                    std::string ctx = "prenorm algorithm=" + alg + ", init=" + init +
                                      ", order=" + order;

                    // Run with normalize_data = yes
                    da_handle h_on = nullptr;
                    EXPECT_EQ(da_handle_init<TypeParam>(&h_on, da_handle_kmeans),
                              da_status_success);
                    EXPECT_EQ(da_options_set_int(h_on, "n_clusters", n_clusters),
                              da_status_success);
                    EXPECT_EQ(da_options_set_string(h_on, "algorithm", alg.c_str()),
                              da_status_success);
                    EXPECT_EQ(da_options_set_string(h_on, "distance", "cosine"),
                              da_status_success);
                    EXPECT_EQ(da_options_set_string(h_on, "storage order", order.c_str()),
                              da_status_success);
                    EXPECT_EQ(da_options_set_string(h_on, "normalize data", "yes"),
                              da_status_success);
                    EXPECT_EQ(da_options_set_string(h_on, "initialization method",
                                                    init.c_str()),
                              da_status_success);
                    EXPECT_EQ(da_options_set_int(h_on, "seed", seed), da_status_success);
                    EXPECT_EQ(da_options_set_int(h_on, "n_init", 1), da_status_success);
                    EXPECT_EQ(da_kmeans_set_data(h_on, n_samples, n_features, A_ptr, lda),
                              da_status_success);
                    EXPECT_EQ(da_kmeans_compute<TypeParam>(h_on), da_status_success);

                    da_int dim_on = n_clusters * n_features;
                    std::vector<TypeParam> centres_on(dim_on);
                    EXPECT_EQ(da_handle_get_result(h_on, da_kmeans_cluster_centres,
                                                   &dim_on, centres_on.data()),
                              da_status_success);
                    da_int dim_labels_on = n_samples;
                    std::vector<da_int> labels_on(dim_labels_on);
                    EXPECT_EQ(da_handle_get_result(h_on, da_kmeans_labels, &dim_labels_on,
                                                   labels_on.data()),
                              da_status_success);

                    // Run with normalize_data = no
                    da_handle h_off = nullptr;
                    EXPECT_EQ(da_handle_init<TypeParam>(&h_off, da_handle_kmeans),
                              da_status_success);
                    EXPECT_EQ(da_options_set_int(h_off, "n_clusters", n_clusters),
                              da_status_success);
                    EXPECT_EQ(da_options_set_string(h_off, "algorithm", alg.c_str()),
                              da_status_success);
                    EXPECT_EQ(da_options_set_string(h_off, "distance", "cosine"),
                              da_status_success);
                    EXPECT_EQ(da_options_set_string(h_off, "normalize data", "no"),
                              da_status_success);
                    EXPECT_EQ(
                        da_options_set_string(h_off, "storage order", order.c_str()),
                        da_status_success);
                    EXPECT_EQ(da_options_set_string(h_off, "initialization method",
                                                    init.c_str()),
                              da_status_success);
                    EXPECT_EQ(da_options_set_int(h_off, "seed", seed), da_status_success);
                    EXPECT_EQ(da_options_set_int(h_off, "n_init", 1), da_status_success);
                    EXPECT_EQ(
                        da_kmeans_set_data(h_off, n_samples, n_features, A_ptr, lda),
                        da_status_success);
                    EXPECT_EQ(da_kmeans_compute<TypeParam>(h_off), da_status_success);

                    da_int dim_off = n_clusters * n_features;
                    std::vector<TypeParam> centres_off(dim_off);
                    EXPECT_EQ(da_handle_get_result(h_off, da_kmeans_cluster_centres,
                                                   &dim_off, centres_off.data()),
                              da_status_success);
                    da_int dim_labels_off = n_samples;
                    std::vector<da_int> labels_off(dim_labels_off);
                    EXPECT_EQ(da_handle_get_result(h_off, da_kmeans_labels,
                                                   &dim_labels_off, labels_off.data()),
                              da_status_success);

                    // Labels should be identical
                    for (da_int i = 0; i < n_samples; i++) {
                        EXPECT_EQ(labels_on[i], labels_off[i])
                            << "Label mismatch at sample " << i << " for " << ctx;
                    }

                    // Centres should match
                    for (da_int i = 0; i < n_clusters * n_features; i++) {
                        EXPECT_NEAR(centres_on[i], centres_off[i], tol)
                            << "Centre mismatch at index " << i << " for " << ctx;
                    }

                    da_handle_destroy(&h_on);
                    da_handle_destroy(&h_off);
                }
            }
        }
    }

    // --- Test empty cluster split handling with spherical k-means ---
    // All 5 data points point roughly in the (1,1) direction. One supplied centre
    // is orthogonal, so all points go to cluster 0, leaving cluster 1 empty.
    // The "split" strategy should recover a valid (unit-normalized) second centre.
    {
        da_int ec_n_samples = 5;
        da_int ec_n_features = 2;
        da_int ec_n_clusters = 2;

        std::vector<TypeParam> ec_A_col = {2.13, 2.11, 2.12, 2.13, 2.14,
                                           2.11, 2.14, 2.13, 2.12, 2.13};
        std::vector<TypeParam> ec_A_row(ec_n_samples * ec_n_features);
        for (da_int i = 0; i < ec_n_samples; i++)
            for (da_int j = 0; j < ec_n_features; j++)
                ec_A_row[i * ec_n_features + j] = ec_A_col[i + j * ec_n_samples];

        // Centre 0 ~ (1,1), centre 1 ~ (-1,1): orthogonal to data
        std::vector<TypeParam> ec_C_col = {1.0, -1.0, 1.0, 1.0};
        std::vector<TypeParam> ec_C_row = {1.0, 1.0, -1.0, 1.0};

        for (const auto &order : orders) {
            const TypeParam *ec_A_ptr, *ec_C_ptr;
            da_int ec_lda, ec_ldc;
            bool is_row_major = (order == "row-major");
            if (is_row_major) {
                ec_A_ptr = ec_A_row.data();
                ec_lda = ec_n_features;
                ec_C_ptr = ec_C_row.data();
                ec_ldc = ec_n_features;
            } else {
                ec_A_ptr = ec_A_col.data();
                ec_lda = ec_n_samples;
                ec_C_ptr = ec_C_col.data();
                ec_ldc = ec_n_clusters;
            }

            for (const auto &alg : algorithms) {
                std::string ctx =
                    "empty_cluster_split algorithm=" + alg + ", order=" + order;

                da_handle handle = nullptr;
                EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans),
                          da_status_success);
                EXPECT_EQ(da_options_set_int(handle, "n_clusters", ec_n_clusters),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle, "algorithm", alg.c_str()),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle, "distance", "cosine"),
                          da_status_success);
                EXPECT_EQ(da_options_set_string(handle, "storage order", order.c_str()),
                          da_status_success);
                EXPECT_EQ(
                    da_options_set_string(handle, "initialization method", "supplied"),
                    da_status_success);
                EXPECT_EQ(da_options_set_string(handle, "empty clusters", "split"),
                          da_status_success);
                EXPECT_EQ(da_kmeans_set_data(handle, ec_n_samples, ec_n_features,
                                             ec_A_ptr, ec_lda),
                          da_status_success);
                EXPECT_EQ(da_kmeans_set_init_centres(handle, ec_C_ptr, ec_ldc),
                          da_status_success);
                EXPECT_EQ(da_kmeans_compute<TypeParam>(handle), da_status_success);

                da_int dim = ec_n_clusters * ec_n_features;
                std::vector<TypeParam> centres(dim);
                EXPECT_EQ(da_handle_get_result(handle, da_kmeans_cluster_centres, &dim,
                                               centres.data()),
                          da_status_success);

                // Check all clusters are non-empty after split recovery
                da_int dim_labels = ec_n_samples;
                std::vector<da_int> labels(dim_labels);
                EXPECT_EQ(da_handle_get_result_int(handle, da_kmeans_labels, &dim_labels,
                                                   labels.data()),
                          da_status_success);
                for (da_int i = 0; i < ec_n_clusters; i++) {
                    da_int count = 0;
                    for (da_int s = 0; s < ec_n_samples; s++)
                        count += (labels[s] == i) ? 1 : 0;
                    EXPECT_GT(count, 0) << "Cluster " << i << " is empty for " << ctx;
                }

                // Both centres should be unit-normalized after split recovery
                for (da_int i = 0; i < ec_n_clusters; i++) {
                    TypeParam norm_sq = (TypeParam)0.0;
                    for (da_int j = 0; j < ec_n_features; j++) {
                        da_int idx =
                            is_row_major ? i * ec_n_features + j : i + j * ec_n_clusters;
                        norm_sq += centres[idx] * centres[idx];
                    }
                    EXPECT_NEAR(norm_sq, (TypeParam)1.0, tol)
                        << "Centre " << i << " not unit-normalized for " << ctx;
                }

                da_handle_destroy(&handle);
            }
        }
    }
}

/*
 * Test that compute does not crash on non-finite or extreme-magnitude
 * input. Covers NaN with n_init = 1, NaN with n_init > 1 (exercises a
 * different swap path inside perform_kmeans), and finite input whose
 * squared distances overflow to +inf.
 */
TYPED_TEST(KMeansTest, NonFiniteInputDoesNotCrash) {
    struct Scenario {
        const char *name;
        bool use_nan;
        da_int n_init;
    };

    // Magnitude whose square overflows to +inf for TypeParam.
    const TypeParam M = std::sqrt(std::numeric_limits<TypeParam>::max()) * TypeParam(2);

    const Scenario scenarios[] = {
        {"NaN, n_init=1", true, 1},
        {"NaN, n_init=4", true, 4},
        {"Inf-inertia, n_init=1", false, 1},
    };

    for (const auto &sc : scenarios) {
        std::vector<TypeParam> A{1.0, 1.1, 0.5,  0.49, -2.0, -2.0, 0.53, 0.9,  1.2, -1.8,
                                 1.0, 1.2, -2.0, -1.9, 0.5,  0.51, -2.1, 0.95, 0.8, 0.6};
        if (sc.use_nan) {
            A[3] = std::numeric_limits<TypeParam>::quiet_NaN();
        } else {
            for (da_int i = 0; i < 10; ++i) {
                TypeParam sign = (i < 5) ? TypeParam(1) : TypeParam(-1);
                A[i] = sign * M;
                A[i + 10] = sign * M;
            }
        }

        da_int n_samples = 10, n_features = 2, n_clusters = 3, lda = 10;

        da_handle handle = nullptr;
        EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans), da_status_success)
            << sc.name;
        EXPECT_EQ(da_kmeans_set_data(handle, n_samples, n_features, A.data(), lda),
                  da_status_success)
            << sc.name;
        EXPECT_EQ(da_options_set_int(handle, "n_clusters", n_clusters), da_status_success)
            << sc.name;
        EXPECT_EQ(da_options_set_string(handle, "algorithm", "lloyd"), da_status_success)
            << sc.name;
        EXPECT_EQ(da_options_set_int(handle, "n_init", sc.n_init), da_status_success)
            << sc.name;
        EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success) << sc.name;

        // Reaching this line without crashing is the assertion.
        da_status s = da_kmeans_compute<TypeParam>(handle);
        (void)s;

        da_handle_destroy(&handle);
    }
}

/*
 * Test that labels returned after max_iter = 1 agree with predict()
 * on the training data and with argmin against the final centres.
 */
TYPED_TEST(KMeansTest, MaxIter1LabelsConsistentWithPredict) {
    da_handle handle = nullptr;

    std::vector<TypeParam> A{1.0, 1.1, 0.5,  0.49, -2.0, -2.0, 0.53, 0.9,  1.2, -1.8,
                             1.0, 1.2, -2.0, -1.9, 0.5,  0.51, -2.1, 0.95, 0.8, 0.6};
    // Initial centres chosen so the first Lloyd step moves some points
    // between clusters.
    std::vector<TypeParam> C{1.0, 0.5, -2.0, 1.0, 0.5, -2.0};

    da_int n_samples = 10, n_features = 2, n_clusters = 3, lda = 10, ldc = 3;

    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_kmeans_set_data(handle, n_samples, n_features, A.data(), lda),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_clusters", n_clusters), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "lloyd"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "initialization method", "supplied"),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_init", 1), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 1), da_status_success);
    EXPECT_EQ(da_kmeans_set_init_centres(handle, C.data(), ldc), da_status_success);

    da_status s = da_kmeans_compute<TypeParam>(handle);
    ASSERT_TRUE(s == da_status_success || s == da_status_maxit);

    da_int ldim = n_samples;
    std::vector<da_int> labels(n_samples, -1);
    EXPECT_EQ(da_handle_get_result_int(handle, da_kmeans_labels, &ldim, labels.data()),
              da_status_success);

    da_int cdim = n_clusters * n_features;
    std::vector<TypeParam> C_final(cdim, TypeParam(0));
    EXPECT_EQ(
        da_handle_get_result(handle, da_kmeans_cluster_centres, &cdim, C_final.data()),
        da_status_success);

    std::vector<da_int> predicted(n_samples, -1);
    EXPECT_EQ(
        da_kmeans_predict(handle, n_samples, n_features, A.data(), lda, predicted.data()),
        da_status_success);

    for (da_int i = 0; i < n_samples; ++i) {
        // Labels must match predict() on the training data.
        EXPECT_EQ(predicted[i], labels[i]);

        // Labels must match argmin against the final centres.
        TypeParam best = std::numeric_limits<TypeParam>::infinity();
        da_int best_j = 0;
        for (da_int j = 0; j < n_clusters; ++j) {
            TypeParam d = TypeParam(0);
            for (da_int f = 0; f < n_features; ++f) {
                TypeParam diff = A[i + f * n_samples] - C_final[j + f * n_clusters];
                d += diff * diff;
            }
            if (d < best) {
                best = d;
                best_j = j;
            }
        }
        EXPECT_EQ(labels[i], best_j);
    }

    da_handle_destroy(&handle);
}
