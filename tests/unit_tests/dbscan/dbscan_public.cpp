/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <stdio.h>
#include <string.h>

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "dbscan_test_data.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

/* Given a vector of labels, this function renames the labels so that the first label encountered is 0,
   the second label encountered is 1, and so on. This is to enable parallel implementations of DBSCAN,
   in which labels may have different names, to be tested. */
void fix_labels(std::vector<da_int> &labels) {
    std::vector<da_int> unique_labels;
    for (auto &label : labels) {
        if (label != -1 && std::find(unique_labels.begin(), unique_labels.end(), label) ==
                               unique_labels.end()) {
            unique_labels.push_back(
                label); // Add all labels that aren't noise (-1) to unique_labels
        }
    }

    da_int n_clusters = (da_int)unique_labels.size();
    std::map<da_int, da_int> label_map;
    label_map[-1] = -1; // Noise label remains -1
    for (da_int i = 0; i < n_clusters; i++) {
        label_map[unique_labels[i]] = i;
    }

    for (auto &label : labels) {
        auto it = label_map.find(label);
        if (it != label_map.end()) {
            label = it->second;
        }
    }
}

template <typename T> std::vector<DBSCANParamType<T>> getParams() {
    std::vector<DBSCANParamType<T>> params;
    GetDBSCANData(params);
    return params;
}

template <typename T> void test_functionality(const DBSCANParamType<T> &param) {
    da_handle handle = nullptr;

    std::vector<std::string> cluster_list{"serial", "parallel"};

    // Loop over the parallel and serial versions of the actual DBSCAN cluster phase
    for (const auto &cluster_method : cluster_list) {
        EXPECT_EQ(da_debug_set("dbscan.cluster_methods", cluster_method.c_str()),
                  da_status_success);
        std::cout << "Functionality test: " << param.test_name << ", (" << cluster_method
                  << " clustering phase)" << std::endl;

        EXPECT_EQ(da_handle_init<T>(&handle, da_handle_dbscan), da_status_success)
            << "handle_init call failed.";
        EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
                  da_status_success)
            << "Set string 'algorithm' failed.";
        EXPECT_EQ(da_options_set_string(handle, "metric", param.metric.c_str()),
                  da_status_success)
            << "Set string 'metric' failed.";
        EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
                  da_status_success)
            << "Set string 'storage order' failed.";
        EXPECT_EQ(da_options_set_int(handle, "min samples", param.min_samples),
                  da_status_success)
            << "Set option 'min samples' failed.";
        EXPECT_EQ(da_options_set_int(handle, "leaf size", param.leaf_size),
                  da_status_success)
            << "Set option 'leaf size' failed.";
        EXPECT_EQ(da_options_set(handle, "eps", param.eps), da_status_success)
            << "Set options 'eps' failed.";
        EXPECT_EQ(da_options_set(handle, "power", param.power), da_status_success)
            << "Set string 'power' failed.";

        EXPECT_EQ(da_dbscan_set_data(handle, param.n_samples, param.n_features,
                                     param.A.data(), param.lda),
                  da_status_success)
            << "Call to set_data failed.";

        EXPECT_EQ(da_dbscan_compute<T>(handle), param.expected_status)
            << "Call to compute failed.";

        // Check that the serial vs parallel cluster path is correct
        char answer[100];
        EXPECT_EQ(da_debug_get("dbscan.setup", 100, answer), da_status_success);
        std::string expect = "clustering=" + cluster_method;

        EXPECT_EQ(std::string(answer), expect);

        da_int n_clusters, n_core_samples, one = 1;
        EXPECT_EQ(da_handle_get_result(handle, da_dbscan_n_clusters, &one, &n_clusters),
                  da_status_success)
            << "Get result 'n_clusters' failed.";
        EXPECT_EQ(n_clusters, param.expected_n_clusters)
            << "n_clusters failed to match expected value.";

        EXPECT_EQ(
            da_handle_get_result(handle, da_dbscan_n_core_samples, &one, &n_core_samples),
            da_status_success)
            << "Get result 'n_core_samples' failed.";
        EXPECT_EQ(n_core_samples, param.expected_n_core_samples)
            << "n_core_samples failed to match expected value.";

        std::vector<da_int> labels(param.n_samples);
        std::vector<da_int> core_sample_indices(n_core_samples);
        da_int rinfo_size = 9;
        std::vector<T> rinfo(rinfo_size);

        EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &rinfo_size, rinfo.data()),
                  da_status_success)
            << "Get result 'da_rinfo' failed.";

        EXPECT_ARR_EQ(rinfo_size, rinfo.data(), param.expected_rinfo.data(), 1, 1, 0, 0);
        da_int n_samples = param.n_samples;
        EXPECT_EQ(
            da_handle_get_result(handle, da_dbscan_labels, &n_samples, labels.data()),
            da_status_success)
            << "Get result 'da_dbscan_labels' failed.";

        // Parallel implementations may encounter samples in a different order so labels may have different names, so we fix this here
        fix_labels(labels);
        std::vector<da_int> expected_labels = param.expected_labels;
        fix_labels(expected_labels);

        EXPECT_ARR_EQ(param.n_samples, labels.data(), expected_labels.data(), 1, 1, 0, 0);

        if (n_core_samples > 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_dbscan_core_sample_indices,
                                           &n_core_samples, core_sample_indices.data()),
                      da_status_success)
                << "Get result 'da_dbscan_core_sample_indices' failed.";
            // Since parallel implementations may encounter samples in a different orders, we reorder
            std::sort(core_sample_indices.begin(), core_sample_indices.end());
            std::vector<da_int> expected_core_sample_indices =
                param.expected_core_sample_indices;
            std::sort(expected_core_sample_indices.begin(),
                      expected_core_sample_indices.end());
            EXPECT_ARR_EQ(n_core_samples, core_sample_indices.data(),
                          expected_core_sample_indices.data(), 1, 1, 0, 0);
        }
        da_handle_destroy(&handle);
    }
}

class DoubleFunctionalityTest : public testing::TestWithParam<DBSCANParamType<double>> {};
class FloatFunctionalityTest : public testing::TestWithParam<DBSCANParamType<float>> {};

template <typename T> void PrintTo(const DBSCANParamType<T> &param, ::std::ostream *os) {
    *os << param.test_name;
}

TEST_P(DoubleFunctionalityTest, ParameterizedTest) {
    const DBSCANParamType<double> &p = GetParam();
    test_functionality(p);
}

TEST_P(FloatFunctionalityTest, ParameterizedTest) {
    const DBSCANParamType<float> &p = GetParam();
    test_functionality(p);
}

INSTANTIATE_TEST_SUITE_P(DBSCAN_Functionality_Tests_Double, DoubleFunctionalityTest,
                         ::testing::ValuesIn(getParams<double>()));
INSTANTIATE_TEST_SUITE_P(DBSCAN_Functionality_Tests_Float, FloatFunctionalityTest,
                         ::testing::ValuesIn(getParams<float>()));

template <typename T> class DBSCANTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DBSCANTest, FloatTypes);

TYPED_TEST(DBSCANTest, MultipleCalls) {
    // Check we can repeatedly compute DBSCAN with the same single handle

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_dbscan), da_status_success);

    std::vector<DBSCANParamType<TypeParam>> params;
    Get30by3Data(params);
    Get25by2Data(params);

    for (auto &param : params) {
        std::cout << "Functionality test: " << param.test_name << std::endl;

        EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
                  da_status_success)
            << "Set string 'algorithm' failed.";
        EXPECT_EQ(da_options_set_string(handle, "metric", param.metric.c_str()),
                  da_status_success)
            << "Set string 'metric' failed.";
        EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
                  da_status_success)
            << "Set string 'storage order' failed.";
        EXPECT_EQ(da_options_set_int(handle, "min samples", param.min_samples),
                  da_status_success)
            << "Set option 'min samples' failed.";
        EXPECT_EQ(da_options_set_int(handle, "leaf size", param.leaf_size),
                  da_status_success)
            << "Set option 'leaf size' failed.";
        EXPECT_EQ(da_options_set(handle, "eps", param.eps), da_status_success)
            << "Set options 'eps' failed.";
        EXPECT_EQ(da_options_set(handle, "power", param.power), da_status_success)
            << "Set string 'power' failed.";

        EXPECT_EQ(da_dbscan_set_data(handle, param.n_samples, param.n_features,
                                     param.A.data(), param.lda),
                  da_status_success)
            << "Call to set_data failed.";

        EXPECT_EQ(da_dbscan_compute<TypeParam>(handle), param.expected_status)
            << "Call to compute failed.";

        da_int n_clusters, n_core_samples, one = 1;
        EXPECT_EQ(da_handle_get_result(handle, da_dbscan_n_clusters, &one, &n_clusters),
                  da_status_success)
            << "Get result 'n_clusters' failed.";
        EXPECT_EQ(n_clusters, param.expected_n_clusters)
            << "n_clusters failed to match expected value.";

        EXPECT_EQ(
            da_handle_get_result(handle, da_dbscan_n_core_samples, &one, &n_core_samples),
            da_status_success)
            << "Get result 'n_core_samples' failed.";
        EXPECT_EQ(n_core_samples, param.expected_n_core_samples)
            << "n_core_samples failed to match expected value.";

        std::vector<da_int> labels(param.n_samples);
        std::vector<da_int> core_sample_indices(n_core_samples);
        da_int rinfo_size = 9;
        std::vector<TypeParam> rinfo(rinfo_size);

        EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &rinfo_size, rinfo.data()),
                  da_status_success)
            << "Get result 'da_rinfo' failed.";

        EXPECT_ARR_EQ(rinfo_size, rinfo.data(), param.expected_rinfo.data(), 1, 1, 0, 0);
        da_int n_samples = param.n_samples;
        EXPECT_EQ(
            da_handle_get_result(handle, da_dbscan_labels, &n_samples, labels.data()),
            da_status_success)
            << "Get result 'da_dbscan_labels' failed.";

        // Parallel implementations may encounter samples in a different order so labels may have different names, so we fix this here
        fix_labels(labels);
        std::vector<da_int> expected_labels = param.expected_labels;
        fix_labels(expected_labels);

        EXPECT_ARR_EQ(param.n_samples, labels.data(), expected_labels.data(), 1, 1, 0, 0);

        if (n_core_samples > 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_dbscan_core_sample_indices,
                                           &n_core_samples, core_sample_indices.data()),
                      da_status_success)
                << "Get result 'da_dbscan_core_sample_indices' failed.";
            // Since parallel implementations may encounter samples in a different orders, we reorder
            std::sort(core_sample_indices.begin(), core_sample_indices.end());
            std::vector<da_int> expected_core_sample_indices =
                param.expected_core_sample_indices;
            std::sort(expected_core_sample_indices.begin(),
                      expected_core_sample_indices.end());
            EXPECT_ARR_EQ(n_core_samples, core_sample_indices.data(),
                          expected_core_sample_indices.data(), 1, 1, 0, 0);
        }
    }
    da_handle_destroy(&handle);
}

TYPED_TEST(DBSCANTest, ErrorExits) {

    da_handle handle = nullptr;
    std::vector<TypeParam> A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    da_int n_samples = 4, n_features = 3, lda = 4;
    da_int results_arr_int[1];
    da_int *null_results_arr_int = nullptr;
    TypeParam results_arr[1];
    TypeParam *null_results_arr = nullptr;
    da_int dim = 1;

    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_dbscan), da_status_success);

    // Error exits to do with routines called in the wrong order

    EXPECT_EQ(da_dbscan_compute<TypeParam>(handle), da_status_no_data);
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, results_arr),
              da_status_no_data);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, &dim, results_arr_int),
              da_status_no_data);

    // Test error exits in compute

    // k-d tree with cosine metic
    std::string s1 = "kd tree";
    EXPECT_EQ(da_options_set_string(handle, "algorithm", s1.c_str()), da_status_success);
    std::string s2 = "cosine";
    EXPECT_EQ(da_options_set_string(handle, "metric", s2.c_str()), da_status_success);
    EXPECT_EQ(da_dbscan_set_data(handle, n_samples, n_features, A.data(), lda),
              da_status_success);
    EXPECT_EQ(da_dbscan_compute<TypeParam>(handle), da_status_incompatible_options);

    // k-d tree with Minkowski metric and p < 1
    s2 = "minkowski";
    EXPECT_EQ(da_options_set_string(handle, "algorithm", s1.c_str()), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", s2.c_str()), da_status_success);
    EXPECT_EQ(da_options_set(handle, "power", (TypeParam)0.5), da_status_success);
    EXPECT_EQ(da_dbscan_set_data(handle, n_samples, n_features, A.data(), lda),
              da_status_success);
    EXPECT_EQ(da_dbscan_compute<TypeParam>(handle), da_status_incompatible_options);

    s1 = "brute";
    s2 = "Euclidean";
    EXPECT_EQ(da_options_set_string(handle, "algorithm", s1.c_str()), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", s2.c_str()), da_status_success);
    EXPECT_EQ(da_options_set(handle, "min samples", (da_int)2), da_status_success);
    EXPECT_EQ(da_options_set(handle, "eps", (TypeParam)20.0), da_status_success);
    EXPECT_EQ(da_dbscan_set_data(handle, n_samples, n_features, A.data(), lda),
              da_status_success);
    EXPECT_EQ(da_dbscan_compute<TypeParam>(handle), da_status_success);

    // get results error exits
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, null_results_arr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, null_results_arr_int, results_arr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, &dim, null_results_arr_int),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, null_results_arr_int,
                                       null_results_arr_int),
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
    EXPECT_EQ(dim, 9);
    dim = 0;
    EXPECT_EQ(da_handle_get_result_int(handle, da_dbscan_labels, &dim, results_arr_int),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, 4);
    dim = 1;
    EXPECT_EQ(da_handle_get_result_int(handle, da_dbscan_core_sample_indices, &dim,
                                       results_arr_int),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, 4);

    da_handle_destroy(&handle);
}

TYPED_TEST(DBSCANTest, BadHandleTests) {

    // handle not initialized
    da_handle handle = nullptr;
    TypeParam A = 1;

    EXPECT_EQ(da_dbscan_set_data(handle, 1, 1, &A, 1), da_status_handle_not_initialized);
    EXPECT_EQ(da_dbscan_compute<TypeParam>(handle), da_status_handle_not_initialized);

    // Incorrect handle type
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_linmod), da_status_success);

    EXPECT_EQ(da_dbscan_set_data(handle, 1, 1, &A, 1), da_status_invalid_handle_type);
    EXPECT_EQ(da_dbscan_compute<TypeParam>(handle), da_status_invalid_handle_type);

    da_handle_destroy(&handle);
}

TEST(DBSCANTest, IncorrectHandlePrecision) {
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_dbscan), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_dbscan), da_status_success);

    double Ad = 0.0;
    float As = 0.0f;

    EXPECT_EQ(da_dbscan_set_data_d(handle_s, 1, 1, &Ad, 1), da_status_wrong_type);
    EXPECT_EQ(da_dbscan_set_data_s(handle_d, 1, 1, &As, 1), da_status_wrong_type);

    EXPECT_EQ(da_dbscan_compute_d(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_dbscan_compute_s(handle_d), da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}