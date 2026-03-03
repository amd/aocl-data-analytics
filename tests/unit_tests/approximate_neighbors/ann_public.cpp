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

#include "../utest_utils.hpp"
#include "ann_tests.hpp"
#include "aoclda.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstring>
#include <iostream>
#include <numeric>

template <typename T> class ANNTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> void PrintTo(const ANNParamType<T> &param, ::std::ostream *os) {
    *os << param.test_name;
}

template <typename T> std::vector<ANNParamType<T>> getFunctionalityParams() {
    std::vector<ANNParamType<T>> params;
    GetANNFunctionalityData(params);
    return params;
}

// Basic functionality tests
template <typename T> void test_functionality(const ANNParamType<T> &param) {
    da_handle handle = nullptr;
    std::vector<da_int> k_ind, k_ind_twice;
    std::vector<T> k_dist, k_dist_twice;
    k_ind.resize(param.n_queries * param.k, 0);
    k_dist.resize(param.n_queries * param.k, 0.0);

    std::cout << "Functionality test: " << param.test_name << std::endl;
    // Basic usage
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_approx_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", param.metric.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", param.k),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_list", param.nlist), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_probe", param.nprobe), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", param.seed), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "k-means_iter", param.kmeans_iter),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "train fraction", param.train_fraction),
              da_status_success);
    EXPECT_EQ(da_approx_nn_set_training_data(handle, param.n_samples, param.n_features,
                                             param.X_train.data(), param.ldx_train),
              da_status_success);
    EXPECT_EQ(da_approx_nn_train<T>(handle), da_status_success);

    // check list_sizes is zero post training
    da_int size_nlist = param.nlist;
    std::vector<da_int> list_sizes(size_nlist);
    std::vector<da_int> zeros(size_nlist);

    EXPECT_EQ(da_handle_get_result(handle, da_approx_nn_list_sizes, &size_nlist,
                                   list_sizes.data()),
              da_status_success);

    EXPECT_ARR_EQ(size_nlist, list_sizes.data(), zeros.data(), 1, 1, 0, 0);

    EXPECT_EQ(da_approx_nn_add(handle, param.n_samples, param.n_features,
                               param.X_train.data(), param.ldx_train),
              da_status_success);

    if (!param.allow_empty_lists) {
        // Check there are no empty lists after adding data
        EXPECT_EQ(da_handle_get_result(handle, da_approx_nn_list_sizes, &size_nlist,
                                       list_sizes.data()),
                  da_status_success);

        for (da_int i = 0; i < size_nlist; i++) {
            EXPECT_GT(list_sizes[i], 0);
        }
    }

    EXPECT_EQ(da_approx_nn_kneighbors(handle, param.n_queries, param.n_features,
                                      param.X_test.data(), param.ldx_test, k_ind.data(),
                                      k_dist.data(), param.k, true),
              da_status_success);

    da_order param_order = (param.order == "column-major") ? column_major : row_major;
    if (!param.expected_kind.empty()) {
        EXPECT_ARR_EQ(param.n_queries * param.k, param.expected_kind, k_ind, 1, 1, 0, 0);
    } else {
        EXPECT_TRUE(validate_indices<T>(k_ind, param.n_queries, param.k, param.n_samples,
                                        param_order));
    }

    T eps = 3 * 1e3 * std::numeric_limits<T>::epsilon();
    EXPECT_ARR_NEAR(param.n_queries * param.k, param.expected_kdist, k_dist, eps);

    // Reset training data and check we get the same results through calling train_and_add
    EXPECT_EQ(da_approx_nn_set_training_data(handle, param.n_samples, param.n_features,
                                             param.X_train.data(), param.ldx_train),
              da_status_success);
    EXPECT_EQ(da_approx_nn_train_and_add<T>(handle), da_status_success);

    // Call with return_distance=false
    EXPECT_EQ(da_approx_nn_kneighbors(handle, param.n_queries, param.n_features,
                                      param.X_test.data(), param.ldx_test, k_ind.data(),
                                      nullptr, param.k, false),
              da_status_success);

    if (!param.expected_kind.empty()) {
        EXPECT_ARR_EQ(param.n_queries * param.k, param.expected_kind, k_ind, 1, 1, 0, 0);
    } else {
        EXPECT_TRUE(validate_indices<T>(k_ind, param.n_queries, param.k, param.n_samples,
                                        param_order));
    }

    // Call with return_distance=true
    EXPECT_EQ(da_approx_nn_kneighbors(handle, param.n_queries, param.n_features,
                                      param.X_test.data(), param.ldx_test, k_ind.data(),
                                      k_dist.data(), param.k, true),
              da_status_success);

    if (!param.expected_kind.empty()) {
        EXPECT_ARR_EQ(param.n_queries * param.k, param.expected_kind, k_ind, 1, 1, 0, 0);
    } else {
        EXPECT_TRUE(validate_indices<T>(k_ind, param.n_queries, param.k, param.n_samples,
                                        param_order));
    }

    EXPECT_ARR_NEAR(param.n_queries * param.k, param.expected_kdist, k_dist, eps);

    da_int size_centroids = param.nlist * param.n_features;
    da_int size_rinfo = 4;

    std::vector<T> centroids(size_centroids);
    std::vector<T> rinfo(size_rinfo);

    // Want a stricter tol for get_result comparisons
    T small_eps = 2 * std::numeric_limits<T>::epsilon();

    // Check get_result
    // For now we don't expose spherical kmeans through the public API so won't test
    // those results (cosine or inner product) here

    if (param.metric == "inner product" || param.metric == "cosine") {
        if (!param.expected_rinfo.empty()) {
            EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size_rinfo, rinfo.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_rinfo - 1, rinfo.data(), param.expected_rinfo.data(),
                            small_eps);
        }
    } else {
        if (!param.expected_centroids.empty()) {
            EXPECT_EQ(da_handle_get_result(handle, da_approx_nn_cluster_centroids,
                                           &size_centroids, centroids.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_centroids, centroids, param.expected_centroids,
                            small_eps);
        }

        if (!param.expected_rinfo.empty()) {
            EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size_rinfo, rinfo.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_rinfo, rinfo.data(), param.expected_rinfo.data(),
                            small_eps);
        }
    }

    // check list sizes sum to n_samples
    EXPECT_EQ(da_handle_get_result(handle, da_approx_nn_list_sizes, &size_nlist,
                                   list_sizes.data()),
              da_status_success);

    EXPECT_EQ(std::accumulate(list_sizes.begin(), list_sizes.end(), da_int(0)),
              param.n_samples);

    // Test adding the same data twice
    if (!param.expected_kind_two_adds.empty()) {
        std::vector<da_int> k_ind_two_adds;
        std::vector<T> k_dist_two_adds;
        k_ind_two_adds.resize(2 * param.n_queries * param.k, 0);
        k_dist_two_adds.resize(2 * param.n_queries * param.k, 0.0);
        // Second add call on the same handle after training
        EXPECT_EQ(da_approx_nn_add(handle, param.n_samples, param.n_features,
                                   param.X_train.data(), param.ldx_train),
                  da_status_success);

        // Search with 2x the number of neighbors
        EXPECT_EQ(da_approx_nn_kneighbors(handle, param.n_queries, param.n_features,
                                          param.X_test.data(), param.ldx_test,
                                          k_ind_two_adds.data(), k_dist_two_adds.data(),
                                          2 * param.k, true),
                  da_status_success);

        if (!param.expected_kind.empty()) {
            SortDuplicateResults(param_order, k_ind_two_adds.data(), param.n_queries,
                                 param.k * 2);

            EXPECT_ARR_EQ(2 * param.n_queries * param.k, param.expected_kind_two_adds,
                          k_ind_two_adds, 1, 1, 0, 0);
        } else {
            EXPECT_TRUE(validate_indices<T>(k_ind_two_adds, param.n_queries, param.k * 2,
                                            param.n_samples * 2, param_order));
        }
        EXPECT_ARR_NEAR(2 * param.n_queries * param.k, param.expected_kdist_two_adds,
                        k_dist_two_adds, eps);

        // Check list sizes sum to 2 * n_samples
        EXPECT_EQ(da_handle_get_result(handle, da_approx_nn_list_sizes, &size_nlist,
                                       list_sizes.data()),
                  da_status_success);

        EXPECT_EQ(std::accumulate(list_sizes.begin(), list_sizes.end(), da_int(0)),
                  2 * param.n_samples);
    }

    da_handle_destroy(&handle);
}

class DoubleFunctionalityTest : public testing::TestWithParam<ANNParamType<double>> {};
class FloatFunctionalityTest : public testing::TestWithParam<ANNParamType<float>> {};

TEST_P(DoubleFunctionalityTest, ParameterizedTest) {
    const ANNParamType<double> &p = GetParam();
    test_functionality(p);
}

TEST_P(FloatFunctionalityTest, ParameterizedTest) {
    const ANNParamType<float> &p = GetParam();
    test_functionality(p);
}

INSTANTIATE_TEST_SUITE_P(ANN_Functionality_Tests_Double, DoubleFunctionalityTest,
                         ::testing::ValuesIn(getFunctionalityParams<double>()));
INSTANTIATE_TEST_SUITE_P(ANN_Functionality_Tests_Float, FloatFunctionalityTest,
                         ::testing::ValuesIn(getFunctionalityParams<float>()));

template <typename T> std::vector<ANNParamType<T>> getRecallParams() {
    std::vector<ANNParamType<T>> params;
    GetANNRecallData(params);
    return params;
}

// Larger tests to check recall is satisfactory
// Recall = proportion of true nearest neighbors found by approximate search
template <typename T> void test_ann_recall(const ANNParamType<T> &param) {

    // Create handle and set options
    da_handle ann_handle = nullptr;

    EXPECT_EQ(da_handle_init<T>(&ann_handle, da_handle_approx_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(ann_handle, "metric", param.metric.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(ann_handle, "storage order", param.order.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(ann_handle, "algorithm", param.algorithm.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_int(ann_handle, "number of neighbors", param.k),
              da_status_success);
    EXPECT_EQ(da_options_set_int(ann_handle, "n_list", param.nlist), da_status_success);
    EXPECT_EQ(da_options_set_int(ann_handle, "n_probe", param.nprobe), da_status_success);
    EXPECT_EQ(da_options_set_int(ann_handle, "seed", param.seed), da_status_success);
    EXPECT_EQ(da_options_set_int(ann_handle, "k-means_iter", param.kmeans_iter),
              da_status_success);
    EXPECT_EQ(da_options_set(ann_handle, "train fraction", param.train_fraction),
              da_status_success);

    da_order order = (param.order == "column-major") ? column_major : row_major;

    // Get the training data
    std::string input_data_fname =
        std::string(DATA_DIR) + "/ann_data/" + param.csvname + "_data.csv";
    da_datastore csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(csv_store, "datastore precision", prec_name<T>()),
        da_status_success);
    EXPECT_EQ(da_data_load_from_csv(csv_store, input_data_fname.c_str()),
              da_status_success);

    da_int ncols, nrows;
    EXPECT_EQ(da_data_get_n_cols(csv_store, &ncols), da_status_success);
    EXPECT_EQ(da_data_get_n_rows(csv_store, &nrows), da_status_success);
    EXPECT_EQ(da_data_select_columns(csv_store, "features", 0, ncols - 1),
              da_status_success);

    da_int nfeat = ncols;
    da_int nsamples_train = nrows;
    // Extract the selections
    std::vector<T> X_train(nfeat * nsamples_train);
    da_int ldx_train = (order == column_major) ? nsamples_train : nfeat;

    EXPECT_EQ(da_data_extract_selection(csv_store, "features", order, X_train.data(),
                                        ldx_train),
              da_status_success);

    da_datastore_destroy(&csv_store);

    // Create the model
    EXPECT_EQ(da_approx_nn_set_training_data(ann_handle, nsamples_train, nfeat,
                                             X_train.data(), ldx_train),
              da_status_success);
    // Train and add to the model
    EXPECT_EQ(da_approx_nn_train_and_add<T>(ann_handle), da_status_success);

    // Get the test data
    input_data_fname = std::string(DATA_DIR) + "/ann_data/" + param.csvname + "_test.csv";
    csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(csv_store, "datastore precision", prec_name<T>()),
        da_status_success);
    EXPECT_EQ(da_data_load_from_csv(csv_store, input_data_fname.c_str()),
              da_status_success);

    EXPECT_EQ(da_data_get_n_cols(csv_store, &ncols), da_status_success);
    EXPECT_EQ(da_data_get_n_rows(csv_store, &nrows), da_status_success);
    EXPECT_EQ(da_data_select_columns(csv_store, "features", 0, ncols - 1),
              da_status_success);

    nfeat = ncols;
    da_int nsamples_test = nrows;
    // Extract the selections
    std::vector<T> X_test(nfeat * nsamples_test);
    da_int ldx_test = (order == column_major) ? nsamples_test : nfeat;

    EXPECT_EQ(
        da_data_extract_selection(csv_store, "features", order, X_test.data(), ldx_test),
        da_status_success);
    da_datastore_destroy(&csv_store);

    // Check that the recall is good enough
    da_int k_neigh = param.k;

    std::vector<da_int> computed_indices(k_neigh * nsamples_test);

    T recall;
    EXPECT_EQ(da_approx_nn_kneighbors(ann_handle, nsamples_test, nfeat, X_test.data(),
                                      ldx_test, computed_indices.data(), nullptr, k_neigh,
                                      false),
              da_status_success);

    // Compute recall based on inputs and computed_indices
    recall =
        compute_recall(param, computed_indices.data(), X_train.data(), nsamples_train,
                       nfeat, ldx_train, X_test.data(), nsamples_test, ldx_test);

    EXPECT_GT(recall, param.target_recall);
    std::cout << "Recall on the test data: " << recall << std::endl;

    da_handle_destroy(&ann_handle);
}

class DoubleRecallTest : public testing::TestWithParam<ANNParamType<double>> {};
class FloatRecallTest : public testing::TestWithParam<ANNParamType<float>> {};

TEST_P(DoubleRecallTest, ParameterizedTest) {
    const ANNParamType<double> &p = GetParam();
    test_ann_recall(p);
}

TEST_P(FloatRecallTest, ParameterizedTest) {
    const ANNParamType<float> &p = GetParam();
    test_ann_recall(p);
}

INSTANTIATE_TEST_SUITE_P(ANN_Recall_Tests_Double, DoubleRecallTest,
                         ::testing::ValuesIn(getRecallParams<double>()));
INSTANTIATE_TEST_SUITE_P(ANN_Recall_Tests_Float, FloatRecallTest,
                         ::testing::ValuesIn(getRecallParams<float>()));

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ANNTest, FloatTypes);

// Test IP + cosine centroids are unit norm
// And test get_result for centroids as it is not tested in the functionality test
TYPED_TEST(ANNTest, UnitNormIPCentroids) {
    std::vector<ANNParamType<TypeParam>> params;
    // Check both row and column major, IP + cosine
    WideColIP(params);
    OneListRowIP(params);
    OneListColCosine(params);
    WideRowCosine(params);

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_approx_nn), da_status_success);

    for (auto &param : params) {
        da_int size_centroids = param.nlist * param.n_features;
        std::vector<TypeParam> centroids(size_centroids);
        std::vector<TypeParam> centroid_norms(param.nlist, 0);
        std::vector<TypeParam> ones(param.nlist, 1);

        // Set options
        EXPECT_EQ(da_options_set_string(handle, "metric", param.metric.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "number of neighbors", param.k),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_list", param.nlist), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_probe", param.nprobe), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", param.seed), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "k-means_iter", param.kmeans_iter),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "train fraction", param.train_fraction),
                  da_status_success);
        EXPECT_EQ(da_approx_nn_set_training_data(handle, param.n_samples,
                                                 param.n_features, param.X_train.data(),
                                                 param.ldx_train),
                  da_status_success);
        EXPECT_EQ(da_approx_nn_train<TypeParam>(handle), da_status_success);

        EXPECT_EQ(da_handle_get_result(handle, da_approx_nn_cluster_centroids,
                                       &size_centroids, centroids.data()),
                  da_status_success);

        // Compute centroid norms
        if (param.order == "column-major") {
            for (da_int i = 0; i < param.nlist; i++) {
                centroid_norms[i] = datest_blas::cblas_nrm2(
                    param.n_features, centroids.data() + i, param.nlist);
            }
        } else {
            for (da_int i = 0; i < param.nlist; i++) {
                centroid_norms[i] = datest_blas::cblas_nrm2(
                    param.n_features, centroids.data() + i * param.n_features, 1);
            }
        }
        TypeParam eps = 1e2 * std::numeric_limits<TypeParam>::epsilon();
        EXPECT_ARR_NEAR(param.nlist, ones.data(), centroid_norms.data(), eps);
    }

    da_handle_destroy(&handle);
}

TYPED_TEST(ANNTest, ErrorExits) {
    // Get some data to use
    std::vector<ANNParamType<TypeParam>> params;
    ColSqEuclidean(params);

    std::vector<TypeParam> dist_arr(params[0].k * params[0].n_queries);
    std::vector<da_int> ind_arr(params[0].k * params[0].n_queries);

    TypeParam *null_arr = nullptr;
    da_int *null_arr_int = nullptr;

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_approx_nn), da_status_success);
    // Set some options
    EXPECT_EQ(da_options_set_string(handle, "metric", params[0].metric.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", params[0].order.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", params[0].algorithm.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", params[0].k),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_list", params[0].nlist), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_probe", params[0].nprobe), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", params[0].seed), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "k-means_iter", params[0].kmeans_iter),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "train fraction", params[0].train_fraction),
              da_status_success);

    // Check da_approx_nn_train(_and_add) early call error exits
    EXPECT_EQ(da_approx_nn_train<TypeParam>(handle), da_status_no_data);
    EXPECT_EQ(da_approx_nn_train_and_add<TypeParam>(handle), da_status_no_data);

    // Check da_approx_nn_set_training_data error exits
    EXPECT_EQ(
        da_approx_nn_set_training_data(handle, params[0].n_samples, params[0].n_features,
                                       params[0].X_train.data(), params[0].n_samples - 1),
        da_status_invalid_leading_dimension);
    EXPECT_EQ(da_approx_nn_set_training_data(handle, 0, params[0].n_features,
                                             params[0].X_train.data(),
                                             params[0].ldx_train),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_approx_nn_set_training_data(handle, params[0].n_samples, 0,
                                             params[0].X_train.data(),
                                             params[0].ldx_train),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_approx_nn_set_training_data(handle, params[0].n_samples,
                                             params[0].n_features, null_arr,
                                             params[0].ldx_train),
              da_status_invalid_pointer);

    // Call da_approx_nn_set_training_data successfully
    EXPECT_EQ(
        da_approx_nn_set_training_data(handle, params[0].n_samples, params[0].n_features,
                                       params[0].X_train.data(), params[0].ldx_train),
        da_status_success);

    // Check add and kneighbors early call error exits
    EXPECT_EQ(da_approx_nn_add(handle, params[0].n_samples, params[0].n_features,
                               params[0].X_train.data(), params[0].ldx_train),
              da_status_no_data);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      ind_arr.data(), dist_arr.data(), params[0].k, true),
              da_status_no_data);

    // Check we exit with n_list > n_samples
    EXPECT_EQ(da_options_set_int(handle, "n_list", params[0].n_samples + 10),
              da_status_success);
    EXPECT_EQ(da_approx_nn_train<TypeParam>(handle), da_status_invalid_array_dimension);
    // Reset n_list
    EXPECT_EQ(da_options_set_int(handle, "n_list", params[0].nlist), da_status_success);

    // Call da_approx_nn_train successfully
    EXPECT_EQ(da_approx_nn_train<TypeParam>(handle), da_status_success);

    // Check we must call da_approx_nn_add before da_approx_nn_kneighbors
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      ind_arr.data(), dist_arr.data(), params[0].k, true),
              da_status_no_data);

    // Check da_approx_nn_add error exits
    EXPECT_EQ(da_approx_nn_add(handle, params[0].n_samples, params[0].n_features,
                               params[0].X_train.data(), params[0].ldx_train - 1),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_approx_nn_add(handle, 0, params[0].n_features, params[0].X_train.data(),
                               params[0].ldx_train),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_approx_nn_add(handle, params[0].n_samples, 0, params[0].X_train.data(),
                               params[0].ldx_train),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_approx_nn_add(handle, params[0].n_samples, params[0].n_features,
                               null_arr, params[0].ldx_train),
              da_status_invalid_pointer);
    EXPECT_EQ(da_approx_nn_add(handle, params[0].n_samples, params[0].n_features + 1,
                               params[0].X_train.data(), params[0].ldx_train),
              da_status_invalid_input);

    // Call da_approx_nn_add successfully
    EXPECT_EQ(da_approx_nn_add(handle, params[0].n_samples, params[0].n_features,
                               params[0].X_train.data(), params[0].ldx_train),
              da_status_success);

    // Check da_approx_nn_kneighbors error exits
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test - 1,
                                      ind_arr.data(), dist_arr.data(), params[0].k, true),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, 0, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      ind_arr.data(), dist_arr.data(), params[0].k, true),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, 0,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      ind_arr.data(), dist_arr.data(), params[0].k, true),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries,
                                      params[0].n_features + 1, params[0].X_test.data(),
                                      params[0].ldx_test, ind_arr.data(), dist_arr.data(),
                                      params[0].k, true),
              da_status_invalid_input);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      null_arr, params[0].ldx_test, ind_arr.data(),
                                      dist_arr.data(), params[0].k, true),
              da_status_invalid_pointer);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      null_arr_int, dist_arr.data(), params[0].k, true),
              da_status_invalid_pointer);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      ind_arr.data(), null_arr, params[0].k, true),
              da_status_invalid_pointer);

    // Check we exit with more neighbors than data added
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      ind_arr.data(), dist_arr.data(),
                                      params[0].n_samples + 1, true),
              da_status_invalid_input);

    // Check we exit with n_probe > n_list
    EXPECT_EQ(da_options_set_int(handle, "n_probe", params[0].nlist + 1),
              da_status_success);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      ind_arr.data(), null_arr, params[0].k, true),
              da_status_invalid_input);
    // Reset n_probe
    EXPECT_EQ(da_options_set_int(handle, "n_probe", params[0].nprobe), da_status_success);

    // Check we can't update n_list after training
    EXPECT_EQ(da_options_set_int(handle, "n_list", params[0].nlist + 1),
              da_status_success);
    EXPECT_EQ(da_approx_nn_add(handle, params[0].n_samples, params[0].n_features,
                               params[0].X_train.data(), params[0].ldx_train),
              da_status_option_locked);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      ind_arr.data(), null_arr, params[0].k, false),
              da_status_option_locked);

    // Reset n_list
    EXPECT_EQ(da_options_set_int(handle, "n_list", params[0].nlist), da_status_success);

    // Check we can't update metric after training
    std::string new_metric =
        (params[0].metric == "sqeuclidean") ? "inner product" : "sqeuclidean";
    EXPECT_EQ(da_options_set_string(handle, "metric", new_metric.c_str()),
              da_status_success);
    EXPECT_EQ(da_approx_nn_add(handle, params[0].n_samples, params[0].n_features,
                               params[0].X_train.data(), params[0].ldx_train),
              da_status_option_locked);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      ind_arr.data(), null_arr, params[0].k, false),
              da_status_option_locked);
    // Reset metric
    EXPECT_EQ(da_options_set_string(handle, "metric", params[0].metric.c_str()),
              da_status_success);

    // Check we call kneighbors successfully
    EXPECT_EQ(da_approx_nn_kneighbors(handle, params[0].n_queries, params[0].n_features,
                                      params[0].X_test.data(), params[0].ldx_test,
                                      ind_arr.data(), dist_arr.data(), params[0].k, true),
              da_status_success);

    // Check da_handle_get_result error exits
    TypeParam result;
    da_int int_result;
    da_int dim = 2;
    // dim too small for rinfo
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, &result),
              da_status_invalid_array_dimension);
    // dim too small for centroids
    dim = params[0].nlist * params[0].n_features - 1;
    EXPECT_EQ(da_handle_get_result(handle, da_approx_nn_cluster_centroids, &dim, &result),
              da_status_invalid_array_dimension);
    // int query fail
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, &int_result),
              da_status_unknown_query);
    // unknown query fail
    EXPECT_EQ(da_handle_get_result(handle, da_pca_scores, &dim, &result),
              da_status_unknown_query);

    da_handle_destroy(&handle);

    // Incompatible options
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_approx_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "fake metric"),
              da_status_option_invalid_value);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "fake algorithm"),
              da_status_option_invalid_value);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", -1),
              da_status_option_invalid_value);
    EXPECT_EQ(da_options_set_int(handle, "n_list", -2), da_status_option_invalid_value);
    EXPECT_EQ(da_options_set_int(handle, "n_probe", -3), da_status_option_invalid_value);
    EXPECT_EQ(da_options_set_int(handle, "seed", -4), da_status_option_invalid_value);
    EXPECT_EQ(da_options_set_int(handle, "k-means_iter", -5),
              da_status_option_invalid_value);
    EXPECT_EQ(da_options_set(handle, "train fraction", (TypeParam)1.5),
              da_status_option_invalid_value);

    da_handle_destroy(&handle);
}

TYPED_TEST(ANNTest, MultipleCalls) {
    // Check we can repeatedly use the same handle
    std::vector<ANNParamType<TypeParam>> params;

    // Get some data
    PaddedColSqEuclidean(params);
    WideRowIP(params);
    OneListColEuclidean(params);

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_approx_nn), da_status_success);

    for (auto &param : params) {
        std::vector<da_int> k_ind, k_ind_twice;
        std::vector<TypeParam> k_dist, k_dist_twice;
        k_ind.resize(param.n_queries * param.k, 0);
        k_dist.resize(param.n_queries * param.k, 0.0);
        k_ind_twice.resize(2 * param.n_queries * param.k, 0);
        k_dist_twice.resize(2 * param.n_queries * param.k, 0.0);

        // Set options and perform functionality checks
        EXPECT_EQ(da_options_set_string(handle, "metric", param.metric.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "number of neighbors", param.k),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_list", param.nlist), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_probe", param.nprobe), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", param.seed), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "k-means_iter", param.kmeans_iter),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "train fraction", param.train_fraction),
                  da_status_success);
        EXPECT_EQ(da_approx_nn_set_training_data(handle, param.n_samples,
                                                 param.n_features, param.X_train.data(),
                                                 param.ldx_train),
                  da_status_success);
        EXPECT_EQ(da_approx_nn_train<TypeParam>(handle), da_status_success);

        EXPECT_EQ(da_approx_nn_add(handle, param.n_samples, param.n_features,
                                   param.X_train.data(), param.ldx_train),
                  da_status_success);

        EXPECT_EQ(da_approx_nn_kneighbors(handle, param.n_queries, param.n_features,
                                          param.X_test.data(), param.ldx_test,
                                          k_ind.data(), k_dist.data(), param.k, true),
                  da_status_success);

        EXPECT_ARR_EQ(param.n_queries * param.k, param.expected_kind, k_ind, 1, 1, 0, 0);

        TypeParam eps = 3 * 1e3 * std::numeric_limits<TypeParam>::epsilon();
        EXPECT_ARR_NEAR(param.n_queries * param.k, param.expected_kdist, k_dist, eps);

        EXPECT_EQ(da_approx_nn_kneighbors(handle, param.n_queries, param.n_features,
                                          param.X_test.data(), param.ldx_test,
                                          k_ind.data(), k_dist.data(), param.k, false),
                  da_status_success);

        EXPECT_ARR_EQ(param.n_queries * param.k, param.expected_kind, k_ind, 1, 1, 0, 0);

        // Add the same data twice
        if (!param.expected_kind_two_adds.empty()) {
            EXPECT_EQ(da_approx_nn_add(handle, param.n_samples, param.n_features,
                                       param.X_train.data(), param.ldx_train),
                      da_status_success);

            // Search with 2x the number of neighbors
            EXPECT_EQ(da_approx_nn_kneighbors(handle, param.n_queries, param.n_features,
                                              param.X_test.data(), param.ldx_test,
                                              k_ind_twice.data(), k_dist_twice.data(),
                                              2 * param.k, true),
                      da_status_success);

            da_order param_order =
                (param.order == "column-major") ? column_major : row_major;
            SortDuplicateResults(param_order, k_ind_twice.data(), param.n_queries,
                                 param.k * 2);

            EXPECT_ARR_EQ(2 * param.n_queries * param.k, param.expected_kind_two_adds,
                          k_ind_twice, 1, 1, 0, 0);
            EXPECT_ARR_NEAR(2 * param.n_queries * param.k, param.expected_kdist_two_adds,
                            k_dist_twice, eps);
        }
    }

    da_handle_destroy(&handle);
}

// Bad handle tests
TYPED_TEST(ANNTest, BadHandleTests) {

    // Handle not initialized
    da_handle handle = nullptr;
    EXPECT_EQ(da_approx_nn_train<TypeParam>(handle), da_status_handle_not_initialized);

    TypeParam X = 1;
    da_int I = 1;
    EXPECT_EQ(da_approx_nn_set_training_data(handle, 1, 1, &X, 1),
              da_status_handle_not_initialized);

    EXPECT_EQ(da_approx_nn_add(handle, 1, 1, &X, 1), da_status_handle_not_initialized);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, 1, 1, &X, 1, &I, nullptr, 1, false),
              da_status_handle_not_initialized);

    // Incorrect handle type
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_linmod), da_status_success);

    EXPECT_EQ(da_approx_nn_set_training_data(handle, 1, 1, &X, 1),
              da_status_invalid_handle_type);

    EXPECT_EQ(da_approx_nn_add(handle, 1, 1, &X, 1), da_status_invalid_handle_type);
    EXPECT_EQ(da_approx_nn_kneighbors(handle, 1, 1, &X, 1, &I, nullptr, 1, false),
              da_status_invalid_handle_type);

    da_handle_destroy(&handle);
}

// Incorrect handle precision
TEST(ANNTest, IncorrectHandlePrecision) {

    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_approx_nn), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_approx_nn), da_status_success);

    double Xd = 1.0;
    float Xs = 1.0f;
    da_int I = 1;

    EXPECT_EQ(da_approx_nn_set_training_data_d(handle_s, 1, 1, &Xd, 1),
              da_status_wrong_type);
    EXPECT_EQ(da_approx_nn_set_training_data_s(handle_d, 1, 1, &Xs, 1),
              da_status_wrong_type);

    EXPECT_EQ(da_approx_nn_train_d(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_approx_nn_train_s(handle_d), da_status_wrong_type);

    EXPECT_EQ(da_approx_nn_train_and_add_d(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_approx_nn_train_and_add_s(handle_d), da_status_wrong_type);

    EXPECT_EQ(da_approx_nn_add_d(handle_s, 1, 1, &Xd, 1), da_status_wrong_type);
    EXPECT_EQ(da_approx_nn_add_s(handle_d, 1, 1, &Xs, 1), da_status_wrong_type);

    EXPECT_EQ(da_approx_nn_kneighbors_d(handle_s, 1, 1, &Xd, 1, &I, nullptr, 1, false),
              da_status_wrong_type);

    EXPECT_EQ(da_approx_nn_kneighbors_s(handle_d, 1, 1, &Xs, 1, &I, nullptr, 1, false),
              da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}
