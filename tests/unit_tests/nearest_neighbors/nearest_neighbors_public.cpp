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
#include "nearest_neighbors_tests.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T> class NearestNeighborsTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> std::vector<NearestNeighborsParamType<T>> getParams() {
    std::vector<NearestNeighborsParamType<T>> params;
    GetNearestNeighborsData(params);
    return params;
}

template <typename T> void test_functionality(const NearestNeighborsParamType<T> &param) {

    da_handle handle = nullptr;

    std::cout << "Functionality test: " << param.name << std::endl;
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_nn), da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_options_set_string(handle, "metric", param.metric.c_str()),
              da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_options_set_string(handle, "weights", param.weights.c_str()),
              da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
              da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
              da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", param.n_neigh_knn),
              da_status_success)
        << da_handle_print_error_message(handle);

    EXPECT_EQ(da_options_set(handle, "minkowski parameter",
                             T(2.0) + 3 * std::numeric_limits<T>::epsilon()),
              da_status_success)
        << da_handle_print_error_message(handle);

    EXPECT_EQ(da_options_set_int(handle, "leaf size", param.leaf_size), da_status_success)
        << da_handle_print_error_message(handle);

    // Test classification
    EXPECT_EQ(da_nn_set_data(handle, param.n_samples, param.n_features,
                             param.X_train.data(), param.ldx_train),
              da_status_success)
        << da_handle_print_error_message(handle);

    // Compute the k-nearest neighbors and return the distances
    std::vector<T> kdist(param.n_neigh_kneighbors * param.n_queries);
    std::vector<da_int> kind(param.n_neigh_kneighbors * param.n_queries);

    EXPECT_EQ(da_nn_kneighbors(handle, param.n_queries, param.n_features,
                               param.X_test.data(), param.ldx_test, kind.data(),
                               kdist.data(), param.n_neigh_kneighbors, 1),
              da_status_success)
        << da_handle_print_error_message(handle);
    // In case of ties in distances the indices may differ, especially when using different algorithms.
    // Therefore, we do not check for exact equality of indices.
    EXPECT_ARR_NEAR(param.n_neigh_kneighbors * param.n_queries, kdist.data(),
                    param.expected_kdist.data(), param.tol);

    EXPECT_EQ(da_nn_set_labels<T>(handle, param.n_samples, param.y_train_class.data()),
              da_status_success)
        << da_handle_print_error_message(handle);

    da_int n_classes = 0; // Set n_classes to zero to do query for the required memory
    EXPECT_EQ(da_nn_classes<T>(handle, &n_classes, nullptr), da_status_success)
        << da_handle_print_error_message(handle);

    std::vector<T> proba(n_classes * param.n_queries);
    EXPECT_EQ(da_nn_classifier_predict_proba(handle, param.n_queries, param.n_features,
                                             param.X_test.data(), param.n_queries,
                                             proba.data(), knn_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);

    EXPECT_ARR_NEAR(n_classes * param.n_queries, proba.data(),
                    param.expected_proba_knn.data(), param.tol);

    std::vector<da_int> labels(param.n_queries);
    EXPECT_EQ(da_nn_classifier_predict(handle, param.n_queries, param.n_features,
                                       param.X_test.data(), param.n_queries,
                                       labels.data(), knn_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);

    EXPECT_ARR_NEAR(param.n_queries, labels.data(), param.expected_labels_knn.data(), 0);

    EXPECT_EQ(da_nn_set_targets(handle, param.n_samples, param.y_train_regression.data()),
              da_status_success)
        << da_handle_print_error_message(handle);

    std::vector<T> targets(param.n_queries);
    EXPECT_EQ(da_nn_regressor_predict(handle, param.n_queries, param.n_features,
                                      param.X_test.data(), param.n_queries,
                                      targets.data(), knn_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);

    EXPECT_ARR_NEAR(param.n_queries, targets.data(), param.expected_targets_knn.data(),
                    param.tol);

    EXPECT_EQ(da_options_set(handle, "radius", param.radius_constructor),
              da_status_success)
        << da_handle_print_error_message(handle);
    // Test radius neighbors
    da_int return_distance = 1;
    // Always sort the results for testing since trees can return neighbors in any order.
    da_int sort_results = 1;
    // Compute radius neighbors for each query point. Last element has the total number of radius neighbors and can be used for memory allocation.
    EXPECT_EQ(da_nn_radius_neighbors(handle, param.n_queries, param.n_features,
                                     param.X_test.data(), param.ldx_test,
                                     param.radius_neigh, return_distance, sort_results),
              da_status_success)
        << da_handle_print_error_message(handle);
    da_int array_size = param.n_queries + 1;
    std::vector<da_int> radius_neigh_count(array_size);
    // Return array that contains the number of radius neighbors for each query point and the total number of radius neighbors.
    EXPECT_EQ(da_handle_get_result(handle, da_nn_radius_neighbors_count, &array_size,
                                   radius_neigh_count.data()),
              da_status_success)
        << da_handle_print_error_message(handle);
    // Return the offsets to locate the radius neighbors for each query point.
    std::vector<da_int> radius_neigh_offsets(array_size);
    EXPECT_EQ(da_handle_get_result(handle, da_nn_radius_neighbors_offsets, &array_size,
                                   radius_neigh_offsets.data()),
              da_status_success)
        << da_handle_print_error_message(handle);
    // Allocate memory and extract the indices of the radius neighbors for each query point.
    da_int total_radius_neighbors = radius_neigh_count[param.n_queries];
    std::vector<da_int> radius_neigh_indices(total_radius_neighbors);
    EXPECT_EQ(da_handle_get_result(handle, da_nn_radius_neighbors_indices,
                                   &total_radius_neighbors, radius_neigh_indices.data()),
              da_status_success)
        << da_handle_print_error_message(handle);
    std::vector<T> radius_neigh_distances(total_radius_neighbors);
    EXPECT_EQ(da_handle_get_result(handle, da_nn_radius_neighbors_distances,
                                   &total_radius_neighbors,
                                   radius_neigh_distances.data()),
              da_status_success)
        << da_handle_print_error_message(handle);
    // Compare the distances with the expected ones.
    // The expected results are stored in std::vector<std::vector<T>> format, so we need to check that
    // the indexing logic works as expected.
    for (da_int i = 0; i < param.n_queries; i++) {
        // If there are no neighbors, radius_neigh_count[i] will be zero and the inner loop will not be executed.
        for (da_int j = 0; j < radius_neigh_count[i]; j++) {
            EXPECT_NEAR(radius_neigh_distances[radius_neigh_offsets[i] + j],
                        param.expected_radius_dist[i][j], param.tol);
        }
    }

    // Test probability estimates, labels, and targets for radius neighbors
    n_classes = 0; // Set n_classes to zero to do query for the required memory
    EXPECT_EQ(da_nn_classes<T>(handle, &n_classes, nullptr), da_status_success)
        << da_handle_print_error_message(handle);

    proba.resize(n_classes * param.n_queries);
    EXPECT_EQ(da_nn_classifier_predict_proba(handle, param.n_queries, param.n_features,
                                             param.X_test.data(), param.n_queries,
                                             proba.data(), radius_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);

    EXPECT_ARR_NEAR(n_classes * param.n_queries, proba.data(),
                    param.expected_proba_radius.data(), param.tol);

    EXPECT_EQ(da_nn_classifier_predict(handle, param.n_queries, param.n_features,
                                       param.X_test.data(), param.n_queries,
                                       labels.data(), radius_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);

    EXPECT_ARR_NEAR(param.n_queries, labels.data(), param.expected_labels_radius.data(),
                    0);

    EXPECT_EQ(da_nn_regressor_predict(handle, param.n_queries, param.n_features,
                                      param.X_test.data(), param.n_queries,
                                      targets.data(), radius_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);

    EXPECT_ARR_NEAR(param.n_queries, targets.data(), param.expected_targets_radius.data(),
                    param.tol);

    da_handle_destroy(&handle);
}

class DoubleFunctionalityTest
    : public testing::TestWithParam<NearestNeighborsParamType<double>> {};
class FloatFunctionalityTest
    : public testing::TestWithParam<NearestNeighborsParamType<float>> {};

template <typename T>
void PrintTo(const NearestNeighborsParamType<T> &param, ::std::ostream *os) {
    *os << param.name;
}

TEST_P(DoubleFunctionalityTest, ParameterizedTest) {
    const NearestNeighborsParamType<double> &p = GetParam();
    test_functionality(p);
}

TEST_P(FloatFunctionalityTest, ParameterizedTest) {
    const NearestNeighborsParamType<float> &p = GetParam();
    test_functionality(p);
}

INSTANTIATE_TEST_SUITE_P(nn_Functionality_Tests_Double, DoubleFunctionalityTest,
                         ::testing::ValuesIn(getParams<double>()));
INSTANTIATE_TEST_SUITE_P(nn_Functionality_Tests_Float, FloatFunctionalityTest,
                         ::testing::ValuesIn(getParams<float>()));

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NearestNeighborsTest, FloatTypes);

// When weights equal distance we could end up with divisions by zero.
// Check that these cases are handled correctly.
TYPED_TEST(NearestNeighborsTest, AccuracyTestingZeroData_knn) {
    da_handle handle = nullptr;
    da_int n_samples = 4;
    da_int n_features = 3;
    da_int n_queries = 3;
    std::vector<TypeParam> X_train{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    std::vector<da_int> y_train_class{1, 2, 0, 1};
    std::vector<TypeParam> y_train_regression{2.2, 4.5, 0.5, 2.2};
    std::vector<TypeParam> X_test{0., 0., 0., 0., 0., 0., 0., 0., 0.};
    TypeParam tol = 100 * std::numeric_limits<TypeParam>::epsilon();
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_nn), da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_options_set_string(handle, "weights", "distance"), da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", 3), da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_nn_set_targets(handle, n_samples, y_train_regression.data()),
              da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_nn_set_labels<TypeParam>(handle, n_samples, y_train_class.data()),
              da_status_success)
        << da_handle_print_error_message(handle);

    // Test Regression
    EXPECT_EQ(da_nn_set_data(handle, n_samples, n_features, X_train.data(), n_samples),
              da_status_success)
        << da_handle_print_error_message(handle);
    std::vector<TypeParam> targets(n_queries);
    EXPECT_EQ(da_nn_regressor_predict(handle, n_queries, n_features, X_test.data(),
                                      n_queries, targets.data(), knn_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);
    std::vector<TypeParam> expected_targets(n_queries, 2.4);
    EXPECT_ARR_NEAR(n_queries, targets.data(), expected_targets.data(), tol);

    // Test Classification
    da_int n_classes = 0; // Set n_classes to zero to do query for the required memory
    EXPECT_EQ(da_nn_classes<TypeParam>(handle, &n_classes, nullptr), da_status_success)
        << da_handle_print_error_message(handle);

    std::vector<TypeParam> expected_proba(n_classes * n_queries, 0.3333333333333333);
    std::vector<TypeParam> proba(n_classes * n_queries);
    EXPECT_EQ(da_nn_classifier_predict_proba(handle, n_queries, n_features, X_test.data(),
                                             n_queries, proba.data(), knn_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_ARR_NEAR(n_classes * n_queries, proba.data(), expected_proba.data(), tol);

    std::vector<da_int> labels(n_queries);
    EXPECT_EQ(da_nn_classifier_predict(handle, n_queries, n_features, X_test.data(),
                                       n_queries, labels.data(), knn_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);
    std::vector<da_int> expected_labels(n_queries, 0);
    EXPECT_ARR_NEAR(n_queries, labels.data(), expected_labels.data(), 0);

    da_handle_destroy(&handle);
}

// When weights equal distance we could end up with divisions by zero.
// Check that these cases are handled correctly.
TYPED_TEST(NearestNeighborsTest, AccuracyTestingZeroData_radius) {
    da_handle handle = nullptr;
    da_int n_samples = 4;
    da_int n_features = 3;
    da_int n_queries = 3;
    std::vector<TypeParam> X_train{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    std::vector<da_int> y_train_class{1, 2, 0, 1};
    std::vector<TypeParam> y_train_regression{2.2, 4.5, 0.5, 2.2};
    std::vector<TypeParam> X_test{0., 0., 0., 0., 0., 0., 0., 0., 0.};
    TypeParam tol = 100 * std::numeric_limits<TypeParam>::epsilon();
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_nn), da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_options_set_string(handle, "weights", "distance"), da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_options_set(handle, "radius", TypeParam(0.5)), da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_nn_set_labels<TypeParam>(handle, n_samples, y_train_class.data()),
              da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_EQ(da_nn_set_targets(handle, n_samples, y_train_regression.data()),
              da_status_success)
        << da_handle_print_error_message(handle);

    // Test Regression
    EXPECT_EQ(da_nn_set_data(handle, n_samples, n_features, X_train.data(), n_samples),
              da_status_success)
        << da_handle_print_error_message(handle);

    // Test radius neighbors
    TypeParam radius = 0.1;
    da_int return_distance = 1;
    da_int sort_results = 0; // No need to sort for this test
    EXPECT_EQ(da_nn_radius_neighbors(handle, n_queries, n_features, X_test.data(),
                                     n_queries, radius, return_distance, sort_results),
              da_status_success)
        << da_handle_print_error_message(handle);

    std::vector<TypeParam> targets(n_queries);
    EXPECT_EQ(da_nn_regressor_predict(handle, n_queries, n_features, X_test.data(),
                                      n_queries, targets.data(), radius_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);

    std::vector<TypeParam> expected_targets(n_queries, 2.35);
    EXPECT_ARR_NEAR(n_queries, targets.data(), expected_targets.data(), tol);

    // Test Classification
    da_int n_classes = 0; // Set n_classes to zero to do query for the required memory
    EXPECT_EQ(da_nn_classes<TypeParam>(handle, &n_classes, nullptr), da_status_success)
        << da_handle_print_error_message(handle);

    std::vector<TypeParam> expected_proba{0.25, 0.25, 0.25, 0.5, 0.5,
                                          0.5,  0.25, 0.25, 0.25};
    std::vector<TypeParam> proba(n_classes * n_queries);
    EXPECT_EQ(da_nn_classifier_predict_proba(handle, n_queries, n_features, X_test.data(),
                                             n_queries, proba.data(), radius_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);
    EXPECT_ARR_NEAR(n_classes * n_queries, proba.data(), expected_proba.data(), tol);

    std::vector<da_int> labels(n_queries);
    EXPECT_EQ(da_nn_classifier_predict(handle, n_queries, n_features, X_test.data(),
                                       n_queries, labels.data(), radius_search_mode),
              da_status_success)
        << da_handle_print_error_message(handle);
    std::vector<da_int> expected_labels{1, 1, 1};
    EXPECT_ARR_NEAR(n_queries, labels.data(), expected_labels.data(), 0);

    // Test using the count & index retrieval combination
    da_int array_size = -1;
    std::vector<da_int> radius_neigh_count(1);
    EXPECT_EQ(da_handle_get_result(handle, da_nn_radius_neighbors_count, &array_size,
                                   radius_neigh_count.data()),
              da_status_invalid_array_dimension)
        << da_handle_print_error_message(handle);

    EXPECT_EQ(array_size, n_queries + 1);
    radius_neigh_count.resize(array_size);
    EXPECT_EQ(da_handle_get_result(handle, da_nn_radius_neighbors_count, &array_size,
                                   radius_neigh_count.data()),
              da_status_success)
        << da_handle_print_error_message(handle);
    // All results should have 0 as the distances and the indices should be all neighbors (0-3)
    std::vector<da_int> expected_radius_neigh_indices{0, 1, 2, 3};
    std::vector<da_int> radius_ind(n_samples);
    std::vector<TypeParam> radius_dist(n_samples);
    da_int temp_size = 0;
    for (da_int i = 0; i < n_queries; i++) {
        EXPECT_EQ(radius_neigh_count[i], n_samples);
        radius_ind[0] = i;
        // First check that passing an invalid size returns an error and returns the required size
        temp_size = 0;
        EXPECT_EQ(da_handle_get_result(handle, da_nn_radius_neighbors_indices_index,
                                       &temp_size, radius_ind.data()),
                  da_status_invalid_array_dimension)
            << da_handle_print_error_message(handle);
        EXPECT_EQ(temp_size, n_samples);
        EXPECT_EQ(da_handle_get_result(handle, da_nn_radius_neighbors_indices_index,
                                       &n_samples, radius_ind.data()),
                  da_status_success)
            << da_handle_print_error_message(handle);
        radius_dist[0] = (TypeParam)i;
        // First check that passing an invalid size returns an error and returns the required size
        temp_size = 0;
        EXPECT_EQ(da_handle_get_result(handle, da_nn_radius_neighbors_distances_index,
                                       &temp_size, radius_dist.data()),
                  da_status_invalid_array_dimension)
            << da_handle_print_error_message(handle);
        EXPECT_EQ(temp_size, n_samples);
        EXPECT_EQ(da_handle_get_result(handle, da_nn_radius_neighbors_distances_index,
                                       &n_samples, radius_dist.data()),
                  da_status_success)
            << da_handle_print_error_message(handle);
        // Test that all distances are zero
        for (da_int j = 0; j < n_samples; j++) {
            EXPECT_EQ(radius_dist[j], TypeParam(0));
        }
        // Test that all indices are as expected
        EXPECT_ARR_NEAR(n_samples, radius_ind.data(),
                        expected_radius_neigh_indices.data(), 0);
    }
    da_handle_destroy(&handle);
}

std::string ErrorExits_print(std::string param) {
    std::string ss = "Test for invalid value of " + param + " failed.";
    return ss;
}

TYPED_TEST(NearestNeighborsTest, ClassificationErrorExits) {
    NearestNeighborsParamType<TypeParam> param;
    da_handle nn_handle = nullptr;

    TypeParam *X_invalid = nullptr;
    da_int *y_invalid = nullptr;
    std::vector<TypeParam> X(1), proba(1), dist(1);
    std::vector<da_int> y(1), ind(1);
    da_int n_classes = -1;
    EXPECT_EQ(da_handle_init<TypeParam>(&nn_handle, da_handle_nn), da_status_success)
        << da_handle_print_error_message(nn_handle);

    // Try calling functionality before setting training data the model.
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_no_data)
        << "Testing calling kneighbors() before setting data failed.";
    EXPECT_EQ(da_nn_classes<TypeParam>(nn_handle, &n_classes, nullptr), da_status_no_data)
        << "Testing calling classes() before setting data failed.";
    EXPECT_EQ(da_nn_classifier_predict(nn_handle, param.n_queries, param.n_features,
                                       X.data(), param.ldx_test, y.data(),
                                       knn_search_mode),
              da_status_no_data)
        << "Testing calling predict() before setting data failed.";
    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, param.n_queries, param.n_features,
                                             X.data(), param.ldx_test, proba.data(),
                                             knn_search_mode),
              da_status_no_data)
        << "Testing calling predict_proba() before setting data failed.";

    // Try calling functionality before setting training labels to the model.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << da_handle_print_error_message(nn_handle);
    EXPECT_EQ(da_nn_classes<TypeParam>(nn_handle, &n_classes, nullptr), da_status_no_data)
        << "Testing calling classes() before setting labels failed.";
    EXPECT_EQ(da_nn_classifier_predict(nn_handle, param.n_queries, param.n_features,
                                       X.data(), param.ldx_test, y.data(),
                                       knn_search_mode),
              da_status_no_data)
        << "Testing calling predict() before setting labels failed.";
    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, param.n_queries, param.n_features,
                                             X.data(), param.ldx_test, proba.data(),
                                             knn_search_mode),
              da_status_no_data)
        << "Testing calling predict_proba() before setting labels failed.";

    // Reset handle to not have any training data or labels
    da_handle_destroy(&nn_handle);
    EXPECT_EQ(da_handle_init<TypeParam>(&nn_handle, da_handle_nn), da_status_success)
        << da_handle_print_error_message(nn_handle);
    // Tests for Brute force algorithm
    EXPECT_EQ(da_options_set_string(nn_handle, "algorithm", "brute"), da_status_success)
        << "Setting algorithm to brute force failed.";
    // Invalid pointers in set_data
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X_invalid,
                             param.ldx_train),
              da_status_invalid_pointer)
        << ErrorExits_print("X_train");
    // Invalid handle
    EXPECT_EQ(da_nn_set_data(nullptr, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_handle_not_initialized)
        << ErrorExits_print("nn_handle");
    // Invalid dimensions
    EXPECT_EQ(da_nn_set_data(nn_handle, -1, param.n_features, X.data(), param.ldx_train),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_samples");
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, 0, X.data(), param.ldx_train),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(), 0),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_train");
    EXPECT_EQ(da_nn_set_labels<TypeParam>(nn_handle, param.n_samples, y_invalid),
              da_status_invalid_pointer)
        << ErrorExits_print("y_train_class");

    // Put valid training data so that we test effectively the rest of the APIs.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success);
    EXPECT_EQ(da_nn_set_labels<TypeParam>(nn_handle, param.n_samples, y.data()),
              da_status_success);

    // Invalid options
    EXPECT_EQ(da_options_set_int(nn_handle, "number of neighbors", 0),
              da_status_option_invalid_value)
        << "Invalid option 'number of neighbors' test failed.";
    EXPECT_EQ(da_options_set_string(nn_handle, "algorithm", "nonexistent"),
              da_status_option_invalid_value)
        << "Invalid option 'algorithm' test failed.";
    EXPECT_EQ(da_options_set_string(nn_handle, "metric", "nonexistent"),
              da_status_option_invalid_value)
        << "Invalid option 'metric' test failed.";
    EXPECT_EQ(da_options_set_string(nn_handle, "weights", "callable"),
              da_status_option_invalid_value)
        << "Invalid option 'weights' test failed.";

    // Incorrect inputs in classes()
    n_classes = 2;
    EXPECT_EQ(da_nn_classes<TypeParam>(nn_handle, &n_classes, nullptr),
              da_status_invalid_pointer)
        << ErrorExits_print("classes");

    // Check get_result()
    TypeParam fp_result;
    da_int int_result;
    da_int dim = 1;
    EXPECT_EQ(da_handle_get_result(nn_handle, da_rinfo, &dim, &fp_result),
              da_status_unknown_query);
    EXPECT_EQ(da_handle_get_result(nn_handle, da_rinfo, &dim, &int_result),
              da_status_unknown_query);

    // Incorrect inputs in kneighbors()
    EXPECT_EQ(da_nn_kneighbors(nn_handle, -1, param.n_features, X.data(), param.ldx_test,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, 3, X.data(), param.ldx_test,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, nullptr,
                               param.ldx_test, ind.data(), dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("X_test");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(), 0,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(), 3, 1),
              da_status_invalid_input)
        << ErrorExits_print("n_neigh_kneighbors");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(), 0, 1),
              da_status_invalid_input)
        << ErrorExits_print("n_neigh_kneighbors");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, nullptr, dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("k_ind");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("k_dist");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_success)
        << "Testing that if distances are not required, k_dist can be nullptr";

    // Set options so that the rest of the APIs do not fail for the wrong reasons.
    EXPECT_EQ(da_options_set_int(nn_handle, "number of neighbors", 1), da_status_success)
        << "Setting correct number of neighbors failed.";

    // Incorrect inputs in predict_proba()
    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, -1, param.n_features, X.data(),
                                             param.ldx_test, proba.data(),
                                             knn_search_mode),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, param.n_queries, 2, X.data(),
                                             param.ldx_test, proba.data(),
                                             knn_search_mode),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, param.n_queries, -1, X.data(),
                                             param.ldx_test, proba.data(),
                                             knn_search_mode),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, param.n_queries, param.n_features,
                                             X.data(), -1, proba.data(), knn_search_mode),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, param.n_queries, param.n_features,
                                             X.data(), param.ldx_test, nullptr,
                                             knn_search_mode),
              da_status_invalid_pointer)
        << ErrorExits_print("proba");
    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, param.n_queries, param.n_features,
                                             X.data(), param.ldx_test, proba.data(),
                                             da_nn_search_mode(3)),
              da_status_invalid_input)
        << ErrorExits_print("proba");

    // Incorrect inputs in predict()
    EXPECT_EQ(da_nn_classifier_predict(nn_handle, -1, param.n_features, X.data(),
                                       param.ldx_test, y.data(), knn_search_mode),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_classifier_predict(nn_handle, param.n_queries, 2, X.data(),
                                       param.ldx_test, y.data(), knn_search_mode),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_classifier_predict(nn_handle, param.n_queries, param.n_features,
                                       X.data(), -1, y.data(), knn_search_mode),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_classifier_predict(nn_handle, -1, param.n_features, X.data(),
                                       param.ldx_test, nullptr, knn_search_mode),
              da_status_invalid_pointer)
        << ErrorExits_print("y_test");

    // Check that if we set up an option after setting the training data, it does not
    // throw an error. Checking internal check_options_update() function.
    EXPECT_EQ(da_options_set_string(nn_handle, "algorithm", "auto"), da_status_success)
        << "Setting algorithm to k-d tree failed.";
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";

    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(nn_handle, "leaf size", (da_int)10), da_status_success)
        << "Setting leaf size to 10 failed.";
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if leaf size option has been set, it cannot be changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(nn_handle, "metric", "minkowski"), da_status_success)
        << "Setting metric to minkowski failed.";
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if metric option has been set, it cannot be changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(nn_handle, "minkowski parameter", TypeParam(1.0)),
              da_status_success)
        << "Setting metric to minkowski failed.";
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if minkowski parameter option has been set, it cannot be "
           "changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << "Setting training data again with the same data failed.";

    // Since we tested all variables checked in check_options_update(), for da_nn_kneighbors(),
    // for predict() and predict_proba() we can set only one of the options to ensure that
    // the check_options_update() has been called.
    EXPECT_EQ(da_options_set(nn_handle, "algorithm", "kd tree"), da_status_success)
        << "Setting algorithm to k-d tree failed.";
    EXPECT_EQ(da_nn_classifier_predict(nn_handle, param.n_queries, param.n_features,
                                       X.data(), param.ldx_test, y.data(),
                                       knn_search_mode),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";
    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, param.n_queries, param.n_features,
                                             X.data(), param.ldx_test, proba.data(),
                                             knn_search_mode),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";

    // Testing that calling da_nn_set_labels with a different number of samples throws an error.
    EXPECT_EQ(da_nn_set_labels<TypeParam>(nn_handle, param.n_samples + 1, y.data()),
              da_status_invalid_array_dimension)
        << "Testing that setting y_train with different number of samples than the one "
           "passed in set_data() fails.";
    // Reset handle to test the other way around
    da_handle_destroy(&nn_handle);
    EXPECT_EQ(da_handle_init<TypeParam>(&nn_handle, da_handle_nn), da_status_success);
    // Testing that calling da_nn_set_labels with a different number of samples throws an error.
    EXPECT_EQ(da_nn_set_labels<TypeParam>(nn_handle, param.n_samples, y.data()),
              da_status_success);
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples + 1, param.n_features, X.data(),
                             param.ldx_train),
              da_status_invalid_array_dimension)
        << "Testing that setting y_train with different number of samples than the one "
           "passed in set_data() fails.";

    da_handle_destroy(&nn_handle);
}

TYPED_TEST(NearestNeighborsTest, RegressionErrorExits) {
    NearestNeighborsParamType<TypeParam> param;
    da_handle nn_handle = nullptr;

    TypeParam *X_invalid = nullptr;
    TypeParam *y_invalid = nullptr;
    std::vector<TypeParam> X(1), proba(1), dist(1), y(1);
    std::vector<da_int> ind(1);
    EXPECT_EQ(da_handle_init<TypeParam>(&nn_handle, da_handle_nn), da_status_success);

    // Try calling functionality before setting training data the model.
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_no_data)
        << "Testing calling kneighbors() before setting data failed.";
    EXPECT_EQ(da_nn_regressor_predict(nn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y.data(),
                                      knn_search_mode),
              da_status_no_data)
        << "Testing calling predict() before setting data failed.";

    // Try calling functionality before setting training targets to the model.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << da_handle_print_error_message(nn_handle);
    EXPECT_EQ(da_nn_regressor_predict(nn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y.data(),
                                      knn_search_mode),
              da_status_no_data)
        << "Testing calling predict() before setting targets failed.";

    EXPECT_EQ(da_options_set_string(nn_handle, "algorithm", "brute"), da_status_success)
        << "Setting algorithm to brute force failed.";
    // Invalid pointers in set_data
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X_invalid,
                             param.ldx_train),
              da_status_invalid_pointer)
        << ErrorExits_print("X_train");
    // Invalid handle
    EXPECT_EQ(da_nn_set_data(nullptr, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_handle_not_initialized)
        << ErrorExits_print("nn_handle");
    // Invalid dimensions
    EXPECT_EQ(da_nn_set_data(nn_handle, -1, param.n_features, X.data(), param.ldx_train),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_samples");
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, 0, X.data(), param.ldx_train),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(), 0),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_train");

    EXPECT_EQ(da_nn_set_targets(nn_handle, param.n_samples, y_invalid),
              da_status_invalid_pointer)
        << ErrorExits_print("y_train_regression");
    // Put valid training data so that we test effectively the rest of the APIs.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success);
    EXPECT_EQ(da_nn_set_targets(nn_handle, param.n_samples, y.data()), da_status_success);
    // Incorrect inputs in kneighbors()
    EXPECT_EQ(da_nn_kneighbors(nn_handle, -1, param.n_features, X.data(), param.ldx_test,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, 3, X.data(), param.ldx_test,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, nullptr,
                               param.ldx_test, ind.data(), dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("X_test");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(), 0,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(), 3, 1),
              da_status_invalid_input)
        << ErrorExits_print("n_neigh_kneighbors");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, nullptr, dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("k_ind");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("k_dist");
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_success)
        << "Testing that if distances are not required, k_dist can be nullptr";

    // Set options so that the rest of the APIs do not fail for the wrong reasons.
    EXPECT_EQ(da_options_set_int(nn_handle, "number of neighbors", 1), da_status_success)
        << "Setting correct number of neighbors failed.";

    // Incorrect inputs in predict()
    EXPECT_EQ(da_nn_regressor_predict(nn_handle, -1, param.n_features, X.data(),
                                      param.ldx_test, y.data(), knn_search_mode),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_regressor_predict(nn_handle, param.n_queries, 2, X.data(),
                                      param.ldx_test, y.data(), knn_search_mode),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_regressor_predict(nn_handle, param.n_queries, param.n_features,
                                      X.data(), -1, y.data(), knn_search_mode),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_regressor_predict(nn_handle, -1, param.n_features, X.data(),
                                      param.ldx_test, nullptr, knn_search_mode),
              da_status_invalid_pointer)
        << ErrorExits_print("y_test");

    // Check that if we set up an option after setting the training data, it does not
    // throw an error. Checking internal check_options_update() function.
    EXPECT_EQ(da_options_set_string(nn_handle, "algorithm", "auto"), da_status_success)
        << "Setting algorithm to k-d tree failed.";
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";

    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(nn_handle, "leaf size", (da_int)10), da_status_success)
        << "Setting leaf size to 10 failed.";
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if leaf size option has been set, it cannot be changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(nn_handle, "metric", "minkowski"), da_status_success)
        << "Setting metric to minkowski failed.";
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if metric option has been set, it cannot be changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(nn_handle, "minkowski parameter", TypeParam(1.0)),
              da_status_success)
        << "Setting metric to minkowski failed.";
    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if minkowski parameter option has been set, it cannot be "
           "changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << "Setting training data again with the same data failed.";

    // Since we tested all variables checked in check_options_update(), for da_nn_kneighbors(),
    // for predict() and predict_proba() we can set only one of the options to ensure that
    // the check_options_update() has been called.
    EXPECT_EQ(da_options_set(nn_handle, "algorithm", "kd tree"), da_status_success)
        << "Setting algorithm to k-d tree failed.";
    EXPECT_EQ(da_nn_regressor_predict(nn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y.data(),
                                      knn_search_mode),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";

    da_handle_destroy(&nn_handle);

    // Test locked options when auto option is set
    EXPECT_EQ(da_handle_init<TypeParam>(&nn_handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_options_set(nn_handle, "algorithm", "auto"), da_status_success)
        << "Setting auto option to true failed.";
    EXPECT_EQ(da_options_set(nn_handle, "metric", "manhattan"), da_status_success)
        << "Setting metric to manhattan failed.";
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success)
        << "Setting training data again with the same data failed.";
    EXPECT_EQ(da_nn_set_targets(nn_handle, param.n_samples, y.data()), da_status_success)
        << "Setting training targets data again with the same data failed.";
    // Change metric after set_data with auto
    EXPECT_EQ(da_options_set(nn_handle, "metric", "euclidean"), da_status_success)
        << "Setting metric to euclidean failed.";
    EXPECT_EQ(da_nn_regressor_predict(nn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y.data(),
                                      knn_search_mode),
              da_status_option_locked)
        << "Testing that if metric option has been set when algorithm is auto, it cannot "
           "be changed "
           "after setting the training data failed.";
    // Change minkowski parameter after set_data with auto
    EXPECT_EQ(da_options_set(nn_handle, "minkowski parameter", TypeParam(2.0)),
              da_status_success)
        << "Setting minkowski parameter to 2.0 failed.";
    EXPECT_EQ(da_nn_regressor_predict(nn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y.data(),
                                      knn_search_mode),
              da_status_option_locked)
        << "Testing that if minkowski parameter option has been set when algorithm is "
           "auto, it cannot be changed "
           "after setting the training data failed.";

    // Testing that calling da_nn_set_targets with a different number of samples throws an error.
    EXPECT_EQ(da_nn_set_targets(nn_handle, param.n_samples + 1, y.data()),
              da_status_invalid_array_dimension)
        << "Testing that setting y_train with different number of samples than the one "
           "passed in set_data() fails.";

    // Reset handle to test the other way around
    da_handle_destroy(&nn_handle);
    EXPECT_EQ(da_handle_init<TypeParam>(&nn_handle, da_handle_nn), da_status_success);
    // Testing that calling da_nn_set_targets with a different number of samples throws an error.
    EXPECT_EQ(da_nn_set_targets(nn_handle, param.n_samples, y.data()), da_status_success);
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples + 1, param.n_features, X.data(),
                             param.ldx_train),
              da_status_invalid_array_dimension)
        << "Testing that setting y_train with different number of samples than the one "
           "passed in set_data() fails.";

    da_handle_destroy(&nn_handle);
}

TYPED_TEST(NearestNeighborsTest, InvalidHandleErrorExits) {
    NearestNeighborsParamType<TypeParam> param;
    da_handle nn_handle = nullptr;

    std::vector<TypeParam> X(1), proba(1), dist(1), y_reg(1);
    std::vector<da_int> ind(1), y_class(1);
    da_int n_classes = 0;

    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_set_data() with invalid handle "
           "failed.";

    EXPECT_EQ(da_nn_set_targets(nn_handle, param.n_samples, y_reg.data()),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_set_targets() with invalid handle "
           "failed.";

    EXPECT_EQ(da_nn_set_labels<TypeParam>(nn_handle, param.n_samples, y_class.data()),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_set_labels() with invalid handle "
           "failed.";

    EXPECT_EQ(da_nn_regressor_predict(nn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y_reg.data(),
                                      knn_search_mode),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_regressor_predict() with invalid handle failed.";

    EXPECT_EQ(da_nn_classifier_predict(nn_handle, param.n_queries, param.n_features,
                                       X.data(), param.ldx_test, y_class.data(),
                                       knn_search_mode),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_classifier_predict() with invalid handle failed.";

    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, param.n_queries, param.n_features,
                                             X.data(), param.ldx_test, X.data(),
                                             knn_search_mode),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_classifier_predict_proba() with invalid handle "
           "failed.";

    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_kneighbors() with invalid handle failed.";

    EXPECT_EQ(da_nn_classes<TypeParam>(nn_handle, &n_classes, nullptr),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_classes() with invalid handle failed.";

    // If TypeParam is float set handle to double, otherwise set it to float
    if constexpr (std::is_same_v<TypeParam, float>) {
        EXPECT_EQ(da_handle_init<double>(&nn_handle, da_handle_nn), da_status_success);
    } else {
        EXPECT_EQ(da_handle_init<float>(&nn_handle, da_handle_nn), da_status_success);
    }

    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_wrong_type)
        << "Testing calling da_nn_set_data() with wrong handle type "
           "failed.";

    EXPECT_EQ(da_nn_set_targets(nn_handle, param.n_samples, y_reg.data()),
              da_status_wrong_type)
        << "Testing calling da_nn_set_targets() with wrong handle type "
           "failed.";

    EXPECT_EQ(da_nn_set_labels<TypeParam>(nn_handle, param.n_samples, y_class.data()),
              da_status_wrong_type)
        << "Testing calling da_nn_set_labels() with wrong handle type "
           "failed.";

    EXPECT_EQ(da_nn_regressor_predict(nn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y_reg.data(),
                                      knn_search_mode),
              da_status_wrong_type)
        << "Testing calling da_nn_regressor_predict() with wrong handle type failed.";

    EXPECT_EQ(da_nn_classifier_predict(nn_handle, param.n_queries, param.n_features,
                                       X.data(), param.ldx_test, y_class.data(),
                                       knn_search_mode),
              da_status_wrong_type)
        << "Testing calling da_nn_classifier_predict() with wrong handle type failed.";

    EXPECT_EQ(da_nn_classifier_predict_proba(nn_handle, param.n_queries, param.n_features,
                                             X.data(), param.ldx_test, X.data(),
                                             knn_search_mode),
              da_status_wrong_type)
        << "Testing calling da_nn_classifier_predict_proba() with wrong handle type "
           "failed.";

    EXPECT_EQ(da_nn_kneighbors(nn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_wrong_type)
        << "Testing calling da_nn_kneighbors() with wrong handle type failed.";

    EXPECT_EQ(da_nn_classes<TypeParam>(nn_handle, &n_classes, nullptr),
              da_status_wrong_type)
        << "Testing calling da_nn_classes() with wrong handle type failed.";

    da_handle_destroy(&nn_handle);
}

TYPED_TEST(NearestNeighborsTest, IncompatibleMetrics) {
    NearestNeighborsParamType<TypeParam> param;
    da_handle nn_handle = nullptr;

    std::vector<TypeParam> X(1), proba(1), dist(1), y(1);
    std::vector<da_int> ind(1);
    EXPECT_EQ(da_handle_init<TypeParam>(&nn_handle, da_handle_nn), da_status_success);

    EXPECT_EQ(da_options_set_string(nn_handle, "algorithm", "kd tree"),
              da_status_success);
    EXPECT_EQ(da_options_set_string(nn_handle, "metric", "cosine"), da_status_success);

    // Incompatible options
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_incompatible_options);

    EXPECT_EQ(da_options_set_string(nn_handle, "metric", "sqeuclidean"),
              da_status_success);

    // Incompatible options
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_incompatible_options);

    EXPECT_EQ(da_options_set_string(nn_handle, "metric", "minkowski"), da_status_success);
    EXPECT_EQ(da_options_set(nn_handle, "minkowski parameter", TypeParam(0.5)),
              da_status_success);

    // Incompatible options
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_incompatible_options);

    EXPECT_EQ(da_options_set_string(nn_handle, "algorithm", "ball tree"),
              da_status_success);

    // Incompatible options
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_incompatible_options);

    EXPECT_EQ(da_options_set_string(nn_handle, "metric", "sqeuclidean"),
              da_status_success);

    // Incompatible options
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_incompatible_options);

    EXPECT_EQ(da_options_set_string(nn_handle, "metric", "cosine"), da_status_success);
    // Incompatible options
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_incompatible_options);

    da_handle_destroy(&nn_handle);
}

TYPED_TEST(NearestNeighborsTest, RadiusNeighborsErrorExits) {
    NearestNeighborsParamType<TypeParam> param;
    da_handle nn_handle = nullptr;

    std::vector<TypeParam> X(1, 1.0);
    std::vector<da_int> y(1, 0);
    TypeParam radius = TypeParam(1.0);

    EXPECT_EQ(da_handle_init<TypeParam>(&nn_handle, da_handle_nn), da_status_success);

    // Try calling functionality before setting training data the model.
    EXPECT_EQ(da_nn_radius_neighbors(nn_handle, param.n_queries, param.n_features,
                                     X.data(), param.ldx_test, radius, 0, 0),
              da_status_no_data)
        << "Testing calling radius_neighbors() before setting data failed.";

    // Put valid training data so that we test effectively the rest of the APIs.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success);

    // Invalid options
    if constexpr (std::is_same_v<TypeParam, float>) {
        EXPECT_EQ(da_options_set_real_s(nn_handle, "radius", -1.0),
                  da_status_option_invalid_value)
            << "Invalid option 'radius' test failed.";
    } else {
        EXPECT_EQ(da_options_set_real_d(nn_handle, "radius", -1.0),
                  da_status_option_invalid_value)
            << "Invalid option 'radius' test failed.";
    }

    // Incorrect inputs in radius_cneighbors()
    EXPECT_EQ(da_nn_radius_neighbors(nn_handle, -1, param.n_features, X.data(),
                                     param.ldx_test, radius, 0, 0),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_radius_neighbors(nn_handle, param.n_queries, 3, X.data(),
                                     param.ldx_test, radius, 0, 0),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    if constexpr (std::is_same_v<TypeParam, float>) {
        EXPECT_EQ(da_nn_radius_neighbors_s(nn_handle, param.n_queries, param.n_features,
                                           nullptr, param.ldx_test, radius, 0, 0),
                  da_status_invalid_pointer)
            << ErrorExits_print("X_test");
    } else {
        EXPECT_EQ(da_nn_radius_neighbors_d(nn_handle, param.n_queries, param.n_features,
                                           nullptr, param.ldx_test, radius, 0, 0),
                  da_status_invalid_pointer)
            << ErrorExits_print("X_test");
    }
    EXPECT_EQ(da_nn_radius_neighbors(nn_handle, param.n_queries, param.n_features,
                                     X.data(), 0, radius, 0, 0),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_radius_neighbors(nn_handle, param.n_queries, param.n_features,
                                     X.data(), param.ldx_test, radius, 0, 1),
              da_status_invalid_input)
        << ErrorExits_print("sort_results");
    // Check that if we set up an option after setting the training data, it does not
    // throw an error. Checking internal check_options_update() function.
    EXPECT_EQ(da_options_set_string(nn_handle, "algorithm", "kd tree"), da_status_success)
        << "Setting algorithm to kd tree failed.";
    EXPECT_EQ(da_nn_radius_neighbors(nn_handle, param.n_queries, param.n_features,
                                     X.data(), param.ldx_test, radius, 0, 0),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";

    EXPECT_EQ(da_options_set_string(nn_handle, "algorithm", "kd tree"), da_status_success)
        << "Setting algorithm to kd tree failed.";
    // Put valid training data so that we test effectively the rest of the APIs.
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success);
    EXPECT_EQ(da_options_set(nn_handle, "leaf size", (da_int)10), da_status_success)
        << "Setting leaf size to 10 failed.";
    EXPECT_EQ(da_nn_radius_neighbors(nn_handle, param.n_queries, param.n_features,
                                     X.data(), param.ldx_test, radius, 0, 0),
              da_status_option_locked)
        << "Testing that if leaf size option has been set, it cannot be changed "
           "after setting the training data failed.";

    // Check get_result when it is called before radius_neighbors()
    TypeParam fp_result = 0;
    da_int int_result = 0;
    da_int dim = 1;
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_distances, &dim,
                                   &fp_result),
              da_status_no_data)
        << "Testing get_result with da_nn_radius_neighbors_distances before "
           "radius_neighbors() failed.";
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_distances_index,
                                   &dim, &fp_result),
              da_status_no_data)
        << "Testing get_result with da_nn_radius_neighbors_distances_index before "
           "radius_neighbors() failed.";
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_indices, &dim,
                                   &int_result),
              da_status_no_data)
        << "Testing get_result with da_nn_radius_neighbors_indices before "
           "radius_neighbors() failed.";
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_indices_index, &dim,
                                   &int_result),
              da_status_no_data)
        << "Testing get_result with da_nn_radius_neighbors_indices_index before "
           "radius_neighbors() failed.";
    EXPECT_EQ(
        da_handle_get_result(nn_handle, da_nn_radius_neighbors_count, &dim, &int_result),
        da_status_no_data)
        << "Testing get_result with da_nn_radius_neighbors_count before "
           "radius_neighbors() failed.";
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_offsets, &dim,
                                   &int_result),
              da_status_no_data)
        << "Testing get_result with da_nn_radius_neighbors_offsets before "
           "radius_neighbors() failed.";

    // Set training data and radius neighbors so that we can test get_result error exits
    EXPECT_EQ(da_nn_set_data(nn_handle, param.n_samples, param.n_features, X.data(),
                             param.ldx_train),
              da_status_success);
    EXPECT_EQ(da_nn_radius_neighbors(nn_handle, param.n_queries, param.n_features,
                                     X.data(), param.ldx_test, radius, 1, 1),
              da_status_success)
        << da_handle_print_error_message(nn_handle);
    // Call get_result with the wrong size
    dim = -1;
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_distances, &dim,
                                   &fp_result),
              da_status_invalid_array_dimension)
        << "Testing get_result with da_nn_radius_neighbors_distances with invalid dim "
           "failed.";
    // dim should be 1 for distances
    EXPECT_EQ(dim, 1);
    dim = -1;
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_indices, &dim,
                                   &int_result),
              da_status_invalid_array_dimension)
        << "Testing get_result with da_nn_radius_neighbors_indices with invalid dim "
           "failed.";
    EXPECT_EQ(dim, 1);
    EXPECT_EQ(
        da_handle_get_result(nn_handle, da_nn_radius_neighbors_count, &dim, &int_result),
        da_status_invalid_array_dimension)
        << "Testing get_result with da_nn_radius_neighbors_count with invalid dim "
           "failed.";
    EXPECT_EQ(dim, 2);
    dim = -1;
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_indices_index, &dim,
                                   &int_result),
              da_status_invalid_array_dimension)
        << "Testing get_result with da_nn_radius_neighbors_indices_index with invalid "
           "dim failed.";
    EXPECT_EQ(dim, 1);
    dim = -1;
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_distances_index,
                                   &dim, &fp_result),
              da_status_invalid_array_dimension)
        << "Testing get_result with da_nn_radius_neighbors_distances_index with invalid "
           "dim failed.";
    EXPECT_EQ(dim, 1);
    dim = -1;
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_offsets, &dim,
                                   &int_result),
              da_status_invalid_array_dimension)
        << "Testing get_result with da_nn_radius_neighbors_offsets with invalid dim "
           "failed.";

    // Call get_result with invalid index
    dim = 1;
    // Negative index
    int_result = -1;
    fp_result = TypeParam(-1);
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_indices_index, &dim,
                                   &int_result),
              da_status_invalid_input)
        << "Testing get_result with da_nn_radius_neighbors_indices_index with invalid "
           "index failed.";
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_distances_index,
                                   &dim, &fp_result),
              da_status_invalid_input)
        << "Testing get_result with da_nn_radius_neighbors_distances_index with invalid "
           "index failed.";
    // too large index
    int_result = 3;
    fp_result = TypeParam(3);
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_indices_index, &dim,
                                   &int_result),
              da_status_invalid_input)
        << "Testing get_result with da_nn_radius_neighbors_indices_index with invalid "
           "index failed.";
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_distances_index,
                                   &dim, &fp_result),
              da_status_invalid_input)
        << "Testing get_result with da_nn_radius_neighbors_distances_index with invalid "
           "index failed.";

    // Test that trying to extract neighbor distances when they were not requested returns an error
    EXPECT_EQ(da_nn_radius_neighbors(nn_handle, param.n_queries, param.n_features,
                                     X.data(), param.ldx_test, radius, 0, 0),
              da_status_success);
    fp_result = 0.0;
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_distances, &dim,
                                   &fp_result),
              da_status_no_data)
        << "Testing get_result with da_nn_radius_neighbors_distances when distances were "
           "not requested failed.";
    EXPECT_EQ(da_handle_get_result(nn_handle, da_nn_radius_neighbors_distances_index,
                                   &dim, &fp_result),
              da_status_no_data)
        << "Testing get_result with da_nn_radius_neighbors_distances_index when "
           "distances were not requested failed.";

    da_handle_destroy(&nn_handle);
}