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

#include <iostream>
#include <limits>
#include <list>
#include <stdio.h>
#include <string.h>

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "knn_tests.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T> class knnTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> std::vector<KNNParamType<T>> getParams() {
    std::vector<KNNParamType<T>> params;
    GetKNNData(params);
    return params;
}

template <typename T> void test_functionality(const KNNParamType<T> &param) {

    da_handle handle = nullptr;

    std::cout << "Functionality test: " << param.name << std::endl;
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", param.metric.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "weights", param.weights.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", param.n_neigh_knn),
              da_status_success);

    EXPECT_EQ(da_options_set(handle, "minkowski parameter",
                             T(2.0) + 3 * std::numeric_limits<T>::epsilon()),
              da_status_success);

    EXPECT_EQ(da_options_set_int(handle, "leaf size", param.leaf_size),
              da_status_success);

    // Test classification
    EXPECT_EQ(da_nn_classifier_set_training_data(
                  handle, param.n_samples, param.n_features, param.X_train.data(),
                  param.ldx_train, param.y_train_class.data()),
              da_status_success);

    // Compute the k-nearest neighbors and return the distances
    std::vector<T> kdist(param.n_neigh_kneighbors * param.n_queries);
    std::vector<da_int> kind(param.n_neigh_kneighbors * param.n_queries);

    EXPECT_EQ(da_nn_kneighbors(handle, param.n_queries, param.n_features,
                               param.X_test.data(), param.ldx_test, kind.data(),
                               kdist.data(), param.n_neigh_kneighbors, 1),
              da_status_success);

    // In case of ties in distances the indices may differ, especially when using different algorithms.
    // Therefore, we do not check for exact equality of indices.
    EXPECT_ARR_NEAR(param.n_neigh_kneighbors * param.n_queries, kdist.data(),
                    param.expected_kdist.data(), param.tol);

    da_int n_classes = 0; // Set n_classes to zero to do query for the required memory
    EXPECT_EQ(da_nn_classes<T>(handle, &n_classes, nullptr), da_status_success);

    std::vector<T> proba(n_classes * param.n_queries);
    EXPECT_EQ(da_nn_classifier_predict_proba(handle, param.n_queries, param.n_features,
                                             param.X_test.data(), param.n_queries,
                                             proba.data()),
              da_status_success);

    EXPECT_ARR_NEAR(n_classes * param.n_queries, proba.data(),
                    param.expected_proba.data(), param.tol);

    std::vector<da_int> labels(param.n_queries);
    EXPECT_EQ(da_nn_classifier_predict(handle, param.n_queries, param.n_features,
                                       param.X_test.data(), param.n_queries,
                                       labels.data()),
              da_status_success);

    EXPECT_ARR_NEAR(param.n_queries, labels.data(), param.expected_labels.data(), 0);

    // Test regression
    EXPECT_EQ(da_nn_regressor_set_training_data(handle, param.n_samples, param.n_features,
                                                param.X_train.data(), param.ldx_train,
                                                param.y_train_regression.data()),
              da_status_success);

    // Zero out distances and indices to test the regression prediction
    std::fill(kdist.begin(), kdist.end(), 0.0);
    std::fill(kind.begin(), kind.end(), 0);
    EXPECT_EQ(da_nn_kneighbors(handle, param.n_queries, param.n_features,
                               param.X_test.data(), param.ldx_test, kind.data(),
                               kdist.data(), param.n_neigh_kneighbors, 1),
              da_status_success);
    // In case of ties in distances the indices may differ, especially when using different algorithms.
    // Therefore, we do not check for exact equality of indices.
    EXPECT_ARR_NEAR(param.n_neigh_kneighbors * param.n_queries, kdist.data(),
                    param.expected_kdist.data(), param.tol);

    std::vector<T> targets(param.n_queries);
    EXPECT_EQ(da_nn_regressor_predict(handle, param.n_queries, param.n_features,
                                      param.X_test.data(), param.n_queries,
                                      targets.data()),
              da_status_success);

    EXPECT_ARR_NEAR(param.n_queries, targets.data(), param.expected_targets.data(),
                    param.tol);

    da_handle_destroy(&handle);
}

class DoubleFunctionalityTest : public testing::TestWithParam<KNNParamType<double>> {};
class FloatFunctionalityTest : public testing::TestWithParam<KNNParamType<float>> {};

template <typename T> void PrintTo(const KNNParamType<T> &param, ::std::ostream *os) {
    *os << param.name;
}

TEST_P(DoubleFunctionalityTest, ParameterizedTest) {
    const KNNParamType<double> &p = GetParam();
    test_functionality(p);
}

TEST_P(FloatFunctionalityTest, ParameterizedTest) {
    const KNNParamType<float> &p = GetParam();
    test_functionality(p);
}

INSTANTIATE_TEST_SUITE_P(knn_Functionality_Tests_Double, DoubleFunctionalityTest,
                         ::testing::ValuesIn(getParams<double>()));
INSTANTIATE_TEST_SUITE_P(knn_Functionality_Tests_Float, FloatFunctionalityTest,
                         ::testing::ValuesIn(getParams<float>()));

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(knnTest, FloatTypes);

// When weights equal distance we could end up with divisions by zero.
// Check that these cases are handled correctly.
TYPED_TEST(knnTest, AccuracyTestingZeroData) {
    da_handle handle = nullptr;
    da_int n_samples = 4;
    da_int n_features = 3;
    da_int n_queries = 3;
    std::vector<TypeParam> X_train{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    std::vector<da_int> y_train_class{1, 2, 0, 1};
    std::vector<TypeParam> y_train_regression{2.2, 4.5, 0.5, 2.2};
    std::vector<TypeParam> X_test{0., 0., 0., 0., 0., 0., 0., 0., 0.};
    TypeParam tol = 100 * std::numeric_limits<TypeParam>::epsilon();
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "weights", "distance"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", 3), da_status_success);

    // Test Regression
    EXPECT_EQ(da_nn_regressor_set_training_data(handle, n_samples, n_features,
                                                X_train.data(), n_samples,
                                                y_train_regression.data()),
              da_status_success);

    std::vector<TypeParam> targets(n_queries);
    EXPECT_EQ(da_nn_regressor_predict(handle, n_queries, n_features, X_test.data(),
                                      n_queries, targets.data()),
              da_status_success);

    std::vector<TypeParam> expected_targets(n_queries, 2.4);
    EXPECT_ARR_NEAR(n_queries, targets.data(), expected_targets.data(), tol);

    // Test Classification
    EXPECT_EQ(da_nn_classifier_set_training_data(handle, n_samples, n_features,
                                                 X_train.data(), n_samples,
                                                 y_train_class.data()),
              da_status_success);

    da_int n_classes = 0; // Set n_classes to zero to do query for the required memory
    EXPECT_EQ(da_nn_classes<TypeParam>(handle, &n_classes, nullptr), da_status_success);

    std::vector<TypeParam> expected_proba(n_classes * n_queries, 0.3333333333333333);
    std::vector<TypeParam> proba(n_classes * n_queries);
    EXPECT_EQ(da_nn_classifier_predict_proba(handle, n_queries, n_features, X_test.data(),
                                             n_queries, proba.data()),
              da_status_success);
    EXPECT_ARR_NEAR(n_classes * n_queries, proba.data(), expected_proba.data(), tol);

    std::vector<da_int> labels(n_queries);
    EXPECT_EQ(da_nn_classifier_predict(handle, n_queries, n_features, X_test.data(),
                                       n_queries, labels.data()),
              da_status_success);
    std::vector<da_int> expected_labels(n_queries, 0);
    EXPECT_ARR_NEAR(n_queries, labels.data(), expected_labels.data(), 0);

    da_handle_destroy(&handle);
}

std::string ErrorExits_print(std::string param) {
    std::string ss = "Test for invalid value of " + param + " failed.";
    return ss;
}

TYPED_TEST(knnTest, ClassificationErrorExits) {
    KNNParamType<TypeParam> param;
    da_handle knn_handle = nullptr;

    TypeParam *X_invalid = nullptr;
    da_int *y_invalid = nullptr;
    std::vector<TypeParam> X(1), proba(1), dist(1);
    std::vector<da_int> y(1), ind(1);
    da_int n_classes = -1;
    EXPECT_EQ(da_handle_init<TypeParam>(&knn_handle, da_handle_nn), da_status_success);

    // Try calling functionality before setting training data the model.
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_no_data)
        << "Testing calling kneighbors() before setting data failed.";
    EXPECT_EQ(da_nn_classes<TypeParam>(knn_handle, &n_classes, nullptr),
              da_status_no_data)
        << "Testing calling classes() before setting data failed.";
    EXPECT_EQ(da_nn_classifier_predict(knn_handle, param.n_queries, param.n_features,
                                       X.data(), param.ldx_test, y.data()),
              da_status_no_data)
        << "Testing calling predict() before setting data failed.";
    EXPECT_EQ(da_nn_classifier_predict_proba(knn_handle, param.n_queries,
                                             param.n_features, X.data(), param.ldx_test,
                                             proba.data()),
              da_status_no_data)
        << "Testing calling predict_proba() before setting data failed.";

    EXPECT_EQ(da_options_set_string(knn_handle, "algorithm", "brute"), da_status_success)
        << "Setting algorithm to brute force failed.";
    // Invalid pointers in set_training_data
    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples,
                                                 param.n_features, X_invalid,
                                                 param.ldx_train, y.data()),
              da_status_invalid_pointer)
        << ErrorExits_print("X_train");
    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples,
                                                 param.n_features, X.data(),
                                                 param.ldx_train, y_invalid),
              da_status_invalid_pointer)
        << ErrorExits_print("y_train_class");
    // Invalid handle
    EXPECT_EQ(da_nn_classifier_set_training_data(nullptr, param.n_samples,
                                                 param.n_features, X.data(),
                                                 param.ldx_train, y.data()),
              da_status_handle_not_initialized)
        << ErrorExits_print("knn_handle");
    // Invalid dimensions
    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, -1, param.n_features,
                                                 X.data(), param.ldx_train, y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_samples");
    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples, 0, X.data(),
                                                 param.ldx_train, y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples,
                                                 param.n_features, X.data(), 0, y.data()),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_train");

    // Put valid training data so that we test effectively the rest of the APIs.
    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples,
                                                 param.n_features, X.data(),
                                                 param.ldx_test, y.data()),
              da_status_success);

    // Invalid options
    EXPECT_EQ(da_options_set_int(knn_handle, "number of neighbors", 0),
              da_status_option_invalid_value)
        << "Invalid option 'number of neighbors' test failed.";
    EXPECT_EQ(da_options_set_string(knn_handle, "algorithm", "nonexistent"),
              da_status_option_invalid_value)
        << "Invalid option 'algorithm' test failed.";
    EXPECT_EQ(da_options_set_string(knn_handle, "metric", "nonexistent"),
              da_status_option_invalid_value)
        << "Invalid option 'metric' test failed.";
    EXPECT_EQ(da_options_set_string(knn_handle, "weights", "callable"),
              da_status_option_invalid_value)
        << "Invalid option 'weights' test failed.";

    // Incorrect inputs in classes()
    n_classes = 2;
    EXPECT_EQ(da_nn_classes<TypeParam>(knn_handle, &n_classes, nullptr),
              da_status_invalid_pointer)
        << ErrorExits_print("classes");

    // Check get_result()
    TypeParam fp_result;
    da_int int_result;
    da_int dim = 1;
    EXPECT_EQ(da_handle_get_result(knn_handle, da_rinfo, &dim, &fp_result),
              da_status_unknown_query);
    EXPECT_EQ(da_handle_get_result(knn_handle, da_rinfo, &dim, &int_result),
              da_status_unknown_query);

    // Incorrect inputs in kneighbors()
    EXPECT_EQ(da_nn_kneighbors(knn_handle, -1, param.n_features, X.data(), param.ldx_test,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, 3, X.data(), param.ldx_test,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(), 0,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(), 3, 1),
              da_status_invalid_input)
        << ErrorExits_print("n_neigh_kneighbors");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(), 0, 1),
              da_status_invalid_input)
        << ErrorExits_print("n_neigh_kneighbors");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, nullptr, dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("k_ind");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("k_dist");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_success)
        << "Testing that if distances are not required, k_dist can be nullptr";

    // Set options so that the rest of the APIs do not fail for the wrong reasons.
    EXPECT_EQ(da_options_set_int(knn_handle, "number of neighbors", 1), da_status_success)
        << "Setting correct number of neighbors failed.";

    // Incorrect inputs in predict_proba()
    EXPECT_EQ(da_nn_classifier_predict_proba(knn_handle, -1, param.n_features, X.data(),
                                             param.ldx_test, proba.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_classifier_predict_proba(knn_handle, param.n_queries, 2, X.data(),
                                             param.ldx_test, proba.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_classifier_predict_proba(knn_handle, param.n_queries, -1, X.data(),
                                             param.ldx_test, proba.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_classifier_predict_proba(knn_handle, param.n_queries,
                                             param.n_features, X.data(), -1,
                                             proba.data()),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_classifier_predict_proba(knn_handle, param.n_queries,
                                             param.n_features, X.data(), param.ldx_test,
                                             nullptr),
              da_status_invalid_pointer)
        << ErrorExits_print("proba");

    // Incorrect inputs in predict()
    EXPECT_EQ(da_nn_classifier_predict(knn_handle, -1, param.n_features, X.data(),
                                       param.ldx_test, y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_classifier_predict(knn_handle, param.n_queries, 2, X.data(),
                                       param.ldx_test, y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_classifier_predict(knn_handle, param.n_queries, param.n_features,
                                       X.data(), -1, y.data()),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_classifier_predict(knn_handle, -1, param.n_features, X.data(),
                                       param.ldx_test, nullptr),
              da_status_invalid_pointer)
        << ErrorExits_print("y_test");

    // Check that if we set up an option after setting the training data, it does not
    // throw an error. Checking internal check_options_update() function.
    EXPECT_EQ(da_options_set_string(knn_handle, "algorithm", "auto"), da_status_success)
        << "Setting algorithm to k-d tree failed.";
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";

    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples,
                                                 param.n_features, X.data(),
                                                 param.ldx_test, y.data()),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(knn_handle, "leaf size", (da_int)10), da_status_success)
        << "Setting leaf size to 10 failed.";
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if leaf size option has been set, it cannot be changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples,
                                                 param.n_features, X.data(),
                                                 param.ldx_test, y.data()),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(knn_handle, "metric", "minkowski"), da_status_success)
        << "Setting metric to minkowski failed.";
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if metric option has been set, it cannot be changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples,
                                                 param.n_features, X.data(),
                                                 param.ldx_test, y.data()),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(knn_handle, "minkowski parameter", TypeParam(1.0)),
              da_status_success)
        << "Setting metric to minkowski failed.";
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if minkowski parameter option has been set, it cannot be "
           "changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples,
                                                 param.n_features, X.data(),
                                                 param.ldx_test, y.data()),
              da_status_success)
        << "Setting training data again with the same data failed.";

    // Since we tested all variables checked in check_options_update(), for da_nn_kneighbors(),
    // for predict() and predict_proba() we can set only one of the options to ensure that
    // the check_options_update() has been called.
    EXPECT_EQ(da_options_set(knn_handle, "algorithm", "kd tree"), da_status_success)
        << "Setting algorithm to k-d tree failed.";
    EXPECT_EQ(da_nn_classifier_predict(knn_handle, param.n_queries, param.n_features,
                                       X.data(), param.ldx_test, y.data()),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";
    EXPECT_EQ(da_nn_classifier_predict_proba(knn_handle, param.n_queries,
                                             param.n_features, X.data(), param.ldx_test,
                                             proba.data()),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";

    da_handle_destroy(&knn_handle);
}

TYPED_TEST(knnTest, RegressionErrorExits) {
    KNNParamType<TypeParam> param;
    da_handle knn_handle = nullptr;

    TypeParam *X_invalid = nullptr;
    TypeParam *y_invalid = nullptr;
    std::vector<TypeParam> X(1), proba(1), dist(1), y(1);
    std::vector<da_int> ind(1);
    EXPECT_EQ(da_handle_init<TypeParam>(&knn_handle, da_handle_nn), da_status_success);

    // Try calling functionality before setting training data the model.
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_no_data)
        << "Testing calling kneighbors() before setting data failed.";
    EXPECT_EQ(da_nn_regressor_predict(knn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y.data()),
              da_status_no_data)
        << "Testing calling predict() before setting data failed.";
    EXPECT_EQ(da_options_set_string(knn_handle, "algorithm", "brute"), da_status_success)
        << "Setting algorithm to brute force failed.";
    // Invalid pointers in set_training_data
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X_invalid,
                                                param.ldx_train, y.data()),
              da_status_invalid_pointer)
        << ErrorExits_print("X_train");
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y_invalid),
              da_status_invalid_pointer)
        << ErrorExits_print("y_train_class");
    // Invalid handle
    EXPECT_EQ(da_nn_regressor_set_training_data(nullptr, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y.data()),
              da_status_handle_not_initialized)
        << ErrorExits_print("knn_handle");
    // Invalid dimensions
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, -1, param.n_features,
                                                X.data(), param.ldx_train, y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_samples");
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples, 0, X.data(),
                                                param.ldx_train, y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(), 0, y.data()),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_train");

    // Put valid training data so that we test effectively the rest of the APIs.
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_test, y.data()),
              da_status_success);

    // Incorrect inputs in kneighbors()
    EXPECT_EQ(da_nn_kneighbors(knn_handle, -1, param.n_features, X.data(), param.ldx_test,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, 3, X.data(), param.ldx_test,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(), 0,
                               ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(), 3, 1),
              da_status_invalid_input)
        << ErrorExits_print("n_neigh_kneighbors");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, nullptr, dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("k_ind");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("k_dist");
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_success)
        << "Testing that if distances are not required, k_dist can be nullptr";

    // Set options so that the rest of the APIs do not fail for the wrong reasons.
    EXPECT_EQ(da_options_set_int(knn_handle, "number of neighbors", 1), da_status_success)
        << "Setting correct number of neighbors failed.";

    // Incorrect inputs in predict()
    EXPECT_EQ(da_nn_regressor_predict(knn_handle, -1, param.n_features, X.data(),
                                      param.ldx_test, y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_nn_regressor_predict(knn_handle, param.n_queries, 2, X.data(),
                                      param.ldx_test, y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_nn_regressor_predict(knn_handle, param.n_queries, param.n_features,
                                      X.data(), -1, y.data()),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_nn_regressor_predict(knn_handle, -1, param.n_features, X.data(),
                                      param.ldx_test, nullptr),
              da_status_invalid_pointer)
        << ErrorExits_print("y_test");

    // Check that if we set up an option after setting the training data, it does not
    // throw an error. Checking internal check_options_update() function.
    EXPECT_EQ(da_options_set_string(knn_handle, "algorithm", "auto"), da_status_success)
        << "Setting algorithm to k-d tree failed.";
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";

    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_test, y.data()),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(knn_handle, "leaf size", (da_int)10), da_status_success)
        << "Setting leaf size to 10 failed.";
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if leaf size option has been set, it cannot be changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_test, y.data()),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(knn_handle, "metric", "minkowski"), da_status_success)
        << "Setting metric to minkowski failed.";
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if metric option has been set, it cannot be changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_test, y.data()),
              da_status_success)
        << "Setting training data again with the same data failed.";

    EXPECT_EQ(da_options_set(knn_handle, "minkowski parameter", TypeParam(1.0)),
              da_status_success)
        << "Setting metric to minkowski failed.";
    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), nullptr,
                               param.n_neigh_kneighbors, 0),
              da_status_option_locked)
        << "Testing that if minkowski parameter option has been set, it cannot be "
           "changed "
           "after setting the training data failed.";
    // Call training_data with the same data again, it should not throw an error.
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_test, y.data()),
              da_status_success)
        << "Setting training data again with the same data failed.";

    // Since we tested all variables checked in check_options_update(), for da_nn_kneighbors(),
    // for predict() and predict_proba() we can set only one of the options to ensure that
    // the check_options_update() has been called.
    EXPECT_EQ(da_options_set(knn_handle, "algorithm", "kd tree"), da_status_success)
        << "Setting algorithm to k-d tree failed.";
    EXPECT_EQ(da_nn_regressor_predict(knn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y.data()),
              da_status_option_locked)
        << "Testing that if algorithm option has been set, it cannot be changed "
           "after setting the training data failed.";

    da_handle_destroy(&knn_handle);

    // Test locked options when auto option is set
    EXPECT_EQ(da_handle_init<TypeParam>(&knn_handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_options_set(knn_handle, "algorithm", "auto"), da_status_success)
        << "Setting auto option to true failed.";
    EXPECT_EQ(da_options_set(knn_handle, "metric", "manhattan"), da_status_success)
        << "Setting metric to manhattan failed.";
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y.data()),
              da_status_success)
        << "Setting training data again with the same data failed.";

    // Change metric after set_data with auto
    EXPECT_EQ(da_options_set(knn_handle, "metric", "euclidean"), da_status_success)
        << "Setting metric to euclidean failed.";
    EXPECT_EQ(da_nn_regressor_predict(knn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y.data()),
              da_status_option_locked)
        << "Testing that if metric option has been set when algorithm is auto, it cannot "
           "be changed "
           "after setting the training data failed.";
    // Change minkowski parameter after set_data with auto
    EXPECT_EQ(da_options_set(knn_handle, "minkowski parameter", TypeParam(2.0)),
              da_status_success)
        << "Setting minkowski parameter to 2.0 failed.";
    EXPECT_EQ(da_nn_regressor_predict(knn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y.data()),
              da_status_option_locked)
        << "Testing that if minkowski parameter option has been set when algorithm is "
           "auto, it cannot be changed "
           "after setting the training data failed.";

    da_handle_destroy(&knn_handle);
}

TYPED_TEST(knnTest, InvalidHandleErrorExits) {
    KNNParamType<TypeParam> param;
    da_handle knn_handle = nullptr;

    std::vector<TypeParam> X(1), proba(1), dist(1), y_reg(1);
    std::vector<da_int> ind(1), y_class(1);
    da_int n_classes = 0;

    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y_reg.data()),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_regressor_set_training_data() with invalid handle "
           "failed.";

    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples,
                                                 param.n_features, X.data(),
                                                 param.ldx_train, y_class.data()),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_classifier_set_training_data() with invalid handle "
           "failed.";

    EXPECT_EQ(da_nn_regressor_predict(knn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y_reg.data()),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_regressor_predict() with invalid handle failed.";

    EXPECT_EQ(da_nn_classifier_predict(knn_handle, param.n_queries, param.n_features,
                                       X.data(), param.ldx_test, y_class.data()),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_classifier_predict() with invalid handle failed.";

    EXPECT_EQ(da_nn_classifier_predict_proba(knn_handle, param.n_queries,
                                             param.n_features, X.data(), param.ldx_test,
                                             X.data()),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_classifier_predict_proba() with invalid handle "
           "failed.";

    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_kneighbors() with invalid handle failed.";

    EXPECT_EQ(da_nn_classes<TypeParam>(knn_handle, &n_classes, nullptr),
              da_status_handle_not_initialized)
        << "Testing calling da_nn_classes() with invalid handle failed.";

    // If TypeParam is float set handle to double, otherwise set it to float
    if constexpr (std::is_same_v<TypeParam, float>) {
        EXPECT_EQ(da_handle_init<double>(&knn_handle, da_handle_nn), da_status_success);
    } else {
        EXPECT_EQ(da_handle_init<float>(&knn_handle, da_handle_nn), da_status_success);
    }

    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y_reg.data()),
              da_status_wrong_type)
        << "Testing calling da_nn_regressor_set_training_data() with wrong handle type "
           "failed.";

    EXPECT_EQ(da_nn_classifier_set_training_data(knn_handle, param.n_samples,
                                                 param.n_features, X.data(),
                                                 param.ldx_train, y_class.data()),
              da_status_wrong_type)
        << "Testing calling da_nn_classifier_set_training_data() with wrong handle type "
           "failed.";

    EXPECT_EQ(da_nn_regressor_predict(knn_handle, param.n_queries, param.n_features,
                                      X.data(), param.ldx_test, y_reg.data()),
              da_status_wrong_type)
        << "Testing calling da_nn_regressor_predict() with wrong handle type failed.";

    EXPECT_EQ(da_nn_classifier_predict(knn_handle, param.n_queries, param.n_features,
                                       X.data(), param.ldx_test, y_class.data()),
              da_status_wrong_type)
        << "Testing calling da_nn_classifier_predict() with wrong handle type failed.";

    EXPECT_EQ(da_nn_classifier_predict_proba(knn_handle, param.n_queries,
                                             param.n_features, X.data(), param.ldx_test,
                                             X.data()),
              da_status_wrong_type)
        << "Testing calling da_nn_classifier_predict_proba() with wrong handle type "
           "failed.";

    EXPECT_EQ(da_nn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                               param.ldx_test, ind.data(), dist.data(),
                               param.n_neigh_kneighbors, 1),
              da_status_wrong_type)
        << "Testing calling da_nn_kneighbors() with wrong handle type failed.";

    EXPECT_EQ(da_nn_classes<TypeParam>(knn_handle, &n_classes, nullptr),
              da_status_wrong_type)
        << "Testing calling da_nn_classes() with wrong handle type failed.";

    da_handle_destroy(&knn_handle);
}

TYPED_TEST(knnTest, IncompatibleMetrics) {
    KNNParamType<TypeParam> param;
    da_handle knn_handle = nullptr;

    std::vector<TypeParam> X(1), proba(1), dist(1), y(1);
    std::vector<da_int> ind(1);
    EXPECT_EQ(da_handle_init<TypeParam>(&knn_handle, da_handle_nn), da_status_success);

    EXPECT_EQ(da_options_set_string(knn_handle, "algorithm", "kd tree"),
              da_status_success);
    EXPECT_EQ(da_options_set_string(knn_handle, "metric", "cosine"), da_status_success);

    // Incompatible options
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y.data()),
              da_status_incompatible_options);

    EXPECT_EQ(da_options_set_string(knn_handle, "metric", "sqeuclidean"),
              da_status_success);

    // Incompatible options
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y.data()),
              da_status_incompatible_options);

    EXPECT_EQ(da_options_set_string(knn_handle, "metric", "minkowski"),
              da_status_success);
    EXPECT_EQ(da_options_set(knn_handle, "minkowski parameter", TypeParam(0.5)),
              da_status_success);

    // Incompatible options
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y.data()),
              da_status_incompatible_options);

    EXPECT_EQ(da_options_set_string(knn_handle, "algorithm", "ball tree"),
              da_status_success);

    // Incompatible options
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y.data()),
              da_status_incompatible_options);

    EXPECT_EQ(da_options_set_string(knn_handle, "metric", "sqeuclidean"),
              da_status_success);

    // Incompatible options
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y.data()),
              da_status_incompatible_options);

    EXPECT_EQ(da_options_set_string(knn_handle, "metric", "cosine"), da_status_success);
    // Incompatible options
    EXPECT_EQ(da_nn_regressor_set_training_data(knn_handle, param.n_samples,
                                                param.n_features, X.data(),
                                                param.ldx_train, y.data()),
              da_status_incompatible_options);

    da_handle_destroy(&knn_handle);
}