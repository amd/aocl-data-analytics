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
#include "knn_tests.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T> class knnTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(knnTest, FloatTypes);

TYPED_TEST(knnTest, AccuracyTesting) {
    std::vector<KNNParamType<TypeParam>> params;
    GetKNNData(params);
    da_handle handle = nullptr;
    da_int count = 0;
    for (auto &param : params) {
        count++;

        std::cout << "Functionality test " << std::to_string(count) << ": " << param.name
                  << std::endl;
        EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_knn), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "metric", param.metric.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "weights", param.weights.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "algorithm", param.algorithm.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "number of neighbors", param.n_neigh_knn),
                  da_status_success);

        EXPECT_EQ(da_knn_set_training_data(handle, param.n_samples, param.n_features,
                                           param.X_train.data(), param.n_samples,
                                           param.y_train.data()),
                  da_status_success);

        // Compute the k-nearest neighbors and return the distances
        std::vector<TypeParam> kdist(param.n_neigh_kneighbors * param.n_queries);
        std::vector<da_int> kind(param.n_neigh_kneighbors * param.n_queries);
        EXPECT_EQ(da_knn_kneighbors(handle, param.n_queries, param.n_features,
                                    param.X_test.data(), param.n_queries, kind.data(),
                                    kdist.data(), param.n_neigh_kneighbors, 1),
                  da_status_success);
        EXPECT_ARR_NEAR(param.n_neigh_kneighbors * param.n_queries, kdist.data(),
                        param.expected_kdist.data(), param.tol);

        da_int n_classes = 0; // Set n_classes to zero to do query for the required memory
        EXPECT_EQ(da_knn_classes<TypeParam>(handle, &n_classes, nullptr),
                  da_status_success);

        std::vector<TypeParam> proba(n_classes * param.n_queries);
        EXPECT_EQ(da_knn_predict_proba(handle, param.n_queries, param.n_features,
                                       param.X_test.data(), param.n_queries,
                                       proba.data()),
                  da_status_success);

        EXPECT_ARR_NEAR(n_classes * param.n_queries, proba.data(),
                        param.expected_proba.data(), param.tol);

        std::vector<da_int> labels(param.n_queries);
        EXPECT_EQ(da_knn_predict(handle, param.n_queries, param.n_features,
                                 param.X_test.data(), param.n_queries, labels.data()),
                  da_status_success);

        EXPECT_ARR_NEAR(param.n_queries, labels.data(), param.expected_labels.data(), 0);
        da_handle_destroy(&handle);
    }
}

// When weights equal distance we could end up with divisions by zero.
// Check that these cases are handled correctly.
TYPED_TEST(knnTest, AccuracyTestingZeroData) {
    std::cout << "Functionality test for zero data:\n";
    da_handle handle = nullptr;
    da_int n_samples = 4;
    da_int n_features = 3;
    da_int n_queries = 3;
    std::vector<TypeParam> X_train{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    std::vector<da_int> y_train{1, 2, 0, 1};
    std::vector<TypeParam> X_test{0., 0., 0., 0., 0., 0., 0., 0., 0.};
    TypeParam tol = 10 * std::numeric_limits<TypeParam>::epsilon();
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_knn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "weights", "distance"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", 3), da_status_success);

    EXPECT_EQ(da_knn_set_training_data(handle, n_samples, n_features, X_train.data(),
                                       n_samples, y_train.data()),
              da_status_success);

    da_int n_classes = 0; // Set n_classes to zero to do query for the required memory
    EXPECT_EQ(da_knn_classes<TypeParam>(handle, &n_classes, nullptr), da_status_success);

    std::vector<TypeParam> expected_proba(n_classes * n_queries, 0.3333333333333333);
    std::vector<TypeParam> proba(n_classes * n_queries);
    EXPECT_EQ(da_knn_predict_proba(handle, n_queries, n_features, X_test.data(),
                                   n_queries, proba.data()),
              da_status_success);
    EXPECT_ARR_NEAR(n_classes * n_queries, proba.data(), expected_proba.data(), tol);

    std::vector<da_int> labels(n_queries);
    EXPECT_EQ(da_knn_predict(handle, n_queries, n_features, X_test.data(), n_queries,
                             labels.data()),
              da_status_success);
    std::vector<da_int> expected_labels(n_queries, 0);
    EXPECT_ARR_NEAR(n_queries, labels.data(), expected_labels.data(), 0);
    da_handle_destroy(&handle);
}

std::string ErrorExits_print(std::string param) {
    std::string ss = "Test for invalid value of " + param + " failed.";
    return ss;
}

TYPED_TEST(knnTest, ErrorExits) {
    KNNParamType<TypeParam> param;
    da_handle knn_handle = nullptr;

    TypeParam *X_invalid = nullptr;
    da_int *y_invalid = nullptr;
    std::vector<TypeParam> X(1), proba(1), dist(1);
    std::vector<da_int> y(1), ind(1);
    da_int n_classes = -1;
    EXPECT_EQ(da_handle_init<TypeParam>(&knn_handle, da_handle_knn), da_status_success);

    // Try calling functionality before setting training data the model.
    EXPECT_EQ(da_knn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                                param.ldx_test, ind.data(), dist.data(),
                                param.n_neigh_kneighbors, 1),
              da_status_no_data)
        << "Testing calling kneighbors() before setting data failed.";
    EXPECT_EQ(da_knn_classes<TypeParam>(knn_handle, &n_classes, nullptr),
              da_status_no_data)
        << "Testing calling classes() before setting data failed.";
    EXPECT_EQ(da_knn_predict_proba(knn_handle, param.n_queries, param.n_features,
                                   X.data(), param.ldx_test, proba.data()),
              da_status_no_data)
        << "Testing calling predict_proba() before setting data failed.";
    EXPECT_EQ(da_knn_predict(knn_handle, param.n_queries, param.n_features, X.data(),
                             param.ldx_test, y.data()),
              da_status_no_data)
        << "Testing calling predict() before setting data failed.";

    // Invalid pointers in set_training_data
    EXPECT_EQ(da_knn_set_training_data(knn_handle, param.n_samples, param.n_features,
                                       X_invalid, param.ldx_train, y.data()),
              da_status_invalid_pointer)
        << ErrorExits_print("X_train");
    EXPECT_EQ(da_knn_set_training_data(knn_handle, param.n_samples, param.n_features,
                                       X.data(), param.ldx_train, y_invalid),
              da_status_invalid_pointer)
        << ErrorExits_print("y_train");
    // Invalid handle
    EXPECT_EQ(da_knn_set_training_data(nullptr, param.n_samples, param.n_features,
                                       X.data(), param.ldx_train, y.data()),
              da_status_handle_not_initialized)
        << ErrorExits_print("knn_handle");
    // Invalid dimensions
    EXPECT_EQ(da_knn_set_training_data(knn_handle, -1, param.n_features, X.data(),
                                       param.ldx_train, y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_samples");
    EXPECT_EQ(da_knn_set_training_data(knn_handle, param.n_samples, 0, X.data(),
                                       param.ldx_train, y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_knn_set_training_data(knn_handle, param.n_samples, param.n_features,
                                       X.data(), 0, y.data()),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_train");

    // Put valid training data so that we test effectively the rest of the APIs.
    EXPECT_EQ(da_knn_set_training_data(knn_handle, param.n_samples, param.n_features,
                                       X.data(), param.ldx_test, y.data()),
              da_status_success);

    // Invalid options
    EXPECT_EQ(da_options_set_int(knn_handle, "number of neighbors", 0),
              da_status_option_invalid_value)
        << "Invalid option 'number of neighbors' test failed.";
    EXPECT_EQ(da_options_set_string(knn_handle, "algorithm", "kdtree"),
              da_status_option_invalid_value)
        << "Invalid option 'algorithm' test failed.";
    EXPECT_EQ(da_options_set_string(knn_handle, "metric", "manhattan"),
              da_status_option_invalid_value)
        << "Invalid option 'metric' test failed.";
    EXPECT_EQ(da_options_set_string(knn_handle, "weights", "callable"),
              da_status_option_invalid_value)
        << "Invalid option 'weights' test failed.";

    // Incorrect inputs in classes()
    n_classes = 2;
    EXPECT_EQ(da_knn_classes<TypeParam>(knn_handle, &n_classes, nullptr),
              da_status_invalid_pointer)
        << ErrorExits_print("classes");

    // Incorrect inputs in kneighbors()
    EXPECT_EQ(da_knn_kneighbors(knn_handle, -1, param.n_features, X.data(),
                                param.ldx_test, ind.data(), dist.data(),
                                param.n_neigh_kneighbors, 1),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_knn_kneighbors(knn_handle, param.n_queries, 3, X.data(), param.ldx_test,
                                ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_knn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                                0, ind.data(), dist.data(), param.n_neigh_kneighbors, 1),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_knn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                                param.ldx_test, ind.data(), dist.data(), 3, 1),
              da_status_invalid_input)
        << ErrorExits_print("n_neigh_kneighbors");
    EXPECT_EQ(da_knn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                                param.ldx_test, nullptr, dist.data(),
                                param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("k_ind");
    EXPECT_EQ(da_knn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                                param.ldx_test, ind.data(), nullptr,
                                param.n_neigh_kneighbors, 1),
              da_status_invalid_pointer)
        << ErrorExits_print("k_dist");
    EXPECT_EQ(da_knn_kneighbors(knn_handle, param.n_queries, param.n_features, X.data(),
                                param.ldx_test, ind.data(), nullptr,
                                param.n_neigh_kneighbors, 0),
              da_status_success)
        << "Testing that if distances are not required, k_dist can be nullptr";

    // Set options so that the rest of the APIs do not fail for the wrong reasons.
    EXPECT_EQ(da_options_set_int(knn_handle, "number of neighbors", 1), da_status_success)
        << "Setting correct number of neighbors failed.";

    // Incorrect inputs in predict_proba()
    EXPECT_EQ(da_knn_predict_proba(knn_handle, -1, param.n_features, X.data(),
                                   param.ldx_test, proba.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_knn_predict_proba(knn_handle, param.n_queries, 2, X.data(),
                                   param.ldx_test, proba.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_knn_predict_proba(knn_handle, param.n_queries, param.n_features,
                                   X.data(), -1, proba.data()),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_knn_predict_proba(knn_handle, -1, param.n_features, X.data(),
                                   param.ldx_test, nullptr),
              da_status_invalid_pointer)
        << ErrorExits_print("proba");

    // Incorrect inputs in predict()
    EXPECT_EQ(da_knn_predict(knn_handle, -1, param.n_features, X.data(), param.ldx_test,
                             y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_queries");
    EXPECT_EQ(da_knn_predict(knn_handle, param.n_queries, 2, X.data(), param.ldx_test,
                             y.data()),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n_features");
    EXPECT_EQ(da_knn_predict(knn_handle, param.n_queries, param.n_features, X.data(), -1,
                             y.data()),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx_test");
    EXPECT_EQ(da_knn_predict(knn_handle, -1, param.n_features, X.data(), param.ldx_test,
                             nullptr),
              da_status_invalid_pointer)
        << ErrorExits_print("y_test");
    da_handle_destroy(&knn_handle);
}
