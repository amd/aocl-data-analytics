/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "decision_tree_positive.hpp"
#include "decision_tree_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <limits>
#include <list>
#include <string>

template <typename T> class decision_tree_public_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(decision_tree_public_test, FloatTypes);

TYPED_TEST(decision_tree_public_test, trivial_trees) {
    std::function<void(test_data_type<TypeParam> & data)> set_test_data[] = {
        set_test_data_8x2_nonunique_const_feat<TypeParam>};
    test_data_type<TypeParam> data;

    da_int i = 0;
    for (auto &data_fun : set_test_data) {
        std::cout << "Testing function: " << i << std::endl;
        data_fun(data);
        da_handle tree_handle = nullptr;
        EXPECT_EQ(da_handle_init<TypeParam>(&tree_handle, da_handle_decision_tree),
                  da_status_success);
        EXPECT_EQ(da_tree_set_training_data(tree_handle, data.n_samples_train,
                                            data.n_feat, 0, data.X_train.data(),
                                            data.ldx_train, data.y_train.data()),
                  da_status_success);
        EXPECT_EQ(da_tree_fit<TypeParam>(tree_handle), da_status_success);
        TypeParam accuracy;
        EXPECT_EQ(da_tree_score(tree_handle, data.n_samples_test, data.n_feat,
                                data.X_test.data(), data.ldx_test, data.y_test.data(),
                                &accuracy),
                  da_status_success);
        EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
        std::vector<da_int> y_pred(data.n_samples_test);
        EXPECT_EQ(da_tree_predict(tree_handle, data.n_samples_test, data.n_feat,
                                  data.X_test.data(), data.ldx_test, y_pred.data()),
                  da_status_success);
        EXPECT_ARR_EQ(data.n_samples_test, y_pred, data.y_test, 1, 1, 0, 0);

        da_handle_destroy(&tree_handle);
        i++;
    }
}

TYPED_TEST(decision_tree_public_test, categorical_features) {
    test_data_type<TypeParam> data;
    set_test_data_6x2_categorical(data);
    da_handle tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    EXPECT_EQ(da_tree_set_training_data(
                  tree_handle, data.n_samples_train, data.n_feat, 0, data.X_train.data(),
                  data.ldx_train, data.y_train.data(), data.categorical_feat.data()),
              da_status_success);
    EXPECT_EQ(da_tree_fit<TypeParam>(tree_handle), da_status_success);
    TypeParam accuracy;
    EXPECT_EQ(da_tree_score(tree_handle, data.n_samples_test, data.n_feat,
                            data.X_test.data(), data.ldx_test, data.y_test.data(),
                            &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
    std::vector<da_int> y_pred(data.n_samples_test);
    EXPECT_EQ(da_tree_predict(tree_handle, data.n_samples_test, data.n_feat,
                              data.X_test.data(), data.ldx_test, y_pred.data()),
              da_status_success);
    EXPECT_ARR_EQ(data.n_samples_test, y_pred, data.y_test, 1, 1, 0, 0);

    da_handle_destroy(&tree_handle);

    // Test one vs all split strategy
    tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "Category split strategy", "one-vs-all"),
              da_status_success);
    EXPECT_EQ(da_tree_set_training_data(
                  tree_handle, data.n_samples_train, data.n_feat, 0, data.X_train.data(),
                  data.ldx_train, data.y_train.data(), data.categorical_feat.data()),
              da_status_success);
    EXPECT_EQ(da_tree_fit<TypeParam>(tree_handle), da_status_success);
    EXPECT_EQ(da_tree_score(tree_handle, data.n_samples_test, data.n_feat,
                            data.X_test.data(), data.ldx_test, data.y_test.data(),
                            &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
    EXPECT_EQ(da_tree_predict(tree_handle, data.n_samples_test, data.n_feat,
                              data.X_test.data(), data.ldx_test, y_pred.data()),
              da_status_success);
    EXPECT_ARR_EQ(data.n_samples_test, y_pred, data.y_test, 1, 1, 0, 0);
    da_handle_destroy(&tree_handle);

    // Do the same with automatic dectection
    // ordered
    tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "detect categorical data", "yes"),
              da_status_success);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, data.n_samples_train, data.n_feat, 0,
                                        data.X_train.data(), data.ldx_train,
                                        data.y_train.data()),
              da_status_success);
    EXPECT_EQ(da_tree_fit<TypeParam>(tree_handle), da_status_success);
    EXPECT_EQ(da_tree_score(tree_handle, data.n_samples_test, data.n_feat,
                            data.X_test.data(), data.ldx_test, data.y_test.data(),
                            &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
    EXPECT_EQ(da_tree_predict(tree_handle, data.n_samples_test, data.n_feat,
                              data.X_test.data(), data.ldx_test, y_pred.data()),
              da_status_success);
    EXPECT_ARR_EQ(data.n_samples_test, y_pred, data.y_test, 1, 1, 0, 0);
    da_handle_destroy(&tree_handle);
    // one vs all
    tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "detect categorical data", "yes"),
              da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "Category split strategy", "one-vs-all"),
              da_status_success);

    EXPECT_EQ(da_tree_set_training_data(tree_handle, data.n_samples_train, data.n_feat, 0,
                                        data.X_train.data(), data.ldx_train,
                                        data.y_train.data()),
              da_status_success);
    EXPECT_EQ(da_tree_fit<TypeParam>(tree_handle), da_status_success);
    EXPECT_EQ(da_tree_score(tree_handle, data.n_samples_test, data.n_feat,
                            data.X_test.data(), data.ldx_test, data.y_test.data(),
                            &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
    EXPECT_EQ(da_tree_predict(tree_handle, data.n_samples_test, data.n_feat,
                              data.X_test.data(), data.ldx_test, y_pred.data()),
              da_status_success);
    EXPECT_ARR_EQ(data.n_samples_test, y_pred, data.y_test, 1, 1, 0, 0);

    da_handle_destroy(&tree_handle);
}

TYPED_TEST(decision_tree_public_test, small_bins) {
    using T = TypeParam;
    std::vector<T> X_train = {(T)0,   (T)1,   (T)2,  (T)3, (T)4,   (T)0.5, (T)1.5, (T)2.5,
                              (T)3.5, (T)4.5, (T)6,  (T)7, (T)8,   (T)9,   (T)5.5, (T)4,
                              (T)3,   (T)2,   (T)0,  (T)1, (T)6.5, (T)5.5, (T)7.5, (T)8.5,
                              (T)6.,  (T)1.,  (T)2., (T)4, (T)3,   (T)2};
    std::vector<da_int> y_train = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
    std::vector<T> X_test = {(T)0, (T)1, (T)2, (T)2, (T)3, (T)4, (T)6,   (T)7,   (T)8,
                             (T)3, (T)4, (T)1, (T)7, (T)8, (T)9, (T)2.5, (T)3.5, (T)4.2};
    std::vector<da_int> y_test = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    da_int n_samples_train = 15;
    da_int n_features = 2;
    da_int n_samples_test = 9;

    // Solve first without binning
    da_handle tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples_train, n_features, 3,
                                        X_train.data(), n_samples_train, y_train.data()),
              da_status_success);
    EXPECT_EQ(da_tree_fit<T>(tree_handle), da_status_success);
    T accuracy;
    EXPECT_EQ(da_tree_score(tree_handle, n_samples_test, n_features, X_test.data(),
                            n_samples_test, y_test.data(), &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
    std::vector<da_int> y_pred(n_samples_test);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples_test, n_features, X_test.data(),
                              n_samples_test, y_pred.data()),
              da_status_success);
    EXPECT_ARR_EQ(n_samples_test, y_pred, y_test, 1, 1, 0, 0);

    // Solve again, binning the data
    EXPECT_EQ(da_options_set(tree_handle, "histogram", "yes"), da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "Maximum bins", (da_int)5), da_status_success);
    EXPECT_EQ(da_tree_fit<T>(tree_handle), da_status_success);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples_test, n_features, X_test.data(),
                            n_samples_test, y_test.data(), &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples_test, n_features, X_test.data(),
                              n_samples_test, y_pred.data()),
              da_status_success);
    EXPECT_ARR_EQ(n_samples_test, y_pred, y_test, 1, 1, 0, 0);

    // Solve again, binning the data with too few bins
    EXPECT_EQ(da_options_set(tree_handle, "histogram", "yes"), da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "category split strategy", "one-vs-all"),
              da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "Maximum bins", (da_int)2), da_status_success);
    EXPECT_EQ(da_tree_fit<T>(tree_handle), da_status_success);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples_test, n_features, X_test.data(),
                            n_samples_test, y_test.data(), &accuracy),
              da_status_success);
    EXPECT_GT(accuracy, 0.5);

    da_handle_destroy(&tree_handle);
}

TYPED_TEST(decision_tree_public_test, onevall) {
    using T = TypeParam;
    // Simple, 2 categorical features case, tree should train in 2 splits
    std::vector<T> X_train = {(T)0, (T)0, (T)0, (T)0, (T)1, (T)1, (T)1, (T)1,
                              (T)1, (T)1, (T)1, (T)1, (T)0, (T)0, (T)1, (T)1,
                              (T)0, (T)0, (T)1, (T)1, (T)0, (T)0, (T)1, (T)1};
    std::vector<da_int> y_train = {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1};
    std::vector<T> X_test = {(T)0, (T)0, (T)1, (T)1, (T)0, (T)1, (T)0, (T)1};
    std::vector<da_int> y_test = {0, 0, 0, 1};
    da_int n_samples_train = 12;
    da_int n_features = 2;
    da_int n_samples_test = 4;
    std::vector<da_int> cat_var = {2, 2};

    da_handle tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "histogram", "yes"), da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "category split strategy", "one-vs-all"),
              da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "Maximum bins", (da_int)8), da_status_success);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples_train, n_features, 3,
                                        X_train.data(), n_samples_train, y_train.data(),
                                        cat_var.data()),
              da_status_success);
    EXPECT_EQ(da_tree_fit<T>(tree_handle), da_status_success);
    T accuracy;
    EXPECT_EQ(da_tree_score(tree_handle, n_samples_test, n_features, X_test.data(),
                            n_samples_test, y_test.data(), &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);

    // solve again without histograms
    EXPECT_EQ(da_options_set(tree_handle, "histogram", "no"), da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "category split strategy", "one-vs-all"),
              da_status_success);
    EXPECT_EQ(da_tree_fit<T>(tree_handle), da_status_success);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples_test, n_features, X_test.data(),
                            n_samples_test, y_test.data(), &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
    da_int s_rinfo = 100;
    std::vector<T> rinfo(s_rinfo);
    EXPECT_EQ(
        da_handle_get_result(tree_handle, da_result::da_rinfo, &s_rinfo, rinfo.data()),
        da_status_success);
    EXPECT_NEAR(rinfo[4], 2.0, 1.0e-10); // Depth of the tree
    da_handle_destroy(&tree_handle);

    // slightly more complex case with 3 classes, 1 continuous feature and 1 categorical
    // X0 == 0 => y = 0
    // X0 == 1 => y = 0 if X1 > 5, 1 otherwise
    // X0 == 2 => y = 0 if X1 > 3, 1 otherwise
    // clang-format off
    X_train = {0,  0,  0,   0,  0,   1,   1,   1,   1,   1,  1,  1,  1, 1,    2,   2,   2,   2,    2,   2,   2,   2,
               .5, 1., 4., 5., 8.,  .8, 1.5, 2.5, 3.5, 4.5, 6., 7., 8., 9., 1.2, 2.2, 2.8, 3.2,  5.2, 6.2, 7.2, 8.2};
    y_train = {0,  0,  0,  0,  0,   1,   1,   1,   1,   1,  0,  0,  0, 0,    1,   1,   1,   0,    0,   0,   0,   0};
    X_test  = {  0,   0,   1,   1,   1,   2,   2,   2,
               1.5, 6.5, 2.3, 4.3, 8.3, 1.3, 4.3, 8.3};
    y_test  = {  0,   0,   1,   1,   0,   1,   0,   0};
    n_samples_train = 22;
    n_features = 2;
    n_samples_test = 8;
    cat_var = {3, -1};
    // clang-format on
    tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "histogram", "no"), da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "Node minimum samples", (da_int)1),
              da_status_success);
    EXPECT_EQ(da_options_set(tree_handle, "category split strategy", "one-vs-all"),
              da_status_success);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples_train, n_features, 2,
                                        X_train.data(), n_samples_train, y_train.data(),
                                        cat_var.data()),
              da_status_success);
    EXPECT_EQ(da_tree_fit<T>(tree_handle), da_status_success);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples_test, n_features, X_test.data(),
                            n_samples_test, y_test.data(), &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);

    // solve again with histograms
    EXPECT_EQ(da_options_set(tree_handle, "histogram", "yes"), da_status_success);
    EXPECT_EQ(da_tree_fit<T>(tree_handle), da_status_success);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples_test, n_features, X_test.data(),
                            n_samples_test, y_test.data(), &accuracy),
              da_status_success);
    std::vector<da_int> y_pred(n_samples_test);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples_test, n_features, X_test.data(),
                              n_samples_test, y_pred.data()),
              da_status_success);
    std::cout << "Predicted labels: ";
    for (const auto &val : y_pred) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
    da_handle_destroy(&tree_handle);
}

TYPED_TEST(decision_tree_public_test, get_results) {

    test_data_type<TypeParam> data;
    set_test_data_8x2_nonunique<TypeParam>(data);

    // Train the default tree on the small data set
    da_handle tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, data.n_samples_train, data.n_feat, 0,
                                        data.X_train.data(), data.n_samples_train,
                                        data.y_train.data()),
              da_status_success);
    da_int seed = 42;
    EXPECT_EQ(da_options_set(tree_handle, "seed", seed), da_status_success);
    // Check da_trained before fitting
    da_int tr_dim = 1, tr_val = -1;
    EXPECT_EQ(da_handle_get_result(tree_handle, da_result::da_trained, &tr_dim, &tr_val),
              da_status_success);
    EXPECT_EQ(tr_val, 0);
    EXPECT_EQ(da_tree_fit<TypeParam>(tree_handle), da_status_success);
    // Check da_trained after fitting
    EXPECT_EQ(da_handle_get_result(tree_handle, da_result::da_trained, &tr_dim, &tr_val),
              da_status_success);
    EXPECT_EQ(tr_val, 1);
    // Quick check on test data
    std::vector<TypeParam> X_test{0.1, 0.3, 0.7, 0.9, 0.2, 0.6, 0.8, 0.1};
    std::vector<da_int> y_test{0, 1, 1, 1};
    TypeParam accuracy;
    EXPECT_EQ(
        da_tree_score(tree_handle, 4, 2, X_test.data(), 4, y_test.data(), &accuracy),
        da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-03);

    // get the results and check the values
    da_int dim = 100;
    std::vector<TypeParam> rinfo(dim);
    EXPECT_EQ(da_handle_get_result(tree_handle, da_result::da_rinfo, &dim, rinfo.data()),
              da_status_success);
    std::vector<TypeParam> rinfo_exp{(TypeParam)data.n_feat,
                                     (TypeParam)data.n_samples_train,
                                     (TypeParam)data.n_samples_train,
                                     (TypeParam)seed,
                                     (TypeParam)2,
                                     (TypeParam)5,
                                     (TypeParam)3,
                                     (TypeParam)1};
    EXPECT_ARR_NEAR(8, rinfo, rinfo_exp, 1.0e-10);

    // Check that other queries fail
    EXPECT_EQ(
        da_handle_get_result(tree_handle, da_result::da_linmod_coef, &dim, rinfo.data()),
        da_status_unknown_query);

    // Check the wrong dimension
    dim = 1;
    EXPECT_EQ(da_handle_get_result(tree_handle, da_result::da_rinfo, &dim, rinfo.data()),
              da_status_invalid_array_dimension);

    // change an option and check that results are no longer available
    EXPECT_EQ(da_options_set(tree_handle, "seed", (da_int)43), da_status_success);
    dim = 8;
    EXPECT_EQ(da_handle_get_result(tree_handle, da_result::da_rinfo, &dim, rinfo.data()),
              da_status_unknown_query);
    da_handle_destroy(&tree_handle);
}

TYPED_TEST(decision_tree_public_test, invalid_input) {

    std::vector<TypeParam> X{0.0, 1.0, 0.0, 2.0};
    std::vector<da_int> y{0, 1};

    // Initialize the decision tree class and fit model
    da_handle tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&tree_handle, da_handle_decision_tree),
              da_status_success);

    // call set_training_data with invalid values
    da_int n_samples = 2, n_features = 2, n_class = 0;
    TypeParam accuracy;

    // set_training_data
    // Invalid pointers
    TypeParam *X_invalid = nullptr;
    da_int *y_invalid = nullptr;
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, n_features, n_class,
                                        X_invalid, n_samples, y.data()),
              da_status_invalid_pointer);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, n_features, n_class,
                                        X.data(), n_samples, y_invalid),
              da_status_invalid_pointer);
    EXPECT_EQ(da_tree_set_training_data(nullptr, n_samples, n_features, n_class,
                                        X_invalid, n_samples, y.data()),
              da_status_handle_not_initialized);
    // wrong dimensions
    EXPECT_EQ(da_tree_set_training_data(tree_handle, 0, n_features, n_class, X_invalid,
                                        n_samples, y.data()),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, 0, n_class, X_invalid,
                                        n_samples, y.data()),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, n_features, n_class,
                                        X_invalid, 1, y.data()),
              da_status_invalid_pointer);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, n_features, n_class,
                                        X.data(), n_samples, y.data()),
              da_status_success);

    // Number of samples too small for histogram option
    EXPECT_EQ(da_options_set(tree_handle, "histogram", "yes"), da_status_success);
    EXPECT_EQ(da_tree_fit<TypeParam>(tree_handle), da_status_invalid_option);
    EXPECT_EQ(da_options_set(tree_handle, "histogram", "no"), da_status_success);

    // model out of date for evaluation
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, n_features, X.data(), n_samples,
                              y.data()),
              da_status_out_of_date);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X.data(), n_samples,
                            y.data(), &accuracy),
              da_status_out_of_date);
    EXPECT_EQ(da_tree_fit<TypeParam>(nullptr), da_status_handle_not_initialized);
    EXPECT_EQ(da_tree_fit<TypeParam>(tree_handle), da_status_success);

    // predict
    // Invalid pointers
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, n_features, X_invalid, n_samples,
                              y.data()),
              da_status_invalid_pointer);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, n_features, X.data(), n_samples,
                              y_invalid),
              da_status_invalid_pointer);
    EXPECT_EQ(
        da_tree_predict(nullptr, n_samples, n_features, X.data(), n_samples, y.data()),
        da_status_handle_not_initialized);
    // Wrong dimensions
    EXPECT_EQ(da_tree_predict(tree_handle, 0, n_features, X.data(), n_samples, y.data()),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, 0, X.data(), n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, 4, X.data(), n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, n_features, X.data(), 1, y.data()),
              da_status_invalid_leading_dimension);

    // score
    // Invalid pointers
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X_invalid, n_samples,
                            y.data(), &accuracy),
              da_status_invalid_pointer);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X.data(), n_samples,
                            y_invalid, &accuracy),
              da_status_invalid_pointer);
    EXPECT_EQ(da_tree_score<TypeParam>(tree_handle, n_samples, n_features, X.data(),
                                       n_samples, y.data(), nullptr),
              da_status_invalid_pointer);
    EXPECT_EQ(da_tree_score<TypeParam>(nullptr, n_samples, n_features, X.data(),
                                       n_samples, y.data(), &accuracy),
              da_status_handle_not_initialized);
    // Wrong dimensions
    EXPECT_EQ(da_tree_score(tree_handle, 0, n_features, X.data(), n_samples, y.data(),
                            &accuracy),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, 0, X.data(), n_samples, y.data(),
                            &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, 4, X.data(), n_samples, y.data(),
                            &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X.data(), 1, y.data(),
                            &accuracy),
              da_status_invalid_leading_dimension);

    da_handle_destroy(&tree_handle);
}

TEST(decision_tree, incorrect_handle_precision) {

    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_decision_tree), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_decision_tree), da_status_success);

    std::vector<da_int> y{0};
    da_int n_samples = 0, n_features = 0;
    std::vector<double> X_d{0.0};
    double accuracy_d = 0.0;
    std::vector<float> X_s{0.0};
    float accuracy_s = 0.0;

    // incorrect handle precision
    EXPECT_EQ(da_tree_set_training_data_s(handle_d, n_samples, n_features, 0, X_s.data(),
                                          n_samples, y.data(), nullptr),
              da_status_wrong_type);
    EXPECT_EQ(da_tree_set_training_data_d(handle_s, n_samples, n_features, 0, X_d.data(),
                                          n_samples, y.data(), nullptr),
              da_status_wrong_type);

    EXPECT_EQ(da_tree_fit_s(handle_d), da_status_wrong_type);
    EXPECT_EQ(da_tree_fit_d(handle_s), da_status_wrong_type);

    EXPECT_EQ(da_tree_predict_s(handle_d, n_samples, n_features, X_s.data(), n_samples,
                                y.data()),
              da_status_wrong_type);
    EXPECT_EQ(da_tree_predict_d(handle_s, n_samples, n_features, X_d.data(), n_samples,
                                y.data()),
              da_status_wrong_type);

    EXPECT_EQ(da_tree_score_s(handle_d, n_samples, n_features, X_s.data(), n_samples,
                              y.data(), &accuracy_s),
              da_status_wrong_type);
    EXPECT_EQ(da_tree_score_d(handle_s, n_samples, n_features, X_d.data(), n_samples,
                              y.data(), &accuracy_d),
              da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

/***********************************
 ********* Positive tests***********
 ***********************************/
typedef struct decision_tree_param_t {
    std::string test_name; // name of the ctest test
    std::string data_name; // name of the files to read in
    std::vector<option_t<da_int>> iopts;
    std::vector<option_t<std::string>> sopts;
    std::vector<option_t<float>> fopts;
    std::vector<option_t<double>> dopts;
    float target_score;
} decision_tree_param_t;

// clang-format off
const decision_tree_param_t decision_tree_param_pos[] = {

    // Testing scoring functions
    {"iris_gini", "iris", {{"node minimum samples", 1}}, {{"scoring function", "gini"}}, {}, {}, 0.95},
    {"iris_entropy", "iris", {{"node minimum samples", 1}}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.95},
    {"iris_misclass", "iris", {{"node minimum samples", 1}}, {{"scoring function", "misclass"}}, {}, {}, 0.8},
    {"gen1_gini", "gen1", {}, {{"scoring function", "gini"}},
                  {{"minimum impurity decrease", 0.03}}, {{"minimum impurity decrease", 0.03}}, 0.93},
    {"gen1_entropy", "gen1", {{"node minimum samples", 1}}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.93},
    {"gen1_misclass", "gen1", {}, {{"scoring function", "misclass"}}, {{"minimum impurity decrease", 0.03}}, {{"minimum impurity decrease", 0.03}}, 0.93},
    {"gen_200x10_gini", "gen_200x10_3class", {}, {{"scoring function", "gini"}}, {}, {}, 0.93},
    {"gen_200x10_entropy", "gen_200x10_3class", {}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.93},
    {"gen_200x10_misclass", "gen_200x10_3class", {}, {{"scoring function", "misclass"}}, {{"minimum impurity decrease", 0.03}}, {{"minimum impurity decrease", 0.03}}, 0.93},
    {"gen_500x20_gini", "gen_500x20_4class", {}, {{"scoring function", "gini"}}, {{"minimum impurity decrease", 0.03}}, {{"minimum impurity decrease", 0.03}}, 0.9},
    {"gen_500x20_entropy", "gen_500x20_4class", {}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.9},
    {"gen_500x20_misclass", "gen_500x20_4class", {}, {{"scoring function", "misclass"}}, {{"minimum impurity decrease", 0.03}}, {{"minimum impurity decrease", 0.03}},  0.89},


    // maximum splits
    {"gen_200x10_maxsplit", "gen_200x10_3class", {{"maximum depth", 19}}, {{"scoring function", "gini"}},
                            {{"Minimum split score", 0.0}, {"minimum impurity decrease", 0.0}},
                            {{"Minimum split score", 0.0}, {"minimum impurity decrease", 0.0}}, 0.9},
    {"gen_500x20_maxsplit", "gen_500x20_4class", {{"maximum depth", 19}}, {{"scoring function", "misclass"}},
                            {{"Minimum split score", 0.0}, {"minimum impurity decrease", 0.0}},
                            {{"Minimum split score", 0.0}, {"minimum impurity decrease", 0.0}}, 0.87},

    // Test identical train and test sets
    {"overfit_gini", "overfit", {{"maximum depth", 24}, {"node minimum samples", 1}}, {{"scoring function", "gini"}},
                            {}, {}, 0.99},
    {"overfit_misclass", "overfit", {{"maximum depth", 24}, {"node minimum samples", 1}}, {{"scoring function", "misclass"}},
                            {}, {}, 0.99},
    {"overfit_entropy", "overfit", {{"maximum depth", 24}, {"node minimum samples", 1}}, {{"scoring function", "entropy"}},
                            {}, {}, 0.99},
    {"overfit_prune05", "overfit", {{"maximum depth", 24}, {"node minimum samples", 1}}, {{"scoring function", "gini"}},
                            {}, {}, 0.97},
    {"overfit_prune1", "overfit", {{"maximum depth", 24}, {"node minimum samples", 1}}, {{"scoring function", "gini"}},
                            {}, {}, 0.97},


    // splits on fewer than all the features
    {"gen_200x10_split4", "gen_200x10_3class", {{"maximum depth", 19}, {"seed", 42}, {"maximum features", 4}},
      {{"scoring function", "entropy"}}, {}, {}, 0.81},
    {"iris_split2", "iris", {{"node minimum samples", 1}, {"maximum depth", 19}, {"seed", 42}, {"maximum features", 2}}, {{"scoring function", "gini"}}, {}, {}, 0.95},
    {"gen_500x20_split6", "gen_500x20_4class", {{"maximum depth", 19}, {"seed", 42}, {"maximum features", 7}},
      {{"scoring function", "gini"}}, {}, {}, 0.8},


    // smaller tree depth
    {"iris_depth2", "iris", {{"maximum depth", 1}}, {{"scoring function", "gini"}}, {}, {}, 0.6},
    {"gen1_depth2", "gen1", {{"maximum depth", 1}}, {{"scoring function", "entropy"}}, {}, {}, 0.9},
    {"gen200x10_depth2", "gen_200x10_3class", {{"maximum depth", 1}}, {{"scoring function", "gini"}}, {}, {}, 0.6},
    {"gen_500x20_depth3", "gen_500x20_4class", {{"maximum depth", 2}}, {{"scoring function", "gini"}}, {}, {}, 0.7},

    // Histogram tests
    {"iris_hist", "iris", {{"maximum bins", 40}, {"node minimum samples", 1}}, {{"histogram", "yes"}}, {{"minimum impurity decrease", 0.01}}, {{"minimum impurity decrease", 0.01}}, 0.95},
    {"gen_500x20_hist", "gen_500x20_4class", {{"maximum bins", 30}}, {{"histogram", "yes"}}, {{"minimum impurity decrease", 0.01}}, {{"minimum impurity decrease", 0.01}}, 0.89},
    {"gen_200x10_hist", "gen_200x10_3class", {{"maximum bins", 35}}, {{"histogram", "yes"}}, {{"minimum impurity decrease", 0.01}}, {{"minimum impurity decrease", 0.01}}, 0.92},
};
// clang-format on

class decision_tree_positive : public testing::TestWithParam<decision_tree_param_t> {};
// Teach GTest how to print the param type
// in this case use only user's unique testname
// It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
void PrintTo(const decision_tree_param_t &param, ::std::ostream *os) {
    *os << param.test_name;
}

// Positive tests with double and single type
TEST_P(decision_tree_positive, Double) {
    const decision_tree_param_t &param = GetParam();
    test_decision_tree_positive<double>(param.data_name, param.iopts, param.sopts,
                                        param.dopts, (double)param.target_score);
}
TEST_P(decision_tree_positive, Single) {
    const decision_tree_param_t &param = GetParam();
    test_decision_tree_positive<float>(param.data_name, param.iopts, param.sopts,
                                       param.fopts, (float)param.target_score);
}

INSTANTIATE_TEST_SUITE_P(decision_tree_pos_suite, decision_tree_positive,
                         testing::ValuesIn(decision_tree_param_pos));
