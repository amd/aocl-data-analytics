/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "decision_tree.hpp"
#include "decision_tree_positive.hpp"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <list>
#include <string>

using namespace da_decision_tree;

template <typename T> class decision_tree_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(decision_tree_test, FloatTypes);

template <typename T> struct test_data_type {
    std::vector<T> X_train;
    std::vector<da_int> y_train;
    std::vector<T> X_test;
    std::vector<da_int> y_test;
    da_int n_samples_train, n_feat, ldx_train, ldx_test;
    da_int n_samples_test;
};

template <typename T> void set_test_data_8x1(test_data_type<T> &data) {

    // idea is that y = 1 with prob 0.75 when x < 0.5
    // and          y = 1 with prob 0.25 with x > 0.5
    data.X_train = {
        (T)0.1, (T)0.2, (T)0.3, (T)0.4, (T)0.6, (T)0.7, (T)0.8, (T)0.9,
    };
    data.y_train = {0, 1, 0, 0, 1, 1, 0, 1};
    data.X_test = {(T)0.1, (T)0.9};
    data.y_test = {0, 1};
    data.n_samples_train = 8, data.n_feat = 1;
    data.n_samples_test = 2;
    data.ldx_train = 8;
    data.ldx_test = 2;
}

template <typename T> void set_test_data_8x2_unique(test_data_type<T> &data) {

    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    // training data values are unique
    data.X_train = {
        (T)0.12, (T)0.11, (T)0.42, (T)0.41,
        (T)0.62, (T)0.61, (T)0.92, (T)0.91, // first column of data
        (T)0.39, (T)0.79, (T)0.38, (T)0.78,
        (T)0.37, (T)0.77, (T)0.36, (T)0.76 // second column of data
    };

    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    data.y_train = {0, 1, 0, 1, 1, 1, 1, 1};
    data.X_test = {(T)0.25, (T)0.25, (T)0.75, (T)0.75,
                   (T)0.25, (T)0.75, (T)0.25, (T)0.75};
    data.y_test = {0, 1, 1, 1};
    data.n_samples_train = 8, data.n_feat = 2;
    data.n_samples_test = 4;
    data.ldx_train = 8;
    data.ldx_test = 4;
}

template <typename T> void set_test_data_8x2_ldx(test_data_type<T> &data) {

    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    // training data values are unique
    data.X_train = {(T)0.12, (T)0.11, (T)0.42, (T)0.41, (T)0.62,  (T)0.61, (T)0.92,
                    (T)0.91, (T)-50., (T)-50., (T)0.39, (T)0.79,  (T)0.38, (T)0.78,
                    (T)0.37, (T)0.77, (T)0.36, (T)0.76, (T)-100., (T)-100.};

    data.y_train = {0, 1, 0, 1, 1, 1, 1, 1};
    data.X_test = {(T)0.25, (T)0.25, (T)0.75, (T)0.75, (T)50., (T)50.,
                   (T)0.25, (T)0.75, (T)0.25, (T)0.75, (T)50., (T)50.};
    data.y_test = {0, 1, 1, 1};
    data.n_samples_train = 8, data.n_feat = 2;
    data.n_samples_test = 4;
    data.ldx_train = 10;
    data.ldx_test = 6;
}

template <typename T> void set_test_data_8x2_nonunique(test_data_type<T> &data) {
    // y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    // training data values are not unique
    data.X_train = {0.1, 0.4, 0.4, 0.6, 0.6, 0.9, 0.9, 0.1, 0.6, 0.1, 0.8,  0.2,
                    0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.4, 0.1, 0.45, 0.45};
    data.y_train = {1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0};
    data.X_test = {(T)0.25, (T)0.25, (T)0.75, (T)0.75,
                   (T)0.25, (T)0.75, (T)0.25, (T)0.75};
    data.y_test = {0, 1, 1, 1};
    data.n_samples_train = 12, data.n_feat = 2;
    data.n_samples_test = 4;
    data.ldx_train = 12;
    data.ldx_test = 4;
}

template <typename T> void set_data_identical(test_data_type<T> &data) {
    //  X contains all 1, No splitting should be done
    data.X_train = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    data.y_train = {1, 0, 1};
    data.X_test = {2.0, 3.0, -2.0, -2.5};
    data.y_test = {1, 1};
    data.n_samples_train = 3, data.n_feat = 2;
    data.n_samples_test = 2;
    data.ldx_train = 3;
    data.ldx_test = 2;
}

TYPED_TEST(decision_tree_test, scorefun) {
    // artificial class counts to check manually the score functions.
    da_int nclass = 2, nsamples = 100;
    std::vector<da_int> count_classes(nclass);
    TypeParam tol = (TypeParam)1.0e-05;

    // fully random class partition
    count_classes[0] = 50;
    count_classes[1] = 50;
    TypeParam res_gini = gini_score<TypeParam>(nsamples, nclass, count_classes);
    EXPECT_NEAR(res_gini, 0.5, tol);
    TypeParam res_misclass =
        misclassification_score<TypeParam>(nsamples, nclass, count_classes);
    EXPECT_NEAR(res_misclass, 0.5, tol);
    TypeParam res_entropy = entropy_score<TypeParam>(nsamples, nclass, count_classes);
    EXPECT_NEAR(res_entropy, 1.0, tol);

    // pure nodes
    count_classes[0] = nsamples;
    count_classes[1] = 0;
    res_gini = gini_score<TypeParam>(nsamples, nclass, count_classes);
    EXPECT_NEAR(res_gini, 0., tol);
    res_misclass = misclassification_score<TypeParam>(nsamples, nclass, count_classes);
    EXPECT_NEAR(res_misclass, 0., tol);
    res_entropy = entropy_score<TypeParam>(nsamples, nclass, count_classes);
    EXPECT_NEAR(res_entropy, 0., tol);

    // misc 2-class partition
    count_classes[0] = 20;
    count_classes[1] = 80;
    res_gini = gini_score<TypeParam>(nsamples, nclass, count_classes);
    EXPECT_NEAR(res_gini, 0.32, tol);
    res_misclass = misclassification_score<TypeParam>(nsamples, nclass, count_classes);
    EXPECT_NEAR(res_misclass, 0.2, tol);
    res_entropy = entropy_score<TypeParam>(nsamples, nclass, count_classes);
    EXPECT_NEAR(res_entropy, 0.72192809, 1.0e-5);
}

TYPED_TEST(decision_tree_test, sort_samples) {
    // Set up a new tree with the small data set defined above
    test_data_type<TypeParam> data;
    set_test_data_8x2_nonunique(data);

    da_errors::da_error_t err(da_errors::DA_RECORD);
    decision_tree<TypeParam> df(err);
    df.set_training_data(data.n_samples_train, data.n_feat, data.X_train.data(),
                         data.n_samples_train, data.y_train.data(), 2);
    EXPECT_EQ(df.opts.set("maximum depth", (da_int)1), da_status_success);
    df.fit();

    // Create a node with all the samples
    node<TypeParam> node_ex;
    node_ex.start_idx = 0;
    node_ex.end_idx = data.n_samples_train - 1;
    node_ex.n_samples = data.n_samples_train;

    // Sort the samples indices according to both features
    std::vector<da_int> expected_idx{0, 7, 9, 11, 1, 2, 3, 4, 8, 10, 5, 6};
    std::vector<TypeParam> expected_val{0.1, 0.1, 0.1, 0.2, 0.4, 0.4,
                                        0.6, 0.6, 0.6, 0.8, 0.9, 0.9};
    df.sort_samples(node_ex, 0);
    EXPECT_ARR_EQ(data.n_samples_train, df.get_samples_idx(), expected_idx, 1, 1, 0, 0);
    EXPECT_ARR_NEAR(data.n_samples_train, df.get_features_values(), expected_val,
                    1.0e-10);

    // node on partial samples
    node_ex.start_idx = 1;
    node_ex.end_idx = 5;
    node_ex.n_samples = 5;
    df.sort_samples(node_ex, 1);
    expected_idx = {0, 9, 7, 1, 11, 2, 3, 4, 8, 10, 5, 6};
    expected_val = {0.1, 0.1, 0.3, 0.3, 0.45, 0.7, 0.6, 0.6, 0.6, 0.8, 0.9, 0.9};
    EXPECT_ARR_EQ(data.n_samples_train, df.get_samples_idx(), expected_idx, 1, 1, 0, 0);
    EXPECT_ARR_NEAR(data.n_samples_train, df.get_features_values(), expected_val,
                    1.0e-10);

    // All the last elements
    // Already sorted
    node_ex.start_idx = 4;
    node_ex.end_idx = 11;
    node_ex.n_samples = 8;
    df.sort_samples(node_ex, 0);
    expected_idx = {0, 9, 7, 1, 11, 2, 3, 4, 8, 10, 5, 6};
    EXPECT_ARR_EQ(data.n_samples_train, df.get_samples_idx(), expected_idx, 1, 1, 0, 0);

    // Start again with only 3 observations as a subset of the data set
    da_int n_obs = 3;
    std::vector<da_int> samples_subset{4, 9, 11};
    df.set_bootstrap(true);
    df.set_training_data(data.n_samples_train, data.n_feat, data.X_train.data(),
                         data.n_samples_train, data.y_train.data(), 2, n_obs,
                         samples_subset.data());
    df.fit();
    node_ex.start_idx = 0;
    node_ex.end_idx = 2;
    node_ex.n_samples = 3;
    df.sort_samples(node_ex, 0);
    expected_idx = {9, 11, 4};
    expected_val = {0.1, 0.2, 0.6};
    EXPECT_ARR_EQ(3, df.get_samples_idx(), expected_idx, 1, 1, 0, 0);
    EXPECT_ARR_EQ(3, df.get_features_values(), expected_val, 1, 1, 0, 0);
}

TYPED_TEST(decision_tree_test, individual_splits) {
    // Set up a new tree with the small data set defined above
    test_data_type<TypeParam> data;
    set_test_data_8x2_nonunique(data);

    da_errors::da_error_t err(da_errors::DA_RECORD);
    decision_tree<TypeParam> tree(err);
    tree.set_training_data(data.n_samples_train, data.n_feat, data.X_train.data(),
                           data.n_samples_train, data.y_train.data());
    TypeParam tol = 1.0e-05;

    // Set maximum depth to 1 to only have 1 node.
    EXPECT_EQ(tree.opts.set("maximum depth", (da_int)1), da_status_success);
    EXPECT_EQ(tree.opts.set("scoring function", "gini"), da_status_success);
    tree.fit();
    // Check that no nodes were added
    EXPECT_EQ(tree.get_tree()[0].left_child_idx, -1);
    EXPECT_EQ(tree.get_tree()[0].right_child_idx, -1);

    // Only one level of children
    EXPECT_EQ(tree.opts.set("maximum depth", (da_int)2), da_status_success);
    tree.refresh();
    tree.fit();
    EXPECT_EQ(tree.get_tree()[0].left_child_idx, 2);
    EXPECT_EQ(tree.get_tree()[0].right_child_idx, 1);
    EXPECT_NEAR(tree.get_tree()[2].score, 0.444444, tol);
    EXPECT_NEAR(tree.get_tree()[1].score, 0., tol);

    // Only 1.0 in the training data, no splitting should occur
    set_data_identical(data);
    tree.set_training_data(data.n_samples_train, data.n_feat, data.X_train.data(),
                           data.n_samples_train, data.y_train.data());
    tree.fit();
    EXPECT_EQ(tree.get_tree()[0].left_child_idx, -1);
    EXPECT_EQ(tree.get_tree()[0].right_child_idx, -1);
    EXPECT_EQ(tree.get_tree()[0].y_pred, 1);
}

TYPED_TEST(decision_tree_test, trivial_trees) {
    std::function<void(test_data_type<TypeParam> & data)> set_test_data[] = {
        set_test_data_8x1<TypeParam>, set_test_data_8x2_unique<TypeParam>,
        set_test_data_8x2_nonunique<TypeParam>, set_test_data_8x2_ldx<TypeParam>};
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

TYPED_TEST(decision_tree_test, small_multiclass) {
    // Data with 4 trivially separated classes
    // [0, 4]x[0, 4] : 0
    // [0, 4]x[6,10] : 1
    // [6,10]x[0, 4] : 2
    // [6,10]x[6,10] : 3
    // clang-format off
    std::vector<TypeParam> X {
        0, 2, 8, 9, 2, 2, 9, 7, 0, 1 , 7, 8 , 3, 3, 8, 9, 4, 0, 6, 10,
        2, 7, 4, 7, 2, 6, 1, 7, 0, 10, 1, 10, 4, 6, 4, 6, 3, 9, 2, 10};
    std::vector<da_int> y {
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    // clang-format on

    da_int nclass = 4, nsamples = 20, nfeat = 2;
    da_errors::da_error_t err(da_errors::DA_RECORD);
    decision_tree<TypeParam> dec_tree(err);
    EXPECT_EQ(dec_tree.set_training_data(nsamples, nfeat, X.data(), nsamples, y.data()),
              da_status_success);
    EXPECT_EQ(dec_tree.fit(), da_status_success);

    std::vector<TypeParam> X_test{1, 3, 6, 9, 2, 7, 1, 10};
    std::vector<da_int> y_pred(4);
    std::vector<da_int> y_expected{0, 1, 2, 3};
    da_int nsamp = 4;
    dec_tree.predict(nsamp, nfeat, X_test.data(), nsamp, y_pred.data());
    EXPECT_ARR_NEAR(nsamp, y_pred, y_expected, 1.0e-10);
    for (da_int i = 0; i < nsamp; i++)
        std::cout << y_pred[i] << " ";
    std::cout << std::endl;

    // Test the clear function
    std::vector<da_int> const &samples_idx = dec_tree.get_samples_idx();
    std::vector<da_int> const &features_idx = dec_tree.get_features_idx();
    std::vector<da_int> const &count_classes = dec_tree.get_count_classes();
    std::vector<da_int> const &count_left_classes = dec_tree.get_count_left_classes();
    std::vector<da_int> const &count_right_classes = dec_tree.get_count_right_classes();
    std::vector<TypeParam> const &features_values = dec_tree.get_features_values();

    EXPECT_GT(samples_idx.capacity(), 1);
    EXPECT_GT(features_idx.capacity(), 1);
    EXPECT_GT(count_classes.capacity(), 1);
    EXPECT_GT(count_left_classes.capacity(), 1);
    EXPECT_GT(count_right_classes.capacity(), 1);
    EXPECT_GT(features_values.capacity(), 1);
    dec_tree.clear_working_memory();
    EXPECT_EQ(samples_idx.capacity(), 0);
    EXPECT_EQ(features_idx.capacity(), 0);
    EXPECT_EQ(count_classes.capacity(), 0);
    EXPECT_EQ(count_left_classes.capacity(), 0);
    EXPECT_EQ(count_right_classes.capacity(), 0);
    EXPECT_EQ(features_values.capacity(), 0);
}

TYPED_TEST(decision_tree_test, get_results) {

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
    EXPECT_EQ(da_tree_fit<TypeParam>(tree_handle), da_status_success);
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
                                     (TypeParam)3,
                                     (TypeParam)5,
                                     (TypeParam)3};
    EXPECT_ARR_NEAR(7, rinfo, rinfo_exp, 1.0e-10);

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
    EXPECT_EQ(
        da_handle_get_result(tree_handle, da_result::da_linmod_coef, &dim, rinfo.data()),
        da_status_unknown_query);
    da_handle_destroy(&tree_handle);
}

TYPED_TEST(decision_tree_test, invalid_input) {

    da_status status;

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
              da_status_invalid_input);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, n_features, n_class,
                                        X.data(), n_samples, y_invalid),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_set_training_data(nullptr, n_samples, n_features, n_class,
                                        X_invalid, n_samples, y.data()),
              da_status_handle_not_initialized);
    // wrong dimensions
    EXPECT_EQ(da_tree_set_training_data(tree_handle, 0, n_features, n_class, X_invalid,
                                        n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, 0, n_class, X_invalid,
                                        n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, n_features, n_class,
                                        X_invalid, 1, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, n_features, n_class,
                                        X.data(), n_samples, y.data()),
              da_status_success);

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
              da_status_invalid_input);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, n_features, X.data(), n_samples,
                              y_invalid),
              da_status_invalid_input);
    EXPECT_EQ(
        da_tree_predict(nullptr, n_samples, n_features, X.data(), n_samples, y.data()),
        da_status_handle_not_initialized);
    // Wrong dimensions
    EXPECT_EQ(da_tree_predict(tree_handle, 0, n_features, X.data(), n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, 0, X.data(), n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, 4, X.data(), n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, n_features, X.data(), 1, y.data()),
              da_status_invalid_input);

    // score
    // Invalid pointers
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X_invalid, n_samples,
                            y.data(), &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X.data(), n_samples,
                            y_invalid, &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X.data(), n_samples,
                            y.data(), nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_score(nullptr, n_samples, n_features, X.data(), n_samples, y.data(),
                            &accuracy),
              da_status_handle_not_initialized);
    // Wrong dimensions
    EXPECT_EQ(da_tree_score(tree_handle, 0, n_features, X.data(), n_samples, y.data(),
                            &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, 0, X.data(), n_samples, y.data(),
                            &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, 4, X.data(), n_samples, y.data(),
                            &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X.data(), 1, y.data(),
                            &accuracy),
              da_status_invalid_input);

    da_handle_destroy(&tree_handle);
}

TEST(decision_tree, incorrect_handle_precision) {
    da_status status;

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
                                          n_samples, y.data()),
              da_status_wrong_type);
    EXPECT_EQ(da_tree_set_training_data_d(handle_s, n_samples, n_features, 0, X_d.data(),
                                          n_samples, y.data()),
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

TYPED_TEST(decision_tree_test, multiple_solve) {
    test_data_type<TypeParam> data;
    set_test_data_8x2_nonunique<TypeParam>(data);

    // Solve a first time
    da_errors::da_error_t err(da_errors::DA_RECORD);
    decision_tree<TypeParam> tree(err);
    EXPECT_EQ(tree.set_training_data(data.n_samples_train, data.n_feat,
                                     data.X_train.data(), data.n_samples_train,
                                     data.y_train.data()),
              da_status_success);
    EXPECT_EQ(tree.fit(), da_status_success);
    TypeParam accuracy;
    EXPECT_EQ(tree.score(data.n_samples_test, data.n_feat, data.X_test.data(),
                         data.n_samples_test, data.y_test.data(), &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);

    // Check that the model is trained
    EXPECT_EQ(tree.model_is_trained(), true);
    // solve again, no work should be performed
    EXPECT_EQ(tree.fit(), da_status_success);

    // Change an option and solve again
    EXPECT_EQ(tree.opts.set("maximum depth", (da_int)3), da_status_success);
    tree.refresh(); // refressh is only called by the public interfaces
    EXPECT_EQ(tree.model_is_trained(), false);
    EXPECT_EQ(tree.fit(), da_status_success);
    EXPECT_EQ(tree.score(data.n_samples_test, data.n_feat, data.X_test.data(),
                         data.n_samples_test, data.y_test.data(), &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
}

/***********************************
 ********* Positive tests***********
 ***********************************/
typedef struct dectree_param_t {
    std::string test_name; // name of the ctest test
    std::string data_name; // name of the files to read in
    std::vector<option_t<da_int>> iopts;
    std::vector<option_t<std::string>> sopts;
    std::vector<option_t<float>> fopts;
    std::vector<option_t<double>> dopts;
    float target_score;
} dectree_param_t;

// clang-format off
const dectree_param_t dectree_param_pos[] = {
    // Testing scoring functions
    {"iris_gini", "iris", {}, {{"scoring function", "gini"}}, {}, {}, 0.95},
    {"iris_entropy", "iris", {}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.95},
    {"iris_misclass", "iris", {}, {{"scoring function", "misclass"}}, {}, {}, 0.8},
    {"gen1_gini", "gen1", {}, {{"scoring function", "gini"}}, {}, {}, 0.93},
    {"gen1_entropy", "gen1", {}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.93},
    {"gen1_misclass", "gen1", {}, {{"scoring function", "misclass"}}, {}, {}, 0.93},
    {"gen_200x10_gini", "gen_200x10_3class", {}, {{"scoring function", "gini"}}, {}, {}, 0.93},
    {"gen_200x10_entropy", "gen_200x10_3class", {}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.93},
    {"gen_200x10_misclass", "gen_200x10_3class", {}, {{"scoring function", "misclass"}}, {}, {}, 0.93},
    {"gen_500x20_gini", "gen_500x20_4class", {}, {{"scoring function", "gini"}}, {}, {}, 0.9},
    {"gen_500x20_entropy", "gen_500x20_4class", {}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.9},
    {"gen_500x20_misclass", "gen_500x20_4class", {}, {{"scoring function", "misclass"}}, {}, {}, 0.89},


    // maximum splits
    {"gen_200x10_maxsplit", "gen_200x10_3class", {{"maximum depth", 20}}, {{"scoring function", "gini"}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}}, 0.9},
    {"gen_500x20_maxsplit", "gen_500x20_4class", {{"maximum depth", 20}}, {{"scoring function", "misclass"}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}}, 0.88},

    // Test identical train and test sets
    {"overfit_gini", "overfit", {{"maximum depth", 25}}, {{"scoring function", "gini"}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}}, 0.99},
    {"overfit_misclass", "overfit", {{"maximum depth", 25}}, {{"scoring function", "misclass"}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}}, 0.99},
    {"overfit_entropy", "overfit", {{"maximum depth", 25}}, {{"scoring function", "entropy"}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}}, 0.99},
    {"overfit_prune05", "overfit", {{"maximum depth", 25}}, {{"scoring function", "gini"}},
                            {{"Minimum split score", 0.05}, {"Minimum split improvement", 0.05}},
                            {{"Minimum split score", 0.05}, {"Minimum split improvement", 0.05}}, 0.97},
    {"overfit_prune1", "overfit", {{"maximum depth", 25}}, {{"scoring function", "gini"}},
                            {{"Minimum split score", 0.1}, {"Minimum split improvement", 0.1}},
                            {{"Minimum split score", 0.1}, {"Minimum split improvement", 0.1}}, 0.97},


    // splits on fewer than all the features
    {"gen_200x10_split4", "gen_200x10_3class", {{"maximum depth", 20}, {"seed", 42}, {"maximum features", 4}},
      {{"scoring function", "entropy"}}, {}, {}, 0.88},
    {"iris_split2", "iris", {{"maximum depth", 20}, {"seed", 42}, {"maximum features", 2}}, {{"scoring function", "gini"}}, {}, {}, 0.95},
    {"gen_500x20_split6", "gen_500x20_4class", {{"maximum depth", 20}, {"seed", 42}, {"maximum features", 7}},
      {{"scoring function", "gini"}}, {}, {}, 0.8},


    // smaller tree depth
    {"iris_depth2", "iris", {{"maximum depth", 2}}, {{"scoring function", "gini"}}, {}, {}, 0.6},
    {"gen1_depth2", "gen1", {{"maximum depth", 2}}, {{"scoring function", "entropy"}}, {}, {}, 0.9},
    {"gen200x10_depth2", "gen_200x10_3class", {{"maximum depth", 2}}, {{"scoring function", "gini"}}, {}, {}, 0.6},
    {"gen_500x20_depth3", "gen_500x20_4class", {{"maximum depth", 3}}, {{"scoring function", "gini"}}, {}, {}, 0.7},

};
// clang-format on

class dectree_positive : public testing::TestWithParam<dectree_param_t> {};
// Teach GTest how to print the param type
// in this case use only user's unique testname
// It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
void PrintTo(const dectree_param_t &param, ::std::ostream *os) { *os << param.test_name; }

// Positive tests with double and single type
TEST_P(dectree_positive, Double) {
    const dectree_param_t &param = GetParam();
    test_decision_tree_positive<double>(param.data_name, param.iopts, param.sopts,
                                        param.dopts, (double)param.target_score);
}
TEST_P(dectree_positive, Single) {
    const dectree_param_t &param = GetParam();
    test_decision_tree_positive<float>(param.data_name, param.iopts, param.sopts,
                                       param.fopts, (float)param.target_score);
}

INSTANTIATE_TEST_SUITE_P(decision_tree_pos_suite, dectree_positive,
                         testing::ValuesIn(dectree_param_pos));
