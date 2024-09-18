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

#include "../utest_utils.hpp"
#include "decision_tree.hpp"
#include "dectree_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <list>

using namespace da_decision_tree;

template <typename T> class dectree_internal_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(dectree_internal_test, FloatTypes);

TYPED_TEST(dectree_internal_test, scorefun) {
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

TYPED_TEST(dectree_internal_test, sort_samples) {
    // Set up a new tree with the small data set defined above
    test_data_type<TypeParam> data;
    set_test_data_8x2_nonunique(data);

    da_errors::da_error_t err(da_errors::DA_RECORD);
    decision_tree<TypeParam> df(err);
    df.set_training_data(data.n_samples_train, data.n_feat, data.X_train.data(),
                         data.n_samples_train, data.y_train.data(), 2);
    EXPECT_EQ(df.opts.set("maximum depth", (da_int)0), da_status_success);
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

TYPED_TEST(dectree_internal_test, individual_splits) {
    // Set up a new tree with the small data set defined above
    test_data_type<TypeParam> data;
    set_test_data_8x2_nonunique(data);

    da_errors::da_error_t err(da_errors::DA_RECORD);
    decision_tree<TypeParam> tree(err);
    tree.set_training_data(data.n_samples_train, data.n_feat, data.X_train.data(),
                           data.n_samples_train, data.y_train.data());
    TypeParam tol = 1.0e-05;

    // Set maximum depth to 1 to only have 1 node.
    EXPECT_EQ(tree.opts.set("maximum depth", (da_int)0), da_status_success);
    EXPECT_EQ(tree.opts.set("scoring function", "gini"), da_status_success);
    tree.fit();
    // Check that no nodes were added
    EXPECT_EQ(tree.get_tree()[0].left_child_idx, -1);
    EXPECT_EQ(tree.get_tree()[0].right_child_idx, -1);

    // Only one level of children
    EXPECT_EQ(tree.opts.set("maximum depth", (da_int)1), da_status_success);
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

TYPED_TEST(dectree_internal_test, small_multiclass) {
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

    da_int nsamples = 20, nfeat = 2;
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

TYPED_TEST(dectree_internal_test, multiple_solve) {
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
    EXPECT_EQ(tree.opts.set("maximum depth", (da_int)2), da_status_success);
    tree.refresh(); // refresh is only called by the public interfaces
    EXPECT_EQ(tree.model_is_trained(), false);
    EXPECT_EQ(tree.fit(), da_status_success);
    EXPECT_EQ(tree.score(data.n_samples_test, data.n_feat, data.X_test.data(),
                         data.n_samples_test, data.y_test.data(), &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
}
