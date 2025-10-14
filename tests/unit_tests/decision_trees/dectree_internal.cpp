/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "decision_forest.hpp"
#include "decision_tree_misc.hpp"
#include "dectree_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <list>

using namespace TEST_ARCH::da_decision_forest;

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

TEST(dectree_internal, bucket_sort) {

    da_int n_samples = 10;
    std::vector<float> X = {1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 2, 2, 2, 1, 0, 0, 1, 1};
    std::vector<float> feature_values;
    std::vector<da_int> y = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    std::vector<da_int> categorical_features = {2, 3};
    std::vector<da_int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<da_int> buckets(n_samples * 3);
    std::vector<da_int> bucket_idx(3);

    // First column
    feature_values = {1, 0, 1, 1, 0, 0, 0, 1, 1, 1};
    bucket_sort_samples(indices, feature_values, categorical_features[0], 0, n_samples,
                        n_samples, buckets, bucket_idx);
    std::vector<float> samples_idx_exp = {1, 4, 5, 6, 0, 2, 3, 7, 8, 9};
    EXPECT_ARR_EQ(n_samples, indices, samples_idx_exp, 1, 1, 0, 0);

    // Slice of first column
    std::iota(indices.begin(), indices.end(), 0);
    feature_values = {1, 0, 1, 1, 0, 0, 0, 1, 1, 1};
    bucket_sort_samples(indices, feature_values, categorical_features[0], 2, 5, n_samples,
                        buckets, bucket_idx);
    samples_idx_exp = {0, 1, 4, 5, 6, 2, 3, 7, 8, 9};
    EXPECT_ARR_EQ(n_samples, indices, samples_idx_exp, 1, 1, 0, 0);

    // Second column
    std::iota(indices.begin(), indices.end(), 0);
    feature_values = {0, 1, 2, 2, 2, 1, 0, 0, 1, 1};
    bucket_sort_samples(indices, feature_values, categorical_features[1], 0, n_samples,
                        n_samples, buckets, bucket_idx);
    samples_idx_exp = {0, 6, 7, 1, 5, 8, 9, 2, 3, 4};
    EXPECT_ARR_EQ(n_samples, indices, samples_idx_exp, 1, 1, 0, 0);

    // Slice of first column with unsorted samples_idx
    std::iota(indices.begin(), indices.end(), 0);
    feature_values = {1, 0, 1, 1, 0, 0, 0, 1, 1, 1};
    bucket_sort_samples(indices, feature_values, categorical_features[0], 0, n_samples,
                        n_samples, buckets, bucket_idx);
    std::cout << std::endl;
    feature_values = {0, 1, 2, 2, 2, 1, 0, 0, 1, 1};
    bucket_sort_samples(indices, feature_values, categorical_features[1], 3, 5, n_samples,
                        buckets, bucket_idx);
    samples_idx_exp = {1, 4, 5, 3, 7, 2, 6, 0, 8, 9};
    EXPECT_ARR_EQ(n_samples, indices, samples_idx_exp, 1, 1, 0, 0);
}

TYPED_TEST(dectree_internal_test, heapsort) {
    da_int n_samples = 10;
    std::vector<TypeParam> values = {5., 4., 3., 2., 8., 0., 9., 1., 7., 6.};
    std::vector<da_int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    multi_range_heap_sort(indices, values, (da_int)0, n_samples);

    // Sort full array
    std::vector<TypeParam> expected_values = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::vector<da_int> expected_indices = {5, 7, 3, 2, 1, 0, 9, 8, 4, 6};
    EXPECT_ARR_EQ(n_samples, values, expected_values, 1, 1, 0, 0);
    EXPECT_ARR_EQ(n_samples, indices, expected_indices, 1, 1, 0, 0);

    // Reinitialize and sort in indices [2:6]
    std::iota(indices.begin(), indices.end(), 0);
    values = {5., 4., 3., 2., 8., 0., 9., 1., 7., 6.};
    multi_range_heap_sort(indices, values, (da_int)2, (da_int)5);
    expected_values = {5, 4, 0, 2, 3, 8, 9, 1, 7, 6};
    expected_indices = {0, 1, 5, 3, 2, 4, 6, 7, 8, 9};
    EXPECT_ARR_EQ(n_samples, values, expected_values, 1, 1, 0, 0);
    EXPECT_ARR_EQ(n_samples, indices, expected_indices, 1, 1, 0, 0);
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

TEST(dectree_internal, histograms) {
    // Create histograms using 4 bins
    da_int max_bin = 4;
    da_int n_samples = 12;
    da_int n_features = 2;
    bins<float> hist(max_bin, n_samples, n_features);

    // clang-format off
    std::vector<float> X = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2};
    // clang-format on

    // Test Full sort implementation
    std::vector<float> vals(n_samples);
    std::copy(X.begin(), X.begin() + n_samples, vals.begin());
    hist.compute_thresholds(vals, 0, 0, 1.0e-05);
    std::vector<float> thresholds_exp = {3, 6, 9};
    EXPECT_ARR_EQ(max_bin - 1, hist.get_thresholds(), thresholds_exp, 1, 1, 0, 0);
    std::copy(X.begin() + n_samples, X.end(), vals.begin());
    hist.compute_thresholds(vals, 0, 1, 1.0e-05);
    thresholds_exp = {1, 2};
    std::vector<da_int> nbins_exp = {4, 3};
    EXPECT_ARR_EQ(2, hist.get_nbins(), nbins_exp, 1, 1, 0, 0);
    EXPECT_ARR_EQ(2, hist.get_thresholds(), thresholds_exp, 1, 1, 3, 0);

    EXPECT_EQ(hist.compute_histograms(X.data(), n_samples, n_features, n_samples),
              da_status_success);
    std::vector<uint16_t> exp_bins = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                                      0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2};
    EXPECT_ARR_EQ(n_samples * n_features, hist.get_bins(), exp_bins, 1, 1, 0, 0);

    // Test nth element implementation
    // First column
    std::copy(X.begin(), X.begin() + n_samples, vals.begin());
    vals = {2, 6, 3, 11, 1, 9, 0, 8, 5, 4, 10, 7};
    hist.compute_thresholds_continuous(vals, 0, 0, 1.0e-05);
    thresholds_exp = {3, 6, 9};
    EXPECT_ARR_EQ(max_bin - 1, hist.get_thresholds(), thresholds_exp, 1, 1, 0, 0);
    // Second columns
    std::copy(X.begin() + n_samples, X.end(), vals.begin());
    hist.compute_thresholds_continuous(vals, 0, 1, 1.0e-05);
    for (int i = 0; i < max_bin - 1; i++)
        std::cout << hist.get_thresholds()[max_bin - 1 + i] << " ";
    std::cout << std::endl;
    thresholds_exp = {1.0, 2.0};
    EXPECT_ARR_EQ(2, hist.get_thresholds(), thresholds_exp, 1, 1, max_bin - 1, 0);

    // Test too few samples to create bins
    n_samples = 2, n_features = 2;
    vals = {0, 1, 0, 1};
    EXPECT_THROW(bins<float> hist_small(max_bin, n_samples, n_features),
                 std::invalid_argument);
}

TEST(dectree_internal_test, compress_indices) {
    std::vector<da_int> indices = {4, 2, 0, 1, 2, 2, 3, 3};
    std::vector<da_int> count(5);
    compress_count_occurences(indices, count);
    std::vector<da_int> count_exp = {1, 1, 3, 2, 1};
    EXPECT_ARR_EQ(5, count, count_exp, 1, 1, 0, 0);
}

TYPED_TEST(dectree_internal_test, bootstrap) {
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

    // n_samples subset with repeated indices
    da_int n_features = 2, n_samples = 20, n_class = 4;
    std::vector<da_int> samples_subset = {2, 1, 1, 1, 0, 4, 18, 19, 15, 15,
                                          3, 3, 9, 9, 7, 8, 6,  4,  8,  11};
    da_errors::da_error_t err(da_errors::DA_RECORD);
    decision_tree<TypeParam> dec_tree(err);
    EXPECT_EQ(dec_tree.set_training_data(n_samples, n_features, X.data(), n_samples,
                                         y.data(), n_class, (da_int)samples_subset.size(),
                                         samples_subset.data()),
              da_status_success);
    dec_tree.set_bootstrap(true);
    EXPECT_EQ(dec_tree.fit(), da_status_success);
    std::vector<TypeParam> X_test{1, 3, 6, 9, 2, 7, 1, 10};
    std::vector<da_int> y_pred(4);
    std::vector<da_int> y_expected{0, 1, 2, 3};
    da_int nsamp = 4;
    dec_tree.predict(nsamp, n_features, X_test.data(), nsamp, y_pred.data());
    EXPECT_ARR_NEAR(nsamp, y_pred, y_expected, 1.0e-10);
    for (da_int i = 0; i < nsamp; i++)
        std::cout << y_pred[i] << " ";
    std::cout << std::endl;

    // with fewer samples
    samples_subset = {0, 1, 0, 3, 2, 5, 5, 8, 8, 12, 15, 18, 19, 19, 10};
    EXPECT_EQ(dec_tree.set_training_data(n_samples, n_features, X.data(), n_samples,
                                         y.data(), n_class, (da_int)samples_subset.size(),
                                         samples_subset.data()),
              da_status_success);
    dec_tree.set_bootstrap(true);
    EXPECT_EQ(dec_tree.fit(), da_status_success);
    dec_tree.predict(nsamp, n_features, X_test.data(), nsamp, y_pred.data());
    EXPECT_ARR_NEAR(nsamp, y_pred, y_expected, 1.0e-10);
    for (da_int i = 0; i < nsamp; i++)
        std::cout << y_pred[i] << " ";
    std::cout << std::endl;
}