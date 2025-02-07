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

#include "../datests_cblas.hh"
#include "../utest_utils.hpp"
#include "aoclda.h"
#include "dectree_positive.hpp"
#include "dectree_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <limits>
#include <list>
#include <string>

template <typename T> class dectree_public_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(dectree_public_test, FloatTypes);

TYPED_TEST(dectree_public_test, trivial_trees) {
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

TYPED_TEST(dectree_public_test, get_results) {

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
                                     (TypeParam)2,
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
    dim = 7;
    EXPECT_EQ(da_handle_get_result(tree_handle, da_result::da_rinfo, &dim, rinfo.data()),
              da_status_unknown_query);
    da_handle_destroy(&tree_handle);
}

TYPED_TEST(dectree_public_test, invalid_input) {

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
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X.data(), n_samples,
                            y.data(), nullptr),
              da_status_invalid_pointer);
    EXPECT_EQ(da_tree_score(nullptr, n_samples, n_features, X.data(), n_samples, y.data(),
                            &accuracy),
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
    {"gen_200x10_maxsplit", "gen_200x10_3class", {{"maximum depth", 19}}, {{"scoring function", "gini"}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}}, 0.9},
    {"gen_500x20_maxsplit", "gen_500x20_4class", {{"maximum depth", 19}}, {{"scoring function", "misclass"}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}}, 0.88},

    // Test identical train and test sets
    {"overfit_gini", "overfit", {{"maximum depth", 24}}, {{"scoring function", "gini"}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}}, 0.99},
    {"overfit_misclass", "overfit", {{"maximum depth", 24}}, {{"scoring function", "misclass"}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}}, 0.99},
    {"overfit_entropy", "overfit", {{"maximum depth", 24}}, {{"scoring function", "entropy"}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}},
                            {{"Minimum split score", 0.0}, {"Minimum split improvement", 0.0}}, 0.99},
    {"overfit_prune05", "overfit", {{"maximum depth", 24}}, {{"scoring function", "gini"}},
                            {{"Minimum split score", 0.05}, {"Minimum split improvement", 0.05}},
                            {{"Minimum split score", 0.05}, {"Minimum split improvement", 0.05}}, 0.97},
    {"overfit_prune1", "overfit", {{"maximum depth", 24}}, {{"scoring function", "gini"}},
                            {{"Minimum split score", 0.1}, {"Minimum split improvement", 0.1}},
                            {{"Minimum split score", 0.1}, {"Minimum split improvement", 0.1}}, 0.97},


    // splits on fewer than all the features
    {"gen_200x10_split4", "gen_200x10_3class", {{"maximum depth", 19}, {"seed", 42}, {"maximum features", 4}},
      {{"scoring function", "entropy"}}, {}, {}, 0.88},
    {"iris_split2", "iris", {{"maximum depth", 19}, {"seed", 42}, {"maximum features", 2}}, {{"scoring function", "gini"}}, {}, {}, 0.95},
    {"gen_500x20_split6", "gen_500x20_4class", {{"maximum depth", 19}, {"seed", 42}, {"maximum features", 7}},
      {{"scoring function", "gini"}}, {}, {}, 0.8},


    // smaller tree depth
    {"iris_depth2", "iris", {{"maximum depth", 1}}, {{"scoring function", "gini"}}, {}, {}, 0.6},
    {"gen1_depth2", "gen1", {{"maximum depth", 1}}, {{"scoring function", "entropy"}}, {}, {}, 0.9},
    {"gen200x10_depth2", "gen_200x10_3class", {{"maximum depth", 1}}, {{"scoring function", "gini"}}, {}, {}, 0.6},
    {"gen_500x20_depth3", "gen_500x20_4class", {{"maximum depth", 2}}, {{"scoring function", "gini"}}, {}, {}, 0.7},

    // sorting method
    {"iris_gini", "iris", {}, {{"scoring function", "gini"}, {"sorting method", "stl"}}, {}, {}, 0.95},
    {"gen1_entropy", "gen1", {}, {{"scoring function", "cross-entropy"}, {"sorting method", "stl"}}, {}, {}, 0.93},
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

TEST(decision_tree, row_major) {

    // Get the training data
    std::string input_data_fname =
        std::string(DATA_DIR) + "/df_data/gen_200x10_3class_data.csv";
    da_datastore csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(csv_store, "datastore precision", "single"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(csv_store, input_data_fname.c_str()),
              da_status_success);

    da_int ncols, nrows;
    EXPECT_EQ(da_data_get_n_cols(csv_store, &ncols), da_status_success);
    EXPECT_EQ(da_data_get_n_rows(csv_store, &nrows), da_status_success);
    // The first ncols-1 columns contain the feature matrix; the last one the response vector
    // Create the selections in the data store
    EXPECT_EQ(da_data_select_columns(csv_store, "features", 0, ncols - 2),
              da_status_success);
    EXPECT_EQ(da_data_select_columns(csv_store, "response", ncols - 1, ncols - 1),
              da_status_success);

    da_int n_features = ncols - 1;
    da_int n_samples = nrows;
    // Extract the selections
    std::vector<float> X(n_features * n_samples);
    std::vector<da_int> y(n_samples);
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", column_major, X.data(),
                                        n_samples),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", column_major, y.data(),
                                        n_samples),
              da_status_success);
    da_datastore_destroy(&csv_store);
    da_int n_class = *std::max_element(y.begin(), y.end()) + 1;
    std::vector<float> X_test;
    for (float val : X)
        X_test.push_back(1.6 + val / 2 + std::cos(val));
    // X, X_test and y now form our data

    // Create main handle and set options
    da_handle tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<float>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    EXPECT_EQ(da_options_set_int(tree_handle, "maximum depth", 5), da_status_success);
    EXPECT_EQ(da_options_set_int(tree_handle, "seed", 77), da_status_success);
    EXPECT_EQ(da_options_set_string(tree_handle, "scoring function", "gini"),
              da_status_success);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, n_features, n_class,
                                        X.data(), n_samples, y.data()),
              da_status_success);
    EXPECT_EQ(da_tree_fit<float>(tree_handle), da_status_success);
    std::vector<da_int> y_pred(n_samples);
    std::vector<float> y_proba(n_samples * n_class), y_log_proba(n_samples * n_class);
    float mean_accuracy;
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, n_features, X_test.data(),
                              n_samples, y_pred.data()),
              da_status_success);
    EXPECT_EQ(da_tree_predict_proba(tree_handle, n_samples, n_features, X_test.data(),
                                    n_samples, y_proba.data(), n_class, n_samples),
              da_status_success);
    EXPECT_EQ(da_tree_predict_log_proba(tree_handle, n_samples, n_features, X_test.data(),
                                        n_samples, y_log_proba.data(), n_class,
                                        n_samples),
              da_status_success);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X_test.data(), n_samples,
                            y.data(), &mean_accuracy),
              da_status_success);

    da_handle_destroy(&tree_handle);

    //Now repeat with row major data
    datest_blas::imatcopy('T', n_samples, n_features, 1.0, X.data(), n_samples,
                          n_features);
    datest_blas::imatcopy('T', n_samples, n_features, 1.0, X_test.data(), n_samples,
                          n_features);
    EXPECT_EQ(da_handle_init<float>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    EXPECT_EQ(da_options_set_int(tree_handle, "maximum depth", 5), da_status_success);
    EXPECT_EQ(da_options_set_int(tree_handle, "seed", 77), da_status_success);
    EXPECT_EQ(da_options_set_string(tree_handle, "scoring function", "gini"),
              da_status_success);
    EXPECT_EQ(da_options_set_string(tree_handle, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_tree_set_training_data(tree_handle, n_samples, n_features, n_class,
                                        X.data(), n_features, y.data()),
              da_status_success);
    EXPECT_EQ(da_tree_fit<float>(tree_handle), da_status_success);
    std::vector<da_int> y_pred_row(n_samples);
    std::vector<float> y_proba_row(n_samples * n_class),
        y_log_proba_row(n_samples * n_class);
    float mean_accuracy_row;
    EXPECT_EQ(da_tree_predict(tree_handle, n_samples, n_features, X_test.data(),
                              n_features, y_pred_row.data()),
              da_status_success);
    EXPECT_EQ(da_tree_predict_proba(tree_handle, n_samples, n_features, X_test.data(),
                                    n_features, y_proba_row.data(), n_class, n_class),
              da_status_success);
    EXPECT_EQ(da_tree_predict_log_proba(tree_handle, n_samples, n_features, X_test.data(),
                                        n_features, y_log_proba_row.data(), n_class,
                                        n_class),
              da_status_success);
    EXPECT_EQ(da_tree_score(tree_handle, n_samples, n_features, X_test.data(), n_features,
                            y.data(), &mean_accuracy_row),
              da_status_success);
    da_handle_destroy(&tree_handle);

    //Check row and column outputs agree
    datest_blas::imatcopy('T', n_class, n_samples, 1.0, y_proba_row.data(), n_class,
                          n_samples);
    datest_blas::imatcopy('T', n_class, n_samples, 1.0, y_log_proba_row.data(), n_class,
                          n_samples);
    EXPECT_ARR_NEAR(n_samples, y_pred.data(), y_pred_row.data(),
                    10 * std::numeric_limits<float>::epsilon());
    EXPECT_ARR_NEAR(n_samples * n_class, y_proba.data(), y_proba_row.data(),
                    10 * std::numeric_limits<float>::epsilon());
    //Guard against inifnite values
    for (float &value : y_log_proba) {
        if (std::isinf(value)) {
            value = 0.0f;
        }
    }
    for (float &value : y_log_proba_row) {
        if (std::isinf(value)) {
            value = 0.0f;
        }
    }
    EXPECT_ARR_NEAR(n_samples * n_class, y_log_proba.data(), y_log_proba_row.data(),
                    10 * std::numeric_limits<float>::epsilon());
    EXPECT_NEAR(mean_accuracy, mean_accuracy_row,
                10 * std::numeric_limits<float>::epsilon());
}
