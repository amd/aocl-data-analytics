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
#include "random_forest.hpp"
#include "random_forest_positive.hpp"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <list>
#include <string>

using namespace da_random_forest;

template <typename T> class random_forest_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};
using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(random_forest_test, FloatTypes);

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
    data.X_train = {(T)0.12, (T)0.11, (T)0.42, (T)0.41, (T)0.62, (T)0.61, (T)0.92,
                    (T)0.91, (T)0.16, (T)0.30, (T)0.39, (T)0.79, (T)0.38, (T)0.78,
                    (T)0.37, (T)0.77, (T)0.36, (T)0.76, (T)0.30, (T)0.16};

    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    data.y_train = {0, 1, 0, 1, 1, 1, 1, 1, 0, 0};
    data.X_test = {(T)0.25, (T)0.25, (T)0.75, (T)0.75,
                   (T)0.25, (T)0.75, (T)0.25, (T)0.75};
    data.y_test = {0, 1, 1, 1};
    data.n_samples_train = 10, data.n_feat = 2;
    data.n_samples_test = 4;
    data.ldx_train = 10;
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

TYPED_TEST(random_forest_test, trivial_forests) {
    std::function<void(test_data_type<TypeParam> & data)> set_test_data[] = {
        set_test_data_8x1<TypeParam>, set_test_data_8x2_unique<TypeParam>,
        set_test_data_8x2_nonunique<TypeParam>, set_test_data_8x2_ldx<TypeParam>};
    test_data_type<TypeParam> data;

    da_int i = 0;
    for (auto &data_fun : set_test_data) {
        std::cout << "Testing function: " << i << std::endl;
        data_fun(data);
        da_handle tree_handle = nullptr;
        EXPECT_EQ(da_handle_init<TypeParam>(&tree_handle, da_handle_decision_forest),
                  da_status_success);
        EXPECT_EQ(da_forest_set_training_data(tree_handle, data.n_samples_train,
                                              data.n_feat, 0, data.X_train.data(),
                                              data.ldx_train, data.y_train.data()),
                  da_status_success);
        EXPECT_EQ(da_options_set(tree_handle, "features selection", "all"),
                  da_status_success);
        EXPECT_EQ(da_options_set(tree_handle, "bootstrap", "no"), da_status_success);
        EXPECT_EQ(da_forest_fit<TypeParam>(tree_handle), da_status_success);
        TypeParam accuracy;
        EXPECT_EQ(da_forest_score(tree_handle, data.n_samples_test, data.n_feat,
                                  data.X_test.data(), data.ldx_test, data.y_test.data(),
                                  &accuracy),
                  da_status_success);
        EXPECT_NEAR(accuracy, 1.0, 1.0e-05);
        std::vector<da_int> y_pred(data.n_samples_test);
        EXPECT_EQ(da_forest_predict(tree_handle, data.n_samples_test, data.n_feat,
                                    data.X_test.data(), data.ldx_test, y_pred.data()),
                  da_status_success);
        EXPECT_ARR_EQ(data.n_samples_test, y_pred, data.y_test, 1, 1, 0, 0);

        da_handle_destroy(&tree_handle);
        i++;
    }
}

TYPED_TEST(random_forest_test, get_results) {

    test_data_type<TypeParam> data;
    set_test_data_8x2_nonunique<TypeParam>(data);

    // Train the default tree on the small data set
    da_handle forest_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&forest_handle, da_handle_decision_forest),
              da_status_success);
    EXPECT_EQ(da_forest_set_training_data(forest_handle, data.n_samples_train,
                                          data.n_feat, 0, data.X_train.data(),
                                          data.n_samples_train, data.y_train.data()),
              da_status_success);
    da_int seed = 42, n_tree = 5;
    EXPECT_EQ(da_options_set(forest_handle, "seed", seed), da_status_success);
    EXPECT_EQ(da_options_set(forest_handle, "number of trees", n_tree),
              da_status_success);
    EXPECT_EQ(da_forest_fit<TypeParam>(forest_handle), da_status_success);
    // Quick check on test data
    std::vector<TypeParam> X_test{0.1, 0.3, 0.7, 0.9, 0.2, 0.6, 0.8, 0.1};
    std::vector<da_int> y_test{0, 1, 1, 1};
    TypeParam accuracy;
    EXPECT_EQ(
        da_forest_score(forest_handle, 4, 2, X_test.data(), 4, y_test.data(), &accuracy),
        da_status_success);
    EXPECT_NEAR(accuracy, 1.0, 1.0e-03);

    // get the results and check the values
    da_int dim = 100;
    da_int n_obs = (da_int)std::ceil(0.8 * data.n_samples_train);
    std::vector<TypeParam> rinfo(dim);
    EXPECT_EQ(
        da_handle_get_result(forest_handle, da_result::da_rinfo, &dim, rinfo.data()),
        da_status_success);
    std::vector<TypeParam> rinfo_exp{(TypeParam)data.n_feat,
                                     (TypeParam)data.n_samples_train, (TypeParam)n_obs,
                                     (TypeParam)seed, (TypeParam)n_tree};
    EXPECT_ARR_NEAR(4, rinfo, rinfo_exp, 1.0e-10);

    // Check that other queries fail
    EXPECT_EQ(da_handle_get_result(forest_handle, da_result::da_linmod_coef, &dim,
                                   rinfo.data()),
              da_status_unknown_query);

    // Check the wrong dimension
    dim = 1;
    EXPECT_EQ(
        da_handle_get_result(forest_handle, da_result::da_rinfo, &dim, rinfo.data()),
        da_status_invalid_array_dimension);

    // change an option and check that results are no longer available
    EXPECT_EQ(da_options_set(forest_handle, "seed", (da_int)43), da_status_success);
    EXPECT_EQ(da_handle_get_result(forest_handle, da_result::da_linmod_coef, &dim,
                                   rinfo.data()),
              da_status_unknown_query);
    da_handle_destroy(&forest_handle);
}

TYPED_TEST(random_forest_test, invalid_input) {

    std::vector<TypeParam> X{0.0, 1.0, 0.0, 2.0};
    std::vector<da_int> y{0, 1};

    // Initialize the decision forest class and fit model
    da_handle forest_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&forest_handle, da_handle_decision_forest),
              da_status_success);

    // call set_training_data with invalid values
    da_int n_samples = 2, n_features = 2, n_class = 0;
    TypeParam accuracy;

    // set_training_data
    // Invalid pointers
    TypeParam *X_invalid = nullptr;
    da_int *y_invalid = nullptr;
    EXPECT_EQ(da_forest_set_training_data(forest_handle, n_samples, n_features, n_class,
                                          X_invalid, n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_set_training_data(forest_handle, n_samples, n_features, n_class,
                                          X.data(), n_samples, y_invalid),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_set_training_data(nullptr, n_samples, n_features, n_class,
                                          X_invalid, n_samples, y.data()),
              da_status_handle_not_initialized);
    // wrong dimensions
    EXPECT_EQ(da_forest_set_training_data(forest_handle, 0, n_features, n_class,
                                          X_invalid, n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_set_training_data(forest_handle, n_samples, 0, n_class, X_invalid,
                                          n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_set_training_data(forest_handle, n_samples, n_features, n_class,
                                          X_invalid, 1, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_set_training_data(forest_handle, n_samples, n_features, n_class,
                                          X.data(), n_samples, y.data()),
              da_status_success);

    // model out of date for evaluation
    EXPECT_EQ(da_forest_predict(forest_handle, n_samples, n_features, X.data(), n_samples,
                                y.data()),
              da_status_out_of_date);
    EXPECT_EQ(da_forest_score(forest_handle, n_samples, n_features, X.data(), n_samples,
                              y.data(), &accuracy),
              da_status_out_of_date);
    EXPECT_EQ(da_forest_fit<TypeParam>(nullptr), da_status_handle_not_initialized);
    EXPECT_EQ(da_forest_fit<TypeParam>(forest_handle), da_status_success);

    // predict
    // Invalid pointers
    EXPECT_EQ(da_forest_predict(forest_handle, n_samples, n_features, X_invalid,
                                n_samples, y.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_predict(forest_handle, n_samples, n_features, X.data(), n_samples,
                                y_invalid),
              da_status_invalid_input);
    EXPECT_EQ(
        da_forest_predict(nullptr, n_samples, n_features, X.data(), n_samples, y.data()),
        da_status_handle_not_initialized);
    // Wrong dimensions
    EXPECT_EQ(
        da_forest_predict(forest_handle, 0, n_features, X.data(), n_samples, y.data()),
        da_status_invalid_input);
    EXPECT_EQ(
        da_forest_predict(forest_handle, n_samples, 0, X.data(), n_samples, y.data()),
        da_status_invalid_input);
    EXPECT_EQ(
        da_forest_predict(forest_handle, n_samples, 4, X.data(), n_samples, y.data()),
        da_status_invalid_input);
    EXPECT_EQ(
        da_forest_predict(forest_handle, n_samples, n_features, X.data(), 1, y.data()),
        da_status_invalid_input);

    // score
    // Invalid pointers
    EXPECT_EQ(da_forest_score(forest_handle, n_samples, n_features, X_invalid, n_samples,
                              y.data(), &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_score(forest_handle, n_samples, n_features, X.data(), n_samples,
                              y_invalid, &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_score(forest_handle, n_samples, n_features, X.data(), n_samples,
                              y.data(), nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_score(nullptr, n_samples, n_features, X.data(), n_samples,
                              y.data(), &accuracy),
              da_status_handle_not_initialized);
    // Wrong dimensions
    EXPECT_EQ(da_forest_score(forest_handle, 0, n_features, X.data(), n_samples, y.data(),
                              &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_score(forest_handle, n_samples, 0, X.data(), n_samples, y.data(),
                              &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_score(forest_handle, n_samples, 4, X.data(), n_samples, y.data(),
                              &accuracy),
              da_status_invalid_input);
    EXPECT_EQ(da_forest_score(forest_handle, n_samples, n_features, X.data(), 1, y.data(),
                              &accuracy),
              da_status_invalid_input);

    da_handle_destroy(&forest_handle);
}

/***********************************
 ********* Positive tests **********
 ***********************************/

typedef struct forest_param_t {
    std::string test_name; // name of the ctest test
    std::string data_name; // name of the files to read in
    std::vector<option_t<da_int>> iopts;
    std::vector<option_t<std::string>> sopts;
    std::vector<option_t<float>> fopts;
    std::vector<option_t<double>> dopts;
    float target_score;
} forest_param_t;

// clang-format off
const forest_param_t forest_param_pos[] = {
    {"iris_gini", "iris", {{"number of trees", 25}, {"seed", 42}},
        {{"scoring function", "gini"}}, {}, {}, 0.95},
    {"iris_entropy", "iris", {{"number of trees", 25}, {"seed", 42}},
        {{"scoring function", "cross-entropy"}}, {}, {}, 0.95},
    {"iris_misclass", "iris", {{"number of trees", 50}, {"seed", 42}},
        {{"scoring function", "misclass"}, {"features selection", "all"}}, {}, {}, 0.95},
    {"gen1_gini", "gen1", {{"number of trees", 25}, {"seed", 42}},
        {{"scoring function", "gini"}}, {}, {}, 0.93},
    {"gen1_entropy", "gen1", {{"number of trees", 25}, {"seed", 42}},
        {{"scoring function", "cross-entropy"}}, {}, {}, 0.93},
    {"gen1_misclass", "gen1", {{"number of trees", 25}, {"seed", 42}},
        {{"scoring function", "misclass"}}, {}, {}, 0.93},
    {"gen_200x10_gini", "gen_200x10_3class", {{"number of trees", 25}, {"seed", 42}},
        {{"scoring function", "gini"}},
        {{"bootstrap samples factor", 1.0}}, {{"bootstrap samples factor", 1.0}}, 0.93},
    {"gen_200x10_entropy", "gen_200x10_3class", {{"number of trees", 25},
        {"seed", 42}}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.93},
    {"gen_200x10_misclass", "gen_200x10_3class", {{"number of trees", 25},
        {"seed", 42}}, {{"scoring function", "misclass"}}, {}, {}, 0.93},
    {"gen_500x20_gini", "gen_500x20_4class", {{"number of trees", 25},
        {"seed", 42}}, {{"scoring function", "gini"}}, {}, {}, 0.9},
    {"gen_500x20_entropy", "gen_500x20_4class", {{"number of trees", 25},
        {"seed", 42}}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.9},
    {"gen_500x20_misclass", "gen_500x20_4class", {{"number of trees", 25},
        {"seed", 42}}, {{"scoring function", "misclass"}}, {}, {}, 0.9},

    // splits on fewer than all the features
    {"gen_200x10_split4", "gen_200x10_3class",
        {{"number of trees", 25}, {"maximum depth", 19}, {"seed", 42}, {"maximum features", 4}},
        {{"scoring function", "entropy"}}, {}, {}, 0.93},
    {"iris_split2", "iris", {{"number of trees", 25}, {"maximum depth", 19}, {"seed", 42}, {"maximum features", 2}},
        {{"scoring function", "gini"}}, {}, {}, 0.95},
    {"gen_500x20_split6", "gen_500x20_4class",
        {{"number of trees", 25}, {"maximum depth", 19}, {"seed", 42}, {"maximum features", 7}},
        {{"scoring function", "gini"}}, {}, {}, 0.9},

    // Test parallel inference
    {"inference_1_block", "gen_200x10_3class", {{"number of trees", 25},
        {"seed", 42}, {"block size", 400}}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.93},
    {"inference_2_blocks", "gen_200x10_3class", {{"number of trees", 25},
        {"seed", 42}, {"block size", 200}}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.93},
    {"inference_400_blocks", "gen_200x10_3class", {{"number of trees", 25},
        {"seed", 42}, {"block size", 1}}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.93},
    {"inference_37_blocks", "gen_200x10_3class", {{"number of trees", 25},
        {"seed", 42}, {"block size", 37}}, {{"scoring function", "cross-entropy"}}, {}, {}, 0.93},

};
// clang-format on

class forest_positive : public testing::TestWithParam<forest_param_t> {};
// Teach GTest how to print the param type
// in this case use only user's unique testname
// It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
void PrintTo(const forest_param_t &param, ::std::ostream *os) { *os << param.test_name; }

// Positive tests with double and single type
TEST_P(forest_positive, Double) {
    const forest_param_t &param = GetParam();
    test_forest_positive<double>(param.data_name, param.iopts, param.sopts, param.dopts,
                                 (double)param.target_score);
}
TEST_P(forest_positive, Single) {
    const forest_param_t &param = GetParam();
    test_forest_positive<float>(param.data_name, param.iopts, param.sopts, param.fopts,
                                (float)param.target_score);
}

INSTANTIATE_TEST_SUITE_P(forest_pos_suite, forest_positive,
                         testing::ValuesIn(forest_param_pos));
