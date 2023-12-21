/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "aoclda.h"

#include <random>

#include "gtest/gtest.h"

#include "utest_utils.hpp"

template <typename T>
da_status da_df_tree_set_training_data(da_handle handle, da_int n_obs, da_int n_features,
                                       T *x, da_int ldx, uint8_t *y);
template <>
da_status da_df_tree_set_training_data<double>(da_handle handle, da_int n_obs,
                                               da_int n_features, double *x, da_int ldx,
                                               uint8_t *y) {
    return da_df_tree_set_training_data_d(handle, n_obs, n_features, x, ldx, y);
}

template <>
da_status da_df_tree_set_training_data<float>(da_handle handle, da_int n_obs,
                                              da_int n_features, float *x, da_int ldx,
                                              uint8_t *y) {
    return da_df_tree_set_training_data_s(handle, n_obs, n_features, x, ldx, y);
}

template <typename T> da_status da_df_tree_fit(da_handle handle);

template <> da_status da_df_tree_fit<float>(da_handle handle) {
    return da_df_tree_fit_s(handle);
}

template <> da_status da_df_tree_fit<double>(da_handle handle) {
    return da_df_tree_fit_d(handle);
}

template <typename T>
da_status da_df_tree_score(da_handle handle, da_int n_obs, T *x, da_int ldx,
                           uint8_t *y_test, T *score);

template <>
da_status da_df_tree_score<double>(da_handle handle, da_int n_obs, double *x, da_int ldx,
                                   uint8_t *y_test, double *score) {
    return da_df_tree_score_d(handle, n_obs, x, ldx, y_test, score);
}

template <>
da_status da_df_tree_score<float>(da_handle handle, da_int n_obs, float *x, da_int ldx,
                                  uint8_t *y_test, float *score) {
    return da_df_tree_score_s(handle, n_obs, x, ldx, y_test, score);
}

template <typename T>
da_status da_df_tree_predict(da_handle handle, da_int n_obs, T *x, da_int ldx,
                             uint8_t *y_pred);

template <>
da_status da_df_tree_predict<double>(da_handle handle, da_int n_obs, double *x,
                                     da_int ldx, uint8_t *y_pred) {
    return da_df_tree_predict_d(handle, n_obs, x, ldx, y_pred);
}

template <>
da_status da_df_tree_predict<float>(da_handle handle, da_int n_obs, float *x, da_int ldx,
                                    uint8_t *y_pred) {
    return da_df_tree_predict_s(handle, n_obs, x, ldx, y_pred);
}

template <typename T> struct TestDataType {
    std::vector<T> x_train;
    std::vector<uint8_t> y;
    std::vector<T> x_test;
    std::vector<uint8_t> y_test;
    da_int n_obs_train, d;
    da_int n_obs_test;
};

template <typename T> void set_test_data_8x1(TestDataType<T> &data) {

    // idea is that y = 1 with prob 0.75 when x < 0.5
    // and          y = 1 with prob 0.25 with x > 0.5
    data.x_train = {
        (T)0.1, (T)0.2, (T)0.3, (T)0.4,
        (T)0.6, (T)0.7, (T)0.8, (T)0.9, // first column of data
    };

    data.y = {
        0, 1, 0, 0, 1, 1, 0, 1 // labels
    };

    data.x_test = {(T)0.25, (T)0.75};

    data.y_test = {
        0, 1 // labels
    };

    data.n_obs_train = 8, data.d = 1;
    data.n_obs_test = 2;
}

template <typename T> void set_test_data_8x2_unique(TestDataType<T> &data) {

    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    // training data values are unique
    data.x_train = {
        (T)0.12, (T)0.11, (T)0.42, (T)0.41,
        (T)0.62, (T)0.61, (T)0.92, (T)0.91, // first column of data
        (T)0.39, (T)0.79, (T)0.38, (T)0.78,
        (T)0.37, (T)0.77, (T)0.36, (T)0.76 // second column of data
    };

    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    data.y = {
        0, 1, 0, 1, 1, 1, 1, 1 // labels
    };

    data.x_test = {
        (T)0.25, (T)0.25, (T)0.75, (T)0.75, // first column of data
        (T)0.25, (T)0.75, (T)0.25, (T)0.75  // second column of data
    };

    // idea is that if fit is correct we should be able to predict these labels
    // with 100% accuracy
    data.y_test = {
        0, 1, 1, 1 // labels
    };

    data.n_obs_train = 8, data.d = 2;
    data.n_obs_test = 4;
}

template <typename T> void set_test_data_8x2_nonunique(TestDataType<T> &data) {
    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    // training data values are not unique
    data.x_train = {
        (T)0.1, (T)0.1, (T)0.4, (T)0.4,
        (T)0.6, (T)0.6, (T)0.9, (T)0.9, // first column of data
        (T)0.3, (T)0.7, (T)0.3, (T)0.7,
        (T)0.3, (T)0.7, (T)0.3, (T)0.7 // second column of data
    };

    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    data.y = {
        0, 1, 0, 1, 1, 1, 1, 1 // labels
    };

    data.x_test = {
        (T)0.25, (T)0.25, (T)0.75, (T)0.75, // first column of data
        (T)0.25, (T)0.75, (T)0.25, (T)0.75  // second column of data
    };

    // idea is that if fit is correct we should be able to predict these labels
    // with 100% accuracy
    data.y_test = {
        0, 1, 1, 1 // labels
    };

    data.n_obs_train = 8, data.d = 2;
    data.n_obs_test = 4;
}

template <typename T> void test_decision_tree_invalid_input() {
    da_status status;

    std::vector<T> x = {
        0.0,
    };
    std::vector<uint8_t> y = {
        0,
    };
    da_int n_obs = 0, d = 0;

    // Initialize the decision forest class and fit model
    da_handle df_handle = nullptr;
    status = da_handle_init<T>(&df_handle, da_handle_decision_tree);
    EXPECT_EQ(status, da_status_success);

    // call set_training_data with invalid values
    T *x_invalid = nullptr;
    status =
        da_df_tree_set_training_data<T>(df_handle, n_obs, d, x_invalid, n_obs, y.data());
    EXPECT_EQ(status, da_status_invalid_input);

    status =
        da_df_tree_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_invalid_input);

    n_obs = 1;
    d = 1;
    status = da_df_tree_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs - 1,
                                             y.data());
    EXPECT_EQ(status, da_status_invalid_input);

    da_handle_destroy(&df_handle);
}

template <typename T> void test_decision_tree_get_results() {
    da_status status;

    std::vector<T> x = {
        0.0,
    };
    std::vector<uint8_t> y = {
        0,
    };
    da_int n_obs = 1, d = 1;

    // Initialize the decision forest class and fit model
    da_handle df_handle = nullptr;
    status = da_handle_init<T>(&df_handle, da_handle_decision_tree);
    EXPECT_EQ(status, da_status_success);

    // run with random seed
    da_int seed_val = -1;
    EXPECT_EQ(da_options_set_int(df_handle, "seed", seed_val), da_status_success);

    EXPECT_EQ(
        da_df_tree_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data()),
        da_status_success);

    EXPECT_EQ(da_df_tree_fit<T>(df_handle), da_status_success);

    da_int rinfo_size = 3;
    std::vector<T> rinfo(rinfo_size);
    EXPECT_EQ(
        da_handle_get_result(df_handle, da_result::da_rinfo, &rinfo_size, rinfo.data()),
        da_status_success);

    std::cout << "seed_val = " << (da_int)rinfo[0] << std::endl;
    std::cout << "n_obs    = " << (da_int)rinfo[1] << std::endl;
    std::cout << "d        = " << (da_int)rinfo[2] << std::endl;

    // run with the same seed as before
    seed_val = (da_int)rinfo[0];
    EXPECT_EQ(da_options_set_int(df_handle, "seed", seed_val), da_status_success);

    EXPECT_EQ(
        da_df_tree_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data()),
        da_status_success);

    EXPECT_EQ(da_df_tree_fit<T>(df_handle), da_status_success);

    EXPECT_EQ(
        da_handle_get_result(df_handle, da_result::da_rinfo, &rinfo_size, rinfo.data()),
        da_status_success);
    EXPECT_EQ((da_int)rinfo[0], seed_val);
    EXPECT_EQ((da_int)rinfo[1], n_obs);
    EXPECT_EQ((da_int)rinfo[2], d);

    da_handle_destroy(&df_handle);
}

template <typename T>
void test_decision_tree_correctness(TestDataType<T> &data, std::string score_str) {

    std::vector<T> x_train = data.x_train;
    std::vector<uint8_t> y = data.y;
    std::vector<T> x_test = data.x_test;
    std::vector<uint8_t> y_test = data.y_test;

    da_int n_obs_train = data.n_obs_train, d = data.d;
    da_int n_obs_test = data.n_obs_test;

    // Initialize the decision tree class and fit model
    da_handle df_handle = nullptr;

    // status = da_handle_init_s(&df_handle, da_handle_decision_tree);
    EXPECT_EQ(da_handle_init<T>(&df_handle, da_handle_decision_tree), da_status_success);

    EXPECT_EQ(da_options_set_int(df_handle, "depth", 5), da_status_success);
    EXPECT_EQ(da_options_set_int(df_handle, "n_features_to_select", d),
              da_status_success);
    EXPECT_EQ(da_options_set_int(df_handle, "seed", 301), da_status_success);

    EXPECT_EQ(da_options_set_string(df_handle, "scoring function", score_str.data()),
              da_status_success);

    // status = da_options_set_real_s(df_handle, "diff_thres", 1.0);

    // status = da_df_tree_set_training_data_s(df_handle, n_obs_train, d, x_train.data(), n_obs_train, y.data());
    EXPECT_EQ(da_df_tree_set_training_data<T>(df_handle, n_obs_train, d, x_train.data(),
                                              n_obs_train, y.data()),
              da_status_success);

    // status = da_df_tree_fit_s(df_handle);
    EXPECT_EQ(da_df_tree_fit<T>(df_handle), da_status_success);

    T score = 0.0;
    // status = da_df_tree_score_s(df_handle, n_obs_test, x_test.data(), n_obs_test, y_test.data(), &score);
    EXPECT_EQ(da_df_tree_score<T>(df_handle, n_obs_test, x_test.data(), n_obs_test,
                                  y_test.data(), &score),
              da_status_success);

    std::cout << "score_str = " << score_str << ", score    = " << score << std::endl;

    // expect score to be 1.0
    EXPECT_EQ(score, 1.0);

    da_handle_destroy(&df_handle);
}

TEST(decision_tree, correctness0) {
    std::cout << "Test with (8x1) data" << std::endl;
    std::cout << "-----------------------" << std::endl;

    // test with scoring function where we expect score to be 1

    std::string score_str = "misclassification-error";
    // std::string score_str = "gini";
    TestDataType<float> data_s;
    set_test_data_8x1<float>(data_s);
    test_decision_tree_correctness<float>(data_s, score_str);

    TestDataType<double> data_d;
    set_test_data_8x1<double>(data_d);
    test_decision_tree_correctness<double>(data_d, score_str);
}

TEST(decision_tree, correctness1) {
    std::cout << "Test with (8x2, unique) data" << std::endl;
    std::cout << "------------------------------" << std::endl;

    // test with scoring functions where we expect score to be 1
    std::vector<std::string> param_vec = {"gini", "cross-entropy"};

    for (auto &score_str : param_vec) {
        TestDataType<float> data_s;
        set_test_data_8x2_unique<float>(data_s);
        test_decision_tree_correctness<float>(data_s, score_str);

        TestDataType<double> data_d;
        set_test_data_8x2_unique<double>(data_d);
        test_decision_tree_correctness<double>(data_d, score_str);
    }
}

TEST(decision_tree, correctness2) {
    std::cout << "Test with (8x2, non-unique) data" << std::endl;
    std::cout << "------------------------------" << std::endl;

    // test with scoring functions where we expect score to be 1
    std::vector<std::string> param_vec = {"gini", "cross-entropy"};

    for (auto &score_str : param_vec) {
        // training data values are not unique
        TestDataType<float> data_s;
        set_test_data_8x2_nonunique<float>(data_s);
        test_decision_tree_correctness<float>(data_s, score_str);

        TestDataType<double> data_d;
        set_test_data_8x2_nonunique<double>(data_d);
        test_decision_tree_correctness<double>(data_d, score_str);
    }
}

TEST(decision_tree, invalid_input) {
    test_decision_tree_invalid_input<float>();
    test_decision_tree_invalid_input<double>();
}

TEST(decision_tree, get_results) {
    test_decision_tree_get_results<float>();
    test_decision_tree_get_results<double>();
}

template <typename T> void test_decision_tree_bad_handle() {
    da_status status;

    std::vector<T> x = {
        0.0,
    };
    std::vector<uint8_t> y = {
        0,
    };
    da_int n_obs = 0, d = 0;
    T score = 0.0;

    // handle not initialized
    da_handle df_handle = nullptr;
    status =
        da_df_tree_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_handle_not_initialized);

    status = da_df_tree_fit<T>(df_handle);
    EXPECT_EQ(status, da_status_handle_not_initialized);

    status = da_df_tree_predict<T>(df_handle, n_obs, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_handle_not_initialized);

    status = da_df_tree_score<T>(df_handle, n_obs, x.data(), n_obs, y.data(), &score);
    EXPECT_EQ(status, da_status_handle_not_initialized);

    // incorrect handle type
    EXPECT_EQ(da_handle_init<T>(&df_handle, da_handle_linmod), da_status_success);
    status =
        da_df_tree_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_invalid_handle_type);

    status = da_df_tree_fit<T>(df_handle);
    EXPECT_EQ(status, da_status_invalid_handle_type);

    status = da_df_tree_predict<T>(df_handle, n_obs, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_invalid_handle_type);

    status = da_df_tree_score<T>(df_handle, n_obs, x.data(), n_obs, y.data(), &score);
    EXPECT_EQ(status, da_status_invalid_handle_type);

    da_handle_destroy(&df_handle);
}

TEST(decision_tree, bad_handle) {
    test_decision_tree_bad_handle<float>();
    test_decision_tree_bad_handle<double>();
}

TEST(decision_tree, incorrect_handle_precision) {
    da_status status;

    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_decision_tree), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_decision_tree), da_status_success);

    std::vector<uint8_t> y = {
        0,
    };
    da_int n_obs = 0, d = 0;

    std::vector<double> x_d = {
        0.0,
    };
    double score_d = 0.0;

    std::vector<float> x_s = {
        0.0,
    };
    float score_s = 0.0;

    // incorrect handle precision
    status =
        da_df_tree_set_training_data_s(handle_d, n_obs, d, x_s.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_wrong_type);
    status =
        da_df_tree_set_training_data_d(handle_s, n_obs, d, x_d.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_wrong_type);

    status = da_df_tree_fit_s(handle_d);
    EXPECT_EQ(status, da_status_wrong_type);
    status = da_df_tree_fit_d(handle_s);
    EXPECT_EQ(status, da_status_wrong_type);

    status = da_df_tree_predict_s(handle_d, n_obs, x_s.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_wrong_type);
    status = da_df_tree_predict_d(handle_s, n_obs, x_d.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_wrong_type);

    status = da_df_tree_score_s(handle_d, n_obs, x_s.data(), n_obs, y.data(), &score_s);
    EXPECT_EQ(status, da_status_wrong_type);
    status = da_df_tree_score_d(handle_s, n_obs, x_d.data(), n_obs, y.data(), &score_d);
    EXPECT_EQ(status, da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

template <typename T> void test_decision_tree_invalid_array_dim() {
    da_status status;

    std::vector<T> x = {
        0.0,
    };
    std::vector<uint8_t> y = {
        0,
    };
    da_int n_obs = 1, d = 1;

    // Initialize the decision forest class and fit model
    da_handle df_handle = nullptr;
    status = da_handle_init<T>(&df_handle, da_handle_decision_tree);
    EXPECT_EQ(status, da_status_success);

    // run with random seed
    da_int seed_val = -1;
    EXPECT_EQ(da_options_set_int(df_handle, "seed", seed_val), da_status_success);

    EXPECT_EQ(
        da_df_tree_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data()),
        da_status_success);

    EXPECT_EQ(da_df_tree_fit<T>(df_handle), da_status_success);

    da_int rinfo_size = 2;
    std::vector<T> rinfo(rinfo_size);
    EXPECT_EQ(
        da_handle_get_result(df_handle, da_result::da_rinfo, &rinfo_size, rinfo.data()),
        da_status_invalid_array_dimension);

    rinfo_size = 0;
    rinfo.resize(rinfo_size);
    EXPECT_EQ(
        da_handle_get_result(df_handle, da_result::da_rinfo, &rinfo_size, rinfo.data()),
        da_status_invalid_array_dimension);

    da_handle_destroy(&df_handle);
}

TEST(decision_tree, invalid_array_dim) {
    test_decision_tree_invalid_array_dim<float>();
    test_decision_tree_invalid_array_dim<double>();
}
