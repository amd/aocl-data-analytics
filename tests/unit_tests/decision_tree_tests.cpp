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

#include <list>
#include <random>

#include "gtest/gtest.h"

#include "utest_utils.hpp"

template <typename T> struct TestDataType {
    std::vector<T> x_train;
    std::vector<uint8_t> y;
    std::vector<T> x_test;
    std::vector<uint8_t> y_test;
    da_int n_obs_train, d;
    da_int n_obs_test;
};

template <typename T> class DecisionTreeTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DecisionTreeTest, FloatTypes);

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

TYPED_TEST(DecisionTreeTest, invalid_input) {

    std::vector<da_handle_type> handle_type_enum_vec = {da_handle_decision_tree,
                                                        da_handle_decision_forest};

    for (auto &handle_type_enum : handle_type_enum_vec) {
        da_status status;

        std::vector<TypeParam> x = {
            0.0,
        };
        std::vector<uint8_t> y = {
            0,
        };
        da_int n_obs = 0, d = 0;

        // Initialize the decision forest class and fit model
        da_handle df_handle = nullptr;
        status = da_handle_init<TypeParam>(&df_handle, handle_type_enum);
        EXPECT_EQ(status, da_status_success);

        // call set_training_data with invalid values
        TypeParam *x_invalid = nullptr;
        status = da_df_set_training_data<TypeParam>(df_handle, n_obs, d, x_invalid, n_obs,
                                                    y.data());
        EXPECT_EQ(status, da_status_invalid_input);

        status = da_df_set_training_data<TypeParam>(df_handle, n_obs, d, x.data(), n_obs,
                                                    y.data());
        EXPECT_EQ(status, da_status_invalid_input);

        n_obs = 1;
        d = 1;
        status = da_df_set_training_data<TypeParam>(df_handle, n_obs, d, x.data(),
                                                    n_obs - 1, y.data());
        EXPECT_EQ(status, da_status_invalid_input);

        da_handle_destroy(&df_handle);
    }
}

TYPED_TEST(DecisionTreeTest, get_results) {

    std::vector<da_handle_type> handle_type_enum_vec = {da_handle_decision_tree,
                                                        da_handle_decision_forest};

    for (auto &handle_type_enum : handle_type_enum_vec) {
        da_status status;

        std::vector<TypeParam> x = {
            0.0,
        };
        std::vector<uint8_t> y = {
            0,
        };
        da_int n_obs = 1, d = 1;

        // Initialize the decision forest class and fit model
        da_handle df_handle = nullptr;
        status = da_handle_init<TypeParam>(&df_handle, handle_type_enum);
        EXPECT_EQ(status, da_status_success);

        // run with random seed
        da_int seed_val = -1;
        EXPECT_EQ(da_options_set_int(df_handle, "seed", seed_val), da_status_success);

        EXPECT_EQ(da_df_set_training_data<TypeParam>(df_handle, n_obs, d, x.data(), n_obs,
                                                     y.data()),
                  da_status_success);

        EXPECT_EQ(da_df_fit<TypeParam>(df_handle), da_status_success);

        da_int rinfo_size = 3;
        std::vector<TypeParam> rinfo(rinfo_size);
        EXPECT_EQ(da_handle_get_result(df_handle, da_result::da_rinfo, &rinfo_size,
                                       rinfo.data()),
                  da_status_success);

        std::cout << "seed_val = " << (da_int)rinfo[0] << std::endl;
        std::cout << "n_obs    = " << (da_int)rinfo[1] << std::endl;
        std::cout << "d        = " << (da_int)rinfo[2] << std::endl;

        // run with the same seed as before
        seed_val = (da_int)rinfo[0];
        EXPECT_EQ(da_options_set_int(df_handle, "seed", seed_val), da_status_success);

        EXPECT_EQ(da_df_set_training_data<TypeParam>(df_handle, n_obs, d, x.data(), n_obs,
                                                     y.data()),
                  da_status_success);

        EXPECT_EQ(da_df_fit<TypeParam>(df_handle), da_status_success);

        EXPECT_EQ(da_handle_get_result(df_handle, da_result::da_rinfo, &rinfo_size,
                                       rinfo.data()),
                  da_status_success);
        EXPECT_EQ((da_int)rinfo[0], seed_val);
        EXPECT_EQ((da_int)rinfo[1], n_obs);
        EXPECT_EQ((da_int)rinfo[2], d);

        da_handle_destroy(&df_handle);
    }
}

template <typename T>
void test_decision_tree_correctness(TestDataType<T> &data, std::string score_str,
                                    da_handle &df_handle) {

    std::vector<T> x_train = data.x_train;
    std::vector<uint8_t> y = data.y;
    std::vector<T> x_test = data.x_test;
    std::vector<uint8_t> y_test = data.y_test;

    da_int n_obs_train = data.n_obs_train, d = data.d;
    da_int n_obs_test = data.n_obs_test;

    EXPECT_EQ(da_options_set_int(df_handle, "depth", 5), da_status_success);
    EXPECT_EQ(da_options_set_int(df_handle, "n_features_to_select", d),
              da_status_success);
    EXPECT_EQ(da_options_set_int(df_handle, "seed", 301), da_status_success);

    EXPECT_EQ(da_options_set_string(df_handle, "scoring function", score_str.data()),
              da_status_success);

    EXPECT_EQ(da_df_set_training_data<T>(df_handle, n_obs_train, d, x_train.data(),
                                         n_obs_train, y.data()),
              da_status_success);

    EXPECT_EQ(da_df_fit<T>(df_handle), da_status_success);

    T score = 0.0;
    EXPECT_EQ(da_df_score<T>(df_handle, n_obs_test, d, x_test.data(), n_obs_test,
                             y_test.data(), &score),
              da_status_success);

    std::cout << "score_str = " << score_str << ", score    = " << score << std::endl;

    // expect score to be 1.0
    EXPECT_EQ(score, 1.0);
}

TYPED_TEST(DecisionTreeTest, correctness0) {
    std::cout << "Test with (8x1) data" << std::endl;
    std::cout << "-----------------------" << std::endl;

    // test with scoring function where we expect score to be 1

    std::string score_str = "misclassification-error";
    // std::string score_str = "gini";
    TestDataType<TypeParam> data;
    set_test_data_8x1<TypeParam>(data);

    da_handle df_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&df_handle, da_handle_decision_tree),
              da_status_success);
    test_decision_tree_correctness<TypeParam>(data, score_str, df_handle);
    da_handle_destroy(&df_handle);
}

TYPED_TEST(DecisionTreeTest, correctness1) {
    std::cout << "Test with (8x2, unique) data" << std::endl;
    std::cout << "------------------------------" << std::endl;

    // test with scoring functions where we expect score to be 1
    std::vector<std::string> param_vec = {"gini", "cross-entropy"};

    for (auto &score_str : param_vec) {
        TestDataType<TypeParam> data_s;
        set_test_data_8x2_unique<TypeParam>(data_s);

        da_handle df_handle = nullptr;
        EXPECT_EQ(da_handle_init<TypeParam>(&df_handle, da_handle_decision_tree),
                  da_status_success);
        test_decision_tree_correctness<TypeParam>(data_s, score_str, df_handle);
        da_handle_destroy(&df_handle);
    }
}

TYPED_TEST(DecisionTreeTest, correctness2) {
    std::cout << "Test with (8x2, non-unique) data" << std::endl;
    std::cout << "------------------------------" << std::endl;

    // test with scoring functions where we expect score to be 1
    std::vector<std::string> param_vec = {"gini", "cross-entropy"};

    for (auto &score_str : param_vec) {
        // training data values are not unique
        TestDataType<TypeParam> data_s;
        set_test_data_8x2_nonunique<TypeParam>(data_s);

        da_handle df_handle = nullptr;
        EXPECT_EQ(da_handle_init<TypeParam>(&df_handle, da_handle_decision_tree),
                  da_status_success);
        test_decision_tree_correctness<TypeParam>(data_s, score_str, df_handle);
        da_handle_destroy(&df_handle);
    }
}

TYPED_TEST(DecisionTreeTest, reuse_handle) {
    da_handle df_handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&df_handle, da_handle_decision_tree),
              da_status_success);

    std::cout << "Test with (8x1) data" << std::endl;
    std::cout << "-----------------------" << std::endl;

    // test with scoring function where we expect score to be 1

    std::string score_str = "misclassification-error";
    // std::string score_str = "gini";
    TestDataType<TypeParam> data;
    set_test_data_8x1<TypeParam>(data);
    test_decision_tree_correctness<TypeParam>(data, score_str, df_handle);

    std::cout << "Test with (8x2, unique) data" << std::endl;
    std::cout << "------------------------------" << std::endl;

    // test with scoring functions where we expect score to be 1
    std::vector<std::string> param_vec = {"gini", "cross-entropy"};

    for (auto &score_str : param_vec) {
        TestDataType<TypeParam> data_s;
        set_test_data_8x2_unique<TypeParam>(data_s);
        test_decision_tree_correctness<TypeParam>(data_s, score_str, df_handle);
    }

    std::cout << "Test with (8x2, non-unique) data" << std::endl;
    std::cout << "------------------------------" << std::endl;

    for (auto &score_str : param_vec) {
        // training data values are not unique
        TestDataType<TypeParam> data_s;
        set_test_data_8x2_nonunique<TypeParam>(data_s);
        test_decision_tree_correctness<TypeParam>(data_s, score_str, df_handle);
    }
    da_handle_destroy(&df_handle);
}

TYPED_TEST(DecisionTreeTest, bad_handle) {
    da_status status;

    std::vector<TypeParam> x = {
        0.0,
    };
    std::vector<uint8_t> y = {
        0,
    };
    da_int n_obs = 0, d = 0;
    TypeParam score = 0.0;

    // handle not initialized
    da_handle df_handle = nullptr;
    status = da_df_set_training_data<TypeParam>(df_handle, n_obs, d, x.data(), n_obs,
                                                y.data());
    EXPECT_EQ(status, da_status_handle_not_initialized);

    status = da_df_fit<TypeParam>(df_handle);
    EXPECT_EQ(status, da_status_handle_not_initialized);

    status = da_df_predict<TypeParam>(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_handle_not_initialized);

    status =
        da_df_score<TypeParam>(df_handle, n_obs, d, x.data(), n_obs, y.data(), &score);
    EXPECT_EQ(status, da_status_handle_not_initialized);

    // incorrect handle type
    EXPECT_EQ(da_handle_init<TypeParam>(&df_handle, da_handle_linmod), da_status_success);
    status = da_df_set_training_data<TypeParam>(df_handle, n_obs, d, x.data(), n_obs,
                                                y.data());
    EXPECT_EQ(status, da_status_invalid_handle_type);

    status = da_df_fit<TypeParam>(df_handle);
    EXPECT_EQ(status, da_status_invalid_handle_type);

    status = da_df_predict<TypeParam>(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_invalid_handle_type);

    status =
        da_df_score<TypeParam>(df_handle, n_obs, d, x.data(), n_obs, y.data(), &score);
    EXPECT_EQ(status, da_status_invalid_handle_type);

    da_handle_destroy(&df_handle);
}

TEST(DecisionTreeTest, incorrect_handle_precision) {
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
    status = da_df_set_training_data_s(handle_d, n_obs, d, x_s.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_wrong_type);
    status = da_df_set_training_data_d(handle_s, n_obs, d, x_d.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_wrong_type);

    status = da_df_fit_s(handle_d);
    EXPECT_EQ(status, da_status_wrong_type);
    status = da_df_fit_d(handle_s);
    EXPECT_EQ(status, da_status_wrong_type);

    status = da_df_predict_s(handle_d, n_obs, d, x_s.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_wrong_type);
    status = da_df_predict_d(handle_s, n_obs, d, x_d.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_wrong_type);

    status = da_df_score_s(handle_d, n_obs, d, x_s.data(), n_obs, y.data(), &score_s);
    EXPECT_EQ(status, da_status_wrong_type);
    status = da_df_score_d(handle_s, n_obs, d, x_d.data(), n_obs, y.data(), &score_d);
    EXPECT_EQ(status, da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

TYPED_TEST(DecisionTreeTest, invalid_array_dim) {

    std::vector<da_handle_type> handle_type_enum_vec = {da_handle_decision_tree,
                                                        da_handle_decision_forest};

    for (auto &handle_type_enum : handle_type_enum_vec) {
        da_status status;

        std::vector<TypeParam> x = {
            0.0,
        };
        std::vector<uint8_t> y = {
            0,
        };
        da_int n_obs = 1, d = 1;

        // Initialize the decision forest class and fit model
        da_handle df_handle = nullptr;
        status = da_handle_init<TypeParam>(&df_handle, handle_type_enum);
        EXPECT_EQ(status, da_status_success);

        // run with random seed
        da_int seed_val = -1;
        EXPECT_EQ(da_options_set_int(df_handle, "seed", seed_val), da_status_success);

        EXPECT_EQ(da_df_set_training_data<TypeParam>(df_handle, n_obs, d, x.data(), n_obs,
                                                     y.data()),
                  da_status_success);

        EXPECT_EQ(da_df_fit<TypeParam>(df_handle), da_status_success);

        da_int rinfo_size = 2;
        std::vector<TypeParam> rinfo(rinfo_size);
        EXPECT_EQ(da_handle_get_result(df_handle, da_result::da_rinfo, &rinfo_size,
                                       rinfo.data()),
                  da_status_invalid_array_dimension);

        rinfo_size = 0;
        rinfo.resize(rinfo_size);
        EXPECT_EQ(da_handle_get_result(df_handle, da_result::da_rinfo, &rinfo_size,
                                       rinfo.data()),
                  da_status_invalid_array_dimension);

        rinfo_size = 3;
        rinfo.resize(rinfo_size);
        EXPECT_EQ(da_handle_get_result(df_handle, da_result::da_linmod_coeff, &rinfo_size,
                                       rinfo.data()),
                  da_status_unknown_query);

        std::vector<da_int> iinfo(rinfo_size);
        EXPECT_EQ(da_handle_get_result(df_handle, da_result::da_linmod_coeff, &rinfo_size,
                                       iinfo.data()),
                  da_status_unknown_query);

        da_handle_destroy(&df_handle);
    }
}

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

template <typename T> struct DFParamType {
    std::string test_name;
    std::string score_criteria;
    da_handle_type handle_type_enum;
    T expected_score;
};

template <typename T> void GetExampleData(std::vector<DFParamType<T>> &params) {
    DFParamType<T> param;

    param.handle_type_enum = da_handle_decision_tree;
    param.test_name = "decision tree with gini scoring criteria";
    param.score_criteria = "gini";
    param.expected_score = 0.93250;
    params.push_back(param);

    param.handle_type_enum = da_handle_decision_tree;
    param.test_name = "decision tree with cross-entropy scoring criteria";
    param.score_criteria = "cross-entropy";
    param.expected_score = 0.92750;
    params.push_back(param);

    param.handle_type_enum = da_handle_decision_tree;
    param.test_name = "decision tree with misclassification-error scoring criteria";
    param.score_criteria = "misclassification-error";
    param.expected_score = 0.93250;
    params.push_back(param);

    param.handle_type_enum = da_handle_decision_forest;
    param.test_name = "decision forest with gini scoring criteria";
    param.score_criteria = "gini";
    param.expected_score = 0.93250;
    params.push_back(param);

    param.handle_type_enum = da_handle_decision_forest;
    param.test_name = "decision forest with cross-entropy scoring criteria";
    param.score_criteria = "cross-entropy";
    param.expected_score = 0.94250;
    params.push_back(param);

    param.handle_type_enum = da_handle_decision_forest;
    param.test_name = "decision forest with misclassification-error scoring criteria";
    param.score_criteria = "misclassification-error";
    param.expected_score = 0.93750;
    params.push_back(param);
}

TYPED_TEST(DecisionTreeTest, decision_tree_ex) {

    std::vector<DFParamType<TypeParam>> params;
    GetExampleData(params);
    da_int count = 0;

    for (auto &param : params) {
        count++;
        std::cout << "Test " << std::to_string(count) << ": " << param.test_name
                  << std::endl;

        std::string score_criteria = param.score_criteria;
        TypeParam expected_score = param.expected_score;
        da_handle_type handle_type_enum = param.handle_type_enum;

        da_handle df_handle;
        da_datastore csv_handle;
        da_status status;

        // Read in training data
        csv_handle = nullptr;
        status = da_datastore_init(&csv_handle);
        EXPECT_EQ(status, da_status_success);

        char features_fp[256] = DATA_DIR;
        strcat(features_fp, "/df_data/");
        strcat(features_fp, "training_features");
        strcat(features_fp, ".csv");

        char labels_fp[256] = DATA_DIR;
        strcat(labels_fp, "/df_data/");
        strcat(labels_fp, "training_labels");
        strcat(labels_fp, ".csv");

        TypeParam *x = nullptr;
        uint8_t *y = nullptr;
        da_int n_obs = 0, d = 0, nrows_y = 0, ncols_y = 0;
        status = da_read_csv(csv_handle, features_fp, &x, &n_obs, &d, nullptr);
        EXPECT_EQ(status, da_status_success);
        status =
            da_read_csv_uint8(csv_handle, labels_fp, &y, &nrows_y, &ncols_y, nullptr);
        EXPECT_EQ(status, da_status_success);

        // Initialize the decision tree class and fit model
        df_handle = nullptr;
        status = da_handle_init<TypeParam>(&df_handle, handle_type_enum);

        if (handle_type_enum == da_handle_decision_tree) {
            status = da_options_set_int(df_handle, "depth", 5);
            EXPECT_EQ(status, da_status_success);
            status = da_options_set_int(df_handle, "seed", 77);
            EXPECT_EQ(status, da_status_success);
            status = da_options_set_int(df_handle, "n_features_to_select", d);
            EXPECT_EQ(status, da_status_success);
        }

        if (handle_type_enum == da_handle_decision_forest) {
            status = da_options_set_int(df_handle, "seed", 988);
            EXPECT_EQ(status, da_status_success);
            status = da_options_set_int(df_handle, "n_obs_per_tree", 100);
            EXPECT_EQ(status, da_status_success);
            status = da_options_set_int(df_handle, "n_features_to_select", 3);
            EXPECT_EQ(status, da_status_success);
            status = da_options_set_int(df_handle, "n_trees", 20);
            EXPECT_EQ(status, da_status_success);
        }

        status =
            da_options_set_string(df_handle, "scoring function", score_criteria.data());
        EXPECT_EQ(status, da_status_success);

        status = da_df_set_training_data<TypeParam>(df_handle, n_obs, d, x, n_obs, y);
        EXPECT_EQ(status, da_status_success);

        status = da_df_fit<TypeParam>(df_handle);
        EXPECT_EQ(status, da_status_success);

        // Read in data for making predictions
        char test_features_fp[256] = DATA_DIR;
        strcat(test_features_fp, "/df_data/");
        strcat(test_features_fp, "test_features");
        strcat(test_features_fp, ".csv");

        char test_labels_fp[256] = DATA_DIR;
        strcat(test_labels_fp, "/df_data/");
        strcat(test_labels_fp, "test_labels");
        strcat(test_labels_fp, ".csv");

        TypeParam *x_test = nullptr;
        uint8_t *y_test = nullptr;
        n_obs = 0;
        d = 0;
        nrows_y = 0;
        ncols_y = 0;

        status = da_read_csv(csv_handle, test_features_fp, &x_test, &n_obs, &d, nullptr);
        EXPECT_EQ(status, da_status_success);
        status = da_read_csv_uint8(csv_handle, test_labels_fp, &y_test, &nrows_y,
                                   &ncols_y, nullptr);
        EXPECT_EQ(status, da_status_success);

        // Make predictions with model and evaluate score
        std::vector<uint8_t> y_pred(n_obs);
        status =
            da_df_predict<TypeParam>(df_handle, n_obs, d, x_test, n_obs, y_pred.data());
        EXPECT_EQ(status, da_status_success);
        TypeParam score = 0.0;
        status =
            da_df_score<TypeParam>(df_handle, n_obs, d, x_test, n_obs, y_test, &score);
        EXPECT_EQ(status, da_status_success);

        EXPECT_NEAR(score, expected_score, 1e-6);

        if (x_test)
            free(x_test);

        if (y_test)
            free(y_test);

        if (x)
            free(x);

        if (y)
            free(y);

        da_datastore_destroy(&csv_handle);
        da_handle_destroy(&df_handle);
    }
}

TYPED_TEST(DecisionTreeTest, errors) {
    std::vector<da_handle_type> handle_type_enum_vec = {da_handle_decision_tree,
                                                        da_handle_decision_forest};

    for (auto &handle_type_enum : handle_type_enum_vec) {
        da_status status;

        std::vector<TypeParam> x = {
            0.0,
        };
        std::vector<uint8_t> y = {
            0,
        };
        da_int n_obs = 1, d = 1;

        // Initialize the decision forest class
        da_handle df_handle = nullptr;
        status = da_handle_init<TypeParam>(&df_handle, handle_type_enum);
        EXPECT_EQ(status, da_status_success);

        // call fit before set_training_data
        EXPECT_EQ(da_df_fit<TypeParam>(df_handle), da_status_no_data);

        status = da_df_set_training_data<TypeParam>(df_handle, n_obs, d, x.data(), n_obs,
                                                    y.data());
        EXPECT_EQ(status, da_status_success);

        // call predict before fit
        std::vector<uint8_t> y_pred(n_obs);
        status =
            da_df_predict<TypeParam>(df_handle, n_obs, d, x.data(), n_obs, y_pred.data());
        EXPECT_EQ(status, da_status_out_of_date);

        // call score before fit
        TypeParam score = 0.0;
        status = da_df_score<TypeParam>(df_handle, n_obs, d, x.data(), n_obs, y.data(),
                                        &score);
        EXPECT_EQ(status, da_status_out_of_date);

        status = da_df_fit<TypeParam>(df_handle);
        EXPECT_EQ(status, da_status_success);

        // call predict with invalid inputs
        TypeParam *x_invalid = nullptr;
        status = da_df_predict<TypeParam>(df_handle, n_obs, d, x_invalid, n_obs - 1,
                                          y_pred.data());
        EXPECT_EQ(status, da_status_invalid_input);

        status = da_df_predict<TypeParam>(df_handle, 0, d, x.data(), 0, y_pred.data());
        EXPECT_EQ(status, da_status_invalid_input);

        status =
            da_df_predict<TypeParam>(df_handle, n_obs, 0, x.data(), n_obs, y_pred.data());
        EXPECT_EQ(status, da_status_invalid_input);

        status = da_df_predict<TypeParam>(df_handle, n_obs, d, x.data(), n_obs - 1,
                                          y_pred.data());
        EXPECT_EQ(status, da_status_invalid_input);

        // call score with invalid inputs
        status = da_df_score<TypeParam>(df_handle, n_obs, d, x_invalid, n_obs - 1,
                                        y_pred.data(), &score);
        EXPECT_EQ(status, da_status_invalid_input);

        status =
            da_df_score<TypeParam>(df_handle, 0, d, x.data(), 0, y_pred.data(), &score);
        EXPECT_EQ(status, da_status_invalid_input);

        status = da_df_score<TypeParam>(df_handle, n_obs, 0, x.data(), n_obs,
                                        y_pred.data(), &score);
        EXPECT_EQ(status, da_status_invalid_input);

        status = da_df_score<TypeParam>(df_handle, n_obs, d, x.data(), n_obs - 1,
                                        y_pred.data(), &score);
        EXPECT_EQ(status, da_status_invalid_input);

        da_handle_destroy(&df_handle);
    }
}

TYPED_TEST(DecisionTreeTest, illegal_input1) {
    da_status status;
    da_handle df_handle = nullptr;

    TypeParam x[1] = {0.0};
    uint8_t y[1] = {0};
    da_int n_obs = 1, n_features = 1, ldx = 1;
    status = da_handle_init<TypeParam>(&df_handle, da_handle_decision_tree);
    EXPECT_EQ(status, da_status_success);
    status = da_df_set_training_data<TypeParam>(df_handle, n_obs, n_features, x, ldx, y);
    EXPECT_EQ(status, da_status_success);
    status = da_options_set_int(df_handle, "depth", 5);
    EXPECT_EQ(status, da_status_success);
    status = da_options_set_int(df_handle, "seed", 77);
    EXPECT_EQ(status, da_status_success);
    status = da_options_set_int(df_handle, "n_features_to_select", 10);
    EXPECT_EQ(status, da_status_success);
    status = da_options_set_string(df_handle, "scoring function", "gini");
    EXPECT_EQ(status, da_status_success);

    // n_features_to_select > n_features
    status = da_df_fit<TypeParam>(df_handle);
    EXPECT_EQ(status, da_status_invalid_input);

    da_handle_destroy(&df_handle);
}

TYPED_TEST(DecisionTreeTest, zero_input) {
    da_status status;
    da_handle df_handle = nullptr;

    TypeParam x[1] = {0.0};
    uint8_t y[1] = {1};
    da_int n_obs = 1, n_features = 1, ldx = 1;
    status = da_handle_init<TypeParam>(&df_handle, da_handle_decision_tree);
    EXPECT_EQ(status, da_status_success);
    status = da_df_set_training_data<TypeParam>(df_handle, n_obs, n_features, x, ldx, y);
    EXPECT_EQ(status, da_status_success);
    status = da_options_set_int(df_handle, "depth", 5);
    EXPECT_EQ(status, da_status_success);
    status = da_options_set_int(df_handle, "seed", 77);
    EXPECT_EQ(status, da_status_success);
    status = da_options_set_int(df_handle, "n_features_to_select", n_features);
    EXPECT_EQ(status, da_status_success);
    status = da_options_set_string(df_handle, "scoring function", "gini");
    EXPECT_EQ(status, da_status_success);

    status = da_df_fit<TypeParam>(df_handle);
    EXPECT_EQ(status, da_status_success);

    TypeParam x_test[1] = {1.0};
    uint8_t y_pred[1];

    status =
        da_df_predict<TypeParam>(df_handle, n_obs, n_features, x_test, n_obs, y_pred);
    EXPECT_EQ(status, da_status_success);

    EXPECT_EQ(y_pred[0], 1);

    da_handle_destroy(&df_handle);
}

TYPED_TEST(DecisionTreeTest, identical_x) {
    da_status status;
    da_handle handle = nullptr;

    TypeParam x[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    uint8_t y[3] = {1, 0, 1};
    da_int n_obs = 3, n_features = 2, ldx = 3;
    status = da_handle_init<TypeParam>(&handle, da_handle_decision_tree);
    status = da_df_set_training_data<TypeParam>(handle, n_obs, n_features, x, ldx, y);
    status = da_options_set_int(handle, "depth", 10);
    status = da_options_set_int(handle, "seed", 1);
    status = da_options_set_int(handle, "n_features_to_select", 2);
    status = da_options_set_string(handle, "scoring function", "gini");
    status = da_df_fit<TypeParam>(handle);

    n_obs = 2;
    TypeParam x_test[4] = {2.0, 3.0, -2.0, -2.5};
    uint8_t y_pred[2];

    status = da_df_predict<TypeParam>(handle, n_obs, n_features, x_test, n_obs, y_pred);
    EXPECT_EQ(status, da_status_success);

    EXPECT_EQ(y_pred[0], 1);
    EXPECT_EQ(y_pred[1], 1);

    da_handle_destroy(&handle);
}
