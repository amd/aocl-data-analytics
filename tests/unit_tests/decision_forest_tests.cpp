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

#include "aoclda.h"

#include <list>
#include <random>

#include "gtest/gtest.h"

#include "utest_utils.hpp"

template <typename T> class DecisionForestTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DecisionForestTest, FloatTypes);

template <typename T> void test_decision_forest_invalid_input() {
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
    status = da_handle_init<T>(&df_handle, da_handle_decision_forest);
    EXPECT_EQ(status, da_status_success);

    // call set_training_data with invalid values
    T *x_invalid = nullptr;
    status = da_df_set_training_data<T>(df_handle, n_obs, d, x_invalid, n_obs, y.data());
    EXPECT_EQ(status, da_status_invalid_input);

    status = da_df_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_invalid_input);

    n_obs = 1;
    d = 1;
    status =
        da_df_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs - 1, y.data());
    EXPECT_EQ(status, da_status_invalid_input);

    da_handle_destroy(&df_handle);
}

template <typename T> void test_decision_forest_get_results() {
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
    status = da_handle_init<T>(&df_handle, da_handle_decision_forest);
    EXPECT_EQ(status, da_status_success);

    // run with random seed
    da_int seed_val = -1;
    EXPECT_EQ(da_options_set_int(df_handle, "seed", seed_val), da_status_success);

    EXPECT_EQ(da_df_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data()),
              da_status_success);

    EXPECT_EQ(da_df_fit<T>(df_handle), da_status_success);

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

    EXPECT_EQ(da_df_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data()),
              da_status_success);

    EXPECT_EQ(da_df_fit<T>(df_handle), da_status_success);

    EXPECT_EQ(
        da_handle_get_result(df_handle, da_result::da_rinfo, &rinfo_size, rinfo.data()),
        da_status_success);
    EXPECT_EQ((da_int)rinfo[0], seed_val);
    EXPECT_EQ((da_int)rinfo[1], n_obs);
    EXPECT_EQ((da_int)rinfo[2], d);

    da_handle_destroy(&df_handle);
}

TYPED_TEST(DecisionForestTest, invalid_input) {
    test_decision_forest_invalid_input<TypeParam>();
}

TYPED_TEST(DecisionForestTest, get_results) {
    test_decision_forest_get_results<TypeParam>();
}

template <typename T> void test_decision_forest_bad_handle() {
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

    status = da_df_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_handle_not_initialized);

    status = da_df_fit<T>(df_handle);
    EXPECT_EQ(status, da_status_handle_not_initialized);

    status = da_df_predict<T>(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_handle_not_initialized);

    status = da_df_score<T>(df_handle, n_obs, d, x.data(), n_obs, y.data(), &score);
    EXPECT_EQ(status, da_status_handle_not_initialized);

    // incorrect handle type
    EXPECT_EQ(da_handle_init<T>(&df_handle, da_handle_linmod), da_status_success);
    status = da_df_set_training_data<T>(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_invalid_handle_type);

    status = da_df_fit<T>(df_handle);
    EXPECT_EQ(status, da_status_invalid_handle_type);

    status = da_df_predict<T>(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_invalid_handle_type);

    status = da_df_score<T>(df_handle, n_obs, d, x.data(), n_obs, y.data(), &score);
    EXPECT_EQ(status, da_status_invalid_handle_type);

    da_handle_destroy(&df_handle);
}

TYPED_TEST(DecisionForestTest, bad_handle) {
    test_decision_forest_bad_handle<TypeParam>();
}

TEST(DecisionForestTest, incorrect_handle_precision) {
    da_status status;

    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_decision_forest), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_decision_forest), da_status_success);

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

TYPED_TEST(DecisionForestTest, test_decision_forest_invalid_array_dim) {
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
    status = da_handle_init<TypeParam>(&df_handle, da_handle_decision_forest);
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
