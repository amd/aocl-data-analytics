/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclda_cpp_overloads.hpp"
#include "train_test_split_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <string>

class TrainTestSplitFunctionality
    : public testing::TestWithParam<std::tuple<params, da_int>> {};

template <class T>
void train_test_split_test_functionality(const params pr, da_int bigger_ldx) {
    std::tuple<std::vector<da_int>, da_int, da_int, da_int> data =
        get_data(pr.order, bigger_ldx);
    std::vector<T> X = convert_vector<da_int, T>(std::get<0>(data));
    da_int m = std::get<1>(data);
    da_int n = std::get<2>(data);
    da_int ldx = std::get<3>(data);

    const da_int *shuffle_array = (pr.shuffle) ? pr.shuffle_array.data() : nullptr;

    std::vector<T> X_train;
    std::vector<T> X_test;

    if (pr.order == row_major) {
        X_train.resize(pr.ldx_train * pr.train_size, 0.0);
        X_test.resize(pr.ldx_test * pr.test_size, 0.0);
    } else {
        X_train.resize(pr.ldx_train * n, 0.0);
        X_test.resize(pr.ldx_test * n, 0.0);
    }

    EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), ldx, pr.train_size,
                                  pr.test_size, shuffle_array, X_train.data(),
                                  pr.ldx_train, X_test.data(), pr.ldx_test),
              da_status_success);

    if (pr.order == row_major) {
        EXPECT_ARR_EQ(pr.ldx_train * pr.train_size, X_train.data(),
                      pr.expected_train.data(), 1, 1, 0, 0)
            << "Test failure: " << pr.test_name;

        EXPECT_ARR_EQ(pr.ldx_test * pr.test_size, X_test.data(), pr.expected_test.data(),
                      1, 1, 0, 0)
            << "Test failure: " << pr.test_name;
    } else if (pr.order == column_major) {
        EXPECT_ARR_EQ(pr.ldx_train * n, X_train.data(), pr.expected_train.data(), 1, 1, 0,
                      0)
            << "Test failure: " << pr.test_name;

        EXPECT_ARR_EQ(pr.ldx_test * n, X_test.data(), pr.expected_test.data(), 1, 1, 0, 0)
            << "Test failure: " << pr.test_name;
    }

    shuffle_array = nullptr;
}

void PrintTo(const params &param, ::std::ostream *os) { *os << param.test_name; }

TEST_P(TrainTestSplitFunctionality, da_int) {
    const auto &[pr, bigger_ldx] = GetParam();
    train_test_split_test_functionality<da_int>(pr, bigger_ldx);
}

TEST_P(TrainTestSplitFunctionality, double) {
    const auto &[pr, bigger_ldx] = GetParam();
    train_test_split_test_functionality<double>(pr, bigger_ldx);
}

TEST_P(TrainTestSplitFunctionality, float) {
    const auto &[pr, bigger_ldx] = GetParam();
    train_test_split_test_functionality<float>(pr, bigger_ldx);
}

auto bigger_ldx = testing::Values(0, 1);
auto param_thread_combinations =
    testing::Combine(testing::ValuesIn(train_test_split_params), bigger_ldx);

INSTANTIATE_TEST_SUITE_P(traintestSuite, TrainTestSplitFunctionality,
                         param_thread_combinations);

// ******************************

class TrainTestSplitValidation : public testing::TestWithParam<params> {};

template <class T> void train_test_split_test_validation(const params pr) {
    std::tuple<std::vector<da_int>, da_int, da_int, da_int> data = get_data(pr.order, 0);
    std::vector<T> X = convert_vector<da_int, T>(std::get<0>(data));
    da_int m = std::get<1>(data);
    da_int n = std::get<2>(data);
    da_int ldx = std::get<3>(data);

    std::vector<T> X_train;
    std::vector<T> X_test;
    if (pr.order == column_major) {
        X_train.resize(pr.ldx_train * n, 0.0);
        X_test.resize(pr.ldx_test * n, 0.0);
    } else {
        X_train.resize(pr.train_size * pr.ldx_train, 0.0);
        X_test.resize(pr.test_size * pr.ldx_test, 0.0);
    }

    // test invalid X pointer
    EXPECT_EQ(da_train_test_split(pr.order, m, n, nullptr, ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  X_test.data(), pr.ldx_test),
              da_status_invalid_pointer);
    // test invalid X_train pointer
    EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), ldx, pr.train_size,
                                  pr.test_size, nullptr, nullptr, pr.ldx_train,
                                  X_test.data(), pr.ldx_test),
              da_status_invalid_pointer);
    // test invalid X_test pointer
    EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  nullptr, pr.ldx_test),
              da_status_invalid_pointer);
    // test invalid ldx
    EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), 4, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  X_test.data(), pr.ldx_test),
              da_status_invalid_leading_dimension);
    // test invalid ldx_train
    EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), 1, X_test.data(),
                                  pr.ldx_test),
              da_status_invalid_leading_dimension);
    // test invalid ldx_test
    EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  X_test.data(), 1),
              da_status_invalid_leading_dimension);
    // test invalid dimension m
    EXPECT_EQ(da_train_test_split(pr.order, 0, n, X.data(), ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  X_test.data(), pr.ldx_test),
              da_status_invalid_array_dimension);
    // test invalid dimension n
    EXPECT_EQ(da_train_test_split(pr.order, m, 0, X.data(), ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  X_test.data(), pr.ldx_test),
              da_status_invalid_array_dimension);
    // test invalid train_size
    EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), ldx, 0, pr.test_size, nullptr,
                                  X_train.data(), pr.ldx_train, X_test.data(),
                                  pr.ldx_test),
              da_status_invalid_input);
    // test invalid test_size
    EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), ldx, pr.train_size, 0,
                                  nullptr, X_train.data(), pr.ldx_train, X_test.data(),
                                  pr.ldx_test),
              da_status_invalid_input);
    // test invalid train_size + test_size
    if (pr.order == column_major) {
        EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), ldx, 6, 6, nullptr,
                                      X_train.data(), 6, X_test.data(), 6),
                  da_status_invalid_input);
    } else if (pr.order == row_major) {
        EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), ldx, 6, 6, nullptr,
                                      X_train.data(), pr.ldx_train, X_test.data(),
                                      pr.ldx_test),
                  da_status_invalid_input);
    }
}

TEST_P(TrainTestSplitValidation, da_int) {
    const params &pr = GetParam();
    train_test_split_test_validation<da_int>(pr);
}

TEST_P(TrainTestSplitValidation, double) {
    const params &pr = GetParam();
    train_test_split_test_validation<double>(pr);
}

TEST_P(TrainTestSplitValidation, float) {
    const params &pr = GetParam();
    train_test_split_test_validation<float>(pr);
}

INSTANTIATE_TEST_SUITE_P(traintestSuite, TrainTestSplitValidation,
                         testing::ValuesIn(train_test_split_params_validation));