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
#include "train_test_split.hpp"
#include "train_test_split_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <da_omp.hpp>
#include <list>

using namespace TEST_ARCH::da_utils;

class TrainTestSplitInternal
    : public testing::TestWithParam<std::tuple<params, da_int, da_int>> {};

template <class T>
void train_test_split_test_kernels(const params pr, da_int n_threads, da_int bigger_ldx) {
    // Test performed only on row as col has the same kernel for serial and parallel
    if (pr.order == column_major)
        return;

    std::tuple<std::vector<da_int>, da_int, da_int, da_int> data =
        get_data(pr.order, bigger_ldx);
    std::vector<T> X = convert_vector<da_int, T>(std::get<0>(data));
    da_int m = std::get<1>(data);
    da_int n = std::get<2>(data);
    da_int ldx = std::get<3>(data);

    const da_int *shuffle_array = (pr.shuffle) ? pr.shuffle_array.data() : nullptr;

    std::vector<T> X_train_main;
    std::vector<T> X_test_main;
    std::vector<T> X_train_kernel;
    std::vector<T> X_test_kernel;

    X_train_main.resize(pr.ldx_train * pr.train_size, 0.0);
    X_test_main.resize(pr.ldx_test * pr.test_size, 0.0);
    X_train_kernel.resize(pr.ldx_train * pr.train_size, 0.0);
    X_test_kernel.resize(pr.ldx_test * pr.test_size, 0.0);

    omp_set_num_threads(n_threads);

    EXPECT_EQ(da_train_test_split(pr.order, m, n, X.data(), ldx, pr.train_size,
                                  pr.test_size, shuffle_array, X_train_main.data(),
                                  pr.ldx_train, X_test_main.data(), pr.ldx_test),
              da_status_success);

    omp_set_num_threads(n_threads);
    if (n_threads > 1) {
        split_row_data_parallel(n, pr.train_size, X.data(), ldx, X_train_kernel.data(),
                                pr.ldx_train, shuffle_array);

        const da_int *test_indices =
            shuffle_array != nullptr ? shuffle_array + pr.train_size : nullptr;
        const T *X_ptr =
            test_indices == nullptr ? X.data() + (pr.train_size * ldx) : X.data();

        split_row_data_parallel(n, pr.test_size, X_ptr, ldx, X_test_kernel.data(),
                                pr.ldx_test, test_indices);
    } else {
        split_row_data_serial(n, pr.train_size, pr.test_size, X.data(), ldx,
                              X_train_kernel.data(), pr.ldx_train, X_test_kernel.data(),
                              pr.ldx_test, shuffle_array);
    }

    EXPECT_ARR_EQ(pr.ldx_train * pr.train_size, X_train_main.data(),
                  pr.expected_train.data(), 1, 1, 0, 0)
        << "Test failure: " << pr.test_name;

    EXPECT_ARR_EQ(pr.ldx_test * pr.test_size, X_test_main.data(), pr.expected_test.data(),
                  1, 1, 0, 0)
        << "Test failure: " << pr.test_name;

    EXPECT_ARR_EQ(pr.ldx_train * pr.train_size, X_train_kernel.data(),
                  pr.expected_train.data(), 1, 1, 0, 0)
        << "Test failure: " << pr.test_name;

    EXPECT_ARR_EQ(pr.ldx_test * pr.test_size, X_test_kernel.data(),
                  pr.expected_test.data(), 1, 1, 0, 0)
        << "Test failure: " << pr.test_name;

    shuffle_array = nullptr;
}

void PrintTo(const params &param, ::std::ostream *os) { *os << param.test_name; }

TEST_P(TrainTestSplitInternal, da_int) {
    const auto &[pr, n_threads, bigger_ldx] = GetParam();
    train_test_split_test_kernels<da_int>(pr, n_threads, bigger_ldx);
}

TEST_P(TrainTestSplitInternal, double) {
    const auto &[pr, n_threads, bigger_ldx] = GetParam();
    train_test_split_test_kernels<double>(pr, n_threads, bigger_ldx);
}

TEST_P(TrainTestSplitInternal, float) {
    const auto &[pr, n_threads, bigger_ldx] = GetParam();
    train_test_split_test_kernels<float>(pr, n_threads, bigger_ldx);
}

auto thread_counts = testing::Values(1, 32);
auto bigger_ldx = testing::Values(0, 1);
auto param_thread_combinations = testing::Combine(
    testing::ValuesIn(train_test_split_params), thread_counts, bigger_ldx);

INSTANTIATE_TEST_SUITE_P(traintestSuite, TrainTestSplitInternal,
                         param_thread_combinations);
