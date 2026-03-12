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
#include "shuffle_indices_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <omp.h>
#include <stdio.h>
#include <string.h>

class ShuffleIndicesFunctionality : public testing::TestWithParam<params> {};

template <class T> void shuffle_indices_functionality(const params pr) {
    std::vector<T> classes_converted = convert_vector<double, T>(pr.classes);
    const T *classes = classes_converted.size() > 0 ? classes_converted.data() : nullptr;

    da_int shuffled_size;
    if (classes == nullptr) {
        shuffled_size = pr.m;
    } else {
        shuffled_size = pr.train_size + pr.test_size;
    }

    std::vector<da_int> shuffle_array_v(shuffled_size);
    da_int *shuffle_array = shuffle_array_v.data();

    EXPECT_EQ(da_get_shuffled_indices(pr.m, pr.seed, pr.train_size, pr.test_size,
                                      pr.fp_precision, classes, shuffle_array),
              da_status_success);

    EXPECT_ARR_EQ(shuffled_size, shuffle_array, pr.expected_shuffled_indices, 1, 1, 0, 0)
        << "Test failure: " << pr.test_name;
}

void PrintTo(const params &param, ::std::ostream *os) { *os << param.test_name; }

TEST_P(ShuffleIndicesFunctionality, da_int) {
    const params &pr = GetParam();
    shuffle_indices_functionality<da_int>(pr);
}
TEST_P(ShuffleIndicesFunctionality, float) {
    const params &pr = GetParam();
    shuffle_indices_functionality<float>(pr);
}
TEST_P(ShuffleIndicesFunctionality, double) {
    const params &pr = GetParam();
    shuffle_indices_functionality<double>(pr);
}

INSTANTIATE_TEST_SUITE_P(shuffleIndicesSuite, ShuffleIndicesFunctionality,
                         testing::ValuesIn(shuffle_indices_params));

//  *******************************

const params_t shuffle_indices_params_validation[] = {
    {
        "shuffle_indices_validation_1",
        {33, 22, 33, 44, 22, 33, 44, 22, 33, 44},
        {},
        10,
        42,
        5,
        5,
        10,
    },
    {
        "shuffle_indices_validation_2",
        {33, 22, 33, 44, 22, 33, 44, 22, 33, 44},
        {},
        10,
        -1,
        5,
        5,
        10,
    },
};

class ShuffleIndicesValidation : public testing::TestWithParam<params> {};

template <class T> void shuffle_array_validation(const params pr) {
    std::vector<da_int> shuffle_array(pr.m);
    const T *null_classes_pointer = nullptr;

    // test that all data is valid at the start
    EXPECT_EQ(da_get_shuffled_indices(pr.m, pr.seed, pr.train_size, pr.test_size,
                                      pr.fp_precision, null_classes_pointer,
                                      shuffle_array.data()),
              da_status_success);

    // test invalid shuffle_array pointer
    EXPECT_EQ(da_get_shuffled_indices(pr.m, pr.seed, pr.train_size, pr.test_size,
                                      pr.fp_precision, null_classes_pointer, nullptr),
              da_status_invalid_pointer);

    // test invalid m
    EXPECT_EQ(da_get_shuffled_indices(1, pr.seed, pr.train_size, pr.test_size,
                                      pr.fp_precision, null_classes_pointer,
                                      shuffle_array.data()),
              da_status_invalid_array_dimension);

    // test invalid train_size
    EXPECT_EQ(da_get_shuffled_indices(pr.m, pr.seed, 2, pr.test_size, pr.fp_precision,
                                      pr.classes.data(), shuffle_array.data()),
              da_status_invalid_input);

    // test invalid test_size
    EXPECT_EQ(da_get_shuffled_indices(pr.m, pr.seed, pr.train_size, 2, pr.fp_precision,
                                      pr.classes.data(), shuffle_array.data()),
              da_status_invalid_input);

    // test invalid fp_precision
    EXPECT_EQ(da_get_shuffled_indices(pr.m, pr.seed, pr.train_size, 2, 0,
                                      pr.classes.data(), shuffle_array.data()),
              da_status_invalid_input);

    // test invalid size of train_size + test_size
    EXPECT_EQ(da_get_shuffled_indices(pr.m, pr.seed, 6, pr.test_size, pr.fp_precision,
                                      pr.classes.data(), shuffle_array.data()),
              da_status_invalid_input);

    // test invalid size of train_size + test_size
    EXPECT_EQ(da_get_shuffled_indices(pr.m, pr.seed, pr.train_size, 6, pr.fp_precision,
                                      pr.classes.data(), shuffle_array.data()),
              da_status_invalid_input);

    // test invalid classes number
    std::vector<da_int> inv_n_classes = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    EXPECT_EQ(da_get_shuffled_indices(pr.m, pr.seed, pr.train_size, pr.test_size,
                                      pr.fp_precision, inv_n_classes.data(),
                                      shuffle_array.data()),
              da_status_invalid_input);

    // test invalid classes counts
    std::vector<da_int> inv_n_classes_counts = {2, 2, 2, -1, -1, 2, 2, 2, 2, 1};
    EXPECT_EQ(da_get_shuffled_indices(pr.m, pr.seed, pr.train_size, pr.test_size,
                                      pr.fp_precision, inv_n_classes_counts.data(),
                                      shuffle_array.data()),
              da_status_invalid_input);

    // test invalid seed
    EXPECT_EQ(da_get_shuffled_indices(pr.m, -2, pr.train_size, pr.test_size,
                                      pr.fp_precision, null_classes_pointer,
                                      shuffle_array.data()),
              da_status_invalid_input);
}

TEST_P(ShuffleIndicesValidation, da_int) {
    const params &pr = GetParam();
    shuffle_array_validation<da_int>(pr);
}
TEST_P(ShuffleIndicesValidation, float) {
    const params &pr = GetParam();
    shuffle_array_validation<float>(pr);
}
TEST_P(ShuffleIndicesValidation, double) {
    const params &pr = GetParam();
    shuffle_array_validation<double>(pr);
}

INSTANTIATE_TEST_SUITE_P(shuffleIndicesSuite, ShuffleIndicesValidation,
                         testing::ValuesIn(shuffle_indices_params_validation));