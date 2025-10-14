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
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <omp.h>
#include <stdio.h>
#include <string.h>

typedef struct params_t {
    std::string test_name;
    std::vector<double> classes;
    std::vector<da_int> expected_shuffled_indices;
    da_int m;
    da_int seed;
    da_int train_size;
    da_int test_size;
    da_int fp_precision;
} params;

const params_t shuffle_indices_params[] = {
    {
        "shuffle_indices_1",
        {},
        {5, 0, 6, 2, 9, 4, 1, 8, 7, 3},
        10,
        42,
        0,
        0,
        10,
    },
    {
        "shuffle_indices_2",
        {},
        {5, 2, 7, 11, 0, 3, 12, 8, 6, 1, 10, 9, 4},
        13,
        42,
        0,
        0,
        10,
    },
    {
        "shuffle_indices_3",
        {},
        {4, 10, 11, 13, 20, 19, 6, 17, 9, 0, 1, 5, 2, 14, 8, 16, 12, 3, 18, 15, 7},
        21,
        42,
        0,
        0,
        10,
    },
    {
        "shuffle_indices_4",
        {2, 0, 0, 0, 2, 2, 2, 0, 0, 2},
        {0, 8, 6, 2},
        10,
        42,
        2,
        2,
        10,
    },
    {
        "shuffle_indices_5",
        {2, 0, 0, 0, 2, 2, 2, 0, 0, 2},
        {9, 3, 2, 6, 7},
        10,
        42,
        3,
        2,
        10,
    },
    {
        "shuffle_indices_6",
        {2, 0, 0, 0, 2, 2, 2, 0, 0, 2},
        {9, 3, 8, 6, 7},
        10,
        42,
        2,
        3,
        10,
    },
    {
        "shuffle_indices_7",
        {2, 0, 0, 0, 2, 2, 2, 0, 0, 2},
        {0, 8, 5, 6, 3, 4, 9, 2, 7, 1},
        10,
        42,
        2,
        8,
        10,
    },
    {
        "shuffle_indices_8",
        {2, 0, 0, 0, 2, 2, 2, 0, 0, 2},
        {4, 3, 9, 0, 5, 8, 7, 1, 6, 2},
        10,
        42,
        8,
        2,
        10,
    },
    {
        "shuffle_indices_9",
        {2, 0, 0, 0, 2, 2, 2, 0, 0, 2},
        {2, 0, 9, 3, 1, 7, 4, 8, 5, 6},
        10,
        42,
        5,
        5,
        10,
    },
    {
        "shuffle_indices_10",
        {-14, -2, -2, -2, -14, -2, -14, -14, -2, -14},
        {1, 9, 5, 4},
        10,
        42,
        2,
        2,
        10,
    },
    {
        "shuffle_indices_11",
        {-14, -2, -2, -2, -14, -2, -14, -14, -2, -14},
        {8, 6, 4, 5, 7},
        10,
        42,
        3,
        2,
        10,
    },
    {
        "shuffle_indices_12",
        {-14, -2, -2, -2, -14, -2, -14, -14, -2, -14},
        {8, 6, 9, 5, 7},
        10,
        42,
        2,
        3,
        10,
    },
    {
        "shuffle_indices_13",
        {-14, -2, -2, -2, -14, -2, -14, -14, -2, -14},
        {2, 6, 8, 1, 3, 9, 7, 0, 5, 4},
        10,
        42,
        8,
        2,
        10,
    },
    {
        "shuffle_indices_14",
        {-14, -2, -2, -2, -14, -2, -14, -14, -2, -14},
        {1, 9, 3, 5, 6, 2, 8, 4, 7, 0},
        10,
        42,
        2,
        8,
        10,
    },
    {
        "shuffle_indices_15",
        {-14, -2, -2, -2, -14, -2, -14, -14, -2, -14},
        {4, 1, 8, 6, 0, 7, 2, 9, 3, 5},
        10,
        42,
        5,
        5,
        10,
    },
    {
        "shuffle_indices_16",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18},
        {9, 4, 1, 7, 6, 5},
        10,
        42,
        3,
        3,
        10,
    },
    {
        "shuffle_indices_17",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18},
        {7, 2, 0, 3, 8, 9, 4},
        10,
        42,
        4,
        3,
        10,
    },
    {
        "shuffle_indices_18",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18},
        {8, 2, 0, 1, 7, 6, 5},
        10,
        42,
        3,
        4,
        10,
    },
    {
        "shuffle_indices_19",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18},
        {5, 6, 3, 2, 7, 0, 1, 4, 9, 8},
        10,
        42,
        7,
        3,
        10,
    },
    {
        "shuffle_indices_20",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18},
        {8, 2, 0, 7, 3, 1, 9, 6, 5, 4},
        10,
        42,
        3,
        7,
        10,
    },
    {
        "shuffle_indices_21",
        {33, 0, -8, 105, 105, -66, 33, -66, 0, -8},
        {3, 7, 2, 1, 6, 9, 4, 0, 5, 8},
        10,
        42,
        5,
        5,
        10,
    },
    {
        "shuffle_indices_22",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18, 18, 18, -22},
        {1, 10, 5, 8, 12, 2},
        13,
        42,
        3,
        3,
        10,
    },
    {
        "shuffle_indices_23",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18, 18, 18, -22},
        {8, 6, 10, 0, 4, 9, 1},
        13,
        42,
        4,
        3,
        10,
    },
    {
        "shuffle_indices_24",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18, 18, 18, -22},
        {1, 11, 4, 2, 12, 9, 7},
        13,
        42,
        3,
        4,
        10,
    },
    {
        "shuffle_indices_25",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18, 18, 18, -22},
        {3, 6, 8, 2, 10, 0, 12, 9, 4, 1},
        13,
        42,
        7,
        3,
        10,
    },
    {
        "shuffle_indices_26",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18, 18, 18, -22},
        {1, 11, 4, 6, 9, 12, 7, 5, 2, 8},
        13,
        42,
        3,
        7,
        10,
    },
    {
        "shuffle_indices_27",
        {33, 0, -8, 105, 105, -66, 33, -66, 0, -8, 0, 33, 0},
        {11, 4, 12, 2, 5, 3, 10, 8, 7, 6},
        13,
        42,
        5,
        5,
        10,
    },
    {
        "shuffle_indices_28",
        {-22.2467, 0.732, 18.383, 0.732, -22.2467, -22.2467, 18.383, 0.732, 0.732, 18.383,
         18.383, 18.383, -22.2467},
        {1, 10, 5, 8, 12, 2},
        13,
        42,
        3,
        3,
        10,
    },
    {
        "shuffle_indices_29",
        {-22.2467, 0.732, 18.383, 0.732, -22.2467, -22.2467, 18.383, 0.732, 0.732, 18.383,
         18.383, 18.383, -22.2467},
        {8, 6, 10, 0, 4, 9, 1},
        13,
        42,
        4,
        3,
        10,
    },
    {
        "shuffle_indices_30",
        {-22.2467, 0.732, 18.383, 0.732, -22.2467, -22.2467, 18.383, 0.732, 0.732, 18.383,
         18.383, 18.383, -22.2467},
        {1, 11, 4, 2, 12, 9, 7},
        13,
        42,
        3,
        4,
        10,
    },
    {
        "shuffle_indices_31",
        {-22.2467, 0.732, 18.383, 0.732, -22.2467, -22.2467, 18.383, 0.732, 0.732, 18.383,
         18.383, 18.383, -22.2467},
        {3, 6, 8, 2, 10, 0, 12, 9, 4, 1},
        13,
        42,
        7,
        3,
        10,
    },
    {
        "shuffle_indices_32",
        {-22.2467, 0.732, 18.383, 0.732, -22.2467, -22.2467, 18.383, 0.732, 0.732, 18.383,
         18.383, 18.383, -22.2467},
        {1, 11, 4, 6, 9, 12, 7, 5, 2, 8},
        13,
        42,
        3,
        7,
        10,
    },
    {
        "shuffle_indices_33",
        {-22.2467, 0.732, 18.383, 0.732, -22.2467, -22.2467, 18.383, 0.732, 0.732, 18.383,
         18.383, 18.383, -22.2467},
        {10, 9, 1, 4, 5, 6, 8, 3, 12, 2},
        13,
        42,
        5,
        5,
        10,
    },

};

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