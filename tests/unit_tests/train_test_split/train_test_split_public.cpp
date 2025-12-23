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
    std::vector<double> X;
    std::vector<double> expected_train;
    std::vector<double> expected_test;
    da_int shuffle;
    std::vector<da_int> shuffle_array;
    da_int m;
    da_int n;
    da_int train_size;
    da_int test_size;
    da_order order;
    da_int ldx;
    da_int ldx_train;
    da_int ldx_test;
} params;

const params_t train_test_split_params[] = {
    {
        "train_test_split_test_1",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        0,
        {},
        10,
        5,
        1,
        1,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_2",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2},
        0,
        {},
        10,
        5,
        2,
        1,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_3",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 2, 2, 2, 2, 2},
        0,
        {},
        10,
        5,
        1,
        2,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_4",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7},
        {8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        0,
        {},
        10,
        5,
        8,
        2,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_5",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        0,
        {},
        10,
        5,
        2,
        8,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_6",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4},
        {5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        0,
        {},
        10,
        5,
        5,
        5,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_7",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        0,
        {},
        10,
        5,
        1,
        1,
        column_major,
        10,
        1,
        1,
    },
    {
        "train_test_split_test_8",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
        {2, 2, 2, 2, 2},
        0,
        {},
        10,
        5,
        2,
        1,
        column_major,
        10,
        2,
        1,
    },
    {
        "train_test_split_test_9",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 0, 0, 0, 0},
        {1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
        0,
        {},
        10,
        5,
        1,
        2,
        column_major,
        10,
        1,
        2,
    },
    {
        "train_test_split_test_10",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3,
         4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 8, 9, 8, 9, 8, 9, 8, 9},
        0,
        {},
        10,
        5,
        8,
        2,
        column_major,
        10,
        8,
        2,
    },
    {
        "train_test_split_test_11",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
        {2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5,
         6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9},
        0,
        {},
        10,
        5,
        2,
        8,
        column_major,
        10,
        2,
        8,
    },
    {
        "train_test_split_test_12",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4},
        {5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9},
        0,
        {},
        10,
        5,
        5,
        5,
        column_major,
        10,
        5,
        5,
    },
    {
        "train_test_split_test_13",
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 3, 3,
         3, 3, 3, 0, 4, 4, 4, 4, 4, 0, 5, 5, 5, 5, 5, 0, 6, 6, 6, 6,
         6, 0, 7, 7, 7, 7, 7, 0, 8, 8, 8, 8, 8, 0, 9, 9, 9, 9, 9, 0},
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4},
        {5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        0,
        {},
        10,
        5,
        5,
        5,
        row_major,
        6,
        5,
        5,
    },
    {
        "train_test_split_test_14",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2,
         2, 2, 0, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 0},
        {5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        0,
        {},
        10,
        5,
        5,
        5,
        row_major,
        5,
        6,
        5,
    },
    {
        "train_test_split_test_15",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4},
        {5, 5, 5, 5, 5, 0, 6, 6, 6, 6, 6, 0, 7, 7, 7,
         7, 7, 0, 8, 8, 8, 8, 8, 0, 9, 9, 9, 9, 9, 0},
        0,
        {},
        10,
        5,
        5,
        5,
        row_major,
        5,
        5,
        6,
    },
    {
        "train_test_split_test_16",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2,
         2, 2, 0, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 0},
        {5, 5, 5, 5, 5, 0, 6, 6, 6, 6, 6, 0, 7, 7, 7,
         7, 7, 0, 8, 8, 8, 8, 8, 0, 9, 9, 9, 9, 9, 0},
        0,
        {},
        10,
        5,
        5,
        5,
        row_major,
        5,
        6,
        6,
    },
    {
        "train_test_split_test_17",
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 3, 3,
         3, 3, 3, 0, 4, 4, 4, 4, 4, 0, 5, 5, 5, 5, 5, 0, 6, 6, 6, 6,
         6, 0, 7, 7, 7, 7, 7, 0, 8, 8, 8, 8, 8, 0, 9, 9, 9, 9, 9, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2,
         2, 2, 0, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 0},
        {5, 5, 5, 5, 5, 0, 6, 6, 6, 6, 6, 0, 7, 7, 7,
         7, 7, 0, 8, 8, 8, 8, 8, 0, 9, 9, 9, 9, 9, 0},
        0,
        {},
        10,
        5,
        5,
        5,
        row_major,
        6,
        6,
        6,
    },
    {
        "train_test_split_test_18",
        {0, 1, 2,  3, 4, 5,  6, 7, 8, 9, 10, 0, 1, 2,  3, 4, 5, 6, 7,
         8, 9, 10, 0, 1, 2,  3, 4, 5, 6, 7,  8, 9, 10, 0, 1, 2, 3, 4,
         5, 6, 7,  8, 9, 10, 0, 1, 2, 3, 4,  5, 6, 7,  8, 9, 10},
        {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4},
        {5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9},
        0,
        {},
        10,
        5,
        5,
        5,
        column_major,
        11,
        5,
        5,
    },
    {
        "train_test_split_test_19",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2,
         3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0},
        {5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9},
        0,
        {},
        10,
        5,
        5,
        5,
        column_major,
        10,
        6,
        5,
    },
    {
        "train_test_split_test_20",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4},
        {5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9, 0, 5, 6, 7,
         8, 9, 0, 5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9, 0},
        0,
        {},
        10,
        5,
        5,
        5,
        column_major,
        10,
        5,
        6,
    },
    {
        "train_test_split_test_21",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2,
         3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0},
        {5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9, 0, 5, 6, 7,
         8, 9, 0, 5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9, 0},
        0,
        {},
        10,
        5,
        5,
        5,
        column_major,
        10,
        6,
        6,
    },
    {
        "train_test_split_test_22",
        {0, 1, 2,  3, 4, 5,  6, 7, 8, 9, 10, 0, 1, 2,  3, 4, 5, 6, 7,
         8, 9, 10, 0, 1, 2,  3, 4, 5, 6, 7,  8, 9, 10, 0, 1, 2, 3, 4,
         5, 6, 7,  8, 9, 10, 0, 1, 2, 3, 4,  5, 6, 7,  8, 9, 10},
        {0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2,
         3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0},
        {5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9, 0, 5, 6, 7,
         8, 9, 0, 5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9, 0},
        0,
        {},
        10,
        5,
        5,
        5,
        column_major,
        11,
        6,
        6,
    },
    {
        "train_test_split_test_23",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {3, 3, 3, 3, 3},
        {7, 7, 7, 7, 7},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        1,
        1,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_24",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {3, 3, 3, 3, 3, 7, 7, 7, 7, 7},
        {1, 1, 1, 1, 1},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        2,
        1,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_24",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {3, 3, 3, 3, 3},
        {7, 7, 7, 7, 7, 1, 1, 1, 1, 1},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        1,
        2,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_26",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9,
         0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6},
        {2, 2, 2, 2, 2, 5, 5, 5, 5, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        8,
        2,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_27",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {3, 3, 3, 3, 3, 7, 7, 7, 7, 7},
        {1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8,
         4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        2,
        8,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_28",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0},
        {8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_test_29",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {3, 3, 3, 3, 3},
        {7, 7, 7, 7, 7},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        1,
        1,
        column_major,
        10,
        1,
        1,
    },
    {
        "train_test_split_test_30",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {3, 7, 3, 7, 3, 7, 3, 7, 3, 7},
        {1, 1, 1, 1, 1},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        2,
        1,
        column_major,
        10,
        2,
        1,
    },
    {
        "train_test_split_test_31",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {3, 3, 3, 3, 3},
        {7, 1, 7, 1, 7, 1, 7, 1, 7, 1},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        1,
        2,
        column_major,
        10,
        1,
        2,
    },
    {
        "train_test_split_test_32",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {3, 7, 1, 9, 0, 8, 4, 6, 3, 7, 1, 9, 0, 8, 4, 6, 3, 7, 1, 9,
         0, 8, 4, 6, 3, 7, 1, 9, 0, 8, 4, 6, 3, 7, 1, 9, 0, 8, 4, 6},
        {2, 5, 2, 5, 2, 5, 2, 5, 2, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        8,
        2,
        column_major,
        10,
        8,
        2,
    },
    {
        "train_test_split_test_33",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {3, 7, 3, 7, 3, 7, 3, 7, 3, 7},
        {1, 9, 0, 8, 4, 6, 2, 5, 1, 9, 0, 8, 4, 6, 2, 5, 1, 9, 0, 8,
         4, 6, 2, 5, 1, 9, 0, 8, 4, 6, 2, 5, 1, 9, 0, 8, 4, 6, 2, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        2,
        8,
        column_major,
        10,
        2,
        8,
    },
    {
        "train_test_split_test_34",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0},
        {8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        column_major,
        10,
        5,
        5,
    },
    {
        "train_test_split_test_35",
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 3, 3,
         3, 3, 3, 0, 4, 4, 4, 4, 4, 0, 5, 5, 5, 5, 5, 0, 6, 6, 6, 6,
         6, 0, 7, 7, 7, 7, 7, 0, 8, 8, 8, 8, 8, 0, 9, 9, 9, 9, 9, 0},
        {3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0},
        {8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        row_major,
        6,
        5,
        5,
    },
    {
        "train_test_split_test_36",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {3, 3, 3, 3, 3, 0, 7, 7, 7, 7, 7, 0, 1, 1, 1,
         1, 1, 0, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0},
        {8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        row_major,
        5,
        6,
        5,
    },
    {
        "train_test_split_test_37",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0},
        {8, 8, 8, 8, 8, 0, 4, 4, 4, 4, 4, 0, 6, 6, 6,
         6, 6, 0, 2, 2, 2, 2, 2, 0, 5, 5, 5, 5, 5, 0},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        row_major,
        5,
        5,
        6,
    },
    {
        "train_test_split_test_38",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        {3, 3, 3, 3, 3, 0, 7, 7, 7, 7, 7, 0, 1, 1, 1,
         1, 1, 0, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0},
        {8, 8, 8, 8, 8, 0, 4, 4, 4, 4, 4, 0, 6, 6, 6,
         6, 6, 0, 2, 2, 2, 2, 2, 0, 5, 5, 5, 5, 5, 0},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        row_major,
        5,
        6,
        6,
    },
    {
        "train_test_split_test_39",
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 3, 3,
         3, 3, 3, 0, 4, 4, 4, 4, 4, 0, 5, 5, 5, 5, 5, 0, 6, 6, 6, 6,
         6, 0, 7, 7, 7, 7, 7, 0, 8, 8, 8, 8, 8, 0, 9, 9, 9, 9, 9, 0},
        {3, 3, 3, 3, 3, 0, 7, 7, 7, 7, 7, 0, 1, 1, 1,
         1, 1, 0, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0},
        {8, 8, 8, 8, 8, 0, 4, 4, 4, 4, 4, 0, 6, 6, 6,
         6, 6, 0, 2, 2, 2, 2, 2, 0, 5, 5, 5, 5, 5, 0},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        row_major,
        6,
        6,
        6,
    },
    {
        "train_test_split_test_40",
        {0, 1, 2,  3, 4, 5,  6, 7, 8, 9, 10, 0, 1, 2,  3, 4, 5, 6, 7,
         8, 9, 10, 0, 1, 2,  3, 4, 5, 6, 7,  8, 9, 10, 0, 1, 2, 3, 4,
         5, 6, 7,  8, 9, 10, 0, 1, 2, 3, 4,  5, 6, 7,  8, 9, 10},
        {3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0},
        {8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        column_major,
        11,
        5,
        5,
    },
    {
        "train_test_split_test_41",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {3, 7, 1, 9, 0, 0, 3, 7, 1, 9, 0, 0, 3, 7, 1,
         9, 0, 0, 3, 7, 1, 9, 0, 0, 3, 7, 1, 9, 0, 0},
        {8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        column_major,
        10,
        6,
        5,
    },
    {
        "train_test_split_test_42",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0},
        {8, 4, 6, 2, 5, 0, 8, 4, 6, 2, 5, 0, 8, 4, 6,
         2, 5, 0, 8, 4, 6, 2, 5, 0, 8, 4, 6, 2, 5, 0},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        column_major,
        10,
        5,
        6,
    },
    {
        "train_test_split_test_43",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {3, 7, 1, 9, 0, 0, 3, 7, 1, 9, 0, 0, 3, 7, 1,
         9, 0, 0, 3, 7, 1, 9, 0, 0, 3, 7, 1, 9, 0, 0},
        {8, 4, 6, 2, 5, 0, 8, 4, 6, 2, 5, 0, 8, 4, 6,
         2, 5, 0, 8, 4, 6, 2, 5, 0, 8, 4, 6, 2, 5, 0},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        column_major,
        10,
        6,
        6,
    },
    {
        "train_test_split_test_44",
        {0, 1, 2,  3, 4, 5,  6, 7, 8, 9, 10, 0, 1, 2,  3, 4, 5, 6, 7,
         8, 9, 10, 0, 1, 2,  3, 4, 5, 6, 7,  8, 9, 10, 0, 1, 2, 3, 4,
         5, 6, 7,  8, 9, 10, 0, 1, 2, 3, 4,  5, 6, 7,  8, 9, 10},
        {3, 7, 1, 9, 0, 0, 3, 7, 1, 9, 0, 0, 3, 7, 1,
         9, 0, 0, 3, 7, 1, 9, 0, 0, 3, 7, 1, 9, 0, 0},
        {8, 4, 6, 2, 5, 0, 8, 4, 6, 2, 5, 0, 8, 4, 6,
         2, 5, 0, 8, 4, 6, 2, 5, 0, 8, 4, 6, 2, 5, 0},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        10,
        5,
        5,
        5,
        column_major,
        11,
        6,
        6,
    },

};

class TrainTestSplitFunctionality : public testing::TestWithParam<params> {};

template <class T> void train_test_split_test_functionality(const params pr) {
    std::vector<T> X = convert_vector<double, T>(pr.X);

    const da_int *shuffle_array = (pr.shuffle) ? pr.shuffle_array.data() : nullptr;

    std::vector<T> X_train;
    std::vector<T> X_test;

    if (pr.order == row_major) {
        X_train.resize(pr.ldx_train * pr.train_size, 0.0);
        X_test.resize(pr.ldx_test * pr.test_size, 0.0);
    } else if (pr.order == column_major) {
        X_train.resize(pr.ldx_train * pr.n, 0.0);
        X_test.resize(pr.ldx_test * pr.n, 0.0);
    }

    EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, X.data(), pr.ldx, pr.train_size,
                                  pr.test_size, shuffle_array, X_train.data(),
                                  pr.ldx_train, X_test.data(), pr.ldx_test),
              da_status_success);

    if (pr.order == row_major) {
        for (da_int i = 0; i < pr.ldx_train * pr.train_size; i++) {
            std::cout << X_train[i] << ", ";
        }
        std::cout << '\n';

        EXPECT_ARR_EQ(pr.ldx_train * pr.train_size, X_train.data(),
                      pr.expected_train.data(), 1, 1, 0, 0)
            << "Test failure: " << pr.test_name;

        EXPECT_ARR_EQ(pr.ldx_test * pr.test_size, X_test.data(), pr.expected_test.data(),
                      1, 1, 0, 0)
            << "Test failure: " << pr.test_name;
    } else if (pr.order == column_major) {
        EXPECT_ARR_EQ(pr.ldx_train * pr.n, X_train.data(), pr.expected_train.data(), 1, 1,
                      0, 0)
            << "Test failure: " << pr.test_name;

        EXPECT_ARR_EQ(pr.ldx_test * pr.n, X_test.data(), pr.expected_test.data(), 1, 1, 0,
                      0)
            << "Test failure: " << pr.test_name;
    }

    shuffle_array = nullptr;
}

void PrintTo(const params &param, ::std::ostream *os) { *os << param.test_name; }

TEST_P(TrainTestSplitFunctionality, da_int) {
    const params &pr = GetParam();
    train_test_split_test_functionality<da_int>(pr);
}

TEST_P(TrainTestSplitFunctionality, double) {
    const params &pr = GetParam();
    train_test_split_test_functionality<double>(pr);
}

TEST_P(TrainTestSplitFunctionality, float) {
    const params &pr = GetParam();
    train_test_split_test_functionality<float>(pr);
}

INSTANTIATE_TEST_SUITE_P(traintestSuite, TrainTestSplitFunctionality,
                         testing::ValuesIn(train_test_split_params));

// ******************************

const params_t train_test_split_params_validation[] = {
    {
        "train_test_split_validation_1",
        {1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5},
        {},
        {},
        0,
        {},
        5,
        5,
        3,
        2,
        row_major,
        5,
        5,
        5,
    },
    {
        "train_test_split_validation_2",
        {1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5},
        {},
        {},
        0,
        {},
        5,
        5,
        3,
        2,
        column_major,
        5,
        3,
        2,
    },
};

class TrainTestSplitValidation : public testing::TestWithParam<params> {};

template <class T> void train_test_split_test_validation(const params pr) {
    std::vector<T> X = convert_vector<double, T>(pr.X);

    std::vector<T> X_train;
    std::vector<T> X_test;
    if (pr.order == column_major) {
        X_train.resize(pr.train_size * pr.ldx_train, 0.0);
        X_test.resize(pr.test_size * pr.ldx_test, 0.0);
    } else if (pr.order == row_major) {
        X_train.resize(pr.train_size * pr.n, 0.0);
        X_test.resize(pr.test_size * pr.n, 0.0);
    }

    // test invalid X pointer
    EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, nullptr, pr.ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  X_test.data(), pr.ldx_test),
              da_status_invalid_pointer);
    // test invalid X_train pointer
    EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, X.data(), pr.ldx, pr.train_size,
                                  pr.test_size, nullptr, nullptr, pr.ldx_train,
                                  X_test.data(), pr.ldx_test),
              da_status_invalid_pointer);
    // test invalid X_test pointer
    EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, X.data(), pr.ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  nullptr, pr.ldx_test),
              da_status_invalid_pointer);
    // test invalid ldx
    EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, X.data(), 4, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  X_test.data(), pr.ldx_test),
              da_status_invalid_leading_dimension);
    // test invalid ldx_train
    EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, X.data(), pr.ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), 1, X_test.data(),
                                  pr.ldx_test),
              da_status_invalid_leading_dimension);
    // test invalid ldx_test
    EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, X.data(), pr.ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  X_test.data(), 1),
              da_status_invalid_leading_dimension);
    // test invalid dimension m
    EXPECT_EQ(da_train_test_split(pr.order, 0, pr.n, X.data(), pr.ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  X_test.data(), pr.ldx_test),
              da_status_invalid_array_dimension);
    // test invalid dimension n
    EXPECT_EQ(da_train_test_split(pr.order, pr.m, 0, X.data(), pr.ldx, pr.train_size,
                                  pr.test_size, nullptr, X_train.data(), pr.ldx_train,
                                  X_test.data(), pr.ldx_test),
              da_status_invalid_array_dimension);
    // test invalid train_size
    EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, X.data(), pr.ldx, 0, pr.test_size,
                                  nullptr, X_train.data(), pr.ldx_train, X_test.data(),
                                  pr.ldx_test),
              da_status_invalid_input);
    // test invalid test_size
    EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, X.data(), pr.ldx, pr.train_size,
                                  0, nullptr, X_train.data(), pr.ldx_train, X_test.data(),
                                  pr.ldx_test),
              da_status_invalid_input);
    // test invalid train_size + test_size
    if (pr.order == column_major) {
        EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, X.data(), pr.ldx, 3, 3,
                                      nullptr, X_train.data(), pr.ldx_train,
                                      X_test.data(), pr.ldx_test),
                  da_status_invalid_leading_dimension);
    } else if (pr.order == row_major) {
        EXPECT_EQ(da_train_test_split(pr.order, pr.m, pr.n, X.data(), pr.ldx, 3, 3,
                                      nullptr, X_train.data(), pr.ldx_train,
                                      X_test.data(), pr.ldx_test),
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