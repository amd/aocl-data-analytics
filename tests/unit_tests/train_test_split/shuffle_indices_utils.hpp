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

#include <aoclda.h>
#include <string>
#include <vector>

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
        "shuffle_indices_generic_even_m",
        {},
        {5, 0, 6, 2, 9, 4, 1, 8, 7, 3},
        10,
        42,
        0,
        0,
        10,
    },
    {
        "shuffle_indices_generic_odd_m",
        {},
        {4, 10, 11, 13, 20, 19, 6, 17, 9, 0, 1, 5, 2, 14, 8, 16, 12, 3, 18, 15, 7},
        21,
        42,
        0,
        0,
        10,
    },
    {
        "shuffle_indices_stratified_min_split_2_classes",
        {2, 0, 0, 0, 2, 2, 2, 0, 0, 2},
        {0, 8, 6, 2},
        10,
        42,
        2,
        2,
        10,
    },
    {
        "shuffle_indices_stratified_test_bigger_2_classes",
        {2, 0, 0, 0, 2, 2, 2, 0, 0, 2},
        {0, 8, 5, 6, 3, 4, 9, 2, 7, 1},
        10,
        42,
        2,
        8,
        10,
    },
    {
        "shuffle_indices_stratified_train_bigger_2_classes",
        {2, 0, 0, 0, 2, 2, 2, 0, 0, 2},
        {4, 3, 9, 0, 5, 8, 7, 1, 6, 2},
        10,
        42,
        8,
        2,
        10,
    },
    {
        "shuffle_indices_stratified_even_split_2_classes",
        {2, 0, 0, 0, 2, 2, 2, 0, 0, 2},
        {2, 0, 9, 3, 1, 7, 4, 8, 5, 6},
        10,
        42,
        5,
        5,
        10,
    },
    {
        "shuffle_indices_stratified_min_split_2_negative_classes",
        {-14, -2, -2, -2, -14, -2, -14, -14, -2, -14},
        {1, 9, 5, 4},
        10,
        42,
        2,
        2,
        10,
    },
    {
        "shuffle_indices_stratified_train_bigger_2_negative_classes",
        {-14, -2, -2, -2, -14, -2, -14, -14, -2, -14},
        {2, 6, 8, 1, 3, 9, 7, 0, 5, 4},
        10,
        42,
        8,
        2,
        10,
    },
    {
        "shuffle_indices_stratified_test_bigger_2_negative_classes",
        {-14, -2, -2, -2, -14, -2, -14, -14, -2, -14},
        {1, 9, 3, 5, 6, 2, 8, 4, 7, 0},
        10,
        42,
        2,
        8,
        10,
    },
    {
        "shuffle_indices_stratified_even_split_2_negative_classes",
        {-14, -2, -2, -2, -14, -2, -14, -14, -2, -14},
        {4, 1, 8, 6, 0, 7, 2, 9, 3, 5},
        10,
        42,
        5,
        5,
        10,
    },
    {
        "shuffle_indices_stratified_min_split_3_mixed_classes",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18},
        {9, 4, 1, 7, 6, 5},
        10,
        42,
        3,
        3,
        10,
    },
    {
        "shuffle_indices_stratified_train_bigger_3_mixed_classes",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18},
        {5, 6, 3, 2, 7, 0, 1, 4, 9, 8},
        10,
        42,
        7,
        3,
        10,
    },
    {
        "shuffle_indices_stratified_test_bigger_3_mixed_classes",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18},
        {8, 2, 0, 7, 3, 1, 9, 6, 5, 4},
        10,
        42,
        3,
        7,
        10,
    },
    {
        "shuffle_indices_stratified_even_split_3_mixed_classes",
        {-22, 0, 18, 0, -22, -22, 18, 0, 0, 18},
        {3, 8, 2, 4, 0, 6, 9, 7, 1, 5},
        10,
        42,
        5,
        5,
        10,
    },
    {
        "shuffle_indices_stratified_min_split_3_mixed_float_classes",
        {-22.2467, 0.732, 18, 0.732, -22.2467, -22.2467, 18, 0.736, 0.73, 18, 18, 18,
         -22.2467},
        {1, 10, 5, 8, 12, 2},
        13,
        42,
        3,
        3,
        10,
    },
    {
        "shuffle_indices_stratified_train_bigger_3_mixed_float_classes",
        {-22.2467, 0.732, 18, 0.732, -22.2467, -22.2467, 18, 0.736, 0.73, 18, 18, 18,
         -22.2467},
        {3, 6, 8, 2, 10, 0, 12, 9, 4, 1},
        13,
        42,
        7,
        3,
        10,
    },
    {
        "shuffle_indices_stratified_test_bigger_3_mixed_float_classes",
        {-22.2467, 0.732, 18, 0.732, -22.2467, -22.2467, 18, 0.736, 0.73, 18, 18, 18,
         -22.2467},
        {1, 11, 4, 6, 9, 12, 7, 5, 2, 8},
        13,
        42,
        3,
        7,
        10,
    },
    {
        "shuffle_indices_stratified_even_split_3_mixed_float_classes",
        {-22.2467, 0.732, 18, 0.732, -22.2467, -22.2467, 18, 0.736, 0.73, 18, 18, 18,
         -22.2467},
        {10, 9, 1, 4, 5, 6, 8, 3, 12, 2},
        13,
        42,
        5,
        5,
        10,
    },

};