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

// The function should return the dataset, m, n, ldx in that order
std::tuple<std::vector<da_int>, da_int, da_int, da_int> get_data(da_order order,
                                                                 da_int bigger_ldx) {
    if (order == column_major) {
        if (bigger_ldx) {
            return {{0, 1, 2,  3, 4, 5,  6, 7, 8, 9, 10, 0, 1, 2,  3, 4, 5, 6, 7,
                     8, 9, 10, 0, 1, 2,  3, 4, 5, 6, 7,  8, 9, 10, 0, 1, 2, 3, 4,
                     5, 6, 7,  8, 9, 10, 0, 1, 2, 3, 4,  5, 6, 7,  8, 9, 10},
                    10,
                    5,
                    11};
        } else {
            return {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6,
                     7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3,
                     4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                    10,
                    5,
                    10};
        }

    } else {
        if (bigger_ldx) {
            return {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 3, 3,
                     3, 3, 3, 0, 4, 4, 4, 4, 4, 0, 5, 5, 5, 5, 5, 0, 6, 6, 6, 6,
                     6, 0, 7, 7, 7, 7, 7, 0, 8, 8, 8, 8, 8, 0, 9, 9, 9, 9, 9, 0},
                    10,
                    5,
                    6};
        } else {
            return {{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3,
                     3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6,
                     6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
                    10,
                    5,
                    5};
        }
    }
}

typedef struct params_t {
    std::string test_name;
    std::vector<double> expected_train;
    std::vector<double> expected_test;
    da_int shuffle;
    std::vector<da_int> shuffle_array;
    da_int train_size;
    da_int test_size;
    da_order order;
    da_int ldx_train;
    da_int ldx_test;
} params;

const params_t train_test_split_params[] = {
    {
        "tts_row_min_split",
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        0,
        {},
        1,
        1,
        row_major,
        5,
        5,
    },
    {
        "tts_row_train_bigger",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7},
        {8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        0,
        {},
        8,
        2,
        row_major,
        5,
        5,
    },
    {
        "tts_row_test_bigger",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        0,
        {},
        2,
        8,
        row_major,
        5,
        5,
    },
    {
        "tts_row_even_split",
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4},
        {5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9},
        0,
        {},
        5,
        5,
        row_major,
        5,
        5,
    },
    {
        "tts_col_min_split",
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        0,
        {},
        1,
        1,
        column_major,
        1,
        1,
    },
    {
        "tts_col_train_bigger",
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3,
         4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 8, 9, 8, 9, 8, 9, 8, 9},
        0,
        {},
        8,
        2,
        column_major,
        8,
        2,
    },
    {
        "tts_col_test_bigger",
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
        {2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5,
         6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9},
        0,
        {},
        2,
        8,
        column_major,
        2,
        8,
    },
    {
        "tts_col_even_split",
        {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4},
        {5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9},
        0,
        {},
        5,
        5,
        column_major,
        5,
        5,
    },
    {
        "tts_row_ldx_bigger",
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2,
         2, 2, 0, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 0},
        {5, 5, 5, 5, 5, 0, 6, 6, 6, 6, 6, 0, 7, 7, 7,
         7, 7, 0, 8, 8, 8, 8, 8, 0, 9, 9, 9, 9, 9, 0},
        0,
        {},
        5,
        5,
        row_major,
        6,
        6,
    },
    {
        "tts_col_ldx_bigger",
        {0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2,
         3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0},
        {5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9, 0, 5, 6, 7,
         8, 9, 0, 5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9, 0},
        0,
        {},
        5,
        5,
        column_major,
        6,
        6,
    },
    {
        "tts_row_shuffle_min_split",
        {3, 3, 3, 3, 3},
        {7, 7, 7, 7, 7},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        1,
        1,
        row_major,
        5,
        5,
    },
    {
        "tts_row_shuffle_train_bigger",
        {3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9,
         0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6},
        {2, 2, 2, 2, 2, 5, 5, 5, 5, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        8,
        2,
        row_major,
        5,
        5,
    },
    {
        "tts_row_shuffle_test_bigger",
        {3, 3, 3, 3, 3, 7, 7, 7, 7, 7},
        {1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8,
         4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        2,
        8,
        row_major,
        5,
        5,
    },
    {
        "tts_row_shuffle_even_split",
        {3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0},
        {8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        5,
        5,
        row_major,
        5,
        5,
    },
    {
        "tts_col_shuffle_min_split",
        {3, 3, 3, 3, 3},
        {7, 7, 7, 7, 7},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        1,
        1,
        column_major,
        1,
        1,
    },
    {
        "tts_col_shuffle_train_bigger",
        {3, 7, 1, 9, 0, 8, 4, 6, 3, 7, 1, 9, 0, 8, 4, 6, 3, 7, 1, 9,
         0, 8, 4, 6, 3, 7, 1, 9, 0, 8, 4, 6, 3, 7, 1, 9, 0, 8, 4, 6},
        {2, 5, 2, 5, 2, 5, 2, 5, 2, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        8,
        2,
        column_major,
        8,
        2,
    },
    {
        "tts_col_shuffle_test_bigger",
        {3, 7, 3, 7, 3, 7, 3, 7, 3, 7},
        {1, 9, 0, 8, 4, 6, 2, 5, 1, 9, 0, 8, 4, 6, 2, 5, 1, 9, 0, 8,
         4, 6, 2, 5, 1, 9, 0, 8, 4, 6, 2, 5, 1, 9, 0, 8, 4, 6, 2, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        2,
        8,
        column_major,
        2,
        8,
    },
    {
        "tts_col_shuffle_even_split",
        {3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0, 3, 7, 1, 9, 0},
        {8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5, 8, 4, 6, 2, 5},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        5,
        5,
        column_major,
        5,
        5,
    },
    {
        "tts_row_shuffle_ldxs_bigger",
        {3, 3, 3, 3, 3, 0, 7, 7, 7, 7, 7, 0, 1, 1, 1,
         1, 1, 0, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0},
        {8, 8, 8, 8, 8, 0, 4, 4, 4, 4, 4, 0, 6, 6, 6,
         6, 6, 0, 2, 2, 2, 2, 2, 0, 5, 5, 5, 5, 5, 0},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        5,
        5,
        row_major,
        6,
        6,
    },
    {
        "tts_col_shuffle_ldxs_bigger",
        {3, 7, 1, 9, 0, 0, 3, 7, 1, 9, 0, 0, 3, 7, 1,
         9, 0, 0, 3, 7, 1, 9, 0, 0, 3, 7, 1, 9, 0, 0},
        {8, 4, 6, 2, 5, 0, 8, 4, 6, 2, 5, 0, 8, 4, 6,
         2, 5, 0, 8, 4, 6, 2, 5, 0, 8, 4, 6, 2, 5, 0},
        1,
        {3, 7, 1, 9, 0, 8, 4, 6, 2, 5},
        5,
        5,
        column_major,
        6,
        6,
    }

};

const params_t train_test_split_params_validation[] = {
    {
        "tts_test_validation_row",
        {},
        {},
        0,
        {},
        3,
        2,
        row_major,
        5,
        5,
    },
    {
        "tts_test_validation_col",
        {},
        {},
        0,
        {},
        3,
        2,
        column_major,
        3,
        2,
    },
};