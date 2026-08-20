/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "gtest/gtest.h"

/*
 * Tests for utility functions (aoclda_utils.h)
 */
TEST(UtilsCAPI, CheckDataDouble) {
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_EQ(da_check_data_d(column_major, 3, 2, X, 3), da_status_success);
}

TEST(UtilsCAPI, CheckDataFloat) {
    float X[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    EXPECT_EQ(da_check_data_s(column_major, 3, 2, X, 3), da_status_success);
}

TEST(UtilsCAPI, SwitchOrderCopyDouble) {
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double Y[6];
    EXPECT_EQ(da_switch_order_copy_d(column_major, 3, 2, X, 3, Y, 2), da_status_success);
}

TEST(UtilsCAPI, SwitchOrderCopyFloat) {
    float X[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float Y[6];
    EXPECT_EQ(da_switch_order_copy_s(column_major, 3, 2, X, 3, Y, 2), da_status_success);
}

TEST(UtilsCAPI, SwitchOrderInPlaceDouble) {
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_EQ(da_switch_order_in_place_d(column_major, 2, 3, X, 2, 3), da_status_success);
}

TEST(UtilsCAPI, SwitchOrderInPlaceFloat) {
    float X[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    EXPECT_EQ(da_switch_order_in_place_s(column_major, 2, 3, X, 2, 3), da_status_success);
}

TEST(UtilsCAPI, TrainTestSplitDouble) {
    double X[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    da_int m = 6, n = 2, ldx = 6;
    da_int train_size = 4, test_size = 2;
    da_int shuffle_array[6] = {0, 1, 2, 3, 4, 5};

    double X_train[8];
    double X_test[4];
    EXPECT_EQ(da_train_test_split_d(column_major, m, n, X, ldx, train_size, test_size,
                                    shuffle_array, X_train, train_size, X_test,
                                    test_size),
              da_status_success);
}

TEST(UtilsCAPI, TrainTestSplitFloat) {
    float X[12] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    da_int m = 6, n = 2;
    da_int train_size = 4, test_size = 2;
    da_int shuffle_array[6] = {0, 1, 2, 3, 4, 5};

    float X_train[8], X_test[4];
    EXPECT_EQ(da_train_test_split_s(column_major, m, n, X, m, train_size, test_size,
                                    shuffle_array, X_train, train_size, X_test,
                                    test_size),
              da_status_success);
}

TEST(UtilsCAPI, TrainTestSplitInt) {
    da_int X[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    da_int m = 6, n = 2;
    da_int train_size = 4, test_size = 2;
    da_int shuffle_array[6] = {0, 1, 2, 3, 4, 5};

    da_int X_train[8], X_test[4];
    EXPECT_EQ(da_train_test_split_int(column_major, m, n, X, m, train_size, test_size,
                                      shuffle_array, X_train, train_size, X_test,
                                      test_size),
              da_status_success);
}

TEST(UtilsCAPI, GetShuffledIndicesDouble) {
    da_int m = 6, train_size = 4, test_size = 2, seed = 42;
    double classes[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    da_int shuffle_array[6];

    EXPECT_EQ(da_get_shuffled_indices_d(m, seed, train_size, test_size, 10, classes,
                                        shuffle_array),
              da_status_success);
}

TEST(UtilsCAPI, GetShuffledIndicesFloat) {
    da_int m = 6, train_size = 4, test_size = 2, seed = 42;
    float classes[6] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    da_int shuffle_array[6];

    EXPECT_EQ(da_get_shuffled_indices_s(m, seed, train_size, test_size, 10, classes,
                                        shuffle_array),
              da_status_success);
}

TEST(UtilsCAPI, GetShuffledIndicesInt) {
    da_int m = 6, train_size = 4, test_size = 2, seed = 42;
    da_int classes[6] = {0, 0, 0, 1, 1, 1};
    da_int shuffle_array[6];

    EXPECT_EQ(da_get_shuffled_indices_int(m, seed, train_size, test_size, 0, classes,
                                          shuffle_array),
              da_status_success);
}

TEST(UtilsCAPI, GetArchInfo) {
    da_int len = 100;
    char arch[100], ns[100];
    EXPECT_EQ(da_get_arch_info(&len, arch, ns), da_status_success);
    EXPECT_GT(len, 0);
}

TEST(UtilsCAPI, GetIntInfo) {
    size_t len = 100;
    char int_type[100];
    EXPECT_EQ(da_get_int_info(&len, int_type), da_status_success);
}

TEST(UtilsCAPI, DebugSetGet) {
    EXPECT_EQ(da_debug_set("test.key", "test_value"), da_status_success);

    char value[100];
    EXPECT_EQ(da_debug_get("test.key", 100, value), da_status_success);
}
