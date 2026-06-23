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
#include <cstdint>
#include <cstdlib>
#include <cstring>

/*
 * Tests for CSV APIs (aoclda_csv.h)
 */
TEST(CsvCAPI, ReadDouble) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/csv_test_float.csv");

    double *A = nullptr;
    da_int n_rows = 0, n_cols = 0;
    char **headings = nullptr;

    EXPECT_EQ(da_read_csv_d(store, filepath, &A, &n_rows, &n_cols, &headings),
              da_status_success);
    EXPECT_EQ(n_rows, 3);
    EXPECT_EQ(n_cols, 5);

    if (A)
        free(A);
    if (headings)
        da_delete_string_array(&headings, n_cols);
    da_datastore_destroy(&store);
}

TEST(CsvCAPI, ReadFloat) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/csv_test_float.csv");

    float *A = nullptr;
    da_int n_rows = 0, n_cols = 0;
    char **headings = nullptr;

    EXPECT_EQ(da_read_csv_s(store, filepath, &A, &n_rows, &n_cols, &headings),
              da_status_success);
    EXPECT_EQ(n_rows, 3);
    EXPECT_EQ(n_cols, 5);

    if (A)
        free(A);
    if (headings)
        da_delete_string_array(&headings, n_cols);
    da_datastore_destroy(&store);
}

TEST(CsvCAPI, ReadInt) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);

    char filepath[256] = DATA_DIR;
#ifdef AOCLDA_ILP64
    strcat(filepath, "csv_data/csv_test_int64_head.csv");
#else
    strcat(filepath, "csv_data/csv_test_int32_head.csv");
#endif

    da_int *A = nullptr;
    da_int n_rows = 0, n_cols = 0;
    char **headings = nullptr;

    EXPECT_EQ(da_read_csv_int(store, filepath, &A, &n_rows, &n_cols, &headings),
              da_status_success);

    if (A)
        free(A);
    if (headings)
        da_delete_string_array(&headings, n_cols);
    da_datastore_destroy(&store);
}

TEST(CsvCAPI, ReadUint8) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/csv_test_bool.csv");

    uint8_t *A = nullptr;
    da_int n_rows = 0, n_cols = 0;
    char **headings = nullptr;

    EXPECT_EQ(da_read_csv_uint8(store, filepath, &A, &n_rows, &n_cols, &headings),
              da_status_success);
    EXPECT_EQ(n_rows, 2);
    EXPECT_EQ(n_cols, 4);

    if (A)
        free(A);
    if (headings)
        da_delete_string_array(&headings, n_cols);
    da_datastore_destroy(&store);
}

TEST(CsvCAPI, ReadString) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/csv_test_char.csv");

    char **A = nullptr;
    da_int n_rows = 0, n_cols = 0;
    char **headings = nullptr;

    EXPECT_EQ(da_read_csv_string(store, filepath, &A, &n_rows, &n_cols, &headings),
              da_status_success);
    EXPECT_EQ(n_rows, 3);
    EXPECT_EQ(n_cols, 4);

    if (A)
        EXPECT_EQ(da_delete_string_array(&A, n_rows * n_cols), da_status_success);
    if (headings)
        da_delete_string_array(&headings, n_cols);
    da_datastore_destroy(&store);
}

TEST(CsvCAPI, DeleteStringArray) {
    // da_delete_string_array with nullptr should be safe
    char **S = nullptr;
    EXPECT_EQ(da_delete_string_array(&S, 2), da_status_success);
}
