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
#include "aoclda.hpp"
#include "gtest/gtest.h"
#include <cstdint>
#include <cstring>

/*
 * Tests for datastore APIs (aoclda_datastore.h)
 */
TEST(DatastoreCAPI, InitDestroy) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_NE(store, nullptr);

    EXPECT_EQ(da_datastore_print_error_message(store), da_status_success);

    da_datastore_destroy(&store);
    EXPECT_EQ(store, nullptr);
}

TEST(DatastoreCAPI, PrintOptions) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    EXPECT_EQ(da_data_print_options(store), da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadColInt) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int block[6] = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(da_data_load_col_int(store, 3, 2, block, column_major, 1),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadColRealD) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    double block[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_EQ(da_data_load_col_real_d(store, 3, 2, block, column_major, 1),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadColRealS) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    float block[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    EXPECT_EQ(da_data_load_col_real_s(store, 3, 2, block, column_major, 1),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadColUint8) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    uint8_t block[6] = {0, 1, 1, 0, 1, 0};
    EXPECT_EQ(da_data_load_col_uint8(store, 3, 2, block, column_major, 1),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadColStr) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    const char *block[6] = {"a", "b", "c", "d", "e", "f"};
    EXPECT_EQ(da_data_load_col_str(store, 3, 2, block, column_major), da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadRowInt) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int row_block[4] = {7, 8, 9, 10};
    EXPECT_EQ(da_data_load_row_int(store, 2, 2, row_block, column_major, 1),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadRowRealD) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    double row_block[4] = {7.0, 8.0, 9.0, 10.0};
    EXPECT_EQ(da_data_load_row_real_d(store, 2, 2, row_block, column_major, 1),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadRowRealS) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    float row_block[4] = {7.0f, 8.0f, 9.0f, 10.0f};
    EXPECT_EQ(da_data_load_row_real_s(store, 2, 2, row_block, column_major, 1),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadRowUint8) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    uint8_t row_block[4] = {1, 0, 1, 1};
    EXPECT_EQ(da_data_load_row_uint8(store, 2, 2, row_block, column_major, 1),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadRowStr) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    const char *row_block[4] = {"g", "h", "i", "j"};
    EXPECT_EQ(da_data_load_row_str(store, 2, 2, row_block, column_major),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LoadFromCsv) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "datastore precision", "double"),
              da_status_success);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/csv_test_float.csv");

    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_success);

    da_int n_rows = 0, n_cols = 0;
    EXPECT_EQ(da_data_get_n_rows(store, &n_rows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &n_cols), da_status_success);
    EXPECT_EQ(n_rows, 3);
    EXPECT_EQ(n_cols, 5);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, GetNRowsNCols) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int block[6] = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(da_data_load_col_int(store, 3, 2, block, column_major, 1),
              da_status_success);

    da_int n_rows = 0, n_cols = 0;
    EXPECT_EQ(da_data_get_n_rows(store, &n_rows), da_status_success);
    EXPECT_EQ(n_rows, 3);
    EXPECT_EQ(da_data_get_n_cols(store, &n_cols), da_status_success);
    EXPECT_EQ(n_cols, 2);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, GetSetElementInt) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int block[6] = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(da_data_load_col_int(store, 3, 2, block, column_major, 1),
              da_status_success);

    da_int elem = 0;
    EXPECT_EQ(da_data_get_element_int(store, 0, 0, &elem), da_status_success);
    EXPECT_EQ(elem, 1);

    EXPECT_EQ(da_data_set_element_int(store, 0, 0, 99), da_status_success);
    EXPECT_EQ(da_data_get_element_int(store, 0, 0, &elem), da_status_success);
    EXPECT_EQ(elem, 99);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, GetSetElementRealD) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    double block[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_EQ(da_data_load_col_real_d(store, 3, 2, block, column_major, 1),
              da_status_success);

    double elem = 0.0;
    EXPECT_EQ(da_data_get_element_real_d(store, 0, 0, &elem), da_status_success);
    EXPECT_EQ(elem, 1.0);

    EXPECT_EQ(da_data_set_element_real_d(store, 0, 0, 99.0), da_status_success);
    EXPECT_EQ(da_data_get_element_real_d(store, 0, 0, &elem), da_status_success);
    EXPECT_EQ(elem, 99.0);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, GetSetElementRealS) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    float block[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    EXPECT_EQ(da_data_load_col_real_s(store, 3, 2, block, column_major, 1),
              da_status_success);

    float elem = 0.0f;
    EXPECT_EQ(da_data_get_element_real_s(store, 0, 0, &elem), da_status_success);
    EXPECT_EQ(elem, 1.0f);

    EXPECT_EQ(da_data_set_element_real_s(store, 0, 0, 99.0f), da_status_success);
    EXPECT_EQ(da_data_get_element_real_s(store, 0, 0, &elem), da_status_success);
    EXPECT_EQ(elem, 99.0f);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, GetSetElementUint8) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    uint8_t block[6] = {0, 1, 1, 0, 1, 0};
    EXPECT_EQ(da_data_load_col_uint8(store, 3, 2, block, column_major, 1),
              da_status_success);

    uint8_t elem = 0;
    EXPECT_EQ(da_data_get_element_uint8(store, 0, 0, &elem), da_status_success);
    EXPECT_EQ(elem, 0);

    EXPECT_EQ(da_data_set_element_uint8(store, 0, 0, 1), da_status_success);
    EXPECT_EQ(da_data_get_element_uint8(store, 0, 0, &elem), da_status_success);
    EXPECT_EQ(elem, 1);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, LabelColumn) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int block[6] = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(da_data_load_col_int(store, 3, 2, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_label_column(store, "col_A", 0), da_status_success);
    EXPECT_EQ(da_data_label_column(store, "col_B", 1), da_status_success);

    da_int col_idx = -1;
    EXPECT_EQ(da_data_get_col_idx(store, "col_A", &col_idx), da_status_success);
    EXPECT_EQ(col_idx, 0);

    da_int label_sz = 100;
    char label[100];
    EXPECT_EQ(da_data_get_col_label(store, 0, &label_sz, label), da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, ExtractColumnInt) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int int_block[6] = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(da_data_load_col_int(store, 3, 2, int_block, column_major, 1),
              da_status_success);

    da_int col[3];
    EXPECT_EQ(da_data_extract_column_int(store, 0, 3, col), da_status_success);
    EXPECT_EQ(col[0], 1);
    EXPECT_EQ(col[1], 2);
    EXPECT_EQ(col[2], 3);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, ExtractColumnRealD) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    double block[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_EQ(da_data_load_col_real_d(store, 3, 2, block, column_major, 1),
              da_status_success);

    double col[3];
    EXPECT_EQ(da_data_extract_column_real_d(store, 0, 3, col), da_status_success);
    EXPECT_EQ(col[0], 1.0);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, ExtractColumnRealS) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    float block[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    EXPECT_EQ(da_data_load_col_real_s(store, 3, 2, block, column_major, 1),
              da_status_success);

    float col[3];
    EXPECT_EQ(da_data_extract_column_real_s(store, 0, 3, col), da_status_success);
    EXPECT_EQ(col[0], 1.0f);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, ExtractColumnUint8) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    uint8_t block[6] = {2, 1, 1, 0, 1, 0};
    EXPECT_EQ(da_data_load_col_uint8(store, 3, 2, block, column_major, 1),
              da_status_success);

    uint8_t col[3];
    EXPECT_EQ(da_data_extract_column_uint8(store, 0, 3, col), da_status_success);
    EXPECT_EQ(col[0], 2);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, ExtractColumnStr) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    const char *block[6] = {"a", "b", "c", "d", "e", "f"};
    EXPECT_EQ(da_data_load_col(store, 3, 2, block, column_major), da_status_success);

    char *col[3];
    EXPECT_EQ(da_data_extract_column(store, 0, 3, col), da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, SelectColumns) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int block[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(da_data_load_col_int(store, 3, 3, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_select_columns(store, "sel1", 0, 1), da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, SelectRows) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int block[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(da_data_load_col_int(store, 3, 3, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_select_rows(store, "sel1", 0, 1), da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, SelectSlice) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int block[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(da_data_load_col_int(store, 3, 3, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_select_slice(store, "sel1", 0, 1, 0, 1), da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, SelectNonMissing) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    double block[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_EQ(da_data_load_col_real_d(store, 3, 2, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_select_columns(store, "sel1", 0, 1), da_status_success);
    EXPECT_EQ(da_data_select_non_missing(store, "sel1", 1), da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, SelectRemoveColumns) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int block[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    EXPECT_EQ(da_data_load_col_int(store, 3, 4, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_select_columns(store, "sel1", 0, 3), da_status_success);
    EXPECT_EQ(da_data_select_remove_columns(store, "sel1", 2, 3), da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, SelectRemoveRows) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int block[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    EXPECT_EQ(da_data_load_col_int(store, 4, 3, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_select_rows(store, "sel1", 0, 3), da_status_success);
    EXPECT_EQ(da_data_select_remove_rows(store, "sel1", 2, 3), da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, ExtractSelectionInt) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int block[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(da_data_load_col_int(store, 3, 3, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_select_slice(store, "sel1", 0, 1, 0, 1), da_status_success);

    da_int data[4];
    EXPECT_EQ(da_data_extract_selection_int(store, "sel1", column_major, data, 2),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, ExtractSelectionRealD) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    double block[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    EXPECT_EQ(da_data_load_col_real_d(store, 3, 3, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_select_slice(store, "sel1", 0, 1, 0, 1), da_status_success);

    double data[4];
    EXPECT_EQ(da_data_extract_selection_real_d(store, "sel1", column_major, data, 2),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, ExtractSelectionRealS) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    float block[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    EXPECT_EQ(da_data_load_col_real_s(store, 3, 3, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_select_slice(store, "sel1", 0, 1, 0, 1), da_status_success);

    float data[4];
    EXPECT_EQ(da_data_extract_selection_real_s(store, "sel1", column_major, data, 2),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, ExtractSelectionUint8) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    uint8_t block[9] = {0, 1, 1, 0, 1, 0, 1, 1, 0};
    EXPECT_EQ(da_data_load_col_uint8(store, 3, 3, block, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_select_slice(store, "sel1", 0, 1, 0, 1), da_status_success);

    uint8_t data[4];
    EXPECT_EQ(da_data_extract_selection_uint8(store, "sel1", column_major, data, 2),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(DatastoreCAPI, HConcat) {
    da_datastore store1 = nullptr, store2 = nullptr;
    EXPECT_EQ(da_datastore_init(&store1), da_status_success);
    EXPECT_EQ(da_datastore_init(&store2), da_status_success);

    da_int block1[6] = {1, 2, 3, 4, 5, 6};
    da_int block2[3] = {7, 8, 9};
    EXPECT_EQ(da_data_load_col_int(store1, 3, 2, block1, column_major, 1),
              da_status_success);
    EXPECT_EQ(da_data_load_col_int(store2, 3, 1, block2, column_major, 1),
              da_status_success);

    EXPECT_EQ(da_data_hconcat(&store1, &store2), da_status_success);

    da_int n_cols = 0;
    EXPECT_EQ(da_data_get_n_cols(store1, &n_cols), da_status_success);
    EXPECT_EQ(n_cols, 3);

    da_datastore_destroy(&store1);
    da_datastore_destroy(&store2);
}
