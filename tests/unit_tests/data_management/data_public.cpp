/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

void get_heterogeneous_data_store_pub(da_datastore store, da_int &mt, da_int &nt,
                                      std::vector<da_int> &idata,
                                      std::vector<float> &fdata,
                                      std::vector<std::string> &sdata) {
    /* create a data_store in ds with heterogeneous data
     * Dimensions: 6 x 7
     *   ------   ------    ------   ------
     *  | int  | | int  |  |float | | str  |
     *  | 4x2  | | 4x2  |  | 5x2  | | 5x1  |
     *   ------   ------   |      | |      |
     *   ------   ------   |      | |      |
     *  | 1x2  | | 1x2  |  |      | |      |
     *   ------   ------    ------   ------
     *   ---------------    ------   ------
     *  |     1x4       |  | 1x2  | | 1x1  |
     *   ---------------    ------   ------
     */
    std::vector<da_int> ib1, ib2, ib3, ib4, ib5;
    std::vector<float> fb1, fb2;
    da_int m, n;

    ib1 = {1, 2, 3, 4, 5, 6, 7, 8};
    m = 4;
    n = 2;
    EXPECT_EQ(da_data_load_col_int(store, m, n, ib1.data(), row_major, true),
              da_status_success);
    ib2 = {1, 2, 3, 4, 5, 6, 7, 8};
    m = 4;
    n = 2;
    EXPECT_EQ(da_data_load_col_int(store, m, n, ib2.data(), column_major, true),
              da_status_success);
    ib3 = {10, 11};
    m = 1;
    n = 2;
    EXPECT_EQ(da_data_load_row_int(store, m, n, ib3.data(), column_major, true),
              da_status_success);
    ib4 = {12, 13};
    m = 1;
    n = 2;
    EXPECT_EQ(da_data_load_row_int(store, m, n, ib4.data(), column_major, true),
              da_status_success);
    fb1 = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
    m = 5;
    n = 2;
    EXPECT_EQ(da_data_load_col_real_s(store, m, n, fb1.data(), column_major, true),
              da_status_success);
    const char *cb1[5];
    cb1[0] = "1";
    cb1[1] = "a2";
    cb1[2] = "bb3";
    cb1[3] = "ccc4";
    cb1[4] = "dddd5";
    m = 5;
    n = 1;
    EXPECT_EQ(da_data_load_col_str(store, m, n, cb1, column_major), da_status_success);
    ib5 = {21, 22, 23, 24};
    m = 1;
    n = 4;
    EXPECT_EQ(da_data_load_row_int(store, m, n, ib5.data(), row_major, true),
              da_status_success);
    fb2 = {10.1f, 20.2f};
    m = 1;
    n = 2;
    EXPECT_EQ(da_data_load_row_real_s(store, m, n, fb2.data(), row_major, true),
              da_status_success);
    const char *cb2[1];
    cb2[0] = "row6_1";
    m = 1;
    n = 1;
    EXPECT_EQ(da_data_load_row_str(store, m, n, cb2, row_major), da_status_success);

    // expected blocks, column major ordering
    idata = {1, 3, 5, 7, 10, 21, 2, 4, 6, 8, 11, 22,
             1, 2, 3, 4, 12, 23, 5, 6, 7, 8, 13, 24};
    fdata = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 10.1f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 20.2f};
    sdata = {"1", "a2", "bb3", "ccc4", "dddd5", "row6_1"};
    mt = 6;
    nt = 7;
}

void get_transition_datastore(da_datastore &store) {
    /* Create a datastore with partially added rows
     *  ------   -------
     * |  int | |  dbl  |
     * |  2x4 | |  2x4  |
     *  ------   -------
     *  ------
     * |  1x4 |   [empty]
     *  ------
     */
    std::vector<da_int> ib1, ib2;
    std::vector<double> db1;

    ib1 = {1, 2, 3, 4, 5, 6, 7, 8};
    ib2 = {1, 2, 3, 4};
    db1 = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
    da_int m = 2, n = 4;
    EXPECT_EQ(da_data_load_col_int(store, m, n, ib1.data(), row_major, true),
              da_status_success);
    EXPECT_EQ(da_data_load_col_real_d(store, m, n, db1.data(), row_major, true),
              da_status_success);
    m = 1;
    EXPECT_EQ(da_data_load_row_int(store, m, n, ib2.data(), row_major, true),
              da_status_success);
}

TEST(datastore, getSetElementPub) {
    da_datastore store = nullptr;
    da_datastore_init(&store);
    da_int m, n;
    std::vector<da_int> idata;
    std::vector<float> fdata;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store_pub(store, m, n, idata, fdata, sdata);

    // add a uint8 column
    uint8_t ui_block[6] = {0, 1, 0, 0, 1, 1};
    EXPECT_EQ(da_data_load_col_uint8(store, 6, 1, ui_block, column_major, 0),
              da_status_success);

    // setters
    EXPECT_EQ(da_data_set_element_int(store, 0, 0, 100), da_status_success);
    EXPECT_EQ(da_data_set_element_uint8(store, 0, 7, 2), da_status_success);
    EXPECT_EQ(da_data_set_element_real_d(store, 2, 4, 100.0), da_status_invalid_input);
    EXPECT_EQ(da_data_set_element_real_s(store, 2, 5, 200.0), da_status_success);

    // getters
    da_int iel;
    EXPECT_EQ(da_data_get_element_int(store, 0, 0, &iel), da_status_success);
    EXPECT_EQ(iel, 100);
    uint8_t uiel;
    EXPECT_EQ(da_data_get_element_uint8(store, 0, 7, &uiel), da_status_success);
    EXPECT_EQ(uiel, 2);
    double del;
    EXPECT_EQ(da_data_get_element_real_d(store, 2, 4, &del), da_status_invalid_input);
    float sel;
    EXPECT_EQ(da_data_get_element_real_s(store, 2, 5, &sel), da_status_success);
    EXPECT_EQ(sel, 200.0);

    da_datastore_destroy(&store);
}

TEST(datastore, invalidLoad) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    EXPECT_EQ(da_data_load_col_int(store, 1, 1, nullptr, column_major, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_col_str(store, 1, 1, nullptr, column_major),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_col_real_d(store, 1, 1, nullptr, column_major, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_col_real_s(store, 1, 1, nullptr, column_major, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_col_uint8(store, 1, 1, nullptr, column_major, 1),
              da_status_invalid_input);

    EXPECT_EQ(da_data_load_row_int(store, 1, 1, nullptr, column_major, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_row_str(store, 1, 1, nullptr, column_major),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_row_real_d(store, 1, 1, nullptr, column_major, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_row_real_s(store, 1, 1, nullptr, column_major, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_row_uint8(store, 1, 1, nullptr, column_major, 1),
              da_status_invalid_input);

    da_datastore_destroy(&store);
}

TEST(dataStore, load) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int m, n, copy_data = 0;
    da_order order = row_major;
    std::vector<da_int> intc_bl, intr_bl;
    m = 2;
    n = 3;
    intc_bl = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(da_data_load_col_int(store, m, n, intc_bl.data(), order, copy_data),
              da_status_success);
    m = 1;
    n = 3;
    intr_bl = {1, 2, 3};
    EXPECT_EQ(da_data_load_row_int(store, m, n, intr_bl.data(), order, copy_data),
              da_status_success);
    order = column_major;
    m = 3;
    n = 1;
    const char *char_bl[3];
    char_bl[0] = "test1";
    char_bl[1] = "bla";
    char_bl[2] = "123";
    EXPECT_EQ(da_data_load_col_str(store, m, n, char_bl, order), da_status_success);
    std::vector<float> sreal_bl;
    m = 3;
    n = 2;
    sreal_bl = {1., 2., 3., 4., 5., 6.};
    copy_data = true;
    EXPECT_EQ(da_data_load_col_real_s(store, m, n, sreal_bl.data(), order, copy_data),
              da_status_success);
    std::vector<double> dreal_bl;
    m = 3;
    n = 1;
    dreal_bl = {4., 5., 6.};
    EXPECT_EQ(da_data_load_col_real_d(store, m, n, dreal_bl.data(), order, copy_data),
              da_status_success);
    std::vector<uint8_t> ui_bl;
    m = 3;
    n = 1;
    ui_bl = {0, 1, 1};
    EXPECT_EQ(da_data_load_col_uint8(store, m, n, ui_bl.data(), order, copy_data),
              da_status_success);
    da_datastore_destroy(&store);

    // Test row insertions for other data types
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    m = 1;
    n = 3;
    order = row_major;
    EXPECT_EQ(da_data_load_row_str(store, m, n, char_bl, order), da_status_success);
    EXPECT_EQ(da_data_load_row_str(store, m, n, char_bl, order), da_status_success);
    EXPECT_EQ(da_data_load_row_str(store, m, n, char_bl, order), da_status_success);
    da_datastore_destroy(&store);

    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    m = 2;
    n = 2;
    dreal_bl = {4., 5., 6., 7.};
    copy_data = true;
    order = row_major;
    EXPECT_EQ(da_data_load_row_real_d(store, m, n, dreal_bl.data(), order, copy_data),
              da_status_success);
    EXPECT_EQ(da_data_load_row_real_d(store, m, n, dreal_bl.data(), order, copy_data),
              da_status_success);
    EXPECT_EQ(da_data_load_row_real_d(store, m, n, dreal_bl.data(), order, copy_data),
              da_status_success);
    da_datastore_destroy(&store);

    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    m = 2;
    n = 2;
    dreal_bl = {4., 5., 6., 7.};
    copy_data = true;
    order = row_major;
    EXPECT_EQ(da_data_load_row_real_d(store, m, n, dreal_bl.data(), order, copy_data),
              da_status_success);
    EXPECT_EQ(da_data_load_row_real_d(store, m, n, dreal_bl.data(), order, copy_data),
              da_status_success);
    EXPECT_EQ(da_data_load_row_real_d(store, m, n, dreal_bl.data(), order, copy_data),
              da_status_success);
    da_datastore_destroy(&store);

    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    m = 2;
    n = 2;
    ui_bl = {0, 1, 0, 1};
    copy_data = true;
    order = row_major;
    EXPECT_EQ(da_data_load_row_uint8(store, m, n, ui_bl.data(), order, copy_data),
              da_status_success);
    EXPECT_EQ(da_data_load_row_uint8(store, m, n, ui_bl.data(), order, copy_data),
              da_status_success);
    EXPECT_EQ(da_data_load_row_uint8(store, m, n, ui_bl.data(), order, copy_data),
              da_status_success);
    da_datastore_destroy(&store);
}

TEST(dataStore, hconcatPub) {
    da_datastore store = nullptr, store1 = nullptr, store2 = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_init(&store1), da_status_success);
    EXPECT_EQ(da_datastore_init(&store2), da_status_success);

    // load the heterogeneous data store in 3 different stores
    da_int m, n;
    std::vector<da_int> idata, coli, coli2, coli3;
    std::vector<float> fdata, colf, colf2, colf3;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store_pub(store, m, n, idata, fdata, sdata);
    get_heterogeneous_data_store_pub(store1, m, n, idata, fdata, sdata);
    get_heterogeneous_data_store_pub(store2, m, n, idata, fdata, sdata);

    // add 2 columns to store1
    std::vector<double> dblock = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    m = 6;
    n = 2;
    EXPECT_EQ(da_data_load_col_real_d(store1, m, n, dblock.data(), column_major, true),
              da_status_success);

    // concatenate [store, store1, store2] into store
    EXPECT_EQ(da_data_hconcat(&store1, &store2), da_status_success);
    EXPECT_EQ(store2, nullptr);
    EXPECT_EQ(da_data_hconcat(&store, &store1), da_status_success);
    EXPECT_EQ(store1, nullptr);

    // Check the integer columns
    da_int startx = 0, starty = 0;
    coli.resize(m);
    coli2.resize(m);
    coli3.resize(m);
    for (da_int col = 0; col < 4; col++) {
        starty = col * m;
        EXPECT_EQ(da_data_extract_column_int(store, col, m, coli.data()),
                  da_status_success);
        EXPECT_EQ(da_data_extract_column_int(store, col + 7, m, coli2.data()),
                  da_status_success);
        EXPECT_EQ(da_data_extract_column_int(store, col + 16, m, coli3.data()),
                  da_status_success);
        EXPECT_ARR_EQ(m, coli, idata, 1, 1, startx, starty);
        EXPECT_ARR_EQ(m, coli2, idata, 1, 1, startx, starty);
        EXPECT_ARR_EQ(m, coli3, idata, 1, 1, startx, starty);
    }

    // Check the integer columns
    colf.resize(m);
    colf2.resize(m);
    colf3.resize(m);
    for (da_int col = 4; col < 6; col++) {
        starty = (col - 4) * m;
        EXPECT_EQ(da_data_extract_column_real_s(store, col, m, colf.data()),
                  da_status_success);
        EXPECT_EQ(da_data_extract_column_real_s(store, col + 7, m, colf2.data()),
                  da_status_success);
        EXPECT_EQ(da_data_extract_column_real_s(store, col + 16, m, colf3.data()),
                  da_status_success);
        EXPECT_ARR_EQ(m, colf, fdata, 1, 1, startx, starty);
        EXPECT_ARR_EQ(m, colf2, fdata, 1, 1, startx, starty);
        EXPECT_ARR_EQ(m, colf3, fdata, 1, 1, startx, starty);
    }

    // check the 2 double columns added to store1
    std::vector<double> cold, cold1;
    cold.resize(m);
    cold1.resize(m);
    EXPECT_EQ(da_data_extract_column_real_d(store, 14, m, cold.data()),
              da_status_success);
    EXPECT_EQ(da_data_extract_column_real_d(store, 15, m, cold1.data()),
              da_status_success);
    EXPECT_ARR_EQ(m, cold, dblock, 1, 1, 0, 0);
    EXPECT_ARR_EQ(m, cold1, dblock, 1, 1, 0, 6);

    da_datastore_destroy(&store);
    da_datastore_destroy(&store1);
    da_datastore_destroy(&store2);
}

TEST(datastore, nullArguments) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    // Load
    EXPECT_EQ(da_data_load_from_csv(store, nullptr), da_status_invalid_input);

    // Select
    EXPECT_EQ(da_data_select_columns(store, nullptr, 0, 0), da_status_invalid_input);
    EXPECT_EQ(da_data_select_rows(store, nullptr, 0, 0), da_status_invalid_input);
    EXPECT_EQ(da_data_select_slice(store, nullptr, 0, 0, 0, 0), da_status_invalid_input);
    EXPECT_EQ(da_data_select_non_missing(store, nullptr, 0), da_status_invalid_input);
    EXPECT_EQ(da_data_select_remove_columns(store, nullptr, 0, 0),
              da_status_invalid_input);
    EXPECT_EQ(da_data_select_remove_rows(store, nullptr, 0, 0), da_status_invalid_input);

    // Extract columns
    EXPECT_EQ(da_data_extract_column_int(store, 0, 0, nullptr), da_status_invalid_input);
    EXPECT_EQ(da_data_extract_column_real_s(store, 0, 0, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_column_real_d(store, 0, 0, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_column_uint8(store, 0, 0, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_column_str(store, 0, 0, nullptr), da_status_invalid_input);

    // Extract selection
    EXPECT_EQ(da_data_extract_selection_int(store, "A", column_major, nullptr, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_selection_real_d(store, "A", column_major, nullptr, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_selection_real_s(store, "A", column_major, nullptr, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_selection_uint8(store, "A", column_major, nullptr, 1),
              da_status_invalid_input);
    da_int i;
    float f;
    double d;
    uint8_t ui;
    EXPECT_EQ(da_data_extract_selection_int(store, nullptr, column_major, &i, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_selection_real_d(store, nullptr, column_major, &d, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_selection_real_s(store, nullptr, column_major, &f, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_selection_uint8(store, nullptr, column_major, &ui, 1),
              da_status_invalid_input);

    // Label
    EXPECT_EQ(da_data_label_column(store, nullptr, 1), da_status_invalid_input);
    da_int col_idx;
    EXPECT_EQ(da_data_get_col_idx(store, nullptr, &col_idx), da_status_invalid_input);
    EXPECT_EQ(da_data_get_col_idx(store, "A", nullptr), da_status_invalid_input);
    da_int label_sz = 1;
    EXPECT_EQ(da_data_get_col_label(store, 0, &label_sz, nullptr),
              da_status_invalid_input);
    char label[2] = "A";
    EXPECT_EQ(da_data_get_col_label(store, 0, nullptr, label), da_status_invalid_input);

    // getters
    EXPECT_EQ(da_data_get_n_rows(store, nullptr), da_status_invalid_input);
    EXPECT_EQ(da_data_get_n_cols(store, nullptr), da_status_invalid_input);
    EXPECT_EQ(da_data_get_element_int(store, 0, 0, nullptr), da_status_invalid_input);
    EXPECT_EQ(da_data_get_element_uint8(store, 0, 0, nullptr), da_status_invalid_input);
    EXPECT_EQ(da_data_get_element_real_d(store, 0, 0, nullptr), da_status_invalid_input);
    EXPECT_EQ(da_data_get_element_real_s(store, 0, 0, nullptr), da_status_invalid_input);

    da_datastore_destroy(&store);
}

TEST(dataStore, nullStore) {
    da_datastore store = nullptr, store1 = nullptr;
    da_int int_block = 1;
    uint8_t uint_block = 1;
    const char *str_block = "A";
    double d_block = 1.0;
    float s_block = 1.0;
    uint8_t ui_block = 1;
    EXPECT_EQ(da_data_hconcat(&store, &store1), da_status_store_not_initialized);

    // load cols/rows
    EXPECT_EQ(da_data_load_col_int(store, 1, 1, &int_block, row_major, false),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_load_col_str(store, 1, 1, &str_block, row_major),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_load_col_real_d(store, 1, 1, &d_block, row_major, false),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_load_col_real_s(store, 1, 1, &s_block, row_major, false),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_load_col_uint8(store, 1, 1, &ui_block, row_major, false),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_load_row_int(store, 1, 1, &int_block, row_major, false),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_load_row_str(store, 1, 1, &str_block, row_major),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_load_row_real_d(store, 1, 1, &d_block, row_major, false),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_load_row_real_s(store, 1, 1, &s_block, row_major, false),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_load_row_uint8(store, 1, 1, &ui_block, row_major, false),
              da_status_store_not_initialized);

    // load CSV
    EXPECT_EQ(da_data_load_from_csv(store, "path/to/file"),
              da_status_store_not_initialized);

    // selection
    EXPECT_EQ(da_data_select_columns(store, "A", 1, 1), da_status_store_not_initialized);
    EXPECT_EQ(da_data_select_rows(store, "A", 1, 1), da_status_store_not_initialized);
    EXPECT_EQ(da_data_select_non_missing(store, "A", 0), da_status_store_not_initialized);
    EXPECT_EQ(da_data_select_slice(store, "A", 1, 1, 1, 1),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_select_remove_rows(store, "A", 1, 1),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_select_remove_columns(store, "A", 1, 1),
              da_status_store_not_initialized);

    // extract selection
    EXPECT_EQ(da_data_extract_selection_int(store, "A", column_major, &int_block, 1),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_extract_selection_real_d(store, "A", column_major, &d_block, 1),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_extract_selection_real_s(store, "A", column_major, &s_block, 1),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_extract_selection_uint8(store, "A", column_major, &uint_block, 1),
              da_status_store_not_initialized);

    // Extract columns
    EXPECT_EQ(da_data_extract_column_int(store, 0, 1, &int_block),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_extract_column_real_s(store, 0, 1, &s_block),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_extract_column_real_d(store, 0, 1, &d_block),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_extract_column_uint8(store, 0, 1, &uint_block),
              da_status_store_not_initialized);
    char *cstr_block = nullptr;
    EXPECT_EQ(da_data_extract_column_str(store, 0, 1, &cstr_block),
              da_status_store_not_initialized);

    // setters/getters
    da_int ielem;
    double delem;
    float selem;
    uint8_t uielem;
    EXPECT_EQ(da_data_get_n_rows(store, &ielem), da_status_store_not_initialized);
    EXPECT_EQ(da_data_get_n_cols(store, &ielem), da_status_store_not_initialized);
    EXPECT_EQ(da_data_get_element_int(store, 1, 1, &ielem),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_get_element_real_d(store, 1, 1, &delem),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_get_element_real_s(store, 1, 1, &selem),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_get_element_uint8(store, 1, 1, &uielem),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_set_element_int(store, 1, 1, ielem),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_set_element_real_d(store, 1, 1, delem),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_set_element_real_s(store, 1, 1, selem),
              da_status_store_not_initialized);
    EXPECT_EQ(da_data_set_element_uint8(store, 1, 1, uielem),
              da_status_store_not_initialized);

    // label
    EXPECT_EQ(da_data_label_column(store, "A", 1), da_status_store_not_initialized);
    da_int col_idx = 0;
    EXPECT_EQ(da_data_get_col_idx(store, "A", &col_idx), da_status_store_not_initialized);
    da_int label_sz = 2;
    char label[2] = "A";
    EXPECT_EQ(da_data_get_col_label(store, 0, &label_sz, label),
              da_status_store_not_initialized);

    EXPECT_EQ(da_data_print_options(store), da_status_store_not_initialized);
}

TEST(dataStore, extractSelPub) {
    da_datastore store = nullptr;

    // nullptr store in all routines

    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    // load 2x2 int block
    std::vector<da_int> iblock = {1, 2, 3, 4};
    EXPECT_EQ(da_data_load_col_int(store, 2, 2, iblock.data(), column_major, true),
              da_status_success);
    EXPECT_EQ(da_data_select_slice(store, "int", 0, 1, 0, 0), da_status_success);
    std::vector<da_int> isel(2);
    EXPECT_EQ(da_data_extract_selection_int(store, "int", column_major, isel.data(), 2),
              da_status_success);
    std::vector<da_int> iexp = {1, 2};
    EXPECT_ARR_EQ(2, isel, iexp, 1, 1, 0, 0);

    // load 2x2 uint_8
    std::vector<uint8_t> uiblock = {1, 2, 3, 4};
    EXPECT_EQ(da_data_load_col_uint8(store, 2, 2, uiblock.data(), column_major, true),
              da_status_success);
    EXPECT_EQ(da_data_select_slice(store, "uint", 0, 1, 2, 3), da_status_success);
    std::vector<uint8_t> uisel(4);
    EXPECT_EQ(
        da_data_extract_selection_uint8(store, "uint", column_major, uisel.data(), 2),
        da_status_success);
    std::vector<uint8_t> uiexp = {1, 2, 3, 4};
    EXPECT_ARR_EQ(2, uisel, uiexp, 1, 1, 0, 0);

    // load 2x2 float
    std::vector<float> sblock = {1, 2, 3, 4};
    EXPECT_EQ(da_data_load_col_real_s(store, 2, 2, sblock.data(), column_major, true),
              da_status_success);
    EXPECT_EQ(da_data_select_slice(store, "float", 0, 1, 4, 4), da_status_success);
    std::vector<float> ssel(2);
    std::vector<double> dsel(2);
    EXPECT_EQ(
        da_data_extract_selection_real_s(store, "float", column_major, ssel.data(), 2),
        da_status_success);
    EXPECT_EQ(
        da_data_extract_selection_real_d(store, "float", column_major, dsel.data(), 2),
        da_status_invalid_input);
    std::vector<float> sexp = {1, 2};
    EXPECT_ARR_EQ(2, ssel, sexp, 1, 1, 0, 0);

    // load 2x2 double
    std::vector<double> dblock = {5, 6, 7, 8};
    EXPECT_EQ(da_data_load_col_real_d(store, 2, 2, dblock.data(), column_major, true),
              da_status_success);
    EXPECT_EQ(da_data_select_rows(store, "double", 0, 1), da_status_success);
    EXPECT_EQ(da_data_select_columns(store, "double", 6, 6), da_status_success);
    EXPECT_EQ(
        da_data_extract_selection_real_d(store, "double", column_major, dsel.data(), 2),
        da_status_success);
    EXPECT_EQ(
        da_data_extract_selection_real_s(store, "double", column_major, ssel.data(), 2),
        da_status_invalid_input);
    std::vector<double> dexp = {5, 6};
    EXPECT_ARR_EQ(2, dsel, dexp, 1, 1, 0, 0);

    da_datastore_destroy(&store);
}

TEST(dataStore, missingDataPub) {
    da_datastore store = nullptr;

    // nullptr store in all routines

    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    da_int m, n;
    std::vector<da_int> idata;
    std::vector<float> fdata;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store_pub(store, m, n, idata, fdata, sdata);

    // set some missing values for integers and floating points
    float missing_float = std::numeric_limits<float>::quiet_NaN();
    da_int int_max = std::numeric_limits<da_int>::max();
    EXPECT_EQ(da_data_set_element_int(store, 0, 2, int_max), da_status_success);
    EXPECT_EQ(da_data_set_element_int(store, 2, 0, int_max), da_status_success);
    EXPECT_EQ(da_data_set_element_int(store, 2, 3, int_max), da_status_success);
    EXPECT_EQ(da_data_set_element_real_s(store, 2, 5, missing_float), da_status_success);
    EXPECT_EQ(da_data_set_element_real_s(store, 4, 4, missing_float), da_status_success);

    // select and extract only the integer columns
    const char *tag = "nonmissing int";
    EXPECT_EQ(da_data_select_non_missing(store, tag, true), da_status_success);
    EXPECT_EQ(da_data_select_columns(store, tag, 0, 3), da_status_success);
    std::vector<da_int> int_sel(12);
    EXPECT_EQ(da_data_extract_selection_int(store, tag, column_major, int_sel.data(), 3),
              da_status_success);
    std::vector<da_int> iexp = {3, 7, 21, 4, 8, 22, 2, 4, 23, 6, 8, 24};
    EXPECT_ARR_EQ(12, int_sel, iexp, 1, 1, 0, 0);

    // row-major extraction
    const char *tag_row = "row-major";
    EXPECT_EQ(da_data_select_non_missing(store, tag_row, true), da_status_success);
    EXPECT_EQ(da_data_select_columns(store, tag_row, 0, 3), da_status_success);
    std::vector<da_int> int_sel_row(12);
    EXPECT_EQ(da_data_extract_selection_int(store, tag, row_major, int_sel_row.data(), 4),
              da_status_success);
    std::vector<da_int> iexp_row = {3, 4, 2, 6, 7, 8, 4, 8, 21, 22, 23, 24};
    EXPECT_ARR_EQ(12, int_sel_row, iexp_row, 1, 1, 0, 0);

    da_datastore_destroy(&store);
}

TEST(dataStore, heading) {
    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_float_head.csv");
    da_datastore store = nullptr;

    // with existing headings
    std::vector<std::string> expected_headings = {"one", "cat two", "three", "FOUR",
                                                  "Five"};
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_success);
    char col_name[64];
    da_int name_sz = 64;
    da_int col_idx;
    for (da_int j = 0; j < 5; j++) {
        EXPECT_EQ(da_data_get_col_label(store, j, &name_sz, col_name), da_status_success);
        EXPECT_STREQ(col_name, expected_headings[j].c_str());
        EXPECT_EQ(da_data_get_col_idx(store, expected_headings[j].c_str(), &col_idx),
                  da_status_success);
        EXPECT_EQ(col_idx, j);
    }

    // re-tag one of the column
    const char *new_tag = "changing column tag";
    da_int idx = 1;
    EXPECT_EQ(da_data_label_column(store, new_tag, idx), da_status_success);
    da_int wrong_name_sz = 2;
    EXPECT_EQ(da_data_get_col_label(store, idx, &wrong_name_sz, col_name),
              da_status_invalid_input);
    EXPECT_EQ(da_data_get_col_label(store, idx, &name_sz, col_name), da_status_success);
    EXPECT_STREQ(col_name, new_tag);
    EXPECT_EQ(da_data_get_col_idx(store, new_tag, &col_idx), da_status_success);
    EXPECT_EQ(col_idx, idx);
    da_datastore_destroy(&store);

    // with no headings
    char filepath2[256] = DATA_DIR;
    strcat(filepath2, "csv_data/csv_test_float.csv");
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_success);
    EXPECT_EQ(da_data_get_col_label(store, 1, &name_sz, col_name), da_status_success);
    EXPECT_STREQ(col_name, "");

    // tag an anonymous column
    idx = 4;
    EXPECT_EQ(da_data_label_column(store, new_tag, idx), da_status_success);
    EXPECT_EQ(da_data_get_col_label(store, idx, &name_sz, col_name), da_status_success);
    EXPECT_STREQ(col_name, new_tag);
    EXPECT_EQ(da_data_get_col_idx(store, new_tag, &col_idx), da_status_success);
    EXPECT_EQ(col_idx, idx);
    da_datastore_destroy(&store);

    da_datastore_destroy(&store);
}

TEST(datastore, incompleteStore) {
    // get a datastore in an intermediate state (partially added row)
    da_datastore store;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    get_transition_datastore(store);

    // call all the functions to check they correctly return an error
    // load columns
    da_int idummy;
    float fdummy;
    uint8_t uidummy;
    double ddummy;
    const char *cdummy[] = {"a"};
    EXPECT_EQ(da_data_load_col_int(store, 1, 1, &idummy, row_major, true),
              da_status_missing_block);
    EXPECT_EQ(da_data_load_col_real_d(store, 1, 1, &ddummy, row_major, true),
              da_status_missing_block);
    EXPECT_EQ(da_data_load_col_real_s(store, 1, 1, &fdummy, row_major, true),
              da_status_missing_block);
    EXPECT_EQ(da_data_load_col_uint8(store, 1, 1, &uidummy, row_major, true),
              da_status_missing_block);
    EXPECT_EQ(da_data_load_col_str(store, 1, 1, cdummy, row_major),
              da_status_missing_block);

    // selection
    EXPECT_EQ(da_data_select_columns(store, "key", 0, 1), da_status_missing_block);
    EXPECT_EQ(da_data_select_rows(store, "key", 0, 1), da_status_missing_block);
    EXPECT_EQ(da_data_select_slice(store, "key", 0, 1, 0, 1), da_status_missing_block);
    EXPECT_EQ(da_data_select_non_missing(store, "key", 0), da_status_missing_block);
    EXPECT_EQ(da_data_select_remove_columns(store, "key", 0, 1), da_status_missing_block);
    EXPECT_EQ(da_data_select_remove_rows(store, "key", 0, 1), da_status_missing_block);

    // extract column
    EXPECT_EQ(da_data_extract_column_int(store, 1, 1, &idummy), da_status_missing_block);
    EXPECT_EQ(da_data_extract_column_real_s(store, 1, 1, &fdummy),
              da_status_missing_block);
    EXPECT_EQ(da_data_extract_column_real_d(store, 1, 1, &ddummy),
              da_status_missing_block);
    EXPECT_EQ(da_data_extract_column_uint8(store, 1, 1, &uidummy),
              da_status_missing_block);
    char *Tc[1];
    EXPECT_EQ(da_data_extract_column_str(store, 1, 1, Tc), da_status_missing_block);

    // extract selection
    EXPECT_EQ(da_data_extract_selection_int(store, "key", column_major, &idummy, 1),
              da_status_missing_block);
    EXPECT_EQ(da_data_extract_selection_real_s(store, "key", column_major, &fdummy, 1),
              da_status_missing_block);
    EXPECT_EQ(da_data_extract_selection_real_d(store, "key", column_major, &ddummy, 1),
              da_status_missing_block);
    EXPECT_EQ(da_data_extract_selection_uint8(store, "key", column_major, &uidummy, 1),
              da_status_missing_block);
    da_datastore_destroy(&store);
}

TEST(datastore, selectInvalid) {
    da_datastore store;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    da_int m, n;
    std::vector<da_int> idata;
    std::vector<float> fdata;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store_pub(store, m, n, idata, fdata, sdata);
    EXPECT_EQ(da_data_select_columns(store, "Valid cols", 0, 0), da_status_success);

    // Selections: Wrong name
    EXPECT_EQ(da_data_select_rows(store, "dainternal_A", 0, 0), da_status_invalid_input);
    EXPECT_EQ(da_data_select_columns(store, "dainternal_A", 0, 0),
              da_status_invalid_input);
    EXPECT_EQ(da_data_select_slice(store, "dainternal_A", 0, 0, 0, 0),
              da_status_invalid_input);
    EXPECT_EQ(da_data_select_non_missing(store, "dainternal_A", 0),
              da_status_invalid_input);

    // Selections: Wrong intervals
    EXPECT_EQ(da_data_select_rows(store, "Valid", 0, 0), da_status_success);
    EXPECT_EQ(da_data_select_rows(store, "Valid", -1, 2), da_status_invalid_input);
    EXPECT_EQ(da_data_select_rows(store, "Valid", 2, 2000), da_status_invalid_input);
    EXPECT_EQ(da_data_select_rows(store, "Valid", 3, 2), da_status_invalid_input);
    EXPECT_EQ(da_data_select_columns(store, "Valid", 0, 0), da_status_success);
    EXPECT_EQ(da_data_select_columns(store, "Valid", -1, 2), da_status_invalid_input);
    EXPECT_EQ(da_data_select_columns(store, "Valid", 2, 2000), da_status_invalid_input);
    EXPECT_EQ(da_data_select_columns(store, "Valid", 3, 2), da_status_invalid_input);
    EXPECT_EQ(da_data_select_slice(store, "Valid", -1, 2, 1, 1), da_status_invalid_input);
    EXPECT_EQ(da_data_select_slice(store, "Valid", 2, 2000, 1, 1),
              da_status_invalid_input);
    EXPECT_EQ(da_data_select_slice(store, "Valid", 3, 2, 1, 1), da_status_invalid_input);
    EXPECT_EQ(da_data_select_slice(store, "Valid", 1, 1, -1, 2), da_status_invalid_input);
    EXPECT_EQ(da_data_select_slice(store, "Valid", 1, 1, 2, 2000),
              da_status_invalid_input);
    EXPECT_EQ(da_data_select_slice(store, "Valid", 1, 1, 3, 2), da_status_invalid_input);
    EXPECT_EQ(da_data_select_columns(store, "Valid", 0, 6), da_status_success);
    EXPECT_EQ(da_data_select_remove_columns(store, "Valid", -1, 2),
              da_status_invalid_input);
    EXPECT_EQ(da_data_select_remove_columns(store, "Valid", 2, 2000),
              da_status_invalid_input);
    EXPECT_EQ(da_data_select_remove_columns(store, "Valid", 3, 2),
              da_status_invalid_input);
    EXPECT_EQ(da_data_select_rows(store, "Valid", 0, 5), da_status_success);
    EXPECT_EQ(da_data_select_remove_rows(store, "Valid", -1, 2), da_status_invalid_input);
    EXPECT_EQ(da_data_select_remove_rows(store, "Valid", 2, 2000),
              da_status_invalid_input);
    EXPECT_EQ(da_data_select_remove_rows(store, "Valid", 3, 2), da_status_invalid_input);

    // Remove from selection: non existing selection
    EXPECT_EQ(da_data_select_remove_columns(store, "Invalid", 0, 0),
              da_status_invalid_input);
    EXPECT_EQ(da_data_select_remove_rows(store, "Invalid", 0, 0),
              da_status_invalid_input);

    // Extraction
    da_int extract;
    EXPECT_EQ(
        da_data_extract_selection_int(store, "Non valid", column_major, &extract, 1),
        da_status_invalid_input);

    da_datastore_destroy(&store);
}
