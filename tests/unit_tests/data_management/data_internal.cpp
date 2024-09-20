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
#include "da_error.hpp"
#include "data_store.hpp"
#include "interval_map.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace da_data;

enum int_block_id {
    test1_rblock1 = 0,
    test1_cblock1,
    test1_2rows,
};

void get_block_data_int(int_block_id bid, da_int &m, da_int &n, std::vector<da_int> &bl,
                        da_order &order) {

    switch (bid) {
    case test1_rblock1:
        m = 5;
        n = 2;
        bl.resize(n * m);
        bl = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        order = row_major;
        break;

    case test1_cblock1:
        m = 5;
        n = 2;
        bl.resize(n * m);
        bl = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
        order = column_major;
        break;

    case test1_2rows:
        m = 2;
        n = 4;
        bl.resize(n * m);
        bl = {2, 4, 6, 8, 3, 5, 7, 9};
        order = row_major;
        break;

    default:
        FAIL() << "Unknown block ID";
    }
}

TEST(block, invalidArgs) {
    da_int data[2] = {1, 2};
    da_errors::da_error_t err(da_errors::DA_RECORD);

    EXPECT_THROW(block_dense<da_int> b(-1, 2, data, err), std::invalid_argument);
    EXPECT_THROW(block_dense<da_int> b(1, 0, data, err), std::invalid_argument);
    EXPECT_THROW(block_dense<da_int> b(1, 2, nullptr, err), std::invalid_argument);
    block_dense<da_int> b(1, 2, data, err);

    da_int *col;
    da_int stride;
    EXPECT_EQ(b.get_col(-1, &col, stride), da_status_invalid_input);
    EXPECT_EQ(b.get_col(5, &col, stride), da_status_invalid_input);
}

void get_heterogeneous_data_store(data_store &ds, da_int &mt, da_int &nt,
                                  std::vector<da_int> &idata, std::vector<float> &fdata,
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
    std::vector<std::string> sb1, sb2;
    da_int m, n;

    ib1 = {1, 2, 3, 4, 5, 6, 7, 8};
    m = 4;
    n = 2;
    EXPECT_EQ(ds.concatenate_columns(m, n, ib1.data(), row_major, true),
              da_status_success);
    ib2 = {1, 2, 3, 4, 5, 6, 7, 8};
    m = 4;
    n = 2;
    EXPECT_EQ(ds.concatenate_columns(m, n, ib2.data(), column_major, true),
              da_status_success);
    ib3 = {10, 11};
    m = 1;
    n = 2;
    EXPECT_EQ(ds.concatenate_rows(m, n, ib3.data(), column_major, true),
              da_status_success);
    ib4 = {12, 13};
    m = 1;
    n = 2;
    EXPECT_EQ(ds.concatenate_rows(m, n, ib4.data(), column_major, true),
              da_status_success);
    fb1 = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
    m = 5;
    n = 2;
    EXPECT_EQ(ds.concatenate_columns(m, n, fb1.data(), column_major, true),
              da_status_success);
    sb1 = {"1", "a2", "bb3", "ccc4", "dddd5"};
    m = 5;
    n = 1;
    EXPECT_EQ(ds.concatenate_columns(m, n, sb1.data(), row_major, true),
              da_status_success);
    ib5 = {21, 22, 23, 24};
    m = 1;
    n = 4;
    EXPECT_EQ(ds.concatenate_rows(m, n, ib5.data(), row_major, true), da_status_success);
    fb2 = {10.1f, 20.2f};
    m = 1;
    n = 2;
    EXPECT_EQ(ds.concatenate_rows(m, n, fb2.data(), row_major, true), da_status_success);
    sb2 = {"row6_1"};
    m = 1;
    n = 1;
    EXPECT_EQ(ds.concatenate_rows(m, n, sb2.data(), row_major, true), da_status_success);

    // expected blocks, column major ordering
    idata = {1, 3, 5, 7, 10, 21, 2, 4, 6, 8, 11, 22,
             1, 2, 3, 4, 12, 23, 5, 6, 7, 8, 13, 24};
    fdata = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 10.1f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 20.2f};
    sdata = {"1", "a2", "bb3", "ccc4", "dddd5", "row6_1"};
    mt = 6;
    nt = 7;
}

TEST(block, getCol) {
    std::vector<da_int> bl, col1_exp, col2_exp;
    da_int m, n;
    da_order order;
    da_int stride, startx = 0, starty = 0;
    da_int *col;
    da_errors::da_error_t err(da_errors::DA_RECORD);

    col1_exp = {1, 3, 5, 7, 9};
    col2_exp = {2, 4, 6, 8, 10};

    // Check column extraction in for the row ordering
    get_block_data_int(test1_rblock1, m, n, bl, order);
    block_dense<da_int> b1(m, n, bl.data(), err, order);
    b1.get_col(0, &col, stride);
    EXPECT_ARR_EQ(m, col, col1_exp, stride, 1, startx, starty);
    b1.get_col(1, &col, stride);
    EXPECT_ARR_EQ(m, col, col2_exp, stride, 1, startx, starty);
    // Check column extraction in for the col ordering
    get_block_data_int(test1_cblock1, m, n, bl, order);
    block_dense<da_int> b2(m, n, bl.data(), err, order);
    b2.get_col(0, &col, stride);
    EXPECT_ARR_EQ(m, col, col1_exp, stride, 1, startx, starty);
    b2.get_col(1, &col, stride);
    EXPECT_ARR_EQ(m, col, col2_exp, stride, 1, startx, starty);
    // out of bounds column index
    EXPECT_EQ(b2.get_col(2, &col, stride), da_status_invalid_input);
    EXPECT_EQ(b2.get_col(-1, &col, stride), da_status_invalid_input);
}

TEST(block, copySlice) {
    std::vector<da_int> bl_col, bl_row, islice, exp_slice;
    da_int m, n;
    da_errors::da_error_t err(da_errors::DA_RECORD);

    m = 5;
    n = 4;
    bl_col = {1, 2, 3, 4, 5, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 6, 7, 8, 9, 10};
    block_dense<da_int> b1(m, n, bl_col.data(), err, column_major);

    // load the data from the middle of the block
    interval cols, rows;
    cols = {1, 2};
    rows = {1, 3};
    islice.resize(6);
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 0, 3, islice.data()), da_status_success);
    exp_slice = {3, 5, 7, 4, 6, 8};
    EXPECT_ARR_EQ(6, islice, exp_slice, 1, 1, 0, 0);

    // try to load the block in the middle of the slice
    islice.resize(15);
    std::fill(islice.begin(), islice.end(), 0);
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 1, 5, &islice[5]), da_status_success);
    exp_slice = {0, 0, 0, 0, 0, 0, 3, 5, 7, 0, 0, 4, 6, 8, 0};
    EXPECT_ARR_EQ(15, islice, exp_slice, 1, 1, 0, 0);

    // row ordering
    bl_row = {1, 1, 2, 6, 2, 3, 4, 7, 3, 5, 6, 8, 4, 7, 8, 9, 5, 9, 10, 10};
    block_dense<da_int> b2(m, n, bl_row.data(), err, row_major);
    islice.resize(6);
    EXPECT_EQ(b2.copy_slice_dense(cols, rows, 0, 3, islice.data()), da_status_success);
    exp_slice = {3, 5, 7, 4, 6, 8};
    EXPECT_ARR_EQ(6, islice, exp_slice, 1, 1, 0, 0);

    // try to load the block in the middle of the slice
    islice.resize(15);
    std::fill(islice.begin(), islice.end(), 0);
    EXPECT_EQ(b2.copy_slice_dense(cols, rows, 1, 5, &islice[5]), da_status_success);
    exp_slice = {0, 0, 0, 0, 0, 0, 3, 5, 7, 0, 0, 4, 6, 8, 0};
    EXPECT_ARR_EQ(15, islice, exp_slice, 1, 1, 0, 0);
}

TEST(block, missingValues) {
    std::vector<bool> valid_rows(10, true);
    da_errors::da_error_t err(da_errors::DA_RECORD);
    da_int m = 5;
    da_int n = 4;
    da_int maxi = std::numeric_limits<da_int>::max();
    da_data::interval cols, rows;

    /* column major ordering */
    std::vector<da_int> bl_col = {1, 2, 3, 4, 5,  1, maxi, 5, 7, 9,
                                  2, 4, 6, 8, 10, 6, maxi, 8, 9, maxi};
    block_dense<da_int> b1(m, n, bl_col.data(), err, column_major);
    cols = {0, n - 1};
    rows = {0, m - 1};
    EXPECT_EQ(b1.missing_rows(valid_rows, 0, rows, cols), da_status_success);
    std::vector<bool> exp_valid_rows = {true, false, true, true, false};
    EXPECT_ARR_EQ(5, valid_rows, exp_valid_rows, 1, 1, 0, 0);
    std::fill(valid_rows.begin(), valid_rows.end(), true);
    cols = {1, 3};
    rows = {1, 3};
    EXPECT_EQ(b1.missing_rows(valid_rows, 5, rows, cols), da_status_success);
    exp_valid_rows = {false, true, true};
    EXPECT_ARR_EQ(3, valid_rows, exp_valid_rows, 1, 1, 5, 0);

    /* row major ordering */
    m = 4;
    n = 5;
    std::vector<da_int> bl_row = {1, 2, 3, 4, 5,  1, maxi, 5, 7, 9,
                                  2, 4, 6, 8, 10, 6, maxi, 8, 9, maxi};
    block_dense<da_int> b2(m, n, bl_row.data(), err, row_major);
    cols = {0, n - 1};
    rows = {0, m - 1};
    std::fill(valid_rows.begin(), valid_rows.end(), true);
    EXPECT_EQ(b2.missing_rows(valid_rows, 0, rows, cols), da_status_success);
    exp_valid_rows = {true, false, true, false};
    EXPECT_ARR_EQ(4, valid_rows, exp_valid_rows, 1, 1, 0, 0);
    std::fill(valid_rows.begin(), valid_rows.end(), true);
    cols = {1, 3};
    rows = {0, 2};
    EXPECT_EQ(b2.missing_rows(valid_rows, 5, rows, cols), da_status_success);
    exp_valid_rows = {true, false, true};
    EXPECT_ARR_EQ(3, valid_rows, exp_valid_rows, 1, 1, 5, 0);

    /* try with a type that does not have a missing value defined */
    class missing_not_def {
      public:
        int a = 0;
    };
    std::vector<missing_not_def> bl_not_missing(10);
    m = 5;
    n = 2;
    cols = {0, n - 1};
    rows = {0, m - 1};
    block_dense<missing_not_def> b3(m, n, bl_not_missing.data(), err, row_major);
    std::fill(valid_rows.begin(), valid_rows.end(), true);
    EXPECT_EQ(b3.missing_rows(valid_rows, 5, rows, cols), da_status_success);
    exp_valid_rows.resize(5);
    std::fill(exp_valid_rows.begin(), exp_valid_rows.end(), true);
    EXPECT_ARR_EQ(5, valid_rows, exp_valid_rows, 1, 1, 5, 0);

    /* input errors */
    EXPECT_EQ(b3.missing_rows(valid_rows, -1, rows, cols), da_status_invalid_input);
    EXPECT_EQ(b3.missing_rows(valid_rows, 9, rows, cols), da_status_invalid_input);
}

TEST(block, copySliceInvalid) {
    std::vector<da_int> bl_col, bl_row, islice, exp_slice;
    da_int m, n;
    da_errors::da_error_t err(da_errors::DA_RECORD);

    m = 5;
    n = 4;
    bl_col = {1, 2, 3, 4, 5, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 6, 7, 8, 9, 10};
    block_dense<da_int> b1(m, n, bl_col.data(), err, column_major);

    interval cols, rows;
    cols = {-1, 2};
    rows = {1, 3};
    islice.resize(30);
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 0, 3, islice.data()),
              da_status_invalid_input);
    cols = {2, 1};
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 0, 3, islice.data()),
              da_status_invalid_input);
    cols = {0, 4};
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 0, 3, islice.data()),
              da_status_invalid_input);
    cols = {4, 4};
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 0, 3, islice.data()),
              da_status_invalid_input);
    cols = {1, 2};
    rows = {-1, 2};
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 0, 3, islice.data()),
              da_status_invalid_input);
    rows = {2, 1};
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 0, 3, islice.data()),
              da_status_invalid_input);
    rows = {0, 5};
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 0, 3, islice.data()),
              da_status_invalid_input);
    rows = {5, 6};
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 0, 3, islice.data()),
              da_status_invalid_input);
}

TEST(dataStore, invalidConcat) {
    std::vector<da_int> bl1, bl2, bl3, bl4;
    da_order order;
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);

    data_store ds = data_store(err);
    da_int m, n;
    order = row_major;
    // negative or zero sizes
    m = 0;
    n = 1;
    EXPECT_EQ(ds.concatenate_columns(m, n, bl1.data(), order), da_status_invalid_input);
    EXPECT_EQ(ds.concatenate_rows(m, n, bl1.data(), order), da_status_invalid_input);
    m = 1;
    n = -1;
    EXPECT_EQ(ds.concatenate_columns(m, n, bl1.data(), order), da_status_invalid_input);
    EXPECT_EQ(ds.concatenate_rows(m, n, bl1.data(), order), da_status_invalid_input);

    // Add a first valid block 5 x 2
    get_block_data_int(test1_rblock1, m, n, bl1, order);
    EXPECT_EQ(ds.concatenate_rows(m, n, bl1.data(), order), da_status_success);

    // try to add a 2 x 4 block to the right or the bottom of the data_store
    get_block_data_int(test1_2rows, m, n, bl1, order);
    EXPECT_EQ(ds.concatenate_columns(m, n, bl1.data(), order), da_status_invalid_input);
    EXPECT_EQ(ds.concatenate_rows(m, n, bl1.data(), order), da_status_invalid_input);

    // try to add a 1 x 2 string block
    std::vector<std::string> strbl = {"d1", "d2"};
    order = row_major;
    m = 1;
    n = 2;
    EXPECT_EQ(ds.concatenate_columns(m, n, strbl.data(), order), da_status_invalid_input);

    // add two valid 1 x 2 rows
    bl2 = {1, 2};
    bl3 = {3, 4};
    m = 1;
    n = 2;
    EXPECT_EQ(ds.concatenate_rows(m, n, bl2.data(), order), da_status_success);
    EXPECT_EQ(ds.concatenate_rows(m, n, bl3.data(), order), da_status_success);

    // add an invalid str row to check data is correctly deallocated
    EXPECT_EQ(ds.concatenate_rows(m, n, strbl.data(), order), da_status_invalid_input);

    // add a 7 x 2 double column
    std::vector<double> dbl = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    m = 7;
    n = 2;
    order = column_major;
    EXPECT_EQ(ds.concatenate_columns(m, n, dbl.data(), order), da_status_success);

    // try to add a 1 x 4 int row.
    // correct dims but should fail because the last 2 cols are not of the correct type
    m = 1;
    n = 4;
    bl4 = {1, 2, 3, 4};
    EXPECT_EQ(ds.concatenate_rows(m, n, bl4.data(), order), da_status_invalid_input);
}

TEST(dataStore, invalidExtract) {
    da_int m, n;
    std::vector<da_int> bl1, bl2, bl3;
    da_order order;
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);

    data_store ds = data_store(err);
    get_block_data_int(test1_rblock1, m, n, bl1, order);
    ds.concatenate_columns(m, n, bl1.data(), order);
    get_block_data_int(test1_cblock1, m, n, bl2, order);
    ds.concatenate_columns(m, n, bl2.data(), order);

    da_int mw = m + 1;
    bl3.resize(m);
    EXPECT_EQ(ds.extract_column(2, mw, bl3.data()), da_status_invalid_input);
    EXPECT_EQ(ds.extract_column(-1, m, bl3.data()), da_status_invalid_input);
    EXPECT_EQ(ds.extract_column(4, m, bl3.data()), da_status_invalid_input);
}

TEST(datastore, getSetElement) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    data_store hds = data_store(err);
    da_int m, n;
    std::vector<da_int> idata;
    std::vector<float> fdata;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store(hds, m, n, idata, fdata, sdata);

    da_int ielem = -10;
    float felem = -1.0;
    // invalid requests
    EXPECT_EQ(hds.get_element(-1, 0, ielem), da_status_invalid_input);
    EXPECT_EQ(hds.get_element(0, -1, ielem), da_status_invalid_input);
    EXPECT_EQ(hds.get_element(6, 0, ielem), da_status_invalid_input);
    EXPECT_EQ(hds.get_element(0, 7, ielem), da_status_invalid_input);
    EXPECT_EQ(hds.get_element(5, 5, ielem), da_status_invalid_input);
    EXPECT_EQ(hds.set_element(-1, 0, (da_int)1), da_status_invalid_input);
    EXPECT_EQ(hds.set_element(0, -1, (da_int)1), da_status_invalid_input);
    EXPECT_EQ(hds.set_element(6, 0, (da_int)1), da_status_invalid_input);
    EXPECT_EQ(hds.set_element(0, 7, (da_int)1), da_status_invalid_input);
    EXPECT_EQ(hds.set_element(5, 5, (da_int)2), da_status_invalid_input);

    // get valid elements
    EXPECT_EQ(hds.get_element(0, 0, ielem), da_status_success);
    EXPECT_EQ(ielem, 1);
    EXPECT_EQ(hds.get_element(4, 2, ielem), da_status_success);
    EXPECT_EQ(ielem, 12);
    EXPECT_EQ(hds.get_element(5, 2, ielem), da_status_success);
    EXPECT_EQ(ielem, 23);
    EXPECT_EQ(hds.get_element(5, 5, felem), da_status_success);
    EXPECT_NEAR(felem, 20.2, std::numeric_limits<float>::epsilon() * 100);

    // set the same elements
    EXPECT_EQ(hds.set_element(0, 0, (da_int)100), da_status_success);
    EXPECT_EQ(hds.get_element(0, 0, ielem), da_status_success);
    EXPECT_EQ(ielem, 100);
    EXPECT_EQ(hds.set_element(4, 2, (da_int)101), da_status_success);
    EXPECT_EQ(hds.get_element(4, 2, ielem), da_status_success);
    EXPECT_EQ(ielem, 101);
    EXPECT_EQ(hds.set_element(5, 5, (float)100.1), da_status_success);
    EXPECT_EQ(hds.get_element(5, 5, felem), da_status_success);
    EXPECT_NEAR(felem, 100.1, std::numeric_limits<float>::epsilon() * 100);
}

TEST(dataStore, extractCol) {
    da_int m, n;
    std::vector<da_int> bl1, bl2, bl3;
    da_order order;
    da_int startx = 0, starty = 0;
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);

    data_store ds = data_store(err);
    get_block_data_int(test1_rblock1, m, n, bl1, order);
    ds.concatenate_columns(m, n, bl1.data(), order);
    get_block_data_int(test1_cblock1, m, n, bl2, order);
    ds.concatenate_columns(m, n, bl2.data(), order);

    std::vector<da_int> col, col1_exp, col2_exp;
    col1_exp = {1, 3, 5, 7, 9};
    col2_exp = {2, 4, 6, 8, 10};
    col.resize(m);
    ds.extract_column(0, m, col.data());
    EXPECT_ARR_EQ(m, col, col1_exp, 1, 1, startx, starty);
    ds.extract_column(1, m, col.data());
    EXPECT_ARR_EQ(m, col, col2_exp, 1, 1, startx, starty);
    ds.extract_column(2, m, col.data());
    EXPECT_ARR_EQ(m, col, col1_exp, 1, 1, startx, starty);
    ds.extract_column(3, m, col.data());
    EXPECT_ARR_EQ(m, col, col2_exp, 1, 1, startx, starty);

    // add 2 rows to the main block (2 x 4 block)
    da_int new_m;
    get_block_data_int(test1_2rows, new_m, n, bl3, order);
    ds.concatenate_rows(new_m, n, bl3.data(), order);
    m += new_m;
    col.resize(m);
    ds.extract_column(0, m, col.data());
    col1_exp = {1, 3, 5, 7, 9, 2, 3};
    EXPECT_ARR_EQ(m, col, col1_exp, 1, 1, startx, starty);
    ds.extract_column(3, m, col.data());
    col2_exp = {2, 4, 6, 8, 10, 8, 9};
    EXPECT_ARR_EQ(m, col, col2_exp, 1, 1, startx, starty);

    // test the heterogeneous data-store columns
    da_errors::da_error_t err2(da_errors::action_t::DA_RECORD);
    data_store hds = data_store(err2);
    std::vector<da_int> idata, coli;
    std::vector<float> fdata;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store(hds, m, n, idata, fdata, sdata);
    coli.resize(m);
    hds.extract_column(0, m, coli.data());
    EXPECT_ARR_EQ(m, coli, idata, 1, 1, startx, starty);
    EXPECT_EQ(hds.extract_column(6, m, coli.data()), da_status_invalid_input);
}

TEST(dataStore, invalidHconcat) {
    da_int m, n;
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD),
        err1(da_errors::action_t::DA_RECORD);
    data_store hds = data_store(err), hds1 = data_store(err1);
    std::vector<da_int> idata;
    std::vector<float> fdata;
    std::vector<std::string> sdata;

    get_heterogeneous_data_store(hds, m, n, idata, fdata, sdata);
    EXPECT_EQ(hds.horizontal_concat(hds1), da_status_invalid_input);

    // add a partial row to hds1
    get_heterogeneous_data_store(hds1, m, n, idata, fdata, sdata);
    std::vector<da_int> iblock = {1, 2, 3, 4};
    EXPECT_EQ(hds1.concatenate_rows(1, 4, iblock.data(), row_major, true),
              da_status_success);

    // Add the same partial row to hds
    EXPECT_EQ(hds.concatenate_rows(1, 4, iblock.data(), row_major, true),
              da_status_success);
    // hds partial row fails concatenation
    EXPECT_EQ(hds.horizontal_concat(hds1), da_status_invalid_input);

    // finish hds row and try concat again
    std::vector<float> fblock = {1., 2.};
    std::vector<std::string> sblock = {"1"};
    EXPECT_EQ(hds.concatenate_rows(1, 2, fblock.data(), row_major, true),
              da_status_success);
    EXPECT_EQ(hds.concatenate_rows(1, 1, sblock.data(), row_major, true),
              da_status_success);
    EXPECT_EQ(hds.horizontal_concat(hds1), da_status_invalid_input);
}

TEST(dataStore, hconcat) {
    // create 2 heterogeneous data_stores
    da_int m, n;
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD),
        err2(da_errors::action_t::DA_RECORD), err3(da_errors::action_t::DA_RECORD);
    data_store hds = data_store(err);
    std::vector<da_int> idata, coli, coli2, coli3;
    std::vector<float> fdata, colf, colf2, colf3;
    std::vector<std::string> sdata, cols, cols2, cols3;
    get_heterogeneous_data_store(hds, m, n, idata, fdata, sdata);
    data_store hds2 = data_store(err2);
    get_heterogeneous_data_store(hds2, m, n, idata, fdata, sdata);
    data_store hds3 = data_store(err3);
    get_heterogeneous_data_store(hds3, m, n, idata, fdata, sdata);

    // concatenate them horizontally
    da_int startx = 0, starty = 0;
    hds2.horizontal_concat(hds3);
    EXPECT_TRUE(hds3.empty());
    hds.horizontal_concat(hds2);
    EXPECT_TRUE(hds2.empty());
    coli.resize(m);
    coli2.resize(m);
    coli3.resize(m);
    for (da_int col = 0; col < 4; col++) {
        starty = col * m;
        hds.extract_column(col, m, coli.data());
        hds.extract_column(col + 7, m, coli2.data());
        hds.extract_column(col + 14, m, coli3.data());
        EXPECT_ARR_EQ(m, coli, idata, 1, 1, startx, starty);
        EXPECT_ARR_EQ(m, coli2, idata, 1, 1, startx, starty);
        EXPECT_ARR_EQ(m, coli3, idata, 1, 1, startx, starty);
    }
    colf.resize(m);
    colf2.resize(m);
    colf3.resize(m);
    for (da_int col = 4; col < 6; col++) {
        starty = (col - 4) * m;
        EXPECT_EQ(hds.extract_column(col, m, colf.data()), da_status_success);
        EXPECT_EQ(hds.extract_column(col + 7, m, colf2.data()), da_status_success);
        EXPECT_EQ(hds.extract_column(col + 14, m, colf3.data()), da_status_success);
        EXPECT_ARR_EQ(m, colf, fdata, 1, 1, startx, starty);
        EXPECT_ARR_EQ(m, colf2, fdata, 1, 1, startx, starty);
        EXPECT_ARR_EQ(m, colf3, fdata, 1, 1, startx, starty);
    }
    cols.resize(m);
    cols2.resize(m);
    cols3.resize(m);
    for (da_int col = 6; col < 7; col++) {
        starty = (col - 6) * m;
        EXPECT_EQ(hds.extract_column(col, m, cols.data()), da_status_success);
        EXPECT_EQ(hds.extract_column(col + 7, m, cols2.data()), da_status_success);
        EXPECT_EQ(hds.extract_column(col + 7, m, cols3.data()), da_status_success);
        EXPECT_ARR_EQ(m, cols, sdata, 1, 1, startx, starty);
        EXPECT_ARR_EQ(m, cols2, sdata, 1, 1, startx, starty);
        EXPECT_ARR_EQ(m, cols3, sdata, 1, 1, startx, starty);
    }
}

TEST(dataStore, extractSlice) {
    da_int m, n, ld;
    std::vector<da_int> bl1, bl2, bl3;
    da_order order;
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    data_store ds = data_store(err);
    get_block_data_int(test1_rblock1, m, n, bl1, order);
    EXPECT_EQ(ds.concatenate_columns(m, n, bl1.data(), order), da_status_success);
    get_block_data_int(test1_cblock1, m, n, bl2, order);
    EXPECT_EQ(ds.concatenate_columns(m, n, bl2.data(), order), da_status_success);

    // Extract the first columns into a slice
    interval col_int(0, 1), row_int(0, m - 1);
    ld = row_int.upper - row_int.lower + 1;
    std::vector<da_int> islice(m * 2);
    ds.extract_slice(row_int, col_int, ld, 0, islice.data());
    std::vector<da_int> expected_slice = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
    EXPECT_ARR_EQ(10, islice, expected_slice, 1, 1, 0, 0);

    // Extract the same columns into a bigger memory block
    ld += 3;
    islice.resize(ld * 4);
    da_int first_idx = ld + 3;
    ds.extract_slice(row_int, col_int, ld, first_idx, islice.data());
    expected_slice = {1, 3, 5, 7, 9};
    EXPECT_ARR_EQ(5, islice, expected_slice, 1, 1, first_idx, 0);
    expected_slice = {2, 4, 6, 8, 10};
    EXPECT_ARR_EQ(5, islice, expected_slice, 1, 1, first_idx + ld, 0);

    // columns spread on 2 blocks
    col_int.upper = 2;
    ld = row_int.upper - row_int.lower + 1;
    islice.resize(3 * m);
    ds.extract_slice(row_int, col_int, ld, 0, islice.data());
    expected_slice = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
    EXPECT_ARR_EQ(15, islice, expected_slice, 1, 1, 0, 0);

    // same datastore, partial rows
    row_int.upper = 2;
    ld = row_int.upper - row_int.lower + 1;
    ds.extract_slice(row_int, col_int, ld, 0, islice.data());
    expected_slice = {1, 3, 5, 2, 4, 6, 1, 3, 5};
    EXPECT_ARR_EQ(9, islice, expected_slice, 1, 1, 0, 0);

    // add rows and extract the first 3 columns
    da_int new_m;
    get_block_data_int(test1_2rows, new_m, n, bl3, order);
    EXPECT_EQ(ds.concatenate_rows(new_m, n, bl3.data(), order, true), da_status_success);
    row_int = {0, 6};
    col_int = {0, 2};
    islice.resize(21);
    std::fill(islice.begin(), islice.end(), 0);
    ld = row_int.upper - row_int.lower + 1;
    ds.extract_slice(row_int, col_int, ld, 0, islice.data());
    expected_slice = {1, 3, 5, 7, 9, 2, 3, 2, 4, 6, 8, 10, 4, 5, 1, 3, 5, 7, 9, 6, 7};
    EXPECT_ARR_EQ(21, islice, expected_slice, 1, 1, 0, 0);

    // Test slice extraction on the heterogeneous data store
    da_errors::da_error_t err2(da_errors::action_t::DA_RECORD);
    data_store hds = data_store(err2);
    std::vector<da_int> idata;
    std::vector<float> fdata;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store(hds, m, n, idata, fdata, sdata);

    islice.resize(100);
    std::fill(islice.begin(), islice.end(), 0);
    row_int = {2, 5};
    col_int = {1, 2};
    ld = row_int.upper - row_int.lower + 1;
    hds.extract_slice(row_int, col_int, ld, 0, islice.data());
    expected_slice = {6, 8, 11, 22, 3, 4, 12, 23};
    EXPECT_ARR_EQ(8, islice, expected_slice, 1, 1, 0, 0);

    // same block, bigger data block
    ld += 5;
    first_idx = ld * 2 + 2;
    hds.extract_slice(row_int, col_int, ld, first_idx, islice.data());
    expected_slice = {6, 8, 11, 22};
    EXPECT_ARR_EQ(4, islice, expected_slice, 1, 1, first_idx, 0);
    expected_slice = {3, 4, 12, 23};
    EXPECT_ARR_EQ(4, islice, expected_slice, 1, 1, first_idx + ld, 0);

    // extract just a row
    std::fill(islice.begin(), islice.end(), 0);
    row_int = {4, 4};
    col_int = {0, 3};
    ld = row_int.upper - row_int.lower + 1;
    hds.extract_slice(row_int, col_int, ld, 0, islice.data());
    expected_slice = {10, 11, 12, 13};
    EXPECT_ARR_EQ(4, islice, expected_slice, 1, 1, 0, 0);

    // Only bottom blocks
    std::fill(islice.begin(), islice.end(), 0);
    row_int = {4, 5};
    col_int = {1, 3};
    ld = row_int.upper - row_int.lower + 1;
    hds.extract_slice(row_int, col_int, ld, 0, islice.data());
    expected_slice = {11, 22, 12, 23, 13, 24};
    EXPECT_ARR_EQ(6, islice, expected_slice, 1, 1, 0, 0);

    // extract floats
    std::vector<float> fslice;
    fslice.resize(5);
    row_int = {1, 5};
    col_int = {5, 5};
    ld = row_int.upper - row_int.lower + 1;
    hds.extract_slice(row_int, col_int, ld, 0, fslice.data());
    std::vector<float> fexpected_slice = {6.5, 7.5, 8.5, 9.5};
    EXPECT_ARR_EQ(4, fslice, fexpected_slice, 1, 1, 0, 0);
}

TEST(dataStore, exSliceInvalid) {
    da_int m, n, ld;
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    data_store hds = data_store(err);
    std::vector<da_int> idata;
    std::vector<float> fdata;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store(hds, m, n, idata, fdata, sdata);

    // out of range intervals
    da_data::interval row_int = {2, 1}, col_int = {0, 1};
    ld = 2;
    std::vector<da_int> islice(100);
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    row_int = {0, 1};
    col_int = {10, 5};
    ld = row_int.upper - row_int.lower + 1;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    col_int = {-1, 2};
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    col_int = {2, 7};
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    col_int = {7, 7};
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    col_int = {0, 1};
    row_int = {-1, 2};
    ld = row_int.upper - row_int.lower + 1;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    row_int = {1, 6};
    ld = row_int.upper - row_int.lower + 1;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    row_int = {7, 10};
    ld = row_int.upper - row_int.lower + 1;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);

    // wrong type expected
    col_int = {4, 5};
    row_int = {0, 2};
    ld = row_int.upper - row_int.lower + 1;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    col_int = {0, 5};
    row_int = {0, 2};
    ld = row_int.upper - row_int.lower + 1;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);

    // wrong leading dimension
    row_int = {1, 3};
    col_int = {2, 3};
    ld = 2;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
}

TEST(dataStore, extractSelection) {
    using namespace da_data;
    da_int m, n, ld;
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    data_store hds = data_store(err);
    std::vector<da_int> idata;
    std::vector<float> fdata;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store(hds, m, n, idata, fdata, sdata);
    std::vector<da_int> expected_slice;
    expected_slice.reserve(100);

    // 1 set of columns and rows
    EXPECT_EQ(hds.select_slice("A", {1, 3}, {1, 3}), da_status_success);
    std::vector<da_int> islice(100);
    ld = 3;
    EXPECT_EQ(hds.extract_selection("A", column_major, ld, islice.data()),
              da_status_success);
    expected_slice = {4, 6, 8, 2, 3, 4, 6, 7, 8};
    EXPECT_ARR_EQ(9, islice, expected_slice, 1, 1, 0, 0);

    // 2 sets of columns and rows
    hds.remove_selection("A");
    std::fill(islice.begin(), islice.end(), 0);
    EXPECT_EQ(hds.select_columns("A", {1, 1}), da_status_success);
    EXPECT_EQ(hds.select_columns("A", {2, 3}), da_status_success);
    EXPECT_EQ(hds.select_rows("A", {3, 3}), da_status_success);
    EXPECT_EQ(hds.select_rows("A", {1, 1}), da_status_success);
    EXPECT_EQ(hds.select_rows("A", {2, 2}), da_status_success);
    EXPECT_EQ(hds.extract_selection("A", column_major, ld, islice.data()),
              da_status_success);
    expected_slice = {4, 6, 8, 2, 3, 4, 6, 7, 8};
    EXPECT_ARR_EQ(9, islice, expected_slice, 1, 1, 0, 0);

    // add the rest of the integer data from hds
    EXPECT_EQ(hds.select_columns("A", {0, 0}), da_status_success);
    EXPECT_EQ(hds.select_rows("A", {0, 0}), da_status_success);
    EXPECT_EQ(hds.select_rows("A", {4, 5}), da_status_success);
    ld = 6;
    EXPECT_EQ(hds.extract_selection("A", column_major, ld, islice.data()),
              da_status_success);
    EXPECT_ARR_EQ(24, islice, idata, 1, 1, 0, 0);

    // start another selection of cols only
    std::fill(islice.begin(), islice.end(), 0);
    EXPECT_EQ(hds.select_columns("colsel", {0, 1}), da_status_success);
    EXPECT_EQ(hds.select_columns("colsel", {3, 3}), da_status_success);
    ld = 6;
    EXPECT_EQ(hds.extract_selection("colsel", column_major, ld, islice.data()),
              da_status_success);
    expected_slice = {1, 3, 5, 7, 10, 21, 2, 4, 6, 8, 11, 22, 5, 6, 7, 8, 13, 24};
    EXPECT_ARR_EQ(18, islice, expected_slice, 1, 1, 0, 0);

    // create a new homogeneous data store and extract without selection
    da_errors::da_error_t err2(da_errors::action_t::DA_RECORD);
    data_store ds = data_store(err2);
    da_order order;
    std::vector<da_int> bl1, bl2, bl3;
    get_block_data_int(test1_rblock1, m, n, bl1, order);
    ds.concatenate_columns(m, n, bl1.data(), order);
    get_block_data_int(test1_cblock1, m, n, bl2, order);
    ds.concatenate_columns(m, n, bl2.data(), order);
    da_int new_m;
    get_block_data_int(test1_2rows, new_m, n, bl3, order);
    ds.concatenate_rows(new_m, n, bl3.data(), order);
    ld = 7;
    EXPECT_EQ(ds.extract_selection("", column_major, ld, islice.data()),
              da_status_full_extraction);
    expected_slice = {1, 3, 5, 7, 9, 2, 3, 2, 4, 6, 8, 10, 4, 5,
                      1, 3, 5, 7, 9, 6, 7, 2, 4, 6, 8, 10, 8, 9};
    EXPECT_ARR_EQ(28, islice, expected_slice, 1, 1, 0, 0);

    // start another selection of rows only
    std::fill(islice.begin(), islice.end(), 0);
    EXPECT_EQ(ds.select_rows("rowsel", {0, 1}), da_status_success);
    EXPECT_EQ(ds.select_rows("rowsel", {3, 5}), da_status_success);
    ld = 5;
    EXPECT_EQ(ds.extract_selection("rowsel", column_major, ld, islice.data()),
              da_status_success);
    expected_slice = {1, 3, 7, 9, 2, 2, 4, 8, 10, 4, 1, 3, 7, 9, 6, 2, 4, 8, 10, 8};
    EXPECT_ARR_EQ(20, islice, expected_slice, 1, 1, 0, 0);

    // remove [1, 4] from the last row selection
    EXPECT_EQ(ds.remove_rows_from_selection("rowsel", {1, 4}), da_status_success);
    ld = 2;
    EXPECT_EQ(ds.extract_selection("rowsel", column_major, ld, islice.data()),
              da_status_success);
    expected_slice = {1, 2, 2, 4, 1, 6, 2, 8};
    EXPECT_ARR_EQ(8, islice, expected_slice, 1, 1, 0, 0);

    // New column selection, remove some columns in multiple calls
    EXPECT_EQ(ds.select_columns("colsel", {0, 3}), da_status_success);
    EXPECT_EQ(ds.remove_columns_from_selection("colsel", {1, 1}), da_status_success);
    EXPECT_EQ(ds.remove_columns_from_selection("colsel", {0, 2}), da_status_success);
    ld = 7;
    EXPECT_EQ(ds.extract_selection("colsel", column_major, ld, islice.data()),
              da_status_success);
    expected_slice = {2, 4, 6, 8, 10, 8, 9};
    EXPECT_ARR_EQ(7, islice, expected_slice, 1, 1, 0, 0);
}

TEST(datastore, missingData) {

    using namespace da_data;
    da_int m, n;
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    data_store hds = data_store(err);
    std::vector<da_int> idata;
    std::vector<float> fdata;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store(hds, m, n, idata, fdata, sdata);

    // check missing data for strings is always false
    std::string val = "";
    EXPECT_EQ(is_missing_value<std::string>(val), false);
    val = "\0";
    EXPECT_EQ(is_missing_value<std::string>(val), false);

    // set some missing values for integers and floating points
    float missing_float = std::numeric_limits<float>::quiet_NaN();
    da_int int_max = std::numeric_limits<da_int>::max();
    EXPECT_EQ(hds.set_element(0, 2, int_max), da_status_success);
    EXPECT_EQ(hds.set_element(2, 0, int_max), da_status_success);
    EXPECT_EQ(hds.set_element(2, 3, int_max), da_status_success);
    EXPECT_EQ(hds.set_element(2, 5, missing_float), da_status_success);
    EXPECT_EQ(hds.set_element(4, 4, missing_float), da_status_success);

    // select all rows with no missing elements - should remove rows 0, 2 and 4
    std::string tag = "no missing element";
    EXPECT_EQ(hds.select_non_missing(tag, true), da_status_success);

    // select and extract only the integer columns
    EXPECT_EQ(hds.select_columns(tag, {0, 3}), da_status_success);
    std::vector<da_int> int_sel(12);
    EXPECT_EQ(hds.extract_selection(tag, column_major, 3, int_sel.data()),
              da_status_success);
    std::vector<da_int> iexp = {3, 7, 21, 4, 8, 22, 2, 4, 23, 6, 8, 24};
    EXPECT_ARR_EQ(12, int_sel, iexp, 1, 1, 0, 0);

    // New selection: first select rows and remove from that the rows with missing data
    tag = "subset";
    EXPECT_EQ(hds.select_rows(tag, {1, 2}), da_status_success);
    EXPECT_EQ(hds.select_rows(tag, {4, 5}), da_status_success);
    EXPECT_EQ(hds.select_columns(tag, {0, 3}), da_status_success);
    EXPECT_EQ(hds.select_non_missing(tag, true), da_status_success);
    int_sel.resize(8);
    std::fill(int_sel.begin(), int_sel.end(), 0);
    EXPECT_EQ(hds.extract_selection(tag, column_major, 2, int_sel.data()),
              da_status_success);
    iexp = {3, 21, 4, 22, 2, 23, 6, 24};
    EXPECT_ARR_EQ(8, int_sel, iexp, 1, 1, 0, 0);

    // try with checking only the columns in the selection
    tag = "int partial rows";
    bool full_rows = false;
    EXPECT_EQ(hds.select_columns(tag, {0, 1}), da_status_success);
    EXPECT_EQ(hds.select_non_missing(tag, full_rows), da_status_success);
    int_sel.resize(10);
    std::fill(int_sel.begin(), int_sel.end(), 0);
    EXPECT_EQ(hds.extract_selection(tag, column_major, 5, int_sel.data()),
              da_status_success);
    iexp = {1, 3, 7, 10, 21, 2, 4, 8, 11, 22};
    EXPECT_ARR_EQ(10, int_sel, iexp, 1, 1, 0, 0);

    // same with the floats
    tag = "float partial rows";
    EXPECT_EQ(hds.select_columns(tag, {4, 4}), da_status_success);
    full_rows = false;
    EXPECT_EQ(hds.select_non_missing(tag, full_rows), da_status_success);
    std::vector<float> float_sel(5);
    EXPECT_EQ(hds.extract_selection(tag, column_major, 5, float_sel.data()),
              da_status_success);
    std::vector<float> fexp = {0.5f, 1.5f, 2.5f, 3.5f, 10.1f};
    EXPECT_ARR_EQ(4, float_sel, fexp, 1, 1, 0, 0);

    // select all rows one by one
    tag = "all rows 1by1";
    EXPECT_EQ(hds.select_rows(tag, {0, 0}), da_status_success);
    EXPECT_EQ(hds.select_rows(tag, {1, 1}), da_status_success);
    EXPECT_EQ(hds.select_rows(tag, {2, 2}), da_status_success);
    EXPECT_EQ(hds.select_rows(tag, {3, 3}), da_status_success);
    EXPECT_EQ(hds.select_rows(tag, {4, 4}), da_status_success);
    EXPECT_EQ(hds.select_rows(tag, {5, 5}), da_status_success);
    EXPECT_EQ(hds.select_columns(tag, {0, 0}), da_status_success);
    EXPECT_EQ(hds.select_columns(tag, {1, 1}), da_status_success);
    full_rows = false;
    EXPECT_EQ(hds.select_non_missing(tag, full_rows), da_status_success);
    int_sel.resize(10);
    std::fill(int_sel.begin(), int_sel.end(), 0);
    EXPECT_EQ(hds.extract_selection(tag, column_major, 5, int_sel.data()),
              da_status_success);
    iexp = {1, 3, 7, 10, 21, 2, 4, 8, 11, 22};
    EXPECT_ARR_EQ(10, int_sel, iexp, 1, 1, 0, 0);
}
