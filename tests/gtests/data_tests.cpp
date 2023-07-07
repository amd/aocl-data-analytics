#include "aoclda.h"
#include "da_error.hpp"
#include "data_store.hpp"
#include "interval_map.hpp"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <type_traits>

using namespace da_data;

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

enum int_block_id {
    test1_rblock1 = 0,
    test1_cblock1,
    test1_2rows,
};

void get_block_data_int(int_block_id bid, da_int &m, da_int &n, std::vector<da_int> &bl,
                        da_ordering &order) {

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
        order = col_major;
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
    EXPECT_EQ(ds.concatenate_columns(m, n, ib2.data(), col_major, true),
              da_status_success);
    ib3 = {10, 11};
    m = 1;
    n = 2;
    EXPECT_EQ(ds.concatenate_rows(m, n, ib3.data(), col_major, true), da_status_success);
    ib4 = {12, 13};
    m = 1;
    n = 2;
    EXPECT_EQ(ds.concatenate_rows(m, n, ib4.data(), col_major, true), da_status_success);
    fb1 = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
    m = 5;
    n = 2;
    EXPECT_EQ(ds.concatenate_columns(m, n, fb1.data(), col_major, true),
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
    fb2 = {10.1, 20.2};
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
    fdata = {0.5, 1.5, 2.5, 3.5, 4.5, 10.1, 5.5, 6.5, 7.5, 8.5, 9.5, 20.2};
    sdata = {"1", "a2", "bb3", "ccc4", "dddd5", "row6_1"};
    mt = 6;
    nt = 7;
}

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
    EXPECT_EQ(da_data_load_col_int(store, m, n, ib2.data(), col_major, true),
              da_status_success);
    ib3 = {10, 11};
    m = 1;
    n = 2;
    EXPECT_EQ(da_data_load_row_int(store, m, n, ib3.data(), col_major, true),
              da_status_success);
    ib4 = {12, 13};
    m = 1;
    n = 2;
    EXPECT_EQ(da_data_load_row_int(store, m, n, ib4.data(), col_major, true),
              da_status_success);
    fb1 = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
    m = 5;
    n = 2;
    EXPECT_EQ(da_data_load_col_real_s(store, m, n, fb1.data(), col_major, true),
              da_status_success);
    const char *cb1[5];
    cb1[0] = "1";
    cb1[1] = "a2";
    cb1[2] = "bb3";
    cb1[3] = "ccc4";
    cb1[4] = "dddd5";
    m = 5;
    n = 1;
    EXPECT_EQ(da_data_load_col_str(store, m, n, cb1, col_major), da_status_success);
    ib5 = {21, 22, 23, 24};
    m = 1;
    n = 4;
    EXPECT_EQ(da_data_load_row_int(store, m, n, ib5.data(), row_major, true),
              da_status_success);
    fb2 = {10.1, 20.2};
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
    fdata = {0.5, 1.5, 2.5, 3.5, 4.5, 10.1, 5.5, 6.5, 7.5, 8.5, 9.5, 20.2};
    sdata = {"1", "a2", "bb3", "ccc4", "dddd5", "row6_1"};
    mt = 6;
    nt = 7;
}

TEST(block, getCol) {
    std::vector<da_int> bl, col1_exp, col2_exp;
    da_int m, n;
    da_ordering order;
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
    // out of bound column index
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
    block_dense<da_int> b1(m, n, bl_col.data(), err, col_major);

    // load the data from the middle of the block
    std::pair<da_int, da_int> cols, rows;
    cols = {1, 2};
    rows = {1, 3};
    islice.resize(6);
    EXPECT_EQ(b1.copy_slice_dense(cols, rows, 0, 3, islice.data()), da_status_success);
    exp_slice = {3, 5, 7, 4, 6, 8};
    EXPECT_ARR_EQ(6, islice, exp_slice, 1, 1, 0, 0);

    // try to load he block in the middle of the slice
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

    // try to load he block in the middle of the slice
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

    /* column major oredering */
    std::vector<da_int> bl_col = {1, 2, 3, 4, 5,  1, maxi, 5, 7, 9,
                                  2, 4, 6, 8, 10, 6, maxi, 8, 9, maxi};
    block_dense<da_int> b1(m, n, bl_col.data(), err, col_major);
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

    /* row major oredering */
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
    block_dense<da_int> b1(m, n, bl_col.data(), err, col_major);

    std::pair<da_int, da_int> cols, rows;
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
    da_ordering order;
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
    order = col_major;
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
    da_ordering order;
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
    da_ordering order;
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

    // test the hterogeneous data-store columns
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

TEST(dataStore, load) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int m, n, copy_data = 0;
    da_ordering order = row_major;
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
    order = col_major;
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
    da_datastore_destroy(&store);

    // Test row insertions for other data  types
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
    // create 2 heterogeneous data_store
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
    EXPECT_EQ(da_data_load_col_real_d(store1, m, n, dblock.data(), col_major, true),
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

TEST(dataStore, extractSlice) {
    da_int m, n, ld;
    std::vector<da_int> bl1, bl2, bl3;
    da_ordering order;
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    data_store ds = data_store(err);
    get_block_data_int(test1_rblock1, m, n, bl1, order);
    EXPECT_EQ(ds.concatenate_columns(m, n, bl1.data(), order), da_status_success);
    get_block_data_int(test1_cblock1, m, n, bl2, order);
    EXPECT_EQ(ds.concatenate_columns(m, n, bl2.data(), order), da_status_success);

    // Extract the first columns into a slice
    std::pair<da_int, da_int> col_int(0, 1), row_int(0, m - 1);
    ld = row_int.second - row_int.first + 1;
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
    col_int.second = 2;
    ld = row_int.second - row_int.first + 1;
    islice.resize(3 * m);
    ds.extract_slice(row_int, col_int, ld, 0, islice.data());
    expected_slice = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
    EXPECT_ARR_EQ(15, islice, expected_slice, 1, 1, 0, 0);

    // same datastore, partial rows
    row_int.second = 2;
    ld = row_int.second - row_int.first + 1;
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
    ld = row_int.second - row_int.first + 1;
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
    ld = row_int.second - row_int.first + 1;
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
    ld = row_int.second - row_int.first + 1;
    hds.extract_slice(row_int, col_int, ld, 0, islice.data());
    expected_slice = {10, 11, 12, 13};
    EXPECT_ARR_EQ(4, islice, expected_slice, 1, 1, 0, 0);

    // Only bottom blocks
    std::fill(islice.begin(), islice.end(), 0);
    row_int = {4, 5};
    col_int = {1, 3};
    ld = row_int.second - row_int.first + 1;
    hds.extract_slice(row_int, col_int, ld, 0, islice.data());
    expected_slice = {11, 22, 12, 23, 13, 24};
    EXPECT_ARR_EQ(6, islice, expected_slice, 1, 1, 0, 0);

    // extract floats
    std::vector<float> fslice;
    fslice.resize(5);
    row_int = {1, 5};
    col_int = {5, 5};
    ld = row_int.second - row_int.first + 1;
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
    ld = row_int.second - row_int.first + 1;
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
    ld = row_int.second - row_int.first + 1;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    row_int = {1, 6};
    ld = row_int.second - row_int.first + 1;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    row_int = {7, 10};
    ld = row_int.second - row_int.first + 1;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);

    // wrong type expected
    col_int = {4, 5};
    row_int = {0, 2};
    ld = row_int.second - row_int.first + 1;
    EXPECT_EQ(hds.extract_slice(row_int, col_int, ld, 0, islice.data()),
              da_status_invalid_input);
    col_int = {0, 5};
    row_int = {0, 2};
    ld = row_int.second - row_int.first + 1;
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
    interval rows, cols;
    std::vector<da_int> expected_slice;
    expected_slice.reserve(100);

    // 1 set of columns and rows
    EXPECT_EQ(hds.select_slice("A", {1, 3}, {1, 3}), da_status_success);
    std::vector<da_int> islice(100);
    ld = 3;
    EXPECT_EQ(hds.extract_selection("A", ld, islice.data()), da_status_success);
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
    EXPECT_EQ(hds.extract_selection("A", ld, islice.data()), da_status_success);
    expected_slice = {4, 6, 8, 2, 3, 4, 6, 7, 8};
    EXPECT_ARR_EQ(9, islice, expected_slice, 1, 1, 0, 0);

    // add the rest of the integer data from hds
    EXPECT_EQ(hds.select_columns("A", {0, 0}), da_status_success);
    EXPECT_EQ(hds.select_rows("A", {0, 0}), da_status_success);
    EXPECT_EQ(hds.select_rows("A", {4, 5}), da_status_success);
    ld = 6;
    EXPECT_EQ(hds.extract_selection("A", ld, islice.data()), da_status_success);
    EXPECT_ARR_EQ(24, islice, idata, 1, 1, 0, 0);

    // start another selection of cols only
    std::fill(islice.begin(), islice.end(), 0);
    EXPECT_EQ(hds.select_columns("colsel", {0, 1}), da_status_success);
    EXPECT_EQ(hds.select_columns("colsel", {3, 3}), da_status_success);
    ld = 6;
    EXPECT_EQ(hds.extract_selection("colsel", ld, islice.data()), da_status_success);
    expected_slice = {1, 3, 5, 7, 10, 21, 2, 4, 6, 8, 11, 22, 5, 6, 7, 8, 13, 24};
    EXPECT_ARR_EQ(18, islice, expected_slice, 1, 1, 0, 0);

    // create a new homogeneous data store and extract without selection
    da_errors::da_error_t err2(da_errors::action_t::DA_RECORD);
    data_store ds = data_store(err2);
    da_ordering order;
    std::vector<da_int> bl1, bl2, bl3;
    get_block_data_int(test1_rblock1, m, n, bl1, order);
    ds.concatenate_columns(m, n, bl1.data(), order);
    get_block_data_int(test1_cblock1, m, n, bl2, order);
    ds.concatenate_columns(m, n, bl2.data(), order);
    da_int new_m;
    get_block_data_int(test1_2rows, new_m, n, bl3, order);
    ds.concatenate_rows(new_m, n, bl3.data(), order);
    ld = 7;
    EXPECT_EQ(ds.extract_selection("", ld, islice.data()), da_status_success);
    expected_slice = {1, 3, 5, 7, 9, 2, 3, 2, 4, 6, 8, 10, 4, 5,
                      1, 3, 5, 7, 9, 6, 7, 2, 4, 6, 8, 10, 8, 9};
    EXPECT_ARR_EQ(28, islice, expected_slice, 1, 1, 0, 0);

    // start another selection of rows only
    std::fill(islice.begin(), islice.end(), 0);
    EXPECT_EQ(ds.select_rows("rowsel", {0, 1}), da_status_success);
    EXPECT_EQ(ds.select_rows("rowsel", {3, 5}), da_status_success);
    ld = 5;
    EXPECT_EQ(ds.extract_selection("rowsel", ld, islice.data()), da_status_success);
    expected_slice = {1, 3, 7, 9, 2, 2, 4, 8, 10, 4, 1, 3, 7, 9, 6, 2, 4, 8, 10, 8};
    EXPECT_ARR_EQ(20, islice, expected_slice, 1, 1, 0, 0);
}

TEST(dataStore, nullStore) {
    da_datastore store = nullptr, store1 = nullptr;
    da_int int_block = 1;
    uint8_t uint_block = 1;
    const char *str_block = "A";
    double d_block = 1.0;
    float s_block = 1.0;
    EXPECT_EQ(da_data_hconcat(&store, &store1), da_status_invalid_input);

    // load cols/rows
    EXPECT_EQ(da_data_load_col_int(store, 1, 1, &int_block, row_major, false),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_col_str(store, 1, 1, &str_block, row_major),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_col_real_d(store, 1, 1, &d_block, row_major, false),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_col_real_s(store, 1, 1, &s_block, row_major, false),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_row_int(store, 1, 1, &int_block, row_major, false),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_row_str(store, 1, 1, &str_block, row_major),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_row_real_d(store, 1, 1, &d_block, row_major, false),
              da_status_invalid_input);
    EXPECT_EQ(da_data_load_row_real_s(store, 1, 1, &s_block, row_major, false),
              da_status_invalid_input);

    // selection
    EXPECT_EQ(da_data_select_columns(store, "A", 1, 1), da_status_invalid_input);
    EXPECT_EQ(da_data_select_rows(store, "A", 1, 1), da_status_invalid_input);
    EXPECT_EQ(da_data_select_slice(store, "A", 1, 1, 1, 1), da_status_invalid_input);

    // extract columns
    EXPECT_EQ(da_data_extract_selection_int(store, "A", 1, &int_block),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_selection_real_d(store, "A", 1, &d_block),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_selection_real_s(store, "A", 1, &s_block),
              da_status_invalid_input);
    EXPECT_EQ(da_data_extract_selection_uint8(store, "A", 1, &uint_block),
              da_status_invalid_input);

    // setters/getters
    da_int ielem;
    double delem;
    float selem;
    uint8_t uielem;
    EXPECT_EQ(da_data_get_num_rows(store, &ielem), da_status_invalid_input);
    EXPECT_EQ(da_data_get_num_cols(store, &ielem), da_status_invalid_input);
    EXPECT_EQ(da_data_get_element_int(store, 1, 1, &ielem), da_status_invalid_input);
    EXPECT_EQ(da_data_get_element_real_d(store, 1, 1, &delem), da_status_invalid_input);
    EXPECT_EQ(da_data_get_element_real_s(store, 1, 1, &selem), da_status_invalid_input);
    EXPECT_EQ(da_data_get_element_uint8(store, 1, 1, &uielem), da_status_invalid_input);
    EXPECT_EQ(da_data_set_element_int(store, 1, 1, ielem), da_status_invalid_input);
    EXPECT_EQ(da_data_set_element_real_d(store, 1, 1, delem), da_status_invalid_input);
    EXPECT_EQ(da_data_set_element_real_s(store, 1, 1, selem), da_status_invalid_input);
    EXPECT_EQ(da_data_set_element_uint8(store, 1, 1, uielem), da_status_invalid_input);
}

TEST(dataStore, extractSelPub) {
    da_datastore store = nullptr;

    // nullptr store in all routines

    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    // load 2x2 int block
    std::vector<da_int> iblock = {1, 2, 3, 4};
    EXPECT_EQ(da_data_load_col_int(store, 2, 2, iblock.data(), col_major, true),
              da_status_success);
    EXPECT_EQ(da_data_select_slice(store, "int", 0, 1, 0, 0), da_status_success);
    std::vector<int> isel(2);
    EXPECT_EQ(da_data_extract_selection_int(store, "int", 2, isel.data()),
              da_status_success);
    std::vector<da_int> iexp = {1, 2};
    EXPECT_ARR_EQ(2, isel, iexp, 1, 1, 0, 0);

    // load 2x2 uint_8
    std::vector<uint8_t> uiblock = {1, 2, 3, 4};
    EXPECT_EQ(da_data_load_col_uint8(store, 2, 2, uiblock.data(), col_major, true),
              da_status_success);
    EXPECT_EQ(da_data_select_slice(store, "uint", 0, 1, 2, 3), da_status_success);
    std::vector<uint8_t> uisel(4);
    EXPECT_EQ(da_data_extract_selection_uint8(store, "uint", 2, uisel.data()),
              da_status_success);
    std::vector<uint8_t> uiexp = {1, 2, 3, 4};
    EXPECT_ARR_EQ(2, uisel, uiexp, 1, 1, 0, 0);

    // load 2x2 float
    std::vector<float> sblock = {1, 2, 3, 4};
    EXPECT_EQ(da_data_load_col_real_s(store, 2, 2, sblock.data(), col_major, true),
              da_status_success);
    EXPECT_EQ(da_data_select_slice(store, "float", 0, 1, 4, 4), da_status_success);
    std::vector<float> ssel(2);
    EXPECT_EQ(da_data_extract_selection_real_s(store, "float", 2, ssel.data()),
              da_status_success);
    std::vector<float> sexp = {1, 2};
    EXPECT_ARR_EQ(2, ssel, sexp, 1, 1, 0, 0);

    // load 2x2 double
    std::vector<double> dblock = {5, 6, 7, 8};
    EXPECT_EQ(da_data_load_col_real_d(store, 2, 2, dblock.data(), col_major, true),
              da_status_success);
    EXPECT_EQ(da_data_select_rows(store, "double", 0, 1), da_status_success);
    EXPECT_EQ(da_data_select_columns(store, "double", 6, 6), da_status_success);
    std::vector<double> dsel(2);
    EXPECT_EQ(da_data_extract_selection_real_d(store, "double", 2, dsel.data()),
              da_status_success);
    std::vector<double> dexp = {5, 6};
    EXPECT_ARR_EQ(2, dsel, dexp, 1, 1, 0, 0);

    da_datastore_destroy(&store);
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

    // set some missing values for integers and floating  points
    float missing_float = std::nan("");
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
    EXPECT_EQ(hds.extract_selection(tag, 3, int_sel.data()), da_status_success);
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
    EXPECT_EQ(hds.extract_selection(tag, 2, int_sel.data()), da_status_success);
    iexp = {3, 21, 4, 22, 2, 23, 6, 24};
    EXPECT_ARR_EQ(8, int_sel, iexp, 1, 1, 0, 0);

    // try with checking only the columns in the selection
    tag = "int partial rows";
    bool full_rows = false;
    EXPECT_EQ(hds.select_columns(tag, {0, 1}), da_status_success);
    EXPECT_EQ(hds.select_non_missing(tag, full_rows), da_status_success);
    int_sel.resize(10);
    std::fill(int_sel.begin(), int_sel.end(), 0);
    EXPECT_EQ(hds.extract_selection(tag, 5, int_sel.data()), da_status_success);
    iexp = {1, 3, 7, 10, 21, 2, 4, 8, 11, 22};
    EXPECT_ARR_EQ(10, int_sel, iexp, 1, 1, 0, 0);

    // same with the floats
    tag = "float partial rows";
    EXPECT_EQ(hds.select_columns(tag, {4, 4}), da_status_success);
    full_rows = false;
    EXPECT_EQ(hds.select_non_missing(tag, full_rows), da_status_success);
    std::vector<float> float_sel(5);
    EXPECT_EQ(hds.extract_selection(tag, 5, float_sel.data()), da_status_success);
    std::vector<float> fexp = {0.5, 1.5, 2.5, 3.5, 10.1};
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
    EXPECT_EQ(hds.extract_selection(tag, 5, int_sel.data()), da_status_success);
    iexp = {1, 3, 7, 10, 21, 2, 4, 8, 11, 22};
    EXPECT_ARR_EQ(10, int_sel, iexp, 1, 1, 0, 0);
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

    // set some missing values for integers and floating  points
    float missing_float = std::nan("");
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
    EXPECT_EQ(da_data_extract_selection_int(store, tag, 3, int_sel.data()),
              da_status_success);
    std::vector<da_int> iexp = {3, 7, 21, 4, 8, 22, 2, 4, 23, 6, 8, 24};
    EXPECT_ARR_EQ(12, int_sel, iexp, 1, 1, 0, 0);

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
    EXPECT_EQ(da_datastore_options_set_int(store, "CSV use header row", 1),
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
