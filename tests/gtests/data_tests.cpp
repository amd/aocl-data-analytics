#include "aoclda.h"
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

    EXPECT_THROW(block_dense<da_int> b(-1, 2, data), std::invalid_argument);
    EXPECT_THROW(block_dense<da_int> b(1, 0, data), std::invalid_argument);
    EXPECT_THROW(block_dense<da_int> b(1, 2, nullptr), std::invalid_argument);
    block_dense<da_int> b(1, 2, data);

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

    col1_exp = {1, 3, 5, 7, 9};
    col2_exp = {2, 4, 6, 8, 10};

    // Check column extraction in for the row ordering
    get_block_data_int(test1_rblock1, m, n, bl, order);
    block_dense<da_int> b1(m, n, bl.data(), order);
    b1.get_col(0, &col, stride);
    EXPECT_ARR_EQ(m, col, col1_exp, stride, 1, startx, starty);
    b1.get_col(1, &col, stride);
    EXPECT_ARR_EQ(m, col, col2_exp, stride, 1, startx, starty);
    // Check column extraction in for the col ordering
    get_block_data_int(test1_cblock1, m, n, bl, order);
    block_dense<da_int> b2(m, n, bl.data(), order);
    b2.get_col(0, &col, stride);
    EXPECT_ARR_EQ(m, col, col1_exp, stride, 1, startx, starty);
    b2.get_col(1, &col, stride);
    EXPECT_ARR_EQ(m, col, col2_exp, stride, 1, startx, starty);
}

TEST(dataStore, invalidConcat) {
    std::vector<da_int> bl1, bl2, bl3, bl4;
    da_ordering order;

    data_store ds = data_store();
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

TEST(dataStore, extractCol) {
    da_int m, n;
    std::vector<da_int> bl1, bl2, bl3;
    da_ordering order;
    da_int startx = 0, starty = 0;

    data_store ds = data_store();
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
    data_store hds = data_store();
    std::vector<da_int> idata, coli;
    std::vector<float> fdata;
    std::vector<std::string> sdata;
    get_heterogeneous_data_store(hds, m, n, idata, fdata, sdata);
    coli.resize(m);
    hds.extract_column(0, m, coli.data());
    EXPECT_ARR_EQ(m, coli, idata, 1, 1, startx, starty);
    EXPECT_EQ(hds.extract_column(6, m, coli.data()), da_status_invalid_input);
}

TEST(intervalMap, invalidInput) {
    using namespace da_interval_map;
    interval_map<double> imap;
    da_int lb, ub;
    double d;

    // invalid bounds
    EXPECT_EQ(imap.insert(2, 0, 1.0), da_status_invalid_input);

    // Insert correct interval [0,2]
    EXPECT_EQ(imap.insert(0, 2, 1.0), da_status_success);

    // find values outside of the inserted intervals
    EXPECT_EQ(imap.find(-1, d, lb, ub), false);
    EXPECT_EQ(imap.find(3, d, lb, ub), false);
    EXPECT_EQ(imap.find(-1), imap.end());
    EXPECT_EQ(imap.find(3), imap.end());
    EXPECT_EQ(imap.find(1, d, lb, ub), true);
    EXPECT_EQ(d, 1.0);
    EXPECT_EQ(lb, 0);
    EXPECT_EQ(ub, 2);
    auto it = imap.find(1);
    EXPECT_EQ(it->second, 1.0);
    EXPECT_EQ(it->first.first, 0);
    EXPECT_EQ(it->first.second, 2);

    // overlapping intervals
    EXPECT_EQ(imap.insert(1, 3, 1.0), da_status_invalid_input);
    EXPECT_EQ(imap.insert(2, 3, 1.0), da_status_invalid_input);
    EXPECT_EQ(imap.insert(-1, 0, 1.0), da_status_invalid_input);
    EXPECT_EQ(imap.insert(0, 0, 1.0), da_status_invalid_input);
    EXPECT_EQ(imap.insert(2, 2, 1.0), da_status_invalid_input);

    // add disjointed interval and try to find a value between them
    EXPECT_EQ(imap.insert(5, 10, 2.0), da_status_success);
    EXPECT_EQ(imap.find(4, d, lb, ub), false);
}

TEST(intervalMap, positive) {

    using namespace da_interval_map;
    interval_map<char> imap;
    char c;
    da_int lb, ub;

    EXPECT_EQ(imap.insert(0, 2, 'a'), da_status_success);
    EXPECT_EQ(imap.insert(4, 9, 'b'), da_status_success);
    EXPECT_EQ(imap.find(0, c, lb, ub), true);
    EXPECT_EQ(c, 'a');
    EXPECT_EQ(lb, 0);
    EXPECT_EQ(ub, 2);
    auto it = imap.find(0);
    EXPECT_EQ(it->second, 'a');
    EXPECT_EQ(imap.find(2, c, lb, ub), true);
    EXPECT_EQ(c, 'a');
    EXPECT_EQ(lb, 0);
    EXPECT_EQ(ub, 2);
    EXPECT_EQ(imap.find(5, c, lb, ub), true);
    EXPECT_EQ(c, 'b');
    EXPECT_EQ(lb, 4);
    EXPECT_EQ(ub, 9);
    it = imap.find(5);
    EXPECT_EQ(it->second, 'b');
    EXPECT_EQ(imap.find(9, c, lb, ub), true);
    EXPECT_EQ(c, 'b');
    EXPECT_EQ(lb, 4);
    EXPECT_EQ(ub, 9);
    it = imap.find(9);
    EXPECT_EQ(it->second, 'b');

    EXPECT_EQ(imap.insert(15, 20, 'c'), da_status_success);
    EXPECT_EQ(imap.find(17, c, lb, ub), true);
    EXPECT_EQ(c, 'c');
    EXPECT_EQ(lb, 15);
    EXPECT_EQ(ub, 20);
    it = imap.find(17);
    EXPECT_EQ(it->second, 'c');
}

TEST(intervalMap, erase) {
    using namespace da_interval_map;
    interval_map<char> imap;

    // insert intervals
    // [0,2] [4,9] [10,11] [12, 22] [24, 28] [30, 35] [55, 60]
    EXPECT_EQ(imap.insert(0, 2, 'a'), da_status_success);
    EXPECT_EQ(imap.insert(4, 9, 'b'), da_status_success);
    EXPECT_EQ(imap.insert(10, 11, 'c'), da_status_success);
    EXPECT_EQ(imap.insert(12, 22, 'd'), da_status_success);
    EXPECT_EQ(imap.insert(55, 60, 'g'), da_status_success);
    EXPECT_EQ(imap.insert(30, 35, 'f'), da_status_success);
    EXPECT_EQ(imap.insert(24, 28, 'e'), da_status_success);

    // erase a few intervals
    // leaves: [0,2] [24, 28] [30, 35] [55, 60]
    EXPECT_EQ(imap.erase(13)->second, 'e');
    EXPECT_EQ(imap.find(15), imap.end());
    interval_map<char>::iterator it1, it2;
    it1 = imap.find(9);
    it2 = imap.find(27);
    imap.erase(it1, it2);
    EXPECT_EQ(imap.find(5), imap.end());
    EXPECT_EQ(imap.find(10), imap.end());
    EXPECT_EQ(imap.find(25)->second, 'e');
    EXPECT_EQ(imap.find(35)->second, 'f');
    EXPECT_EQ(imap.find(55)->second, 'g');

    // try to erase invalid keys or iterators
    EXPECT_EQ(imap.erase(12), imap.end());
    EXPECT_EQ(imap.erase(imap.end(), imap.end()), imap.end());

    // erase all intervals from [30, 35]
    // [0,2] [24, 28]
    it1 = imap.find(31);
    it2 = imap.end();
    imap.erase(it1, it2);
    EXPECT_EQ(imap.find(35), imap.end());
    EXPECT_EQ(imap.find(59), imap.end());
    EXPECT_EQ(imap.find(1)->second, 'a');
    EXPECT_EQ(imap.find(28)->second, 'e');

    // erase [24, 28] with single iterator
    it1 = imap.find(26);
    imap.erase(it1);
    EXPECT_EQ(imap.find(28), imap.end());
}

TEST(intervalMap, iterator) {
    using namespace da_interval_map;
    interval_map<char> imap;

    EXPECT_EQ(imap.insert(0, 2, 'a'), da_status_success);
    EXPECT_EQ(imap.insert(4, 9, 'b'), da_status_success);
    EXPECT_EQ(imap.insert(10, 10, 'c'), da_status_success);
    EXPECT_EQ(imap.insert(12, 20, 'd'), da_status_success);

    char vals[4] = {'a', 'b', 'c', 'd'};
    da_int i = 0;
    for (auto it = imap.begin(); it != imap.end(); ++it) {
        EXPECT_EQ(it->second, vals[i]);
        i++;
    }
    EXPECT_EQ(i, 4);
    i = 0;
    for (auto it = imap.begin(); it != imap.end(); it++) {
        EXPECT_EQ((*it).second, vals[i]);
        i++;
    }
    EXPECT_EQ(i, 4);
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

TEST(dataStore, hconcat) {
    // create 2 heterogeneous data_store
    da_int m, n;
    data_store hds = data_store();
    std::vector<da_int> idata, coli, coli2, coli3;
    std::vector<float> fdata, colf, colf2, colf3;
    std::vector<std::string> sdata, cols, cols2, cols3;
    get_heterogeneous_data_store(hds, m, n, idata, fdata, sdata);
    data_store hds2 = data_store();
    get_heterogeneous_data_store(hds2, m, n, idata, fdata, sdata);
    data_store hds3 = data_store();
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
