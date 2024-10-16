/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

/* disable some MSVC warnings about strcat */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "csv_utils.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <list>
#include <string>

template <typename T> class CSVTest_public : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> class DataStoreTest_public : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using CSVTypes = ::testing::Types<float, double, da_int, uint8_t, char *>;
using DataStoreTypes = ::testing::Types<float, double, da_int, uint8_t, char *>;

TYPED_TEST_SUITE(CSVTest_public, CSVTypes);
TYPED_TEST_SUITE(DataStoreTest_public, DataStoreTypes);

TYPED_TEST(CSVTest_public, basic_no_headings_row_major) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");
    std::cout << filepath << std::endl;

    da_datastore store = nullptr;
    da_status err = da_datastore_init(&store);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "thousands", "f"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    TypeParam *a = nullptr;

    da_int nrows = 0, ncols = 0;

    err = da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr);

    EXPECT_EQ(err, params->expected_status);

    EXPECT_EQ(nrows, params->expected_rows);
    EXPECT_EQ(ncols, params->expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            EXPECT_EQ_overload(a[j + ncols * i], GetExpectedData(params, j + ncols * i));
        }
    }

    da_test::free_data(&a, nrows * ncols);

    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(CSVTest_public, basic_no_headings_column_major) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicDataColMajor(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    da_status err = da_datastore_init(&store);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "thousands", "f"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "column-major"),
              da_status_success);
    TypeParam *a = nullptr;

    da_int nrows = 0, ncols = 0;

    err = da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr);

    EXPECT_EQ(err, params->expected_status);

    EXPECT_EQ(nrows, params->expected_rows);
    EXPECT_EQ(ncols, params->expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            // Check against the transpose since we have switched to column-major in this test
            EXPECT_EQ_overload(a[j + ncols * i], GetExpectedData(params, j + ncols * i));
        }
    }

    da_test::free_data(&a, nrows * ncols);

    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(DataStoreTest_public, datastore_no_headings_row_major) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(store, "datatype", params->datatype.c_str()),
        da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "thousands", "f"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);

    da_int nrows = 0, ncols = 0;

    EXPECT_EQ(da_data_load_from_csv(store, filepath), params->expected_status);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, params->expected_rows);
    EXPECT_EQ(ncols, params->expected_columns);

    TypeParam *T = new TypeParam[nrows];

    for (da_int i = 0; i < ncols; i++) {
        EXPECT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            EXPECT_EQ_overload(T[j], GetExpectedData(params, i + ncols * j));
        }
    }

    delete[] T;
    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(DataStoreTest_public, datastore_no_headings_column_major) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(store, "datatype", params->datatype.c_str()),
        da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "thousands", "f"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "column-major"),
              da_status_success);

    da_int nrows = 0, ncols = 0;

    EXPECT_EQ(da_data_load_from_csv(store, filepath), params->expected_status);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, params->expected_rows);
    EXPECT_EQ(ncols, params->expected_columns);

    TypeParam *T = new TypeParam[nrows];

    for (da_int i = 0; i < ncols; i++) {
        EXPECT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            EXPECT_EQ_overload(T[j], GetExpectedData(params, i + ncols * j));
        }
    }

    delete[] T;
    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(CSVTest_public, basic_headings_row_major) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, "_head");

    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    da_status err = da_datastore_init(&store);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);

    TypeParam *a = nullptr;

    da_int nrows = 0, ncols = 0;
    char **headings = nullptr;

    err = da_read_csv(store, filepath, &a, &nrows, &ncols, &headings);

    EXPECT_EQ(err, params->expected_status);

    EXPECT_EQ(nrows, params->expected_rows);
    EXPECT_EQ(ncols, params->expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            EXPECT_EQ_overload(a[j + ncols * i], GetExpectedData(params, j + ncols * i));
        }
    }

    for (da_int j = 0; j < ncols; j++) {
        EXPECT_EQ_overload(headings[j], params->expected_headings[j].c_str());
    }

    da_test::free_data(&a, nrows * ncols);
    EXPECT_EQ(da_delete_string_array(&headings, ncols), da_status_success);

    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(DataStoreTest_public, datastore_headings_row_major) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, "_head");
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(store, "datatype", params->datatype.c_str()),
        da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);

    da_int nrows = 0, ncols = 0;

    EXPECT_EQ(da_data_load_from_csv(store, filepath), params->expected_status);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, params->expected_rows);
    EXPECT_EQ(ncols, params->expected_columns);

    TypeParam *T = new TypeParam[nrows];

    for (da_int i = 0; i < ncols; i++) {
        EXPECT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            EXPECT_EQ_overload(T[j], GetExpectedData(params, i + ncols * j));
        }
    }

    delete[] T;

    char **headings = new char *[ncols];

    da_int name_sz = 128;
    char col_name[128];
    for (da_int j = 0; j < ncols; j++) {
        EXPECT_EQ(da_data_get_col_label(store, j, &name_sz, col_name), da_status_success);
        EXPECT_EQ_overload(col_name, params->expected_headings[j].c_str());
    }

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    delete params;
    delete[] headings;
}

TYPED_TEST(DataStoreTest_public, datastore_headings_column_major) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, "_head");
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(store, "datatype", params->datatype.c_str()),
        da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "column-major"),
              da_status_success);

    da_int nrows = 0, ncols = 0;

    EXPECT_EQ(da_data_load_from_csv(store, filepath), params->expected_status);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, params->expected_rows);
    EXPECT_EQ(ncols, params->expected_columns);

    TypeParam *T = new TypeParam[nrows];

    for (da_int i = 0; i < ncols; i++) {
        EXPECT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            EXPECT_EQ_overload(T[j], GetExpectedData(params, i + ncols * j));
        }
    }

    delete[] T;

    char **headings = new char *[ncols];

    da_int name_sz = 128;
    char col_name[128];
    for (da_int j = 0; j < ncols; j++) {
        EXPECT_EQ(da_data_get_col_label(store, j, &name_sz, col_name), da_status_success);
        EXPECT_EQ_overload(col_name, params->expected_headings[j].c_str());
    }

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    delete params;
    delete[] headings;
}

TYPED_TEST(CSVTest_public, warn_for_missing_data) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetMissingData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    TypeParam *a = nullptr;

    da_int nrows = 0, ncols = 0;

    EXPECT_EQ(da_datastore_options_set_int(store, "warn for missing data", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr),
              params->expected_status);

    EXPECT_EQ(nrows, params->expected_rows);
    EXPECT_EQ(ncols, params->expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            if (check_nan(GetExpectedData(params, j + ncols * i))) {
                EXPECT_TRUE(check_nan(a[j + ncols * i]));
            } else {
                EXPECT_EQ_overload(a[j + ncols * i],
                                   GetExpectedData(params, j + ncols * i));
            }
        }
    }

    da_test::free_data(&a, nrows * ncols);
    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(DataStoreTest_public, warn_for_missing_data_row_major) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetMissingData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int nrows = 0, ncols = 0;

    EXPECT_EQ(da_datastore_options_set_int(store, "warn for missing data", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(store, "datatype", params->datatype.c_str()),
        da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);

    EXPECT_EQ(da_data_load_from_csv(store, filepath), params->expected_status);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, params->expected_rows);
    EXPECT_EQ(ncols, params->expected_columns);

    TypeParam *T = new TypeParam[nrows];

    for (da_int i = 0; i < ncols; i++) {
        EXPECT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            if (check_nan(GetExpectedData(params, i + ncols * j))) {
                EXPECT_TRUE(check_nan(T[j]));
            } else {
                EXPECT_EQ_overload(T[j], GetExpectedData(params, i + ncols * j));
            }
        }
    }

    delete[] T;

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(DataStoreTest_public, warn_for_missing_data_column_major) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetMissingData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int nrows = 0, ncols = 0;

    EXPECT_EQ(da_datastore_options_set_int(store, "warn for missing data", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(store, "datatype", params->datatype.c_str()),
        da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "column-major"),
              da_status_success);

    EXPECT_EQ(da_data_load_from_csv(store, filepath), params->expected_status);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, params->expected_rows);
    EXPECT_EQ(ncols, params->expected_columns);

    TypeParam *T = new TypeParam[nrows];

    for (da_int i = 0; i < ncols; i++) {
        EXPECT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            if (check_nan(GetExpectedData(params, i + ncols * j))) {
                EXPECT_TRUE(check_nan(T[j]));
            } else {
                EXPECT_EQ_overload(T[j], GetExpectedData(params, i + ncols * j));
            }
        }
    }

    delete[] T;

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    delete params;
}

TEST(CSVTest_public, skip_lines_test1) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_skip_lines.csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    double *a = nullptr;

    da_int nrows = 0, ncols = 0;

    da_int expected_rows = 3;
    da_int expected_columns = 5;

    double expected_data[] = {1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0, 8.0,
                              9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};

    // Set options
    EXPECT_EQ(da_datastore_options_set_int(store, "skip empty lines", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "row start", 3), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);

    EXPECT_EQ(da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_success);

    EXPECT_EQ(nrows, expected_rows);
    EXPECT_EQ(ncols, expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            EXPECT_EQ(a[j + ncols * i], expected_data[j + ncols * i]);
        }
    }

    if (a)
        free(a);

    da_datastore_destroy(&store);
}

TEST(CSVTest_public, skip_lines_test2) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_skip_lines.csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    double *a = nullptr;

    da_int nrows = 0, ncols = 0;

    da_int expected_rows = 3;
    da_int expected_columns = 5;

    double expected_data[] = {1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0, 8.0,
                              9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};

    // Set options
    EXPECT_EQ(da_datastore_options_set_int(store, "skip empty lines", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);

    EXPECT_EQ(da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_success);

    EXPECT_EQ(nrows, expected_rows);
    EXPECT_EQ(ncols, expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            EXPECT_EQ(a[j + ncols * i], expected_data[j + ncols * i]);
        }
    }

    if (a)
        free(a);

    da_datastore_destroy(&store);
}

TEST(CSVTest_public, options) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_options.csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    double *a = nullptr;

    da_int nrows = 0, ncols = 0;

    da_int expected_rows = 3;
    da_int expected_columns = 5;

    double expected_data[] = {1.1,     1e3, 1000000000, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                              -5.6e-7, -10, 0.0,        0.0,    0.0, 0.0, 4.5e+5};

    EXPECT_EQ(da_datastore_options_set_string(store, "delimiter", "x"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "thousands", ","),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "decimal", "p"), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "comment", "}"), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(store, "scientific notation character", "g"),
        da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip empty lines", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip footer", 1), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "row start", 3), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "skip rows", "5 9"),
              da_status_success);

    EXPECT_EQ(da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_success);

    EXPECT_EQ(nrows, expected_rows);
    EXPECT_EQ(ncols, expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            EXPECT_EQ(a[j + ncols * i], expected_data[j + ncols * i]);
        }
    }

    if (a)
        free(a);

    // Now try with a datastore object

    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "double"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_success);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, expected_rows);
    EXPECT_EQ(ncols, expected_columns);

    double *T = new double[nrows];

    for (da_int i = 0; i < ncols; i++) {
        EXPECT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            EXPECT_EQ(T[j], expected_data[i + ncols * j]);
        }
    }

    delete[] T;

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);
}

TEST(csvtest, incorrect_headings) {
    char filename[] = "csv_test_incorrect_headings.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, filename);

    double *a = nullptr;
    da_datastore store = nullptr;
    da_int nrows = 0, ncols = 0;
    char **headings = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, &headings),
              da_status_parsing_error);
    da_datastore_destroy(&store);
    if (headings != nullptr) {
        EXPECT_EQ(da_delete_string_array(&headings, ncols), da_status_success);
    }
}

TEST(csvtest, incorrect_headings2) {
    char filename[] = "csv_test_incorrect_headings2.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, filename);

    double *a = nullptr;
    da_datastore store = nullptr;
    da_int nrows = 0, ncols = 0;
    char **headings = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, &headings),
              da_status_parsing_error);
    da_datastore_destroy(&store);
    if (headings != nullptr) {
        EXPECT_EQ(da_delete_string_array(&headings, ncols), da_status_success);
    }
}

TEST(csvtest, error_exits) {
    char filename[] = "csv_test_errors.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, filename);

    // Check for uninitialized handle
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_options_set_int(store, "whitespace delimiter", 1),
              da_status_store_not_initialized);
    double *a = nullptr;
    float *a_f = nullptr;
    da_int *a_int = nullptr;
    uint8_t *a_uint8 = nullptr;
    char **a_char = nullptr;

    da_int nrows = 0, ncols = 0;
    EXPECT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_store_not_initialized);
    EXPECT_EQ(da_read_csv_s(store, filepath, &a_f, &nrows, &ncols, nullptr),
              da_status_store_not_initialized);
    EXPECT_EQ(da_read_csv_int(store, filepath, &a_int, &nrows, &ncols, nullptr),
              da_status_store_not_initialized);
    EXPECT_EQ(da_read_csv_uint8(store, filepath, &a_uint8, &nrows, &ncols, nullptr),
              da_status_store_not_initialized);
    EXPECT_EQ(da_read_csv_string(store, filepath, &a_char, &nrows, &ncols, nullptr),
              da_status_store_not_initialized);

    // Check for various error exits
    double *a_double = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "whitespace delimiter", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);

    EXPECT_EQ(da_read_csv_d(store, filepath, &a_double, &nrows, &ncols, nullptr),
              da_status_parsing_error);
    if (a_double != nullptr) {
        free(a_double);
        a_double = nullptr;
    }
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "double"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_parsing_error);

    EXPECT_EQ(da_read_csv_int(store, filepath, &a_int, &nrows, &ncols, nullptr),
              da_status_parsing_error);
    if (a_int != nullptr) {
        free(a_int);
        a_int = nullptr;
    }
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "integer"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_parsing_error);

    EXPECT_EQ(da_datastore_options_set_string(store, "skip rows", "0"),
              da_status_success);
    EXPECT_EQ(da_read_csv_uint8(store, filepath, &a_uint8, &nrows, &ncols, nullptr),
              da_status_parsing_error);
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "boolean"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_parsing_error);

    EXPECT_EQ(da_read_csv_d(store, filepath, &a_double, &nrows, &ncols, nullptr),
              da_status_parsing_error);
    if (a_double != nullptr) {
        free(a_double);
        a_double = nullptr;
    }
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "double"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_parsing_error);

    EXPECT_EQ(da_read_csv_int(store, filepath, &a_int, &nrows, &ncols, nullptr),
              da_status_parsing_error);
    if (a_int != nullptr) {
        free(a_int);
        a_int = nullptr;
    }
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "integer"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_parsing_error);

    EXPECT_EQ(da_datastore_options_set_string(store, "skip rows", "0, 1"),
              da_status_success);

    EXPECT_EQ(da_datastore_options_set_int(store, "row start", 3), da_status_success);
    EXPECT_EQ(da_read_csv_int(store, filepath, &a_int, &nrows, &ncols, nullptr),
              da_status_parsing_error);
    if (a_int != nullptr) {
        free(a_int);
        a_int = nullptr;
    }
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "integer"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_parsing_error);

    EXPECT_EQ(da_datastore_options_set_int(store, "row start", 4), da_status_success);
    EXPECT_EQ(da_read_csv_int(store, filepath, &a_int, &nrows, &ncols, nullptr),
              da_status_parsing_error);
    if (a_int != nullptr) {
        free(a_int);
        a_int = nullptr;
    }
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "auto"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_parsing_error);

    EXPECT_EQ(da_delete_string_array(nullptr, (da_int)(-1)),
              da_status_invalid_array_dimension);

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    if (a != nullptr)
        free(a);
    if (a_double != nullptr)
        free(a_double);
}

TEST(csvtest, no_data) {
    char filename[] = "csv_test_empty.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, filename);

    // Check for uninitialized handle
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    double *a = nullptr;
    char **headings = nullptr;
    const char *expected_headings[] = {"one", "two", "three", "four", "five"};
    da_int nrows = 0, ncols = 0;

    // Check we can handle headings but no other data
    da_datastore_options_set_int(store, "use header row", 1);
    EXPECT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, &headings),
              da_status_parsing_error);
    if (a != nullptr) {
        // Added for coverity checks. Should not be exercised.
        free(a);
        a = nullptr;
    }
    EXPECT_EQ(nrows, 0);
    EXPECT_EQ(ncols, 5);
    for (da_int j = 0; j < ncols; j++) {
        EXPECT_STREQ(headings[j], expected_headings[j]);
    }
    EXPECT_EQ(da_delete_string_array(&headings, ncols), da_status_success);

    // Now try the same thing in a datastore but just expect no data error
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "double"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_parsing_error);
    da_datastore_destroy(&store);

    // Check we can deal with removing all rows
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 0),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "row start", 1), da_status_success);
    EXPECT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_parsing_error);
    if (a != nullptr) {
        free(a);
        a = nullptr;
    }
    EXPECT_EQ(nrows, 0);
    EXPECT_EQ(ncols, 0);

    da_datastore_options_set_int(store, "use header row", 1);
    EXPECT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, &headings),
              da_status_parsing_error);
    EXPECT_EQ(nrows, 0);
    EXPECT_EQ(ncols, 0);
    if (a != nullptr) {
        free(a);
        a = nullptr;
    }

    da_datastore_destroy(&store);
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 0),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "row start", 1), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "double"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_parsing_error);

    da_datastore_destroy(&store);

    // Check for non-existent csv file
    da_datastore_destroy(&store);
    strcat(filepath, "does_not_exist");
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_file_reading_error);
    if (a != nullptr) {
        free(a);
        a = nullptr;
    }

    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "double"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_file_reading_error);

    da_datastore_destroy(&store);

    if (a)
        free(a);
    if (headings != nullptr) {
        EXPECT_EQ(da_delete_string_array(&headings, ncols), da_status_success);
    }
}

TEST(CSVTest_public, lineterminator) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_lineterminator.csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    da_int *a = nullptr;

    da_int nrows = 0, ncols = 0;

    EXPECT_EQ(da_datastore_options_set_string(store, "line terminator", "x"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_success);

    da_int expected_rows = 2;
    da_int expected_columns = 3;

    da_int expected_data[] = {1, 2, 3, 4, 5, 6};

    EXPECT_EQ(nrows, expected_rows);
    EXPECT_EQ(ncols, expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            EXPECT_EQ(a[j + ncols * i], expected_data[j + ncols * i]);
        }
    }

    if (a)
        free(a);

    // Check the same thing works when reading into a datastore

    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "double"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_success);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, expected_rows);
    EXPECT_EQ(ncols, expected_columns);

    double *T = new double[nrows];

    for (da_int i = 0; i < ncols; i++) {
        EXPECT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            EXPECT_EQ(T[j], expected_data[i + ncols * j]);
        }
    }

    delete[] T;

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);
}

TEST(CSVTest_public, auto_row_major) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_auto.csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int nrows, ncols, expected_rows = 4, expected_columns = 7;

    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "auto"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);

    EXPECT_EQ(da_data_print_options(store), da_status_success);

    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_success);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, expected_rows);
    EXPECT_EQ(ncols, expected_columns);

    const char *expected_headings[] = {"a", "b", "c", "d", "e", "f", "g"};
    da_int c1[] = {1, 2, 3, 4};
    da_int c2[] = {5, 6, 7, 8};
    double c3[] = {4.0, 3.5, 4.0, 6.7};
    double c4[] = {-3.0, -3.0, 3.0, 0.1};
    float cc1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float cc2[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float cc3[] = {4.0f, 3.5f, 4.0f, 6.7f};
    float cc4[] = {-3.0f, -3.0f, 3.0f, 0.1f};
    const char *c6[] = {"1", "-4", "4.1", "false"};
    const char *c7[] = {"hello", "goodbye", "test", "success"};
    uint8_t c5[] = {1, 1, 0, 1};
    da_int Ti[4];
    double Td[4];
    float Tf[4];
    uint8_t Tu[4];
    char *Tc[4];
    char col_name[128];

    EXPECT_EQ(da_data_extract_column_int(store, 0, nrows, Ti), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c1[j], Ti[j]);
    }

    EXPECT_EQ(da_data_extract_column_int(store, 1, nrows, Ti), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c2[j], Ti[j]);
    }

    EXPECT_EQ(da_data_extract_column_real_d(store, 2, nrows, Td), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c3[j], Td[j]);
    }

    EXPECT_EQ(da_data_extract_column_real_d(store, 3, nrows, Td), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c4[j], Td[j]);
    }

    EXPECT_EQ(da_data_extract_column_uint8(store, 4, nrows, Tu), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c5[j], Tu[j]);
    }

    EXPECT_EQ(da_data_extract_column_str(store, 5, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c6[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ_overload(c6[j], Tc[j]);
    }

    EXPECT_EQ(da_data_extract_column_str(store, 6, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c7[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ_overload(c7[j], Tc[j]);
    }

    for (da_int j = 0; j < ncols; j++) {
        da_int name_sz = 128;
        EXPECT_EQ(da_data_get_col_label(store, j, &name_sz, col_name), da_status_success);
        EXPECT_EQ_overload(col_name, expected_headings[j]);
    }

    da_datastore_destroy(&store);

    // Repeat with option set for integer data being interpreted as float
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "auto"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "integers as floats", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "datastore precision", "single"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_success);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, expected_rows);
    EXPECT_EQ(ncols, expected_columns);

    EXPECT_EQ(da_data_extract_column_real_s(store, 0, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(cc1[j], Tf[j]);
    }

    EXPECT_EQ(da_data_extract_column_real_s(store, 1, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(cc2[j], Tf[j]);
    }

    EXPECT_EQ(da_data_extract_column_real_s(store, 2, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(cc3[j], Tf[j]);
    }

    EXPECT_EQ(da_data_extract_column_real_s(store, 3, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(cc4[j], Tf[j]);
    }

    EXPECT_EQ(da_data_extract_column_uint8(store, 4, nrows, Tu), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c5[j], Tu[j]);
    }

    EXPECT_EQ(da_data_extract_column_str(store, 5, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c6[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ_overload(c6[j], Tc[j]);
    }

    EXPECT_EQ(da_data_extract_column_str(store, 6, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c7[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ_overload(c7[j], Tc[j]);
    }

    da_int label_sz = 64;
    char col_label[64];
    for (da_int j = 0; j < ncols; j++) {
        da_data_get_col_label(store, j, &label_sz, col_label);
        EXPECT_EQ_overload(col_label, expected_headings[j]);
    }

    da_datastore_destroy(&store);
}

TEST(CSVTest_public, auto_column_major) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_auto.csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_int nrows, ncols, expected_rows = 4, expected_columns = 7;

    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "auto"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "column-major"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);

    EXPECT_EQ(da_data_print_options(store), da_status_success);

    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_success);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, expected_rows);
    EXPECT_EQ(ncols, expected_columns);

    const char *expected_headings[] = {"a", "b", "c", "d", "e", "f", "g"};
    da_int c1[] = {1, 2, 3, 4};
    da_int c2[] = {5, 6, 7, 8};
    double c3[] = {4.0, 3.5, 4.0, 6.7};
    double c4[] = {-3.0, -3.0, 3.0, 0.1};
    float cc1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float cc2[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float cc3[] = {4.0f, 3.5f, 4.0f, 6.7f};
    float cc4[] = {-3.0f, -3.0f, 3.0f, 0.1f};
    const char *c6[] = {"1", "-4", "4.1", "false"};
    const char *c7[] = {"hello", "goodbye", "test", "success"};
    uint8_t c5[] = {1, 1, 0, 1};
    da_int Ti[4];
    double Td[4];
    float Tf[4];
    uint8_t Tu[4];
    char *Tc[4];
    char col_name[128];

    EXPECT_EQ(da_data_extract_column_int(store, 0, nrows, Ti), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c1[j], Ti[j]);
    }

    EXPECT_EQ(da_data_extract_column_int(store, 1, nrows, Ti), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c2[j], Ti[j]);
    }

    EXPECT_EQ(da_data_extract_column_real_d(store, 2, nrows, Td), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c3[j], Td[j]);
    }

    EXPECT_EQ(da_data_extract_column_real_d(store, 3, nrows, Td), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c4[j], Td[j]);
    }

    EXPECT_EQ(da_data_extract_column_uint8(store, 4, nrows, Tu), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c5[j], Tu[j]);
    }

    EXPECT_EQ(da_data_extract_column_str(store, 5, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c6[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ_overload(c6[j], Tc[j]);
    }

    EXPECT_EQ(da_data_extract_column_str(store, 6, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c7[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ_overload(c7[j], Tc[j]);
    }

    for (da_int j = 0; j < ncols; j++) {
        da_int name_sz = 128;
        EXPECT_EQ(da_data_get_col_label(store, j, &name_sz, col_name), da_status_success);
        EXPECT_EQ_overload(col_name, expected_headings[j]);
    }

    da_datastore_destroy(&store);

    // Repeat with option set for integer data being interpreted as float
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "datatype", "auto"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "skip initial space", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "integers as floats", 1),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "datastore precision", "single"),
              da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(store, "storage order", "column-major"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_success);

    EXPECT_EQ(da_data_get_n_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(store, &ncols), da_status_success);

    EXPECT_EQ(nrows, expected_rows);
    EXPECT_EQ(ncols, expected_columns);

    EXPECT_EQ(da_data_extract_column_real_s(store, 0, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(cc1[j], Tf[j]);
    }

    EXPECT_EQ(da_data_extract_column_real_s(store, 1, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(cc2[j], Tf[j]);
    }

    EXPECT_EQ(da_data_extract_column_real_s(store, 2, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(cc3[j], Tf[j]);
    }

    EXPECT_EQ(da_data_extract_column_real_s(store, 3, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(cc4[j], Tf[j]);
    }

    EXPECT_EQ(da_data_extract_column_uint8(store, 4, nrows, Tu), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ(c5[j], Tu[j]);
    }

    EXPECT_EQ(da_data_extract_column_str(store, 5, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c6[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ_overload(c6[j], Tc[j]);
    }

    EXPECT_EQ(da_data_extract_column_str(store, 6, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c7[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        EXPECT_EQ_overload(c7[j], Tc[j]);
    }

    da_int label_sz = 64;
    char col_label[64];
    for (da_int j = 0; j < ncols; j++) {
        da_data_get_col_label(store, j, &label_sz, col_label);
        EXPECT_EQ_overload(col_label, expected_headings[j]);
    }

    da_datastore_destroy(&store);
}
