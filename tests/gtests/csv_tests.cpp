/* disable some MSVC warnings about strcat */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "aoclda.h"
#include "da_handle.hpp"
#include "data_store.hpp"
#include "options.hpp"
#include "utest_utils.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <list>
#include <string>

// Overload some ASSERTs nan checks and free'ing functions for potential char* data
template <typename T> void ASSERT_EQ_overload(T d1, T d2) { ASSERT_EQ(d1, d2); }

void ASSERT_EQ_overload(const char *d1, const char *d2) { ASSERT_STREQ(d1, d2); }

template <typename T> bool check_nan(T num) {
    return std::isnan(static_cast<double>(num));
}

bool check_nan([[maybe_unused]] char *num) { return false; }
bool check_nan([[maybe_unused]] const char *num) { return false; }

template <typename T> struct CSVParamType {
    std::string filename;
    da_int expected_rows;
    da_int expected_columns;
    std::vector<T> expected_data;
    std::vector<std::string> expected_char_data;
    std::vector<std::string> expected_headings;
    da_status expected_status;
    std::string datatype;
};

template <typename T> T GetExpectedData(CSVParamType<T> *params, da_int i) {
    return params->expected_data[i];
}

const char *GetExpectedData(CSVParamType<char *> *params, da_int i) {
    return params->expected_char_data[i].c_str();
}

template <typename T> void GetBasicData(CSVParamType<T> *params);

template <> void GetBasicData<double>(CSVParamType<double> *params) {

    params->filename = "csv_test_float";
    params->expected_rows = 3;
    params->expected_columns = 5;
    params->expected_data = {1.1,     1e3, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                             -5.6e-7, -10, 0.0,    0.0,    0.0, 0.0, 4.5e+5};
    params->expected_headings = {"one", "cat two", "three", "FOUR", "Five"};
    params->expected_status = da_status_success;
    params->datatype = "double";
}

template <> void GetBasicData<float>(CSVParamType<float> *params) {

    params->filename = "csv_test_float";
    params->expected_rows = 3;
    params->expected_columns = 5;
    std::vector<float> data{1.1f, 1e3f,  4.1e-3f, 0.03e6f,  2.f,
                            -1.f, -3.2f, -4.5e4f, -5.6e-7f, -10.f,
                            0.0f, 0.0f,  0.0f,    0.0f,     4.5e+5f};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "cat two", "three", "FOUR", "Five"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "float";
}

template <> void GetBasicData<int64_t>(CSVParamType<int64_t> *params) {

    params->filename = "csv_test_int64";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<int64_t> data{1,
                              5,
                              3,
                              0,
                              -0,
                              -43,
                              9223372036854775807,
                              9223372036854775806,
                              +345,
                              -9223372036854775807,
                              -9223372036854775806,
                              67};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "integer";
}

template <> void GetBasicData<int32_t>(CSVParamType<int32_t> *params) {

    params->filename = "csv_test_int32";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<int32_t> data{1,   5,          3,           0,   0,           43,
                              +92, 2147483647, +2147483646, 184, -2147483647, 67};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "integer";
}

template <> void GetBasicData<char *>(CSVParamType<char *> *params) {

    params->filename = "csv_test_char";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<std::string> data = {"lorem", "ipsum",       "dolor",      "sit",
                                     "amet",  "consectetur", "adipiscing", "edit",
                                     "sed",   "do",          "eiusmod",    "tempor"};
    params->expected_char_data = data;
    std::vector<std::string> headings = {"this", "is", "the", "header"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "string";
}

template <> void GetBasicData<uint8_t>(CSVParamType<uint8_t> *params) {

    params->filename = "csv_test_bool";
    params->expected_rows = 2;
    params->expected_columns = 4;
    std::vector<uint8_t> data{1, 1, 1, 1, 1, 0, 0, 0};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "boolean";
}

template <typename T> void GetMissingData(CSVParamType<T> *params);

template <> void GetMissingData<double>(CSVParamType<double> *params) {

    params->filename = "csv_test_float_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 5;
    std::vector<double> data{1.1,     NAN, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                             -5.6e-7, NAN, NAN,    0.0,    0.0, 0.0, 4.5e+5};
    params->expected_data = data;
    params->expected_status = da_status_missing_data;
    params->datatype = "double";
}

template <> void GetMissingData<float>(CSVParamType<float> *params) {

    params->filename = "csv_test_float_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 5;
    params->expected_data = {1.1f,     NAN, 4.1e-3f, 0.03e6f, 2.f,  -1.f, -3.2f,  -4.5e4f,
                             -5.6e-7f, NAN, NAN,     0.0f,    0.0f, 0.0f, 4.5e+5f};
    params->expected_status = da_status_missing_data;
    params->datatype = "float";
}

template <> void GetMissingData<int64_t>(CSVParamType<int64_t> *params) {

    params->filename = "csv_test_int64_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<int64_t> data{1,
                              5,
                              3,
                              INT64_MAX,
                              -0,
                              -43,
                              9223372036854775807,
                              9223372036854775806,
                              INT64_MAX,
                              -9223372036854775807,
                              -9223372036854775806,
                              67};
    params->expected_data = data;
    params->expected_status = da_status_missing_data;
    params->datatype = "integer";
}

template <> void GetMissingData<int32_t>(CSVParamType<int32_t> *params) {

    params->filename = "csv_test_int32_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<int32_t> data{1,   5,   3,         INT32_MAX, -0,   -43,
                              922, 922, INT32_MAX, -922,      -922, 67};
    params->expected_data = data;
    params->expected_status = da_status_missing_data;
    params->datatype = "integer";
}

template <> void GetMissingData<char *>(CSVParamType<char *> *params) {

    params->filename = "csv_test_char_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<std::string> data = {"lorem",       "",      "dolor", "sit", "amet",
                                     "consectetur", "",      "edit",  "sed", "do",
                                     "eiusmod",     "tempor"};
    params->expected_char_data = data;
    params->expected_status = da_status_success;
    params->datatype = "string";
}

template <> void GetMissingData<uint8_t>(CSVParamType<uint8_t> *params) {

    params->filename = "csv_test_bool_missing_data";
    params->expected_rows = 2;
    params->expected_columns = 4;
    std::vector<uint8_t> data{1, 1, UINT8_MAX, 1, 1, 0, UINT8_MAX, 0};
    params->expected_data = data;
    params->expected_status = da_status_missing_data;
    params->datatype = "boolean";
}

template <typename T> class CSVTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> class DataStoreTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using CSVTypes = ::testing::Types<float, double, da_int, uint8_t, char *>;
using DataStoreTypes = ::testing::Types<float, double, da_int, uint8_t, char *>;

TYPED_TEST_SUITE(CSVTest, CSVTypes);
TYPED_TEST_SUITE(DataStoreTest, DataStoreTypes);

TYPED_TEST(CSVTest, basic_no_headings) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    da_status err = da_datastore_init(&store);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV skip initial space", 1),
              da_status_success);
    TypeParam *a = nullptr;

    da_int nrows = 0, ncols = 0;

    err = da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr);

    ASSERT_EQ(err, params->expected_status);

    ASSERT_EQ(nrows, params->expected_rows);
    ASSERT_EQ(ncols, params->expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            ASSERT_EQ_overload(a[j + ncols * i], GetExpectedData(params, j + ncols * i));
        }
    }

    da_csv::free_data(&a, nrows * ncols);

    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(DataStoreTest, datastore_no_headings) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV skip initial space", 1),
              da_status_success);
    ASSERT_EQ(
        da_datastore_options_set_string(store, "CSV datatype", params->datatype.c_str()),
        da_status_success);

    da_int nrows = 0, ncols = 0;

    EXPECT_EQ(da_data_load_from_csv(store, filepath), params->expected_status);

    EXPECT_EQ(da_data_get_num_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_num_cols(store, &ncols), da_status_success);

    ASSERT_EQ(nrows, params->expected_rows);
    ASSERT_EQ(ncols, params->expected_columns);

    TypeParam *T = new TypeParam[nrows];

    for (da_int i = 0; i < ncols; i++) {
        ASSERT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            ASSERT_EQ_overload(T[j], GetExpectedData(params, i + ncols * j));
        }
    }

    delete[] T;
    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(CSVTest, basic_headings) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, "_head");

    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    da_status err = da_datastore_init(&store);
    da_datastore_options_set_int(store, "CSV skip initial space", 1);
    da_datastore_options_set_int(store, "CSV use header row", 1);
    TypeParam *a = nullptr;

    da_int nrows = 0, ncols = 0;
    char **headings = nullptr;

    err = da_read_csv(store, filepath, &a, &nrows, &ncols, &headings);

    ASSERT_EQ(err, params->expected_status);

    ASSERT_EQ(nrows, params->expected_rows);
    ASSERT_EQ(ncols, params->expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            ASSERT_EQ_overload(a[j + ncols * i], GetExpectedData(params, j + ncols * i));
        }
    }

    for (da_int j = 0; j < ncols; j++) {
        ASSERT_EQ_overload(headings[j], params->expected_headings[j].c_str());
    }

    da_csv::free_data(&a, nrows * ncols);
    da_csv::free_data(&headings, ncols);

    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(DataStoreTest, datastore_headings) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, "_head");
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV skip initial space", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV use header row", 1),
              da_status_success);
    ASSERT_EQ(
        da_datastore_options_set_string(store, "CSV datatype", params->datatype.c_str()),
        da_status_success);

    da_int nrows = 0, ncols = 0;

    EXPECT_EQ(da_data_load_from_csv(store, filepath), params->expected_status);

    EXPECT_EQ(da_data_get_num_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_num_cols(store, &ncols), da_status_success);

    ASSERT_EQ(nrows, params->expected_rows);
    ASSERT_EQ(ncols, params->expected_columns);

    TypeParam *T = new TypeParam[nrows];

    for (da_int i = 0; i < ncols; i++) {
        ASSERT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            ASSERT_EQ_overload(T[j], GetExpectedData(params, i + ncols * j));
        }
    }

    delete[] T;

    char **headings = new char *[ncols];

    da_int name_sz = 128;
    char col_name[128];
    for (da_int j = 0; j < ncols; j++) {
        EXPECT_EQ(da_data_get_col_label(store, j, &name_sz, col_name), da_status_success);
        ASSERT_EQ_overload(col_name, params->expected_headings[j].c_str());
    }

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    delete params;
    delete[] headings;
}

TYPED_TEST(CSVTest, warn_for_missing_data) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetMissingData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    ASSERT_EQ(da_datastore_init(&store), da_status_success);
    TypeParam *a = nullptr;

    da_int nrows = 0, ncols = 0;

    ASSERT_EQ(da_datastore_options_set_int(store, "CSV warn for missing data", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV skip initial space", 1),
              da_status_success);
    ASSERT_EQ(da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr),
              params->expected_status);

    ASSERT_EQ(nrows, params->expected_rows);
    ASSERT_EQ(ncols, params->expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            if (check_nan(GetExpectedData(params, j + ncols * i))) {
                ASSERT_TRUE(check_nan(a[j + ncols * i]));
            } else {
                ASSERT_EQ_overload(a[j + ncols * i],
                                   GetExpectedData(params, j + ncols * i));
            }
        }
    }

    da_csv::free_data(&a, nrows * ncols);
    da_datastore_destroy(&store);

    delete params;
}

TYPED_TEST(DataStoreTest, warn_for_missing_data) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetMissingData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_datastore store = nullptr;
    ASSERT_EQ(da_datastore_init(&store), da_status_success);

    da_int nrows = 0, ncols = 0;

    ASSERT_EQ(da_datastore_options_set_int(store, "CSV warn for missing data", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV skip initial space", 1),
              da_status_success);
    ASSERT_EQ(
        da_datastore_options_set_string(store, "CSV datatype", params->datatype.c_str()),
        da_status_success);

    EXPECT_EQ(da_data_load_from_csv(store, filepath), params->expected_status);

    EXPECT_EQ(da_data_get_num_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_num_cols(store, &ncols), da_status_success);

    ASSERT_EQ(nrows, params->expected_rows);
    ASSERT_EQ(ncols, params->expected_columns);

    TypeParam *T = new TypeParam[nrows];

    for (da_int i = 0; i < ncols; i++) {
        ASSERT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            if (check_nan(GetExpectedData(params, i + ncols * j))) {
                ASSERT_TRUE(check_nan(T[j]));
            } else {
                ASSERT_EQ_overload(T[j], GetExpectedData(params, i + ncols * j));
            }
        }
    }

    delete[] T;

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    delete params;
}

TEST(CSVTest, options) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_options.csv");

    da_datastore store = nullptr;
    ASSERT_EQ(da_datastore_init(&store), da_status_success);
    double *a = nullptr;

    da_int nrows = 0, ncols = 0;

    da_int expected_rows = 3;
    da_int expected_columns = 5;

    double expected_data[] = {1.1,     1e3, 1000000000, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                              -5.6e-7, -10, 0.0,        0.0,    0.0, 0.0, 4.5e+5};

    ASSERT_EQ(da_datastore_options_set_string(store, "CSV delimiter", "x"),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV thousands", ","),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV decimal", "p"),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV comment", "}"),
              da_status_success);
    ASSERT_EQ(
        da_datastore_options_set_string(store, "CSV scientific notation character", "g"),
        da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV skip initial space", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV skip empty lines", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV skip footer", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV row start", 3), da_status_success);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV skip rows", "5 9"),
              da_status_success);

    ASSERT_EQ(da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_bad_lines);

    ASSERT_EQ(nrows, expected_rows);
    ASSERT_EQ(ncols, expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], expected_data[j + ncols * i]);
        }
    }

    if (a)
        free(a);

    // Now try with a datastore object

    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "double"),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(store, filepath), da_status_bad_lines);

    EXPECT_EQ(da_data_get_num_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_num_cols(store, &ncols), da_status_success);

    ASSERT_EQ(nrows, expected_rows);
    ASSERT_EQ(ncols, expected_columns);

    double *T = new double[nrows];

    for (da_int i = 0; i < ncols; i++) {
        ASSERT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            ASSERT_EQ(T[j], expected_data[i + ncols * j]);
        }
    }

    delete[] T;

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);
}

TEST(csvtest, error_exits) {
    char filename[] = "csv_test_errors.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, filename);

    // Check for uninitialized handle
    da_datastore store = nullptr;
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV whitespace delimiter", 1),
              da_status_invalid_pointer);
    double *a = nullptr;

    da_int nrows = 0, ncols = 0;
    ASSERT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_store_not_initialized);

    // Check for various error exits
    double *a_double = nullptr;
    da_int *a_int = nullptr;
    uint8_t *a_uint8 = nullptr;
    ASSERT_EQ(da_datastore_init(&store), da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV whitespace delimiter", 1),
              da_status_success);

    ASSERT_EQ(da_read_csv_d(store, filepath, &a_double, &nrows, &ncols, nullptr),
              da_status_range_error);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "double"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_range_error);

    ASSERT_EQ(da_read_csv_int(store, filepath, &a_int, &nrows, &ncols, nullptr),
              da_status_invalid_chars);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "integer"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_invalid_chars);

    ASSERT_EQ(da_datastore_options_set_string(store, "CSV skip rows", "0"),
              da_status_success);
    ASSERT_EQ(da_read_csv_uint8(store, filepath, &a_uint8, &nrows, &ncols, nullptr),
              da_status_invalid_boolean);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "boolean"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_invalid_boolean);

    ASSERT_EQ(da_read_csv_d(store, filepath, &a_double, &nrows, &ncols, nullptr),
              da_status_range_error);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "double"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_range_error);

    ASSERT_EQ(da_read_csv_int(store, filepath, &a_int, &nrows, &ncols, nullptr),
              da_status_no_digits);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "integer"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_no_digits);

    ASSERT_EQ(da_datastore_options_set_string(store, "CSV skip rows", "0, 1"),
              da_status_success);

    ASSERT_EQ(da_datastore_options_set_int(store, "CSV row start", 3), da_status_success);
    ASSERT_EQ(da_read_csv_int(store, filepath, &a_int, &nrows, &ncols, nullptr),
              da_status_overflow);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "integer"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_overflow);

    ASSERT_EQ(da_datastore_options_set_int(store, "CSV row start", 4), da_status_success);
    ASSERT_EQ(da_read_csv_int(store, filepath, &a_int, &nrows, &ncols, nullptr),
              da_status_ragged_csv);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "auto"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_ragged_csv);

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);

    if (a)
        free(a);
    if (a_double)
        free(a_double);
}

TEST(csvtest, no_data) {
    char filename[] = "csv_test_empty.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, filename);

    // Check for uninitialized handle
    da_datastore store = nullptr;
    ASSERT_EQ(da_datastore_init(&store), da_status_success);

    double *a = nullptr;
    char **headings = nullptr;
    const char *expected_headings[] = {"one", "two", "three", "four", "five"};
    da_int nrows = 0, ncols = 0;

    // Check we can handle headings but no other data
    da_datastore_options_set_int(store, "CSV use header row", 1);
    ASSERT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, &headings),
              da_status_no_data);
    ASSERT_EQ(nrows, 0);
    ASSERT_EQ(ncols, 5);
    for (da_int j = 0; j < ncols; j++) {
        EXPECT_STREQ(headings[j], expected_headings[j]);
    }
    da_csv::free_data(&headings, ncols);

    // Now try the same thing in a datastore but just expect no data error
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV use header row", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "double"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_no_data);
    da_datastore_destroy(&store);

    // Check we can deal with removing all rows
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV use header row", 0),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV row start", 1), da_status_success);
    ASSERT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_no_data);
    ASSERT_EQ(nrows, 0);
    ASSERT_EQ(ncols, 0);

    da_datastore_options_set_int(store, "CSV use header row", 1);
    ASSERT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, &headings),
              da_status_no_data);
    ASSERT_EQ(nrows, 0);
    ASSERT_EQ(ncols, 0);

    da_datastore_destroy(&store);
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV use header row", 0),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV row start", 1), da_status_success);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "double"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_no_data);

    da_datastore_destroy(&store);

    // Check for non-existent csv file
    da_datastore_destroy(&store);
    strcat(filepath, "does_not_exist");
    ASSERT_EQ(da_datastore_init(&store), da_status_success);
    ASSERT_EQ(da_read_csv_d(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_file_not_found);

    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "double"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_file_not_found);

    da_datastore_destroy(&store);

    if (a)
        free(a);
}

TEST(CSVTest, lineterminator) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_lineterminator.csv");

    da_datastore store = nullptr;
    ASSERT_EQ(da_datastore_init(&store), da_status_success);
    da_int *a = nullptr;

    da_int nrows = 0, ncols = 0;

    ASSERT_EQ(da_datastore_options_set_string(store, "CSV line terminator", "x"),
              da_status_success);
    ASSERT_EQ(da_read_csv(store, filepath, &a, &nrows, &ncols, nullptr),
              da_status_success);

    da_int expected_rows = 2;
    da_int expected_columns = 3;

    da_int expected_data[] = {1, 2, 3, 4, 5, 6};

    ASSERT_EQ(nrows, expected_rows);
    ASSERT_EQ(ncols, expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], expected_data[j + ncols * i]);
        }
    }

    if (a)
        free(a);

    // Check the same thing works when reading into a datastore

    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "double"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_success);

    EXPECT_EQ(da_data_get_num_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_num_cols(store, &ncols), da_status_success);

    ASSERT_EQ(nrows, expected_rows);
    ASSERT_EQ(ncols, expected_columns);

    double *T = new double[nrows];

    for (da_int i = 0; i < ncols; i++) {
        ASSERT_EQ(da_data_extract_column(store, i, nrows, T), da_status_success);
        for (da_int j = 0; j < nrows; j++) {
            ASSERT_EQ(T[j], expected_data[i + ncols * j]);
        }
    }

    delete[] T;

    da_datastore_destroy(&store);
    da_datastore_destroy(&store);
}

TEST(CSVTest, auto) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_auto.csv");

    da_datastore store = nullptr;
    ASSERT_EQ(da_datastore_init(&store), da_status_success);

    da_int nrows, ncols, expected_rows = 4, expected_columns = 7;

    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "auto"),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV use header row", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV skip initial space", 1),
              da_status_success);

    ASSERT_EQ(da_data_print_options(store), da_status_success);

    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_success);

    EXPECT_EQ(da_data_get_num_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_num_cols(store, &ncols), da_status_success);

    ASSERT_EQ(nrows, expected_rows);
    ASSERT_EQ(ncols, expected_columns);

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

    ASSERT_EQ(da_data_extract_column_int(store, 0, nrows, Ti), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ(c1[j], Ti[j]);
    }

    ASSERT_EQ(da_data_extract_column_int(store, 1, nrows, Ti), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ(c2[j], Ti[j]);
    }

    ASSERT_EQ(da_data_extract_column_real_d(store, 2, nrows, Td), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ(c3[j], Td[j]);
    }

    ASSERT_EQ(da_data_extract_column_real_d(store, 3, nrows, Td), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ(c4[j], Td[j]);
    }

    ASSERT_EQ(da_data_extract_column_uint8(store, 4, nrows, Tu), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ(c5[j], Tu[j]);
    }

    ASSERT_EQ(da_data_extract_column_str(store, 5, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c6[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ_overload(c6[j], Tc[j]);
    }

    ASSERT_EQ(da_data_extract_column_str(store, 6, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c7[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ_overload(c7[j], Tc[j]);
    }

    for (da_int j = 0; j < ncols; j++) {
        da_int name_sz = 128;
        EXPECT_EQ(da_data_get_col_label(store, j, &name_sz, col_name), da_status_success);
        ASSERT_EQ_overload(col_name, expected_headings[j]);
    }

    da_datastore_destroy(&store);

    // Repeat with option set for integer data being interpreted as float
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datatype", "auto"),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV use header row", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV skip initial space", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_int(store, "CSV integers as floats", 1),
              da_status_success);
    ASSERT_EQ(da_datastore_options_set_string(store, "CSV datastore precision", "single"),
              da_status_success);
    ASSERT_EQ(da_data_load_from_csv(store, filepath), da_status_success);

    EXPECT_EQ(da_data_get_num_rows(store, &nrows), da_status_success);
    EXPECT_EQ(da_data_get_num_cols(store, &ncols), da_status_success);

    ASSERT_EQ(nrows, expected_rows);
    ASSERT_EQ(ncols, expected_columns);

    ASSERT_EQ(da_data_extract_column_real_s(store, 0, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ(cc1[j], Tf[j]);
    }

    ASSERT_EQ(da_data_extract_column_real_s(store, 1, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ(cc2[j], Tf[j]);
    }

    ASSERT_EQ(da_data_extract_column_real_s(store, 2, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ(cc3[j], Tf[j]);
    }

    ASSERT_EQ(da_data_extract_column_real_s(store, 3, nrows, Tf), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ(cc4[j], Tf[j]);
    }

    ASSERT_EQ(da_data_extract_column_uint8(store, 4, nrows, Tu), da_status_success);
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ(c5[j], Tu[j]);
    }

    ASSERT_EQ(da_data_extract_column_str(store, 5, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c6[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ_overload(c6[j], Tc[j]);
    }

    ASSERT_EQ(da_data_extract_column_str(store, 6, nrows, Tc), da_status_success);
    std::cout << Tc[0] << c7[0] << std::endl;
    for (da_int j = 0; j < nrows; j++) {
        ASSERT_EQ_overload(c7[j], Tc[j]);
    }

    da_int label_sz = 64;
    char col_label[64];
    for (da_int j = 0; j < ncols; j++) {
        da_data_get_col_label(store, j, &label_sz, col_label);
        ASSERT_EQ_overload(col_label, expected_headings[j]);
    }

    da_datastore_destroy(&store);
}