#include "aoclda.h"
#include "utest_utils.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <list>
#include <string>

template <typename T> struct CSVParamType {
    std::string filename;
    da_int expected_rows;
    da_int expected_columns;
    std::vector<T> expected_data;
    std::vector<std::string> expected_headings;
};

template <typename T> void GetBasicData(CSVParamType<T> *params);

template <> void GetBasicData<double>(CSVParamType<double> *params) {

    params->filename = "csv_test_float";
    params->expected_rows = 3;
    params->expected_columns = 5;
    std::vector<double> data{1.1,     1e3, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                             -5.6e-7, -10, 0.0,    0.0,    0.0, 0.0, 4.5e+5};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "cat two", "three", "FOUR", "Five"};
    params->expected_headings = headings;
}

template <> void GetBasicData<float>(CSVParamType<float> *params) {

    params->filename = "csv_test_float";
    params->expected_rows = 3;
    params->expected_columns = 5;
    std::vector<float> data{1.1,     1e3, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                            -5.6e-7, -10, 0.0,    0.0,    0.0, 0.0, 4.5e+5};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "cat two", "three", "FOUR", "Five"};
    params->expected_headings = headings;
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
}

template <> void GetBasicData<uint64_t>(CSVParamType<uint64_t> *params) {

    params->filename = "csv_test_uint64";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<uint64_t> data{1,
                               5,
                               3,
                               0,
                               0,
                               43,
                               +9223372036854775807,
                               +9223372036854775806,
                               +345,
                               18446744073709551615UL,
                               18446744073709551614UL,
                               67};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
}

template <> void GetBasicData<uint8_t>(CSVParamType<uint8_t> *params) {

    params->filename = "csv_test_bool";
    params->expected_rows = 2;
    params->expected_columns = 4;
    std::vector<uint8_t> data{1, 1, 1, 1, 1, 0, 0, 0};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
}

template <typename T> void GetMissingData(CSVParamType<T> *params);

template <> void GetMissingData<double>(CSVParamType<double> *params) {

    params->filename = "csv_test_float_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 5;
    std::vector<double> data{1.1,     NAN, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                             -5.6e-7, NAN, NAN,    0.0,    0.0, 0.0, 4.5e+5};
    params->expected_data = data;
}

template <> void GetMissingData<float>(CSVParamType<float> *params) {

    params->filename = "csv_test_float_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 5;
    std::vector<float> data{1.1,     NAN, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                            -5.6e-7, NAN, NAN,    0.0,    0.0, 0.0, 4.5e+5};
    params->expected_data = data;
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
}

template <> void GetMissingData<uint64_t>(CSVParamType<uint64_t> *params) {

    params->filename = "csv_test_uint64_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<uint64_t> data{1,
                               5,
                               UINT64_MAX,
                               0,
                               0,
                               43,
                               +9223372036854775807,
                               +9223372036854775806,
                               UINT64_MAX,
                               18446744073709551615UL,
                               18446744073709551614UL,
                               67};
    params->expected_data = data;
}

template <> void GetMissingData<uint8_t>(CSVParamType<uint8_t> *params) {

    params->filename = "csv_test_bool_missing_data";
    params->expected_rows = 2;
    params->expected_columns = 4;
    std::vector<uint8_t> data{1, 1, UINT8_MAX, 1, 1, 0, UINT8_MAX, 0};
    params->expected_data = data;
}

template <typename T> class CSVTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using CSVTypes = ::testing::Types<float, double, int64_t, uint64_t, uint8_t>;

TYPED_TEST_SUITE(CSVTest, CSVTypes);

TYPED_TEST(CSVTest, basic_no_headings) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetBasicData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_handle handle = nullptr;
    da_status err = da_handle_init_d(&handle, da_handle_csv_opts);
    TypeParam *a = nullptr;

    da_int nrows = 0, ncols = 0;

    err = da_read_csv(handle, filepath, &a, &nrows, &ncols);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, params->expected_rows);
    ASSERT_EQ(ncols, params->expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], params->expected_data[j + ncols * i]);
        }
    }

    if (a)
        free(a);

    da_handle_destroy(&handle);

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

    da_handle handle = nullptr;
    da_status err = da_handle_init_d(&handle, da_handle_csv_opts);
    TypeParam *a = nullptr;

    da_int nrows = 0, ncols = 0;
    char **headings = nullptr;

    err = da_read_csv(handle, filepath, &a, &nrows, &ncols, &headings);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, params->expected_rows);
    ASSERT_EQ(ncols, params->expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], params->expected_data[j + ncols * i]);
        }
    }

    for (da_int j = 0; j < ncols; j++) {
        EXPECT_STREQ(headings[j], params->expected_headings[j].c_str());
    }

    if (a)
        free(a);

    if (headings) {
        for (da_int i = 0; i < ncols; i++) {
            if (headings[i])
                free(headings[i]);
        }
        free(headings);
    }

    da_handle_destroy(&handle);

    delete params;
}

TYPED_TEST(CSVTest, warn_for_missing_data) {

    CSVParamType<TypeParam> *params = new CSVParamType<TypeParam>();
    GetMissingData(params);

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, params->filename.c_str());
    strcat(filepath, ".csv");

    da_handle handle = nullptr;
    da_status err = da_handle_init_d(&handle, da_handle_csv_opts);
    TypeParam *a = nullptr;

    da_int nrows = 0, ncols = 0;
    char one[] = "1";
    err = da_handle_set_option(handle, csv_option_warn_for_missing_data, one);
    err = da_read_csv(handle, filepath, &a, &nrows, &ncols);

    ASSERT_EQ(err, da_status_warn_missing_data);

    ASSERT_EQ(nrows, params->expected_rows);
    ASSERT_EQ(ncols, params->expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            if (std::isnan((TypeParam)params->expected_data[j + ncols * i])) {
                ASSERT_TRUE(std::isnan((TypeParam)a[j + ncols * i]));
            } else {
                ASSERT_EQ(a[j + ncols * i], params->expected_data[j + ncols * i]);
            }
        }
    }

    if (a)
        free(a);

    da_handle_destroy(&handle);

    delete params;
}

TEST(CSVTest, options) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_options.csv");

    da_handle handle = nullptr;
    da_status err = da_handle_init_d(&handle, da_handle_csv_opts);
    double *a = nullptr;

    da_int nrows = 0, ncols = 0;
    char decimal[] = "p";
    char thousands[] = ",";
    char comment[] = "}";
    char delimiter[] = "x";
    char sci[] = "g";
    char one[] = "1";
    char three[] = "3";
    char five[] = "5";
    char nine[] = "9";
    err = da_handle_set_option(handle, csv_option_delimiter, delimiter);
    err = da_handle_set_option(handle, csv_option_thousands, thousands);
    err = da_handle_set_option(handle, csv_option_decimal, decimal);
    err = da_handle_set_option(handle, csv_option_comment, comment);
    err = da_handle_set_option(handle, csv_option_sci, sci);
    err = da_handle_set_option(handle, csv_option_skip_initial_space, one);
    err = da_handle_set_option(handle, csv_option_skip_empty_lines, one);
    err = da_handle_set_option(handle, csv_option_skip_footer, one);
    err = da_handle_set_option(handle, csv_option_skip_first_N_rows, three);
    err = da_handle_set_option(handle, csv_option_add_skiprow, five);
    err = da_handle_set_option(handle, csv_option_add_skiprow, nine);
    err = da_read_csv(handle, filepath, &a, &nrows, &ncols);

    ASSERT_EQ(err, da_status_warn_bad_lines);

    da_int expected_rows = 3;
    da_int expected_columns = 5;

    double expected_data[] = {1.1,     1e3, 1000000000, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                              -5.6e-7, -10, 0.0,        0.0,    0.0, 0.0, 4.5e+5};

    ASSERT_EQ(nrows, expected_rows);
    ASSERT_EQ(ncols, expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], expected_data[j + ncols * i]);
        }
    }

    if (a)
        free(a);

    da_handle_destroy(&handle);
}

TEST(csvtest, error_exits) {
    char filename[] = "csv_test_errors.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, filename);

    char zero[] = "0";
    char one[] = "1";
    char three[] = "3";
    char four[] = "4";

    da_status err;

    // Check for uninitialized handle
    da_handle handle = nullptr;
    err = da_handle_set_option(handle, csv_option_delim_whitespace, one);
    ASSERT_EQ(err, da_status_handle_not_initialized);
    double *a = nullptr;

    da_int nrows = 0, ncols = 0;
    err = da_read_csv_d(handle, filepath, &a, &nrows, &ncols);
    ASSERT_EQ(err, da_status_handle_not_initialized);

    // Check for incorrect handle type
    err = da_handle_init_d(&handle, da_handle_linreg);
    err = da_handle_set_option(handle, csv_option_delim_whitespace, one);
    ASSERT_EQ(err, da_status_invalid_handle_type);
    err = da_read_csv_d(handle, filepath, &a, &nrows, &ncols);
    ASSERT_EQ(err, da_status_invalid_handle_type);
    da_handle_destroy(&handle);

    // Check for various error exits
    double *a_double = nullptr;
    int64_t *a_int64 = nullptr;
    uint64_t *a_uint64 = nullptr;
    uint8_t *a_uint8 = nullptr;
    err = da_handle_init_d(&handle, da_handle_csv_opts);
    err = da_handle_set_option(handle, csv_option_delim_whitespace, one);

    err = da_read_csv_d(handle, filepath, &a_double, &nrows, &ncols);
    ASSERT_EQ(err, da_status_range_error);

    err = da_read_csv_int64(handle, filepath, &a_int64, &nrows, &ncols);
    ASSERT_EQ(err, da_status_invalid_chars);

    err = da_handle_set_option(handle, csv_option_add_skiprow, zero);
    err = da_read_csv_uint8(handle, filepath, &a_uint8, &nrows, &ncols);
    ASSERT_EQ(err, da_status_invalid_boolean);

    err = da_read_csv_d(handle, filepath, &a_double, &nrows, &ncols);
    ASSERT_EQ(err, da_status_range_error);

    err = da_read_csv_int64(handle, filepath, &a_int64, &nrows, &ncols);
    ASSERT_EQ(err, da_status_no_digits);

    err = da_handle_set_option(handle, csv_option_add_skiprow, one);
    err = da_read_csv_uint64(handle, filepath, &a_uint64, &nrows, &ncols);
    ASSERT_EQ(err, da_status_sign_error);

    err = da_handle_set_option(handle, csv_option_skip_first_N_rows, three);
    err = da_read_csv_int64(handle, filepath, &a_int64, &nrows, &ncols);
    ASSERT_EQ(err, da_status_overflow);

    err = da_handle_set_option(handle, csv_option_skip_first_N_rows, four);
    err = da_read_csv_int64(handle, filepath, &a_int64, &nrows, &ncols);
    ASSERT_EQ(err, da_status_ragged_csv);

    da_handle_destroy(&handle);

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

    char one[] = "1";

    da_status err;

    // Check for uninitialized handle
    da_handle handle = nullptr;
    err = da_handle_init_d(&handle, da_handle_csv_opts);

    double *a = nullptr;
    char **headings = nullptr;
    const char *expected_headings[] = {"one", "two", "three", "four", "five"};
    da_int nrows = 0, ncols = 0;

    // Check we can handle headings but no other data
    err = da_read_csv_d_h(handle, filepath, &a, &nrows, &ncols, &headings);
    ASSERT_EQ(err, da_status_success);
    ASSERT_EQ(nrows, 0);
    ASSERT_EQ(ncols, 5);
    for (da_int j = 0; j < ncols; j++) {
        EXPECT_STREQ(headings[j], expected_headings[j]);
    }
    
    if (headings) {
        for (da_int i = 0; i < ncols; i++) {
            if (headings[i])
                free(headings[i]);
        }
        free(headings);
    }

    // Check we can deal with removing all rows
    err = da_handle_set_option(handle, csv_option_skip_first_N_rows, one);
    err = da_read_csv_d(handle, filepath, &a, &nrows, &ncols);
    ASSERT_EQ(err, da_status_warn_no_data);
    ASSERT_EQ(nrows, 0);
    ASSERT_EQ(ncols, 0);

    err = da_read_csv_d_h(handle, filepath, &a, &nrows, &ncols, &headings);
    ASSERT_EQ(err, da_status_warn_no_data);
    ASSERT_EQ(nrows, 0);
    ASSERT_EQ(ncols, 0);

    // Check for non-existent csv file
    da_handle_destroy(&handle);
    strcat(filepath, "does_not_exist");
    err = da_handle_init_d(&handle, da_handle_csv_opts);
    err = da_read_csv_d(handle, filepath, &a, &nrows, &ncols);
    ASSERT_EQ(err, da_status_file_not_found);

    da_handle_destroy(&handle);

    if (a)
        free(a);

}

TEST(CSVTest, lineterminator) {

    char filepath[256] = DATA_DIR;
    strcat(filepath, "csv_data/");
    strcat(filepath, "csv_test_lineterminator.csv");

    da_handle handle = nullptr;
    da_status err = da_handle_init_d(&handle, da_handle_csv_opts);
    uint64_t *a = nullptr;

    da_int nrows = 0, ncols = 0;

    char terminator[] = "x";

    err = da_handle_set_option(handle, csv_option_lineterminator, terminator);
    err = da_read_csv(handle, filepath, &a, &nrows, &ncols);

    ASSERT_EQ(err, da_status_success);

    da_int expected_rows = 2;
    da_int expected_columns = 3;

    uint64_t expected_data[] = {1,2,3,4,5,6};

    ASSERT_EQ(nrows, expected_rows);
    ASSERT_EQ(ncols, expected_columns);

    for (da_int i = 0; i < nrows; i++) {
        for (da_int j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], expected_data[j + ncols * i]);
        }
    }

    if (a)
        free(a);

    da_handle_destroy(&handle);
}