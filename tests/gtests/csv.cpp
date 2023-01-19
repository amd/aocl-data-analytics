#include "aoclda.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <list>
#include <string>

TEST(csvtest, single_head) {

    char filename[] = "csv_test_float_head.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    float *a = nullptr;
    char **headings = nullptr;

    size_t nrows_expected = 3, ncols_expected = 5;
    float a_expected[15] = {1.1,     1e3, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                            -5.6e-7, -10, 0.0,    0.0,    0.0, 0.0, 4.5e+5};
    char headings_expected[5][10] = {"one", "cat two", "three", "FOUR", "Five"};

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_s_h(opts, filepath, &a, &nrows, &ncols, &headings);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, nrows_expected);
    ASSERT_EQ(ncols, ncols_expected);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], a_expected[j + ncols * i]);
        }
    }

    for (size_t j = 0; j < ncols; j++) {
        EXPECT_STREQ(headings[j], headings_expected[j]);
    }

    da_csv_destroy(&opts);

    if (a)
        free(a);

    if (headings) {
        for (size_t i = 0; i < ncols; i++) {
            if (headings[i])
                free(headings[i]);
        }
        free(headings);
    }
}

TEST(csvtest, double_head) {

    char filename[] = "csv_test_float_head.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    double *a = nullptr;
    char **headings = nullptr;

    size_t nrows_expected = 3, ncols_expected = 5;
    double a_expected[15] = {1.1,     1e3, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                             -5.6e-7, -10, 0.0,    0.0,    0.0, 0.0, 4.5e+5};
    char headings_expected[5][10] = {"one", "cat two", "three", "FOUR", "Five"};

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_d_h(opts, filepath, &a, &nrows, &ncols, &headings);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, nrows_expected);
    ASSERT_EQ(ncols, ncols_expected);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], a_expected[j + ncols * i]);
        }
    }

    for (size_t j = 0; j < ncols; j++) {
        EXPECT_STREQ(headings[j], headings_expected[j]);
    }

    da_csv_destroy(&opts);

    if (a)
        free(a);

    if (headings) {
        for (size_t i = 0; i < ncols; i++) {
            if (headings[i])
                free(headings[i]);
        }
        free(headings);
    }
}

TEST(csvtest, single) {

    char filename[] = "csv_test_float.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    float *a = nullptr;

    size_t nrows_expected = 3, ncols_expected = 5;
    float a_expected[15] = {1.1,     1e3, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                            -5.6e-7, -10, 0.0,    0.0,    0.0, 0.0, 4.5e+5};

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_s(opts, filepath, &a, &nrows, &ncols);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, nrows_expected);
    ASSERT_EQ(ncols, ncols_expected);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], a_expected[j + ncols * i]);
        }
    }

    da_csv_destroy(&opts);

    if (a)
        free(a);
}

TEST(csvtest, double) {

    char filename[] = "csv_test_float.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    double *a = nullptr;

    size_t nrows_expected = 3, ncols_expected = 5;
    double a_expected[15] = {1.1,     1e3, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                             -5.6e-7, -10, 0.0,    0.0,    0.0, 0.0, 4.5e+5};

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_d(opts, filepath, &a, &nrows, &ncols);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, nrows_expected);
    ASSERT_EQ(ncols, ncols_expected);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], a_expected[j + ncols * i]);
        }
    }

    da_csv_destroy(&opts);

    if (a)
        free(a);
}

TEST(csvtest, int64) {

    char filename[] = "csv_test_int64.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    int64_t *a = nullptr;

    size_t nrows_expected = 3, ncols_expected = 4;
    int64_t a_expected[12] = {1,
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

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_int64(opts, filepath, &a, &nrows, &ncols);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, nrows_expected);
    ASSERT_EQ(ncols, ncols_expected);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], a_expected[j + ncols * i]);
        }
    }

    da_csv_destroy(&opts);

    if (a)
        free(a);
}

TEST(csvtest, int64_head) {

    char filename[] = "csv_test_int64_head.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    int64_t *a = nullptr;
    char **headings = nullptr;

    size_t nrows_expected = 3, ncols_expected = 4;
    int64_t a_expected[12] = {1,
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

    char headings_expected[4][10] = {"one", "two", "three", "four"};

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_int64_h(opts, filepath, &a, &nrows, &ncols, &headings);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, nrows_expected);
    ASSERT_EQ(ncols, ncols_expected);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], a_expected[j + ncols * i]);
        }
    }

    for (size_t j = 0; j < ncols; j++) {
        EXPECT_STREQ(headings[j], headings_expected[j]);
    }

    da_csv_destroy(&opts);

    if (headings) {
        for (size_t i = 0; i < ncols; i++) {
            if (headings[i])
                free(headings[i]);
        }
        free(headings);
    }

    if (a)
        free(a);
}

TEST(csvtest, uint64) {

    char filename[] = "csv_test_uint64.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    uint64_t *a = nullptr;

    size_t nrows_expected = 3, ncols_expected = 4;
    uint64_t a_expected[12] = {1,
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

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_uint64(opts, filepath, &a, &nrows, &ncols);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, nrows_expected);
    ASSERT_EQ(ncols, ncols_expected);

    da_csv_destroy(&opts);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], a_expected[j + ncols * i]);
        }
    }

    if (a)
        free(a);
}

TEST(csvtest, uint64_head) {

    char filename[] = "csv_test_uint64_head.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    uint64_t *a = nullptr;
    char **headings = nullptr;

    size_t nrows_expected = 3, ncols_expected = 4;
    uint64_t a_expected[12] = {1,
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

    char headings_expected[4][10] = {"one", "two", "three", "four"};

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_uint64_h(opts, filepath, &a, &nrows, &ncols, &headings);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, nrows_expected);
    ASSERT_EQ(ncols, ncols_expected);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], a_expected[j + ncols * i]);
        }
    }

    for (size_t j = 0; j < ncols; j++) {
        EXPECT_STREQ(headings[j], headings_expected[j]);
    }

    da_csv_destroy(&opts);

    if (headings) {
        for (size_t i = 0; i < ncols; i++) {
            if (headings[i])
                free(headings[i]);
        }
        free(headings);
    }

    if (a)
        free(a);
}

TEST(csvtest, bool) {

    char filename[] = "csv_test_bool.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    uint8_t *a = nullptr;

    size_t nrows_expected = 2, ncols_expected = 4;
    uint8_t a_expected[8] = {1, 1, 1, 1, 1, 0, 0, 0};

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_uint8(opts, filepath, &a, &nrows, &ncols);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, nrows_expected);
    ASSERT_EQ(ncols, ncols_expected);

    da_csv_destroy(&opts);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], a_expected[j + ncols * i]);
        }
    }

    if (a)
        free(a);
}

TEST(csvtest, bool_head) {

    char filename[] = "csv_test_bool_head.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    uint8_t *a = nullptr;
    char **headings = nullptr;

    size_t nrows_expected = 2, ncols_expected = 4;
    uint8_t a_expected[12] = {1, 1, 1, 1, 1, 0, 0, 0};

    char headings_expected[4][10] = {"one", "two", "three", "four"};

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_uint8_h(opts, filepath, &a, &nrows, &ncols, &headings);

    ASSERT_EQ(err, da_status_success);

    ASSERT_EQ(nrows, nrows_expected);
    ASSERT_EQ(ncols, ncols_expected);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            ASSERT_EQ(a[j + ncols * i], a_expected[j + ncols * i]);
        }
    }

    for (size_t j = 0; j < ncols; j++) {
        EXPECT_STREQ(headings[j], headings_expected[j]);
    }

    da_csv_destroy(&opts);

    if (headings) {
        for (size_t i = 0; i < ncols; i++) {
            if (headings[i])
                free(headings[i]);
        }
        free(headings);
    }

    if (a)
        free(a);
}

TEST(csvtest, missing_file) {
    char filename[] = "does_not_exist.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    double *a = nullptr;

    size_t nrows = 0, ncols = 0;
    err = da_read_csv_d(opts, filepath, &a, &nrows, &ncols);
    ASSERT_EQ(err, da_status_file_not_found);

    da_csv_destroy(&opts);

    if (a)
        free(a);
}

TEST(csvtest, tmptest) {
    char filename[] = "csv_test_tmp.csv";
    char filepath[256] = DATA_DIR;
    strcat(filepath, filename);

    da_csv_opts opts = nullptr;
    da_status err = da_csv_init(&opts);
    double *a = nullptr;
    char **headings = nullptr;

    size_t nrows = 0, ncols = 0;
    char one[] = "1";
    err = da_csv_set_option(opts, csv_option_warn_for_missing_data, one);
    err = da_read_csv_d_h(opts, filepath, &a, &nrows, &ncols, &headings);

    for (size_t i = 0; i < nrows; i++) {
        printf("\n");
        for (size_t j = 0; j < ncols; j++) {
            printf("   %f  ", a[j + ncols * i]);
        }
    }
    printf("\n");

    if (headings) {
        for (size_t j = 0; j < ncols; j++) {
            printf("%s", headings[j]);
        }
    }

    char G[] = "G";
    char G2[] = "45";
    err = da_csv_set_option(opts, csv_option_escapechar, G);
    err = da_csv_set_option(opts, csv_option_header_end, G2);

    da_csv_destroy(&opts);

    if (a)
        free(a);

    if (headings) {
        for (size_t i = 0; i < ncols; i++) {
            if (headings[i])
                free(headings[i]);
        }
        free(headings);
    }

    ASSERT_EQ(err, da_status_success);
}
