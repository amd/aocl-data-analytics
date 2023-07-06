#include "aoclda.h"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <list>

template <typename T> class StatisticsUtilitiesTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> struct StatsParamType {
    da_int n;
    da_int p;
    da_int ldx;
    std::vector<T> x;
    std::vector<T> expected_x_column;
    std::vector<T> column_shift;
    std::vector<T> column_scale;

    std::vector<T> expected_x_row;
    std::vector<T> row_shift;
    std::vector<T> row_scale;

    std::vector<T> expected_x_overall;
    std::vector<T> overall_shift;
    std::vector<T> overall_scale;

    da_status expected_status;
    T epsilon;
};

template <typename T> void Get1by1Data(std::vector<StatsParamType<T>> &params) {
    // Test with 1 x 1 data matrix
    StatsParamType<T> param;
    param.n = 1;
    param.p = 1;
    param.ldx = param.n;

    std::vector<T> x(param.n * param.p, 3);
    param.x = x;

    std::vector<T> column_shift(param.p, 1);
    param.column_shift = column_shift;
    std::vector<T> column_scale(param.p, 2);
    param.column_scale = column_scale;
    std::vector<T> expected_x_column(param.p * param.n, 1);
    param.expected_x_column = expected_x_column;

    std::vector<T> row_shift(param.n, -1);
    param.row_shift = row_shift;
    std::vector<T> row_scale(param.n, 2);
    param.row_scale = row_scale;
    std::vector<T> expected_x_row(param.p * param.n, 2);
    param.expected_x_row = expected_x_row;

    std::vector<T> overall_shift(1, 1);
    param.overall_shift = overall_shift;
    std::vector<T> overall_scale(1, 2);
    param.overall_scale = overall_scale;
    std::vector<T> expected_x_overall(param.p * param.n, 1);
    param.expected_x_overall = expected_x_overall;

    param.expected_status = da_status_success;

    param.epsilon = std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetSingleRowData(std::vector<StatsParamType<T>> &params) {
    // Test with a single row
    StatsParamType<T> param;
    param.n = 1;
    param.p = 7;
    param.ldx = param.n;

    std::vector<T> x{0, 1, 2, 3, 4, 5, 6};
    param.x = x;

    std::vector<T> column_shift{2, 4, 6, 8, 10, 12, 14};
    param.column_shift = column_shift;
    std::vector<T> column_scale{0, 1, 4, 4, 0, 4, 2};
    param.column_scale = column_scale;
    std::vector<T> expected_x_column{-2, -3, -1, -1.25, -6, -1.75, -4};
    param.expected_x_column = expected_x_column;

    std::vector<T> row_shift{-1};
    param.row_shift = row_shift;
    std::vector<T> row_scale{2};
    param.row_scale = row_scale;
    std::vector<T> expected_x_row{0.5, 1, 1.5, 2, 2.5, 3, 3.5};
    param.expected_x_row = expected_x_row;

    std::vector<T> overall_shift{0};
    param.overall_shift = overall_shift;
    std::vector<T> overall_scale{0};
    param.overall_scale = overall_scale;
    std::vector<T> expected_x_overall{0, 1, 2, 3, 4, 5, 6};
    param.expected_x_overall = expected_x_overall;

    param.expected_status = da_status_success;

    param.epsilon = std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetSingleColumnData(std::vector<StatsParamType<T>> &params) {
    // Test with a single column
    StatsParamType<T> param;
    param.n = 7;
    param.p = 1;
    param.ldx = param.n;

    std::vector<T> x{0, 2, 4, 6, 8, 10, 12};
    param.x = x;

    std::vector<T> column_shift{6};
    param.column_shift = column_shift;
    std::vector<T> column_scale{0.5};
    param.column_scale = column_scale;
    std::vector<T> expected_x_column{-12, -8, -4, 0, 4, 8, 12};
    param.expected_x_column = expected_x_column;

    std::vector<T> row_shift{-2, -2, -4, -4, -6, -6, -8};
    param.row_shift = row_shift;
    std::vector<T> row_scale{2, 0, 0, 1, 2, 4, 4};
    param.row_scale = row_scale;
    std::vector<T> expected_x_row{1, 4, 8, 10, 7, 4, 5};
    param.expected_x_row = expected_x_row;

    std::vector<T> overall_shift{-4};
    param.overall_shift = overall_shift;
    std::vector<T> overall_scale{4};
    param.overall_scale = overall_scale;
    std::vector<T> expected_x_overall{1, 1.5, 2, 2.5, 3, 3.5, 4};
    param.expected_x_overall = expected_x_overall;

    param.expected_status = da_status_success;

    param.epsilon = std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetStandardData(std::vector<StatsParamType<T>> &params) {
    // Test with typical data
    StatsParamType<T> param;
    param.n = 5;
    param.p = 6;
    param.ldx = param.n;

    std::vector<T> x{0, 2,   4,   6, 8,   10, 12.2, 4, -8, 4,   2,   7,   -6, 1.2, 5.0,
                     0, 2.2, 4.1, 6, 4.8, 10, 12,   4, -8, 0.4, 1.2, 7.3, -6, 1.2, 5};
    param.x = x;

    std::vector<T> column_shift{6, -2, 0, 1.1, 2, 6};
    param.column_shift = column_shift;
    std::vector<T> column_scale{0.5, 0, 1, 0.5, 0.25, -0.5};
    param.column_scale = column_scale;
    std::vector<T> expected_x_column{-12, -8, -4, 0,   4,    12,   14.2, 6,  -6,  6,
                                     2,   7,  -6, 1.2, 5.0,  -2.2, 2.2,  6,  9.8, 7.4,
                                     32,  40, 8,  -40, -6.4, 9.6,  -2.6, 24, 9.6, 2};
    param.expected_x_column = expected_x_column;

    std::vector<T> row_shift{-1, -2, -3, -4, -5};
    param.row_shift = row_shift;
    std::vector<T> row_scale{2, 0, 1, 0.5, -1};
    param.row_scale = row_scale;
    std::vector<T> expected_x_row{0.5, 4,  7,  20,   -13,   5.5, 14.2, 7,   -8,   -9,
                                  1.5, 9,  -3, 10.4, -10.0, 0.5, 4.2,  7.1, 20,   -9.8,
                                  5.5, 14, 7,  -8,   -5.4,  1.1, 9.3,  -3,  10.4, -10};
    param.expected_x_row = expected_x_row;

    std::vector<T> overall_shift{-4};
    param.overall_shift = overall_shift;
    std::vector<T> overall_scale{2};
    param.overall_scale = overall_scale;
    std::vector<T> expected_x_overall{2, 3,   4,  5,   6,   7,   8.1,  4,    -2,  4,
                                      3, 5.5, -1, 2.6, 4.5, 2,   3.1,  4.05, 5,   4.4,
                                      7, 8,   4,  -2,  2.2, 2.6, 5.65, -1,   2.6, 4.5};
    param.expected_x_overall = expected_x_overall;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetSubarrayData(std::vector<StatsParamType<T>> &params) {
    // Test with matrix stored in a subarray
    StatsParamType<T> param;
    param.n = 5;
    param.p = 6;
    param.ldx = param.n + 1;

    std::vector<T> x{0,  2,  4,  6,   8,   0, 10,  12.2, 4,   -8,  4,   0,
                     2,  7,  -6, 1.2, 5.0, 0, 0,   2.2,  4.1, 6,   4.8, 0,
                     10, 12, 4,  -8,  0.4, 0, 1.2, 7.3,  -6,  1.2, 5,   0};
    param.x = x;

    std::vector<T> column_shift{6, -2, 0, 1.1, 2, 6};
    param.column_shift = column_shift;
    std::vector<T> column_scale{0.5, 0, 1, 0.5, 0.25, -0.5};
    param.column_scale = column_scale;
    std::vector<T> expected_x_column{-12,  -8,   -4, 0,   4,    0,  12,  14.2, 6,
                                     -6,   6,    0,  2,   7,    -6, 1.2, 5.0,  0,
                                     -2.2, 2.2,  6,  9.8, 7.4,  0,  32,  40,   8,
                                     -40,  -6.4, 0,  9.6, -2.6, 24, 9.6, 2,    0};
    param.expected_x_column = expected_x_column;

    std::vector<T> row_shift{-1, -2, -3, -4, -5};
    param.row_shift = row_shift;
    std::vector<T> row_scale{2, 0, 1, 0.5, -1};
    param.row_scale = row_scale;
    std::vector<T> expected_x_row{0.5, 4,    7,   20,  -13,  0,  5.5,  14.2,  7,
                                  -8,  -9,   0,   1.5, 9,    -3, 10.4, -10.0, 0,
                                  0.5, 4.2,  7.1, 20,  -9.8, 0,  5.5,  14,    7,
                                  -8,  -5.4, 0,   1.1, 9.3,  -3, 10.4, -10,   0};
    param.expected_x_row = expected_x_row;

    std::vector<T> overall_shift{-4};
    param.overall_shift = overall_shift;
    std::vector<T> overall_scale{2};
    param.overall_scale = overall_scale;
    std::vector<T> expected_x_overall{
        2, 3,   4,    5, 6,   0, 7, 8.1, 4, -2, 4,   0, 3,   5.5,  -1, 2.6, 4.5, 0,
        2, 3.1, 4.05, 5, 4.4, 0, 7, 8,   4, -2, 2.2, 0, 2.6, 5.65, -1, 2.6, 4.5, 0};
    param.expected_x_overall = expected_x_overall;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetShiftOnlyData(std::vector<StatsParamType<T>> &params) {
    // Test with only shifting
    StatsParamType<T> param;
    param.n = 6;
    param.p = 5;
    param.ldx = param.n;

    std::vector<T> x{0, 2,   4,   6, 8,   10, 12.2, 4, -8, 4,   2,   7,   -6, 1.2, 5.0,
                     0, 2.2, 4.1, 6, 4.8, 10, 12,   4, -8, 0.4, 1.2, 7.3, -6, 1.2, 5};
    param.x = x;

    std::vector<T> column_shift{6, -2, 0, 1.1, 2};
    param.column_shift = column_shift;
    std::vector<T> column_scale(0);
    param.column_scale = column_scale;
    std::vector<T> expected_x_column{
        -6, -4,  -2,  0,   2,   4,   14.2, 6,   -6,   6,    4,    9,   -6, 1.2,  5.0,
        0,  2.2, 4.1, 4.9, 3.7, 8.9, 10.9, 2.9, -9.1, -1.6, -0.8, 5.3, -8, -0.8, 3};
    param.expected_x_column = expected_x_column;

    std::vector<T> row_shift{-1, -2, -3, -4, -5, -6};
    param.row_shift = row_shift;
    std::vector<T> row_scale(0);
    param.row_scale = row_scale;
    std::vector<T> expected_x_row{1,  4,  7,  10,  13,  16,  13.2, 6,    -5,  8,
                                  7,  13, -5, 3.2, 8.0, 4,   7.2,  10.1, 7,   6.8,
                                  13, 16, 9,  -2,  1.4, 3.2, 10.3, -2,   6.2, 11};
    param.expected_x_row = expected_x_row;

    std::vector<T> overall_shift{-4};
    param.overall_shift = overall_shift;
    std::vector<T> overall_scale(0);
    param.overall_scale = overall_scale;
    std::vector<T> expected_x_overall{4,  6,  8,  10,  12,  14,  16.2, 8,   -4,  8,
                                      6,  11, -2, 5.2, 9.0, 4,   6.2,  8.1, 10,  8.8,
                                      14, 16, 8,  -4,  4.4, 5.2, 11.3, -2,  5.2, 9};
    param.expected_x_overall = expected_x_overall;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetScaleOnlyData(std::vector<StatsParamType<T>> &params) {
    // Test with only scaling
    StatsParamType<T> param;
    param.n = 4;
    param.p = 5;
    param.ldx = param.n;

    std::vector<T> x{0,   2.2, 4.1, 6.3, 8,   10, 12.2, 4.1,  -8,  4,
                     2.6, 7.3, -6,  1.2, 5.0, 0,  2.2,  -4.1, 6.8, 4.8};
    param.x = x;

    std::vector<T> column_shift(0);
    param.column_shift = column_shift;
    std::vector<T> column_scale{0.5, 2, 0, 1, 0.25};
    param.column_scale = column_scale;
    std::vector<T> expected_x_column{0,   4.4, 8.2, 12.6, 4,   5, 6.1, 2.05,  -8,   4,
                                     2.6, 7.3, -6,  1.2,  5.0, 0, 8.8, -16.4, 27.2, 19.2};
    param.expected_x_column = expected_x_column;

    std::vector<T> row_shift(0);
    param.row_shift = row_shift;
    std::vector<T> row_scale{1, 2, 0.5, 2};
    param.row_scale = row_scale;
    std::vector<T> expected_x_row{0,   1.1,  8.2, 3.15, 8,    5, 24.4, 2.05,  -8,   2,
                                  5.2, 3.65, -6,  0.6,  10.0, 0, 2.2,  -2.05, 13.6, 2.4};
    param.expected_x_row = expected_x_row;

    std::vector<T> overall_shift(0);
    param.overall_shift = overall_shift;
    std::vector<T> overall_scale{2};
    param.overall_scale = overall_scale;
    std::vector<T> expected_x_overall{0,    1.1, 2.05, 3.15,  4,    5,  6.1,
                                      2.05, -4,  2,    1.3,   3.65, -3, 0.6,
                                      2.5,  0,   1.1,  -2.05, 3.4,  2.4};
    param.expected_x_overall = expected_x_overall;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T>
void GetNullShiftAndScaleData(std::vector<StatsParamType<T>> &params) {
    // Test with only scaling
    StatsParamType<T> param;
    param.n = 4;
    param.p = 4;
    param.ldx = param.n;

    std::vector<T> x{0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 4, 8, 16};
    param.x = x;

    std::vector<T> column_shift(0);
    param.column_shift = column_shift;
    std::vector<T> column_scale(0);
    param.column_scale = column_scale;
    std::vector<T> expected_x_column{0,
                                     0,
                                     0,
                                     0,
                                     -1.161895003862225,
                                     -0.3872983346207417,
                                     0.3872983346207417,
                                     1.161895003862225,
                                     -1.161895003862225,
                                     -0.3872983346207417,
                                     0.3872983346207417,
                                     1.161895003862225,
                                     -1.02469507659596,
                                     -0.4391550328268399,
                                     0.14638501094228, 1.3174650984805198};
    param.expected_x_column = expected_x_column;

    std::vector<T> row_shift(0);
    param.row_shift = row_shift;
    std::vector<T> row_scale(0);
    param.row_scale = row_scale;
    std::vector<T> expected_x_row{
        0, -1.02469507659596,   -1.02469507659596,   -0.8997696884358682,
        0, -0.4391550328268399, -0.4391550328268399, -0.4678802379866515,
        0, 0.14638501094228,    0.14638501094228,    -0.0359907875374347,
        0, 1.3174650984805198,  1.3174650984805198,  1.4036407139599545};
    param.expected_x_row = expected_x_row;

    std::vector<T> overall_shift(0);
    param.overall_shift = overall_shift;
    std::vector<T> overall_scale(0);
    param.overall_scale = overall_scale;
    std::vector<T> expected_x_overall{
        -0.6729865963777508, -0.6729865963777508, -0.6729865963777508,
        -0.6729865963777508, -0.6729865963777508, -0.4389043019854896,
        -0.2048220075932285, 0.0292602867990326,  -0.6729865963777508,
        -0.2048220075932285, 0.2633425811912938,  0.7315071699758161,
        -0.6729865963777508, 0.2633425811912938,  1.1996717587603385,
        3.0723301138984276};
    param.expected_x_overall = expected_x_overall;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetStatsData(std::vector<StatsParamType<T>> &params) {

    GetStandardData(params);
    GetShiftOnlyData(params);
    GetScaleOnlyData(params);
    GetNullShiftAndScaleData(params);
    GetSubarrayData(params);
    GetSingleRowData(params);
    GetSingleColumnData(params);
    Get1by1Data(params);
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(StatisticsUtilitiesTest, FloatTypes);

TYPED_TEST(StatisticsUtilitiesTest, StatisticsUtilitiesFunctionality) {

    std::vector<StatsParamType<TypeParam>> params;
    GetStatsData(params);

    for (auto &param : params) {
        std::vector<TypeParam> x_column(param.x);
        std::vector<TypeParam> x_row(param.x);
        std::vector<TypeParam> x_overall(param.x);

        TypeParam *column_shift = nullptr, *column_scale = nullptr, *row_shift = nullptr,
                  *row_scale = nullptr, *overall_shift = nullptr,
                  *overall_scale = nullptr;

        if (param.column_shift.size() > 0)
            column_shift = param.column_shift.data();
        if (param.column_scale.size() > 0)
            column_scale = param.column_scale.data();
        if (param.row_shift.size() > 0)
            row_shift = param.row_shift.data();
        if (param.row_scale.size() > 0)
            row_scale = param.row_scale.data();
        if (param.overall_shift.size() > 0)
            overall_shift = param.overall_shift.data();
        if (param.overall_scale.size() > 0)
            overall_scale = param.overall_scale.data();

        ASSERT_EQ(da_standardize(da_axis_col, param.n, param.p, x_column.data(), param.ldx,
                                 column_shift, column_scale),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.ldx * param.p, param.expected_x_column.data(),
                        x_column.data(), param.epsilon);
        ASSERT_EQ(da_standardize(da_axis_row, param.n, param.p, x_row.data(), param.ldx,
                                 row_shift, row_scale),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.ldx * param.p, param.expected_x_row.data(), x_row.data(),
                        param.epsilon);
        ASSERT_EQ(da_standardize(da_axis_all, param.n, param.p, x_overall.data(), param.ldx,
                                 overall_shift, overall_scale),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.ldx * param.p, param.expected_x_overall.data(),
                        x_overall.data(), param.epsilon);
    }
}

TYPED_TEST(StatisticsUtilitiesTest, IllegalArgsStatisticsUtilities) {

    std::vector<TypeParam> x{4.7, 1.2, -0.3, 4.5};
    da_int n = 2, p = 2, ldx = 2;
    std::vector<TypeParam> scale(2, 0);
    std::vector<TypeParam> shift(2, 0);

    // Test with illegal value of ldx
    da_int ldx_illegal = 1;
    ASSERT_EQ(
        da_standardize(da_axis_all, n, p, x.data(), ldx_illegal, scale.data(), shift.data()),
        da_status_invalid_leading_dimension);

    // Test with illegal p
    da_int p_illegal = 0;
    ASSERT_EQ(
        da_standardize(da_axis_all, n, p_illegal, x.data(), ldx, scale.data(), shift.data()),
        da_status_invalid_array_dimension);

    // Test with illegal n
    da_int n_illegal = 0;
    ASSERT_EQ(
        da_standardize(da_axis_all, n_illegal, p, x.data(), ldx, scale.data(), shift.data()),
        da_status_invalid_array_dimension);

    // Test illegal pointers
    TypeParam *x_null = nullptr;
    ASSERT_EQ(da_standardize(da_axis_all, n, p, x_null, ldx, scale.data(), shift.data()),
              da_status_invalid_pointer);
}