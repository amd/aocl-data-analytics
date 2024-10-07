/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
    da_int n = 0;
    da_int p = 0;
    da_int ldx = 0;
    da_int dof = 0;
    da_int mode = 0;
    std::vector<T> x;
    std::vector<T> expected_x_column;
    std::vector<T> column_shift;
    std::vector<T> column_scale;
    std::vector<T> expected_column_shift;
    std::vector<T> expected_column_scale;

    std::vector<T> expected_x_row;
    std::vector<T> row_shift;
    std::vector<T> row_scale;
    std::vector<T> expected_row_shift;
    std::vector<T> expected_row_scale;

    std::vector<T> expected_x_overall;
    std::vector<T> overall_shift;
    std::vector<T> overall_scale;
    std::vector<T> expected_overall_shift;
    std::vector<T> expected_overall_scale;

    da_status expected_status = da_status_success;
    T epsilon = std::numeric_limits<T>::epsilon();
};

template <typename T> void Get1by1Data(std::vector<StatsParamType<T>> &params) {
    // Test with 1 x 1 data matrix
    StatsParamType<T> param;
    param.n = 1;
    param.p = 1;
    param.ldx = param.n;

    std::vector<double> x(param.n * param.p, 3);
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift(param.p, 1);
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<double> column_scale(param.p, 2);
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column(param.p * param.n, 1);
    param.expected_x_column = convert_vector<double, T>(expected_x_column);

    std::vector<double> row_shift(param.n, -1);
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<double> row_scale(param.n, 2);
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row(param.p * param.n, 2);
    param.expected_x_row = convert_vector<double, T>(expected_x_row);

    std::vector<double> overall_shift(1, 1);
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<double> overall_scale(1, 2);
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall(param.p * param.n, 1);
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);

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

    std::vector<double> x{0, 1, 2, 3, 4, 5, 6};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift{2, 4, 6, 8, 10, 12, 14};
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<double> column_scale{0, 1, 4, 4, 0, 4, 2};
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column{-2, -3, -1, -1.25, -6, -1.75, -4};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);

    std::vector<double> row_shift{-1};
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<double> row_scale{2};
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row{0.5, 1, 1.5, 2, 2.5, 3, 3.5};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);

    std::vector<double> overall_shift{1};
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<double> overall_scale{1};
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall{-1, 0, 1, 2, 3, 4, 5};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);

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

    std::vector<double> x{0, 2, 4, 6, 8, 10, 12};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift{6};
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<double> column_scale{0.5};
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column{-12, -8, -4, 0, 4, 8, 12};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);

    std::vector<double> row_shift{-2, -2, -4, -4, -6, -6, -8};
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<double> row_scale{2, 0, 0, 1, 2, 4, 4};
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row{1, 4, 8, 10, 7, 4, 5};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);

    std::vector<double> overall_shift{-4};
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<double> overall_scale{4};
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall{1, 1.5, 2, 2.5, 3, 3.5, 4};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);

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

    std::vector<double> x{0,  2,  4,  6,   8,   10,  12.2, 4,   -8,  4,
                          2,  7,  -6, 1.2, 5.0, 0,   2.2,  4.1, 6,   4.8,
                          10, 12, 4,  -8,  0.4, 1.2, 7.3,  -6,  1.2, 5};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift{6, -2, 0, 1.1, 2, 6};
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<double> column_scale{0.5, 0, 1, 0.5, 0.25, -0.5};
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column{
        -12,  -8,  -4, 0,   4,   12, 14.2, 6, -6,  6,    2,   7,    -6, 1.2, 5.0,
        -2.2, 2.2, 6,  9.8, 7.4, 32, 40,   8, -40, -6.4, 9.6, -2.6, 24, 9.6, 2};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);

    std::vector<double> row_shift{-1, -2, -3, -4, -5};
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<double> row_scale{2, 0, 1, 0.5, -1};
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row{
        0.5, 4,   7,   20, -13,  5.5, 14.2, 7, -8, -9,   1.5, 9,   -3, 10.4, -10.0,
        0.5, 4.2, 7.1, 20, -9.8, 5.5, 14,   7, -8, -5.4, 1.1, 9.3, -3, 10.4, -10};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);

    std::vector<double> overall_shift{-4};
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<double> overall_scale{2};
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall{
        2, 3,   4,    5, 6,   7, 8.1, 4, -2, 4,   3,   5.5,  -1, 2.6, 4.5,
        2, 3.1, 4.05, 5, 4.4, 7, 8,   4, -2, 2.2, 2.6, 5.65, -1, 2.6, 4.5};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);

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

    std::vector<double> x{0,  2,  4,  6,   8,   0, 10,  12.2, 4,   -8,  4,   0,
                          2,  7,  -6, 1.2, 5.0, 0, 0,   2.2,  4.1, 6,   4.8, 0,
                          10, 12, 4,  -8,  0.4, 0, 1.2, 7.3,  -6,  1.2, 5,   0};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift{6, -2, 0, 1.1, 2, 6};
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<double> column_scale{0.5, 0, 1, 0.5, 0.25, -0.5};
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column{-12,  -8,   -4, 0,   4,    0,  12,  14.2, 6,
                                          -6,   6,    0,  2,   7,    -6, 1.2, 5.0,  0,
                                          -2.2, 2.2,  6,  9.8, 7.4,  0,  32,  40,   8,
                                          -40,  -6.4, 0,  9.6, -2.6, 24, 9.6, 2,    0};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);

    std::vector<double> row_shift{-1, -2, -3, -4, -5};
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<double> row_scale{2, 0, 1, 0.5, -1};
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row{0.5, 4,    7,   20,  -13,  0,  5.5,  14.2,  7,
                                       -8,  -9,   0,   1.5, 9,    -3, 10.4, -10.0, 0,
                                       0.5, 4.2,  7.1, 20,  -9.8, 0,  5.5,  14,    7,
                                       -8,  -5.4, 0,   1.1, 9.3,  -3, 10.4, -10,   0};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);

    std::vector<double> overall_shift{-4};
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<double> overall_scale{2};
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall{
        2, 3,   4,    5, 6,   0, 7, 8.1, 4, -2, 4,   0, 3,   5.5,  -1, 2.6, 4.5, 0,
        2, 3.1, 4.05, 5, 4.4, 0, 7, 8,   4, -2, 2.2, 0, 2.6, 5.65, -1, 2.6, 4.5, 0};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);

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

    std::vector<double> x{0,  2,  4,  6,   8,   10,  12.2, 4,   -8,  4,
                          2,  7,  -6, 1.2, 5.0, 0,   2.2,  4.1, 6,   4.8,
                          10, 12, 4,  -8,  0.4, 1.2, 7.3,  -6,  1.2, 5};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift{6, -2, 0, 1.1, 2};
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<T> column_scale(0);
    param.column_scale = column_scale;
    std::vector<double> expected_x_column{
        -6, -4,  -2,  0,   2,   4,   14.2, 6,   -6,   6,    4,    9,   -6, 1.2,  5.0,
        0,  2.2, 4.1, 4.9, 3.7, 8.9, 10.9, 2.9, -9.1, -1.6, -0.8, 5.3, -8, -0.8, 3};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);

    std::vector<double> row_shift{-1, -2, -3, -4, -5, -6};
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<T> row_scale(0);
    param.row_scale = row_scale;
    std::vector<double> expected_x_row{1,  4,  7,  10,  13,  16,  13.2, 6,    -5,  8,
                                       7,  13, -5, 3.2, 8.0, 4,   7.2,  10.1, 7,   6.8,
                                       13, 16, 9,  -2,  1.4, 3.2, 10.3, -2,   6.2, 11};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);

    std::vector<double> overall_shift{-4};
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<T> overall_scale(0);
    param.overall_scale = overall_scale;
    std::vector<double> expected_x_overall{4,  6,  8,  10,  12,  14,  16.2, 8,   -4,  8,
                                           6,  11, -2, 5.2, 9.0, 4,   6.2,  8.1, 10,  8.8,
                                           14, 16, 8,  -4,  4.4, 5.2, 11.3, -2,  5.2, 9};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);

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

    std::vector<double> x{0,   2.2, 4.1, 6.3, 8,   10, 12.2, 4.1,  -8,  4,
                          2.6, 7.3, -6,  1.2, 5.0, 0,  2.2,  -4.1, 6.8, 4.8};
    param.x = convert_vector<double, T>(x);

    std::vector<T> column_shift(0);
    param.column_shift = column_shift;
    std::vector<double> column_scale{0.5, 2, 0, 1, 0.25};
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column{0,    4.4, 8.2, 12.6,  4,    5,   6.1,
                                          2.05, -8,  4,   2.6,   7.3,  -6,  1.2,
                                          5.0,  0,   8.8, -16.4, 27.2, 19.2};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);

    std::vector<T> row_shift(0);
    param.row_shift = row_shift;
    std::vector<double> row_scale{1, 2, 0.5, 2};
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row{0,    1.1, 8.2, 3.15,  8,    5,  24.4,
                                       2.05, -8,  2,   5.2,   3.65, -6, 0.6,
                                       10.0, 0,   2.2, -2.05, 13.6, 2.4};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);

    std::vector<T> overall_shift(0);
    param.overall_shift = overall_shift;
    std::vector<double> overall_scale{2};
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall{0,    1.1, 2.05, 3.15,  4,    5,  6.1,
                                           2.05, -4,  2,    1.3,   3.65, -3, 0.6,
                                           2.5,  0,   1.1,  -2.05, 3.4,  2.4};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);

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

    std::vector<double> x{0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 4, 8, 16};
    param.x = convert_vector<double, T>(x);

    std::vector<T> column_shift(0);
    param.column_shift = column_shift;
    std::vector<T> column_scale(0);
    param.column_scale = column_scale;
    std::vector<double> expected_x_column{0,
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
                                          0.14638501094228,
                                          1.3174650984805198};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);

    std::vector<T> row_shift(0);
    param.row_shift = row_shift;
    std::vector<T> row_scale(0);
    param.row_scale = row_scale;
    std::vector<double> expected_x_row{
        0, -1.02469507659596,   -1.02469507659596,   -0.8997696884358682,
        0, -0.4391550328268399, -0.4391550328268399, -0.4678802379866515,
        0, 0.14638501094228,    0.14638501094228,    -0.0359907875374347,
        0, 1.3174650984805198,  1.3174650984805198,  1.4036407139599545};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);

    std::vector<T> overall_shift(0);
    param.overall_shift = overall_shift;
    std::vector<T> overall_scale(0);
    param.overall_scale = overall_scale;
    std::vector<double> expected_x_overall{
        -0.6729865963777508, -0.6729865963777508, -0.6729865963777508,
        -0.6729865963777508, -0.6729865963777508, -0.4389043019854896,
        -0.2048220075932285, 0.0292602867990326,  -0.6729865963777508,
        -0.2048220075932285, 0.2633425811912938,  0.7315071699758161,
        -0.6729865963777508, 0.2633425811912938,  1.1996717587603385,
        3.0723301138984276};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T>
void GetShiftZeroScaleNonZero(std::vector<StatsParamType<T>> &params) {
    // Test with shift full of zeros and scale non zero
    StatsParamType<T> param;
    param.n = 5;
    param.p = 4;
    param.ldx = param.n;

    std::vector<double> x{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift{0, 0, 0, 0};
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<double> column_scale{0.5, 0.0, 1.0, -2.0};
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column{-4, -2, 0, 2, 4, -2, -1,  0, 1,    2,
                                          -2, -1, 0, 1, 2, 1,  0.5, 0, -0.5, -1.0};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);
    std::vector<double> expected_column_shift{3, 8, 13, 18};
    param.expected_column_shift = convert_vector<double, T>(expected_column_shift);
    std::vector<double> expected_column_scale{0.5, 0.0, 1.0, -2.0};
    param.expected_column_scale = convert_vector<double, T>(expected_column_scale);

    std::vector<double> row_shift{0, 0, 0, 0, 0};
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<double> row_scale{2, 0, 1, 0.5, -1};
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row{-3.75, -7.5, -7.5, -15,  7.5, -1.25, -2.5,
                                       -2.5,  -5,   2.5,  1.25, 2.5, 2.5,   5,
                                       -2.5,  3.75, 7.5,  7.5,  15,  -7.5};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);
    std::vector<double> expected_row_shift{8.5, 9.5, 10.5, 11.5, 12.5};
    param.expected_row_shift = convert_vector<double, T>(expected_row_shift);
    std::vector<double> expected_row_scale{2, 0, 1, 0.5, -1};
    param.expected_row_scale = convert_vector<double, T>(expected_row_scale);

    std::vector<double> overall_shift{0};
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<double> overall_scale{2};
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall{
        -4.75, -4.25, -3.75, -3.25, -2.75, -2.25, -1.75, -1.25, -0.75, -0.25,
        0.25,  0.75,  1.25,  1.75,  2.25,  2.75,  3.25,  3.75,  4.25,  4.75};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);
    std::vector<double> expected_overall_shift{10.5};
    param.expected_overall_shift = convert_vector<double, T>(expected_overall_shift);
    std::vector<double> expected_overall_scale{2};
    param.expected_overall_scale = convert_vector<double, T>(expected_overall_scale);

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetShiftZeroScaleNull(std::vector<StatsParamType<T>> &params) {
    // Test with shift full of zeros and scale null
    StatsParamType<T> param;
    param.n = 5;
    param.p = 4;
    param.ldx = param.n;

    std::vector<double> x{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift{0, 0, 0, 0};
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<double> expected_x_column{-2, -1, 0,  1,  2, -2, -1, 0,  1,  2, -2, -1, 0,
                                          1,  2,  -2, -1, 0, 1,  2,  -2, -1, 0, 1,  2};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);
    std::vector<double> expected_column_shift{3, 8, 13, 18};
    param.expected_column_shift = convert_vector<double, T>(expected_column_shift);

    std::vector<double> row_shift{0, 0, 0, 0, 0};
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<double> expected_x_row{-7.5, -7.5, -7.5, -7.5, -7.5, -2.5, -2.5,
                                       -2.5, -2.5, -2.5, 2.5,  2.5,  2.5,  2.5,
                                       2.5,  7.5,  7.5,  7.5,  7.5,  7.5};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);
    std::vector<double> expected_row_shift{8.5, 9.5, 10.5, 11.5, 12.5};
    param.expected_row_shift = convert_vector<double, T>(expected_row_shift);

    std::vector<double> overall_shift{0};
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<double> expected_x_overall{-9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5,
                                           -2.5, -1.5, -0.5, 0.5,  1.5,  2.5,  3.5,
                                           4.5,  5.5,  6.5,  7.5,  8.5,  9.5};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);
    std::vector<double> expected_overall_shift{10.5};
    param.expected_overall_shift = convert_vector<double, T>(expected_overall_shift);

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetShiftNullScaleZero(std::vector<StatsParamType<T>> &params) {
    // Test with shift null and scale zero
    StatsParamType<T> param;
    param.n = 5;
    param.p = 4;
    param.dof = -1;
    param.ldx = param.n;

    std::vector<double> x{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_scale{0, 0, 0, 0};
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column{
        0.7071067811865475, 1.414213562373095,  2.1213203435596424, 2.82842712474619,
        3.5355339059327373, 4.242640687119285,  4.949747468305833,  5.65685424949238,
        6.363961030678928,  7.071067811865475,  7.7781745930520225, 8.48528137423857,
        9.192388155425117,  9.899494936611665,  10.606601717798211, 11.31370849898476,
        12.020815280171307, 12.727922061357855, 13.435028842544401, 14.14213562373095};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);
    std::vector<double> expected_column_scale{1.4142135623730951, 1.4142135623730951,
                                              1.4142135623730951, 1.4142135623730951};
    param.expected_column_scale = convert_vector<double, T>(expected_column_scale);

    std::vector<double> row_scale{0, 0, 0, 0, 0};
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row{
        0.17888543819998318, 0.35777087639996635, 0.5366563145999494, 0.7155417527999327,
        0.8944271909999159,  1.073312629199899,   1.2521980673998823, 1.4310835055998654,
        1.6099689437998486,  1.7888543819998317,  1.9677398201998149, 2.146625258399798,
        2.3255106965997814,  2.5043961347997645,  2.6832815729997477, 2.862167011199731,
        3.041052449399714,   3.219937887599697,   3.3988233257996803, 3.5777087639996634};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);
    std::vector<double> expected_row_scale{5.5901699437494745, 5.5901699437494745,
                                           5.5901699437494745, 5.5901699437494745,
                                           5.5901699437494745};
    param.expected_row_scale = convert_vector<double, T>(expected_row_scale);

    std::vector<double> overall_scale{0};
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall{
        0.173421993904824,  0.346843987809648,  0.5202659817144719, 0.693687975619296,
        0.8671099695241199, 1.0405319634289438, 1.2139539573337679, 1.387375951238592,
        1.560797945143416,  1.7342199390482398, 1.9076419329530638, 2.0810639268578877,
        2.2544859207627117, 2.4279079146675357, 2.60132990857236,   2.774751902477184,
        2.948173896382008,  3.121595890286832,  3.295017884191656,  3.4684398780964796};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);
    std::vector<double> expected_overall_scale{5.766281297335398};
    param.expected_overall_scale = convert_vector<double, T>(expected_overall_scale);

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T>
void GetShiftNonZeroScaleZero(std::vector<StatsParamType<T>> &params) {
    // Test with shift nonzero and scale full of zeros
    StatsParamType<T> param;
    param.n = 5;
    param.p = 4;
    param.dof = -1;
    param.ldx = param.n;

    std::vector<double> x{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift{3, 8, 13, 18};
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<double> column_scale{0, 0, 0, 0};
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column{-1.414213562373095,
                                          -0.7071067811865475,
                                          0,
                                          0.7071067811865475,
                                          1.414213562373095,
                                          -1.414213562373095,
                                          -0.7071067811865475,
                                          0,
                                          0.7071067811865475,
                                          1.414213562373095,
                                          -1.414213562373095,
                                          -0.7071067811865475,
                                          0,
                                          0.7071067811865475,
                                          1.414213562373095,
                                          -1.414213562373095,
                                          -0.7071067811865475,
                                          0,
                                          0.7071067811865475,
                                          1.414213562373095};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);
    std::vector<double> expected_column_scale{1.4142135623730951, 1.4142135623730951,
                                              1.4142135623730951, 1.4142135623730951};
    param.expected_column_scale = convert_vector<double, T>(expected_column_scale);
    std::vector<double> expected_column_shift{3, 8, 13, 18};
    param.expected_column_shift = convert_vector<double, T>(expected_column_shift);
    std::vector<double> row_shift{8.5, 9.5, 10.5, 11.5, 12.5};
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<double> row_scale{0, 0, 0, 0, 0};
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row{
        -1.3416407864998738, -1.3416407864998738, -1.3416407864998738,
        -1.3416407864998738, -1.3416407864998738, -0.4472135954999579,
        -0.4472135954999579, -0.4472135954999579, -0.4472135954999579,
        -0.4472135954999579, 0.4472135954999579,  0.4472135954999579,
        0.4472135954999579,  0.4472135954999579,  0.4472135954999579,
        1.3416407864998738,  1.3416407864998738,  1.3416407864998738,
        1.3416407864998738,  1.3416407864998738};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);
    std::vector<double> expected_row_scale{5.5901699437494745, 5.5901699437494745,
                                           5.5901699437494745, 5.5901699437494745,
                                           5.5901699437494745};
    param.expected_row_scale = convert_vector<double, T>(expected_row_scale);
    std::vector<double> expected_row_shift{8.5, 9.5, 10.5, 11.5, 12.5};
    param.expected_row_shift = convert_vector<double, T>(expected_row_shift);
    std::vector<double> overall_shift{10.5};
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<double> overall_scale{0};
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall{
        -1.647508942095828,  -1.474086948191004,   -1.30066495428618,
        -1.1272429603813559, -0.9538209664765319,  -0.780398972571708,
        -0.6069769786668839, -0.43355498476205995, -0.26013299085723596,
        -0.086710996952412,  0.086710996952412,    0.26013299085723596,
        0.43355498476205995, 0.6069769786668839,   0.780398972571708,
        0.9538209664765319,  1.1272429603813559,   1.30066495428618,
        1.474086948191004,   1.647508942095828};

    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);
    std::vector<double> expected_overall_scale{5.766281297335398};
    param.expected_overall_scale = convert_vector<double, T>(expected_overall_scale);
    std::vector<double> expected_overall_shift{10.5};
    param.expected_overall_shift = convert_vector<double, T>(expected_overall_shift);
    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetShiftZeroScaleZero(std::vector<StatsParamType<T>> &params) {
    // Test with shift full of zeros and scale full of zeros
    StatsParamType<T> param;
    param.n = 5;
    param.p = 4;
    param.dof = -1;
    param.ldx = param.n;

    std::vector<double> x{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift{0, 0, 0, 0};
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<double> column_scale{0, 0, 0, 0};
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column{-1.414213562373095,
                                          -0.7071067811865475,
                                          0,
                                          0.7071067811865475,
                                          1.414213562373095,
                                          -1.414213562373095,
                                          -0.7071067811865475,
                                          0,
                                          0.7071067811865475,
                                          1.414213562373095,
                                          -1.414213562373095,
                                          -0.7071067811865475,
                                          0,
                                          0.7071067811865475,
                                          1.414213562373095,
                                          -1.414213562373095,
                                          -0.7071067811865475,
                                          0,
                                          0.7071067811865475,
                                          1.414213562373095};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);
    std::vector<double> expected_column_shift{3, 8, 13, 18};
    param.expected_column_shift = convert_vector<double, T>(expected_column_shift);
    std::vector<double> expected_column_scale{1.4142135623730951, 1.4142135623730951,
                                              1.4142135623730951, 1.4142135623730951};
    param.expected_column_scale = convert_vector<double, T>(expected_column_scale);

    std::vector<double> row_shift{0, 0, 0, 0, 0};
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<double> row_scale{0, 0, 0, 0, 0};
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row{
        -1.3416407864998738, -1.3416407864998738, -1.3416407864998738,
        -1.3416407864998738, -1.3416407864998738, -0.4472135954999579,
        -0.4472135954999579, -0.4472135954999579, -0.4472135954999579,
        -0.4472135954999579, 0.4472135954999579,  0.4472135954999579,
        0.4472135954999579,  0.4472135954999579,  0.4472135954999579,
        1.3416407864998738,  1.3416407864998738,  1.3416407864998738,
        1.3416407864998738,  1.3416407864998738};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);
    std::vector<double> expected_row_scale{5.5901699437494745, 5.5901699437494745,
                                           5.5901699437494745, 5.5901699437494745,
                                           5.5901699437494745};
    param.expected_row_scale = convert_vector<double, T>(expected_row_scale);
    std::vector<double> expected_row_shift{8.5, 9.5, 10.5, 11.5, 12.5};
    param.expected_row_shift = convert_vector<double, T>(expected_row_shift);

    std::vector<double> overall_shift{0};
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<double> overall_scale{0};
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall{
        -1.647508942095828,  -1.474086948191004,   -1.30066495428618,
        -1.1272429603813559, -0.9538209664765319,  -0.780398972571708,
        -0.6069769786668839, -0.43355498476205995, -0.26013299085723596,
        -0.086710996952412,  0.086710996952412,    0.26013299085723596,
        0.43355498476205995, 0.6069769786668839,   0.780398972571708,
        0.9538209664765319,  1.1272429603813559,   1.30066495428618,
        1.474086948191004,   1.647508942095828};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);
    std::vector<double> expected_overall_shift{10.5};
    param.expected_overall_shift = convert_vector<double, T>(expected_overall_shift);
    std::vector<double> expected_overall_scale{5.766281297335398};
    param.expected_overall_scale = convert_vector<double, T>(expected_overall_scale);

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetModeOne(std::vector<StatsParamType<T>> &params) {
    // Test with mode = 1
    StatsParamType<T> param;
    param.n = 5;
    param.p = 4;
    param.dof = -1;
    param.mode = 1;
    param.ldx = param.n;

    std::vector<double> x{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    param.x = convert_vector<double, T>(x);

    std::vector<double> column_shift{2, 3, 1, 4};
    param.column_shift = convert_vector<double, T>(column_shift);
    std::vector<double> column_scale{2, 10, 1, 3};
    param.column_scale = convert_vector<double, T>(column_scale);
    std::vector<double> expected_x_column{4,  6,  8,  10, 12, 63, 73, 83, 93, 103,
                                          12, 13, 14, 15, 16, 52, 55, 58, 61, 64};
    param.expected_x_column = convert_vector<double, T>(expected_x_column);
    std::vector<double> expected_column_shift{2, 3, 1, 4};
    param.expected_column_shift = convert_vector<double, T>(expected_column_shift);
    std::vector<double> expected_column_scale{2, 10, 1, 3};
    param.expected_column_scale = convert_vector<double, T>(expected_column_scale);

    std::vector<double> row_shift{1, 2, 3, 4, 5};
    param.row_shift = convert_vector<double, T>(row_shift);
    std::vector<double> row_scale{1, 2, 1, 2, 3};
    param.row_scale = convert_vector<double, T>(row_scale);
    std::vector<double> expected_x_row{2,  6,  6,  12, 20, 7,  16, 11, 22, 35,
                                       12, 26, 16, 32, 50, 17, 36, 21, 42, 65};
    param.expected_x_row = convert_vector<double, T>(expected_x_row);
    std::vector<double> expected_row_scale{1, 2, 1, 2, 3};
    param.expected_row_scale = convert_vector<double, T>(expected_row_scale);
    std::vector<double> expected_row_shift{1, 2, 3, 4, 5};
    param.expected_row_shift = convert_vector<double, T>(expected_row_shift);

    std::vector<double> overall_shift{1};
    param.overall_shift = convert_vector<double, T>(overall_shift);
    std::vector<double> overall_scale{2};
    param.overall_scale = convert_vector<double, T>(overall_scale);
    std::vector<double> expected_x_overall{3,  5,  7,  9,  11, 13, 15, 17, 19, 21,
                                           23, 25, 27, 29, 31, 33, 35, 37, 39, 41};
    param.expected_x_overall = convert_vector<double, T>(expected_x_overall);
    std::vector<double> expected_overall_shift{1};
    param.expected_overall_shift = convert_vector<double, T>(expected_overall_shift);
    std::vector<double> expected_overall_scale{2};
    param.expected_overall_scale = convert_vector<double, T>(expected_overall_scale);

    param.expected_status = da_status_success;

    param.epsilon = 100 * std::numeric_limits<T>::epsilon();

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
    GetShiftZeroScaleNonZero(params);
    GetShiftZeroScaleNull(params);
    GetShiftNullScaleZero(params);
    GetShiftNonZeroScaleZero(params);
    GetShiftZeroScaleZero(params);
    GetModeOne(params);
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

        EXPECT_EQ(da_standardize(da_axis_col, param.n, param.p, x_column.data(),
                                 param.ldx, param.dof, param.mode, column_shift,
                                 column_scale),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.ldx * param.p, param.expected_x_column.data(),
                        x_column.data(), param.epsilon);
        EXPECT_EQ(da_standardize(da_axis_row, param.n, param.p, x_row.data(), param.ldx,
                                 param.dof, param.mode, row_shift, row_scale),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.ldx * param.p, param.expected_x_row.data(), x_row.data(),
                        param.epsilon);
        EXPECT_EQ(da_standardize(da_axis_all, param.n, param.p, x_overall.data(),
                                 param.ldx, param.dof, param.mode, overall_shift,
                                 overall_scale),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.ldx * param.p, param.expected_x_overall.data(),
                        x_overall.data(), param.epsilon);

        if (param.expected_column_shift.size() > 0) {
            EXPECT_ARR_NEAR(param.p, param.expected_column_shift.data(), column_shift,
                            param.epsilon);
        }
        if (param.expected_row_shift.size() > 0) {
            EXPECT_ARR_NEAR(param.n, param.expected_row_shift.data(), row_shift,
                            param.epsilon);
        }
        if (param.expected_column_scale.size() > 0) {
            EXPECT_ARR_NEAR(param.p, param.expected_column_scale.data(), column_scale,
                            param.epsilon);
        }
        if (param.expected_row_scale.size() > 0) {
            EXPECT_ARR_NEAR(param.n, param.expected_row_scale.data(), row_scale,
                            param.epsilon);
        }
        if (param.expected_overall_shift.size() > 0) {
            EXPECT_ARR_NEAR(1, param.expected_overall_shift.data(), overall_shift,
                            param.epsilon);
        }
        if (param.expected_overall_scale.size() > 0) {
            EXPECT_ARR_NEAR(1, param.expected_overall_scale.data(), overall_scale,
                            param.epsilon);
        }
    }
}

TYPED_TEST(StatisticsUtilitiesTest, IllegalArgsStatisticsUtilities) {

    std::vector<double> x_d{4.7, 1.2, -0.3, 4.5};
    std::vector<TypeParam> x = convert_vector<double, TypeParam>(x_d);
    da_int n = 2, p = 2, ldx = 2, dof = 0, mode = 0;
    std::vector<double> scale_d(2, 0);
    std::vector<TypeParam> scale = convert_vector<double, TypeParam>(scale_d);
    std::vector<double> shift_d(2, 0);
    std::vector<TypeParam> shift = convert_vector<double, TypeParam>(shift_d);

    // Test with illegal value of ldx
    da_int ldx_illegal = 1;
    EXPECT_EQ(da_standardize(da_axis_all, n, p, x.data(), ldx_illegal, dof, mode,
                             scale.data(), shift.data()),
              da_status_invalid_leading_dimension);

    // Test with illegal p
    da_int p_illegal = 0;
    EXPECT_EQ(da_standardize(da_axis_all, n, p_illegal, x.data(), ldx, dof, mode,
                             scale.data(), shift.data()),
              da_status_invalid_array_dimension);

    // Test with illegal n
    da_int n_illegal = 0;
    EXPECT_EQ(da_standardize(da_axis_all, n_illegal, p, x.data(), ldx, dof, mode,
                             scale.data(), shift.data()),
              da_status_invalid_array_dimension);

    // Test with illegal mode
    da_int mode_illegal = -12;
    EXPECT_EQ(da_standardize(da_axis_all, n, p, x.data(), ldx, dof, mode_illegal,
                             scale.data(), shift.data()),
              da_status_invalid_input);

    // Test illegal pointers
    TypeParam *x_null = nullptr;
    EXPECT_EQ(da_standardize(da_axis_all, n, p, x_null, ldx, dof, mode, scale.data(),
                             shift.data()),
              da_status_invalid_pointer);
}