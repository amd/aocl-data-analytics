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

#include "aoclda.h"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <list>

template <typename T> class OrderStatisticsTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> struct OrderParamType {
    da_int n;
    da_int p;
    da_int ldx;
    T q;
    std::vector<T> x;
    std::vector<T> expected_row_quantiles;
    std::vector<T> expected_column_quantiles;
    T expected_overall_quantile;
    std::vector<T> expected_row_medians;
    std::vector<T> expected_column_medians;
    T expected_overall_median;
    std::vector<T> expected_row_maxima;
    std::vector<T> expected_column_maxima;
    T expected_overall_maximum;
    std::vector<T> expected_row_minima;
    std::vector<T> expected_column_minima;
    T expected_overall_minimum;
    std::vector<T> expected_row_lower_hinges;
    std::vector<T> expected_column_lower_hinges;
    T expected_overall_lower_hinge;
    std::vector<T> expected_row_upper_hinges;
    std::vector<T> expected_column_upper_hinges;
    da_quantile_type quantile_type;
    T expected_overall_upper_hinge;

    da_status expected_status;
    T epsilon;
};

template <typename T> void GetSingleColumnData(std::vector<OrderParamType<T>> &params) {
    // Test a single column
    OrderParamType<T> param;
    param.n = 72;
    param.p = 1;
    param.ldx = param.n;
    param.q = (T)0.1;
    param.quantile_type = da_quantile_type_3;
    std::vector<double> x{
        4.7,  2.6,  7.4,  9.5,  4.6,  5.1,  3.8,  2.1,  -4.7, 1.6,  8.4,  2.5,
        -2.6, -5.0, 8.0,  0.0,  0.0,  -2.6, 5.4,  -9.9, 2.6,  5.2,  -1.8, 1.2,
        1.6,  9.4,  7.6,  -9.5, 1.6,  4.1,  8,    -2.1, -4.7, -2.6, -7.4, 9.1,
        4.2,  5.3,  7.2,  2.1,  7.7,  2.1,  -7.4, -9.1, 4.1,  -3.1, 0.8,  1.2,
        -4.7, 2.6,  -7.4, 6.5,  -4.3, 5.0,  7.1,  -2.0, 4.1,  2.8,  -7.4, 3.5,
        4.6,  -5.9, -8.2, -2,   4.1,  2.8,  -7.4, 3.5,  -4.1, -5.9, 8.3,  -2};
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_column_quantiles{-7.4};
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians{2.1};
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima{9.5};
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima{-9.9};
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges{4.925};
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges{-2.975};
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_row_quantiles = convert_vector<double, T>(x);
    param.expected_row_medians = convert_vector<double, T>(x);
    param.expected_row_maxima = convert_vector<double, T>(x);
    param.expected_row_minima = convert_vector<double, T>(x);
    param.expected_row_upper_hinges = convert_vector<double, T>(x);
    param.expected_row_lower_hinges = convert_vector<double, T>(x);

    param.expected_overall_quantile = (T)-7.4;
    param.expected_overall_maximum = (T)9.5;
    param.expected_overall_minimum = (T)-9.9;
    param.expected_overall_median = (T)2.1;
    param.expected_overall_upper_hinge = (T)4.925;
    param.expected_overall_lower_hinge = (T)-2.975;

    param.expected_status = da_status_success;

    param.epsilon = 50 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetSingleRowData(std::vector<OrderParamType<T>> &params) {
    // Test a single row
    OrderParamType<T> param;
    param.n = 1;
    param.p = 72;
    param.ldx = param.n;
    param.q = (T)0.9;
    param.quantile_type = da_quantile_type_8;
    std::vector<double> x{
        4.7,  2.6,  7.4,  9.5,  4.6,  5.1,  3.8,  2.1,  -4.7, 1.6,  8.4,  2.5,
        -2.6, -5.0, 8.0,  0.0,  0.0,  -2.6, 5.4,  -9.9, 2.6,  5.2,  -1.8, 1.2,
        1.6,  9.4,  7.6,  -9.5, 1.6,  4.1,  8,    -2.1, -4.7, -2.6, -7.4, 9.1,
        4.2,  5.3,  7.2,  2.1,  7.7,  2.1,  -7.4, -9.1, 4.1,  -3.1, 0.8,  1.2,
        -4.7, 2.6,  -7.4, 6.5,  -4.3, 5.0,  7.1,  -2.0, 4.1,  2.8,  -7.4, 3.5,
        4.6,  -5.9, -8.2, -2,   4.1,  2.8,  -7.4, 3.5,  -4.1, -5.9, 8.3,  -2};
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles{7.83};
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians{2.1};
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima{9.5};
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima{-9.9};
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges{4.925};
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges{-2.975};
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    param.expected_column_quantiles = convert_vector<double, T>(x);
    param.expected_column_medians = convert_vector<double, T>(x);
    param.expected_column_maxima = convert_vector<double, T>(x);
    param.expected_column_minima = convert_vector<double, T>(x);
    param.expected_column_upper_hinges = convert_vector<double, T>(x);
    param.expected_column_lower_hinges = convert_vector<double, T>(x);

    param.expected_overall_quantile = (T)7.83;
    param.expected_overall_maximum = (T)9.5;
    param.expected_overall_minimum = (T)-9.9;
    param.expected_overall_median = (T)2.1;
    param.expected_overall_upper_hinge = (T)4.925;
    param.expected_overall_lower_hinge = (T)-2.975;

    param.expected_status = da_status_success;

    param.epsilon = 50 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetShortFatData1(std::vector<OrderParamType<T>> &params) {
    // Test short wide data matrix
    OrderParamType<T> param;
    param.n = 8;
    param.p = 9;
    param.ldx = param.n;
    param.q = (T)0.7;
    param.quantile_type = da_quantile_type_6;
    std::vector<double> x{4.7,  2.6,  7.4,  9.5,  4.6,  5.1,  8,    2,   -4.7, 1.6,  8.4,
                          3.5,  -2.6, 5.0,  8.0,  0.0,  0.0,  -2.6, 5.4, 9.9,  2.6,  5.2,
                          -1.8, 1.2,  1.6,  9.4,  7.6,  -9.5, 1.6,  4.1, 8,    -2.1, -4.7,
                          -2.6, -7.4, 9.1,  4.2,  5.3,  7.2,  2.1,  1.7, 2.1,  -7.4, -9.1,
                          4.1,  3.1,  0.8,  1.2,  -4.7, 2.6,  -7.4, 6.5, -4.3, 5.0,  8.1,
                          -2.0, 4.1,  2.8,  -7.4, 3.5,  4.6,  -5.9, 8.2, -2,   4.1,  2.8,
                          -7.4, 3.5,  -4.1, -5.9, 8.3,  -2};
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles{4.1, 2.8, 7.4, 9.1, 4.2, 5.1, 8.1, 1.2};
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians{1.6, 2.6, -7.4, 3.5, 2.6, 5., 8., 0.};
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima{4.7, 9.4, 8.4, 9.9, 4.6, 5.3, 8.3, 2.1};
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima{-4.7, -2.6, -7.4, -9.5,
                                            -4.3, -5.9, -1.8, -2.1};
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges{4.1, 2.8,  7.5,  9.3,
                                                  4.4, 5.15, 8.15, 1.6};
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges{-4.7,  -0.5, -7.4, -2.8,
                                                  -3.35, -1.4, 4.,   -2.};
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    std::vector<double> expected_column_quantiles{7.58, 5.9,  5.26, 7.72, 5.87,
                                                  2.4,  5.45, 4.25, 3.68};
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians{4.9,  2.55, 1.9,  2.85, 3.15,
                                                1.45, 0.3,  3.15, 0.4};
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima{9.5, 8.4, 9.9, 9.4, 9.1,
                                               4.1, 8.1, 8.2, 8.3};
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima{2.,   -4.7, -2.6, -9.5, -7.4,
                                               -9.1, -7.4, -7.4, -7.4};
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges{7.85, 7.25,  5.35,  7.9, 6.725,
                                                     2.85, 6.125, 4.475, 3.95};
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges{3.1,   -1.95, -1.35,  -1.175, -4.175,
                                                     -5.35, -4.6,  -4.925, -5.45};
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_overall_quantile = (T)4.73;
    param.expected_overall_maximum = (T)9.9;
    param.expected_overall_minimum = (T)-9.5;
    param.expected_overall_median = (T)2.6;
    param.expected_overall_upper_hinge = (T)5.175;
    param.expected_overall_lower_hinge = (T)-2.075;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetShortFatData2(std::vector<OrderParamType<T>> &params) {
    // Test short wide data matrix
    OrderParamType<T> param;
    param.n = 8;
    param.p = 9;
    param.ldx = param.n;
    param.q = (T)0.7;
    param.quantile_type = da_quantile_type_2;
    std::vector<double> x{
        4.7,  2.6,  7.4,  9.5,   4.6,  5.1,  8,    2,    -4.7, 1.6,  8.4,  3.5,
        -2.6, 5.0,  8.0,  0.0,   0.0,  -2.6, 5.4,  9.9,  2.6,  5.2,  -1.8, 1.2,
        1.6,  9.4,  7.6,  -10.5, 1.6,  4.1,  8,    -2.1, -4.7, -2.6, -7.4, 9.1,
        4.2,  5.3,  7.2,  2.1,   1.7,  2.1,  -7.4, -9.1, 4.1,  3.1,  0.8,  1.2,
        -4.7, 2.6,  -7.4, 6.5,   -4.3, 5.0,  8.1,  -2.0, 4.1,  2.8,  -7.4, 3.5,
        4.6,  -5.9, 8.2,  -2,    4.1,  2.8,  -7.4, 3.5,  -4.1, -5.9, 8.3,  -2};
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles{4.1, 2.8, 7.4, 9.1, 4.2, 5.1, 8.1, 1.2};
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians{1.6, 2.6, -7.4, 3.5, 2.6, 5., 8., 0.};
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima{4.7, 9.4, 8.4, 9.9, 4.6, 5.3, 8.3, 2.1};
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima{-4.7, -2.6, -7.4, -10.5,
                                            -4.3, -5.9, -1.8, -2.1};
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges{4.1, 2.8,  7.5,  9.3,
                                                  4.4, 5.15, 8.15, 1.6};
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges{-4.7,  -0.5, -7.4, -2.8,
                                                  -3.35, -1.4, 4.,   -2.};
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    std::vector<double> expected_column_quantiles{7.4, 5., 5.2, 7.6, 5.3,
                                                  2.1, 5., 4.1, 3.5};
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians{4.9,  2.55, 1.9,  2.85, 3.15,
                                                1.45, 0.3,  3.15, 0.4};
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima{9.5, 8.4, 9.9, 9.4, 9.1,
                                               4.1, 8.1, 8.2, 8.3};
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima{2.,   -4.7, -2.6, -10.5, -7.4,
                                               -9.1, -7.4, -7.4, -7.4};
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges{7.85, 7.25,  5.35,  7.9, 6.725,
                                                     2.85, 6.125, 4.475, 3.95};
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges{3.1,   -1.95, -1.35,  -1.175, -4.175,
                                                     -5.35, -4.6,  -4.925, -5.45};
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_overall_quantile = (T)4.7;
    param.expected_overall_maximum = (T)9.9;
    param.expected_overall_minimum = (T)-10.5;
    param.expected_overall_median = (T)2.6;
    param.expected_overall_upper_hinge = (T)5.175;
    param.expected_overall_lower_hinge = (T)-2.075;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetShortFatData3(std::vector<OrderParamType<T>> &params) {
    // Test short wide data matrix
    OrderParamType<T> param;
    param.n = 8;
    param.p = 9;
    param.ldx = param.n;
    param.q = (T)0.7;
    param.quantile_type = da_quantile_type_1;
    std::vector<double> x{4.7,  2.6,  7.4,  9.5,  4.6,  5.1,  8,    2,   -4.7, 1.6,  8.4,
                          3.5,  -2.6, 5.0,  8.0,  0.0,  0.0,  -2.6, 5.4, 11.9, 2.6,  5.2,
                          -1.8, 1.2,  1.6,  9.4,  7.6,  -9.5, 1.6,  4.1, 8,    -2.1, -4.7,
                          -2.6, -7.4, 9.1,  4.2,  5.3,  7.2,  2.1,  1.7, 2.1,  -7.4, -9.1,
                          4.1,  3.1,  0.8,  1.2,  -4.7, 2.6,  -7.4, 6.5, -4.3, 5.0,  8.1,
                          -2.0, 4.1,  2.8,  -7.4, 3.5,  4.6,  -5.9, 8.2, -2,   4.1,  2.8,
                          -7.4, 3.5,  -4.1, -5.9, 8.3,  -2};
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles{4.1, 2.8, 7.4, 9.1, 4.2, 5.1, 8.1, 1.2};
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians{1.6, 2.6, -7.4, 3.5, 2.6, 5., 8., 0.};
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima{4.7, 9.4, 8.4, 11.9, 4.6, 5.3, 8.3, 2.1};
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima{-4.7, -2.6, -7.4, -9.5,
                                            -4.3, -5.9, -1.8, -2.1};
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges{4.1, 2.8,  7.5,  9.3,
                                                  4.4, 5.15, 8.15, 1.6};
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges{-4.7,  -0.5, -7.4, -2.8,
                                                  -3.35, -1.4, 4.,   -2.};
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    std::vector<double> expected_column_quantiles{7.4, 5., 5.2, 7.6, 5.3,
                                                  2.1, 5., 4.1, 3.5};
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians{4.9,  2.55, 1.9,  2.85, 3.15,
                                                1.45, 0.3,  3.15, 0.4};
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima{9.5, 8.4, 11.9, 9.4, 9.1,
                                               4.1, 8.1, 8.2,  8.3};
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima{2.,   -4.7, -2.6, -9.5, -7.4,
                                               -9.1, -7.4, -7.4, -7.4};
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges{7.85, 7.25,  5.35,  7.9, 6.725,
                                                     2.85, 6.125, 4.475, 3.95};
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges{3.1,   -1.95, -1.35,  -1.175, -4.175,
                                                     -5.35, -4.6,  -4.925, -5.45};
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_overall_quantile = (T)4.7;
    param.expected_overall_maximum = (T)11.9;
    param.expected_overall_minimum = (T)-9.5;
    param.expected_overall_median = (T)2.6;
    param.expected_overall_upper_hinge = (T)5.175;
    param.expected_overall_lower_hinge = (T)-2.075;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetSubarrayData(std::vector<OrderParamType<T>> &params) {
    // Test matrix stored in a subarray
    OrderParamType<T> param;
    param.n = 8;
    param.p = 9;
    param.ldx = param.n + 3;
    param.q = (T)0.6;
    param.quantile_type = da_quantile_type_9;
    std::vector<double> x{1.7,  2.6,  7.4,  9.5,  4.6,  5.1,  8,    2,    0, 0, 0,
                          -4.7, 1.6,  8.4,  3.5,  -2.6, 5.0,  8.0,  0.0,  0, 0, 0,
                          0.0,  -2.6, 4.4,  9.9,  2.6,  5.2,  -1.8, 1.2,  0, 0, 0,
                          -1.9, 9.4,  7.6,  -9.5, 1.6,  4.1,  8,    2.1,  0, 0, 0,
                          -4.7, -2.6, -7.4, 9.1,  4.2,  5.3,  7.2,  2.1,  0, 0, 0,
                          1.7,  2.1,  -7.4, -9.1, 4.1,  3.1,  0.8,  6.2,  0, 0, 0,
                          -4.7, 2.6,  -7.4, 6.5,  -4.3, 5.0,  8.1,  -2.0, 0, 0, 0,
                          4.1,  9.8,  -7.4, 3.5,  4.6,  -5.9, 8.2,  -4,   0, 0, 0,
                          4.1,  2.8,  -7.4, 3.5,  -4.1, -5.9, 8.4,  -2,   0, 0, 0};
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles{1.5725, 2.6, 3.515, 6.275,
                                               3.9875, 5.,  8.,    1.94};
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians{0., 2.6, -7.4, 3.5, 2.6, 5., 8., 1.2};
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima{4.1, 9.8, 8.4, 9.9, 4.6, 5.3, 8.4, 6.2};
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima{-4.7, -2.6, -7.4, -9.5,
                                            -4.3, -5.9, -1.8, -4.};
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges{2.9, 6.1,  7.5,  9.3,
                                                  4.4, 5.15, 8.15, 2.1};
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges{-4.7,  -0.5, -7.4, -2.8,
                                                  -3.35, -1.4, 4.,   -2.};
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    std::vector<double> expected_column_quantiles{5.8475, 3.9875, 3.185,  5.2375, 4.5575,
                                                  2.425,  3.38,   4.2625, 3.0275}; //
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians{4.85, 2.55, 1.9, 3.1, 3.15,
                                                1.9,  0.3,  3.8, 0.4};
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima{9.5, 8.4, 9.9, 9.4, 9.1,
                                               6.2, 8.1, 9.8, 8.4};
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima{1.7,  -4.7, -2.6, -9.5, -7.4,
                                               -9.1, -7.4, -7.4, -7.4};
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges{7.85, 7.25,  5.,  7.9, 6.725,
                                                     3.85, 6.125, 7.3, 3.95};
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges{2.15,  -1.95, -1.35,  -1.025, -4.175,
                                                     -5.35, -4.6,  -5.425, -5.45};
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_overall_quantile = (T)4.1;
    param.expected_overall_maximum = (T)9.9;
    param.expected_overall_minimum = (T)-9.5;
    param.expected_overall_median = (T)2.6;
    param.expected_overall_upper_hinge = (T)5.275;
    param.expected_overall_lower_hinge = (T)-2.45;

    param.expected_status = da_status_success;

    param.epsilon = 50 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetTallThinData1(std::vector<OrderParamType<T>> &params) {
    // Test with tall thin data matrix
    OrderParamType<T> param;
    param.n = 18;
    param.p = 4;
    param.ldx = param.n;
    param.q = (T)0.2;
    param.quantile_type = da_quantile_type_5;
    std::vector<double> x{4.7,  2.6,  7.4,  9.5,  4.6,  5.1,  8,    2,   -4.7, 1.6,  8.4,
                          3.5,  -2.6, 5.0,  8.0,  0.0,  0.0,  -2.6, 5.4, 9.9,  2.6,  5.2,
                          -1.8, 1.2,  1.6,  9.4,  7.6,  -9.5, 1.6,  4.1, 8,    -2.1, -4.7,
                          -2.6, -7.4, 9.1,  4.2,  5.3,  7.2,  2.1,  1.7, 2.1,  -7.4, -9.1,
                          4.1,  3.1,  0.8,  1.2,  -4.7, 2.6,  -7.4, 6.5, -4.3, 5.0,  8.1,
                          -2.0, 4.1,  2.8,  -7.4, 3.5,  4.6,  -5.9, 8.2, -2,   4.1,  2.8,
                          -7.4, 3.5,  -4.1, -5.9, 8.3,  -2};
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles{
        4.35,  -0.62, 3.05, 2.31,  -5.72, 1.47,  -4.7,  -8.14, -2.06,
        -7.25, 1.04,  1.68, -6.59, -0.69, -6.59, -4.91, -6.47, -2.42}; //
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians{5.05,  3.95,  5.65, 4.,   -0.05, 2.8,
                                             3.1,   -1.95, 5.85, -0.2, 2.85,  3.15,
                                             -3.65, 3.05,  -4.4, -1.3, -2.15, 1.5};
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima{8.1, 9.9, 7.4, 9.5, 4.6, 5.1, 8.,  9.4, 8.2,
                                            3.1, 8.4, 4.1, 8.,  5.,  8.,  6.5, 8.3, 9.1};
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima{4.2,  -2.,  2.6,  2.1,  -7.4, 1.2,
                                            -7.4, -9.1, -4.7, -9.5, 0.8,  1.2,
                                            -7.4, -2.1, -7.4, -5.9, -7.4, -2.6};
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges{
        7.425, 8.75,  7.35, 8.425, 3.875, 4.7,   7.15,  7.55,  8.05,
        2.725, 7.325, 3.95, 5.35,  4.625, 4.975, 4.875, 6.225, 8.075};
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges{
        4.325,  -0.85, 2.975, 2.275,  -6.,    1.425,  -5.15,  -8.3,   -2.5,
        -7.625, 1.,    1.6,   -6.725, -0.925, -6.725, -5.075, -6.625, -2.45};
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    std::vector<double> expected_column_quantiles{0., -2.55, -4.66, -5.72}; //
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians{4.05, 2.1, 2.1, 2.8};
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima{9.5, 9.9, 7.2, 8.3};
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima{-4.7, -9.5, -9.1, -7.4};
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges{7.55, 7.7, 4.4, 4.225};
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges{0., -2.225, -4.4, -4.55};
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_overall_quantile = (T)-2.75;
    param.expected_overall_maximum = (T)(T)9.9;
    param.expected_overall_minimum = (T)-9.5;
    param.expected_overall_median = (T)2.6;
    param.expected_overall_upper_hinge = (T)5.175;
    param.expected_overall_lower_hinge = (T)-2.075;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetTallThinData2(std::vector<OrderParamType<T>> &params) {
    // Test with tall thin data matrix
    OrderParamType<T> param;
    param.n = 18;
    param.p = 4;
    param.ldx = param.n;
    param.q = (T)0.2;
    param.quantile_type = da_quantile_type_3;
    std::vector<double> x{
        4.7,  2.6,  7.4,  9.5,   4.6,  5.1,  8,    2,     -4.7, 1.6,  8.4,  3.5,
        -2.6, 5.0,  8.0,  0.0,   0.0,  -2.6, 5.4,  9.9,   2.6,  5.2,  -1.8, 1.2,
        1.6,  9.4,  7.6,  -10.5, 1.6,  4.1,  8,    -2.1,  -4.7, -2.6, -7.4, 9.1,
        4.2,  5.3,  7.2,  2.1,   1.7,  2.1,  -7.4, -29.1, 4.1,  3.1,  0.8,  1.2,
        -4.7, 2.6,  -7.4, 6.5,   -4.3, 5.0,  8.1,  -2.0,  4.1,  2.8,  -7.4, 3.5,
        4.6,  -5.9, 8.2,  -2,    4.1,  2.8,  -7.4, 3.5,   -4.1, -5.9, 8.3,  -2};
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles{4.2,  -2.,   2.6,  2.1,   -7.4, 1.2,
                                               -7.4, -29.1, -4.7, -10.5, 0.8,  1.2,
                                               -7.4, -2.1,  -7.4, -5.9,  -7.4, -2.6};
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians{5.05,  3.95,  5.65, 4.,   -0.05, 2.8,
                                             3.1,   -1.95, 5.85, -0.2, 2.85,  3.15,
                                             -3.65, 3.05,  -4.4, -1.3, -2.15, 1.5};
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima{8.1, 9.9, 7.4, 9.5, 4.6, 5.1, 8.,  9.4, 8.2,
                                            3.1, 8.4, 4.1, 8.,  5.,  8.,  6.5, 8.3, 9.1};
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima{4.2,  -2.,   2.6,  2.1,   -7.4, 1.2,
                                            -7.4, -29.1, -4.7, -10.5, 0.8,  1.2,
                                            -7.4, -2.1,  -7.4, -5.9,  -7.4, -2.6};
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges{
        7.425, 8.75,  7.35, 8.425, 3.875, 4.7,   7.15,  7.55,  8.05,
        2.725, 7.325, 3.95, 5.35,  4.625, 4.975, 4.875, 6.225, 8.075};
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges{
        4.325,  -0.85, 2.975, 2.275,  -6.,    1.425,  -5.15,  -23.3,  -2.5,
        -8.375, 1.,    1.6,   -6.725, -0.925, -6.725, -5.075, -6.625, -2.45};
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    std::vector<double> expected_column_quantiles{-2.6, -4.7, -7.4, -5.9}; //
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians{4.05, 2.1, 2.1, 2.8};
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima{9.5, 9.9, 7.2, 8.3};
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima{-4.7, -10.5, -29.1, -7.4};
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges{7.55, 7.7, 4.4, 4.225};
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges{0., -2.225, -4.4, -4.55};
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_overall_quantile = (T)-4.1;
    param.expected_overall_maximum = (T)(T)9.9;
    param.expected_overall_minimum = (T)-29.1;
    param.expected_overall_median = (T)2.6;
    param.expected_overall_upper_hinge = (T)5.175;
    param.expected_overall_lower_hinge = (T)-2.075;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void GetTallThinData3(std::vector<OrderParamType<T>> &params) {
    // Test with tall thin data matrix
    OrderParamType<T> param;
    param.n = 18;
    param.p = 4;
    param.ldx = param.n;
    param.q = (T)0.2;
    param.quantile_type = da_quantile_type_7;
    std::vector<double> x{
        4.7,  2.6,  7.4,  9.5,   4.6,  5.1,  8,    2,    -4.7, 1.6,  8.4,  3.5,
        -2.6, 5.0,  8.0,  0.0,   0.0,  -2.6, 5.4,  10.9, 2.6,  5.2,  -1.8, 1.2,
        1.6,  9.4,  7.6,  -11.5, 1.6,  4.1,  8,    -2.1, -4.7, -2.6, -7.4, 9.1,
        4.2,  5.3,  7.2,  2.1,   1.7,  2.1,  -7.4, -9.1, 4.1,  3.1,  0.8,  1.2,
        -4.7, 2.6,  -7.4, 6.5,   -4.3, 5.0,  8.1,  -2.0, 4.1,  2.8,  -7.4, 3.5,
        4.6,  -5.9, 8.2,  -2,    4.1,  2.8,  -7.4, 3.5,  -4.1, -5.9, 8.3,  -2};
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles{4.5,   0.76,  3.5,   2.52,  -4.04, 1.74,
                                               -2.0,  -7.18, 0.58,  -5.8,  1.28,  2.16,
                                               -5.78, 0.72,  -5.78, -3.92, -5.54, -2.24};
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians{5.05,  3.95,  5.65, 4.,   -0.05, 2.8,
                                             3.1,   -1.95, 5.85, -0.2, 2.85,  3.15,
                                             -3.65, 3.05,  -4.4, -1.3, -2.15, 1.5};
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima{8.1, 10.9, 7.4, 9.5, 4.6, 5.1, 8.,  9.4, 8.2,
                                            3.1, 8.4,  4.1, 8.,  5.,  8.,  6.5, 8.3, 9.1};
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima{4.2,  -2.,  2.6,  2.1,   -7.4, 1.2,
                                            -7.4, -9.1, -4.7, -11.5, 0.8,  1.2,
                                            -7.4, -2.1, -7.4, -5.9,  -7.4, -2.6};
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges{
        7.425, 9.5,   7.35, 8.425, 3.875, 4.7,   7.15,  7.55,  8.05,
        2.725, 7.325, 3.95, 5.35,  4.625, 4.975, 4.875, 6.225, 8.075};
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges{
        4.325,  -0.85, 2.975, 2.275,  -6.,    1.425,  -5.15,  -8.3,   -2.5,
        -9.125, 1.,    1.6,   -6.725, -0.925, -6.725, -5.075, -6.625, -2.45};
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    std::vector<double> expected_column_quantiles{0., -2.4, -4.54, -5.18}; //
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians{4.05, 2.1, 2.1, 2.8};
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima{9.5, 10.9, 7.2, 8.3};
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima{-4.7, -11.5, -9.1, -7.4};
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges{7.55, 7.7, 4.4, 4.225};
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges{0., -2.225, -4.4, -4.55};
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_overall_quantile = (T)-2.6;
    param.expected_overall_maximum = (T)(T)10.9;
    param.expected_overall_minimum = (T)-11.5;
    param.expected_overall_median = (T)2.6;
    param.expected_overall_upper_hinge = (T)5.175;
    param.expected_overall_lower_hinge = (T)-2.075;

    param.expected_status = da_status_success;

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    params.push_back(param);
}

template <typename T> void Get1by1Data(std::vector<OrderParamType<T>> &params) {
    // Test with 1 x 1 data matrix
    OrderParamType<T> param;
    param.n = 1;
    param.p = 1;
    param.ldx = param.n;
    param.q = (T)0.3;
    param.quantile_type = da_quantile_type_4;
    std::vector<double> x(param.n * param.p, 3);
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles(param.n, 3);
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians(param.n, 3);
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima(param.n, 3);
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima(param.n, 3);
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges(param.n, 3);
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges(param.n, 3);
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    std::vector<double> expected_column_quantiles(param.n, 3);
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians(param.n, 3);
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima(param.n, 3);
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima(param.n, 3);
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges(param.n, 3);
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges(param.n, 3);
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_overall_quantile = (T)3.0;
    param.expected_overall_maximum = (T)3.0;
    param.expected_overall_minimum = (T)3.0;
    param.expected_overall_median = (T)3.0;
    param.expected_overall_upper_hinge = (T)3.0;
    param.expected_overall_lower_hinge = (T)3.0;

    param.expected_status = da_status_success;

    param.epsilon = 0;

    params.push_back(param);
}

template <typename T> void GetZeroData(std::vector<OrderParamType<T>> &params) {
    // Test with data matrix full of zeros
    OrderParamType<T> param;
    param.n = 6;
    param.p = 8;
    param.ldx = param.n;
    param.q = (T)0.8;
    param.quantile_type = da_quantile_type_6;
    std::vector<double> x(param.n * param.p, 0);
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles(param.n, 0);
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians(param.n, 0);
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima(param.n, 0);
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima(param.n, 0);
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges(param.n, 0);
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges(param.n, 0);
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    std::vector<double> expected_column_quantiles(param.p, 0);
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians(param.p, 0);
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima(param.p, 0);
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima(param.p, 0);
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges(param.p, 0);
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges(param.p, 0);
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_overall_quantile = (T)0;
    param.expected_overall_maximum = (T)0;
    param.expected_overall_minimum = (T)0;
    param.expected_overall_median = (T)0;
    param.expected_overall_upper_hinge = (T)0;
    param.expected_overall_lower_hinge = (T)0;

    param.expected_status = da_status_success;

    param.epsilon = std::numeric_limits<T>::epsilon();
    ;

    params.push_back(param);
}

template <typename T> void GetIdenticalData(std::vector<OrderParamType<T>> &params) {
    // Test with data matrix full of ones
    OrderParamType<T> param;
    param.n = 12;
    param.p = 10;
    param.ldx = param.n;
    param.q = (T)0.6;
    param.quantile_type = da_quantile_type_4;
    std::vector<double> x(param.n * param.p, 1.0);
    param.x = convert_vector<double, T>(x);
    std::vector<double> expected_row_quantiles(param.n, 1.0);
    param.expected_row_quantiles = convert_vector<double, T>(expected_row_quantiles);
    std::vector<double> expected_row_medians(param.n, 1.0);
    param.expected_row_medians = convert_vector<double, T>(expected_row_medians);
    std::vector<double> expected_row_maxima(param.n, 1.0);
    param.expected_row_maxima = convert_vector<double, T>(expected_row_maxima);
    std::vector<double> expected_row_minima(param.n, 1.0);
    param.expected_row_minima = convert_vector<double, T>(expected_row_minima);
    std::vector<double> expected_row_upper_hinges(param.n, 1.0);
    param.expected_row_upper_hinges =
        convert_vector<double, T>(expected_row_upper_hinges);
    std::vector<double> expected_row_lower_hinges(param.n, 1.0);
    param.expected_row_lower_hinges =
        convert_vector<double, T>(expected_row_lower_hinges);

    std::vector<double> expected_column_quantiles(param.p, 1.0);
    param.expected_column_quantiles =
        convert_vector<double, T>(expected_column_quantiles);
    std::vector<double> expected_column_medians(param.p, 1.0);
    param.expected_column_medians = convert_vector<double, T>(expected_column_medians);
    std::vector<double> expected_column_maxima(param.p, 1.0);
    param.expected_column_maxima = convert_vector<double, T>(expected_column_maxima);
    std::vector<double> expected_column_minima(param.p, 1.0);
    param.expected_column_minima = convert_vector<double, T>(expected_column_minima);
    std::vector<double> expected_column_upper_hinges(param.p, 1.0);
    param.expected_column_upper_hinges =
        convert_vector<double, T>(expected_column_upper_hinges);
    std::vector<double> expected_column_lower_hinges(param.p, 1.0);
    param.expected_column_lower_hinges =
        convert_vector<double, T>(expected_column_lower_hinges);

    param.expected_overall_quantile = (T)1.0;
    param.expected_overall_maximum = (T)1.0;
    param.expected_overall_minimum = (T)1.0;
    param.expected_overall_median = (T)1.0;
    param.expected_overall_upper_hinge = (T)1.0;
    param.expected_overall_lower_hinge = (T)1.0;

    param.expected_status = da_status_success;

    param.epsilon = std::numeric_limits<T>::epsilon();
    ;

    params.push_back(param);
}

template <typename T> void GetOrderData(std::vector<OrderParamType<T>> &params) {

    GetZeroData(params);
    GetIdenticalData(params);
    GetTallThinData1(params);
    GetTallThinData2(params);
    GetTallThinData3(params);
    GetShortFatData1(params);
    GetShortFatData2(params);
    GetShortFatData3(params);
    GetSubarrayData(params);
    GetSingleRowData(params);
    GetSingleColumnData(params);
    Get1by1Data(params);
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(OrderStatisticsTest, FloatTypes);

TYPED_TEST(OrderStatisticsTest, OrderFunctionality) {

    std::vector<OrderParamType<TypeParam>> params;
    GetOrderData(params);

    for (auto &param : params) {
        std::vector<TypeParam> column_quantiles(param.p);
        std::vector<TypeParam> row_quantiles(param.n);
        TypeParam overall_quantile[1];
        std::vector<TypeParam> column_medians(param.p);
        std::vector<TypeParam> row_medians(param.n);
        TypeParam overall_median[1];
        std::vector<TypeParam> column_maxima(param.p);
        std::vector<TypeParam> row_maxima(param.n);
        TypeParam overall_maximum[1];
        std::vector<TypeParam> column_minima(param.p);
        std::vector<TypeParam> row_minima(param.n);
        TypeParam overall_minimum[1];
        std::vector<TypeParam> column_lower_hinges(param.p);
        std::vector<TypeParam> row_lower_hinges(param.n);
        TypeParam overall_lower_hinge[1];
        std::vector<TypeParam> column_upper_hinges(param.p);
        std::vector<TypeParam> row_upper_hinges(param.n);
        TypeParam overall_upper_hinge[1];

        EXPECT_EQ(da_quantile(da_axis_col, param.n, param.p, param.x.data(), param.ldx,
                              param.q, column_quantiles.data(), param.quantile_type),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.p, param.expected_column_quantiles.data(),
                        column_quantiles.data(), param.epsilon);
        EXPECT_EQ(da_quantile(da_axis_row, param.n, param.p, param.x.data(), param.ldx,
                              param.q, row_quantiles.data(), param.quantile_type),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.n, param.expected_row_quantiles.data(),
                        row_quantiles.data(), param.epsilon);
        EXPECT_EQ(da_quantile(da_axis_all, param.n, param.p, param.x.data(), param.ldx,
                              param.q, overall_quantile, param.quantile_type),
                  param.expected_status);
        EXPECT_NEAR(param.expected_overall_quantile, overall_quantile[0], param.epsilon);

        EXPECT_EQ(da_five_point_summary(da_axis_col, param.n, param.p, param.x.data(),
                                        param.ldx, column_minima.data(),
                                        column_lower_hinges.data(), column_medians.data(),
                                        column_upper_hinges.data(), column_maxima.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.p, param.expected_column_minima.data(),
                        column_minima.data(), param.epsilon);
        EXPECT_ARR_NEAR(param.p, param.expected_column_maxima.data(),
                        column_maxima.data(), param.epsilon);
        EXPECT_ARR_NEAR(param.p, param.expected_column_lower_hinges.data(),
                        column_lower_hinges.data(), param.epsilon);
        EXPECT_ARR_NEAR(param.p, param.expected_column_medians.data(),
                        column_medians.data(), param.epsilon);
        EXPECT_ARR_NEAR(param.p, param.expected_column_upper_hinges.data(),
                        column_upper_hinges.data(), param.epsilon);

        EXPECT_EQ(da_five_point_summary(da_axis_row, param.n, param.p, param.x.data(),
                                        param.ldx, row_minima.data(),
                                        row_lower_hinges.data(), row_medians.data(),
                                        row_upper_hinges.data(), row_maxima.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.n, param.expected_row_minima.data(), row_minima.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.n, param.expected_row_maxima.data(), row_maxima.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.n, param.expected_row_lower_hinges.data(),
                        row_lower_hinges.data(), param.epsilon);
        EXPECT_ARR_NEAR(param.n, param.expected_row_medians.data(), row_medians.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.n, param.expected_row_upper_hinges.data(),
                        row_upper_hinges.data(), param.epsilon);

        EXPECT_EQ(da_five_point_summary(da_axis_all, param.n, param.p, param.x.data(),
                                        param.ldx, overall_minimum, overall_lower_hinge,
                                        overall_median, overall_upper_hinge,
                                        overall_maximum),
                  param.expected_status);
        EXPECT_NEAR(param.expected_overall_minimum, overall_minimum[0], param.epsilon);
        EXPECT_NEAR(param.expected_overall_maximum, overall_maximum[0], param.epsilon);
        EXPECT_NEAR(param.expected_overall_median, overall_median[0], param.epsilon);
        EXPECT_NEAR(param.expected_overall_lower_hinge, overall_lower_hinge[0],
                    param.epsilon);
        EXPECT_NEAR(param.expected_overall_upper_hinge, overall_upper_hinge[0],
                    param.epsilon);
    }
}

TYPED_TEST(OrderStatisticsTest, IllegalArgsOrderStatistics) {

    std::vector<double> x_d{4.7, 1.2, -0.3, 4.5};
    std::vector<TypeParam> x = convert_vector<double, TypeParam>(x_d);
    da_int n = 2, p = 2, ldx = 2;
    TypeParam q = (TypeParam)0.5;
    std::vector<double> dummy1_d(10, 0);
    std::vector<double> dummy2_d(10, 0);
    std::vector<double> dummy3_d(10, 0);
    std::vector<double> dummy4_d(10, 0);
    std::vector<double> dummy5_d(10, 0);

    std::vector<TypeParam> dummy1 = convert_vector<double, TypeParam>(dummy1_d);
    std::vector<TypeParam> dummy2 = convert_vector<double, TypeParam>(dummy2_d);
    std::vector<TypeParam> dummy3 = convert_vector<double, TypeParam>(dummy3_d);
    std::vector<TypeParam> dummy4 = convert_vector<double, TypeParam>(dummy4_d);
    std::vector<TypeParam> dummy5 = convert_vector<double, TypeParam>(dummy5_d);

    // Test with illegal value of ldx
    da_int ldx_illegal = 1;
    EXPECT_EQ(da_quantile(da_axis_all, n, p, x.data(), ldx_illegal, q, dummy1.data(),
                          da_quantile_type_1),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_five_point_summary(da_axis_all, n, p, x.data(), ldx_illegal,
                                    dummy1.data(), dummy2.data(), dummy3.data(),
                                    dummy4.data(), dummy5.data()),
              da_status_invalid_leading_dimension);

    // Test with illegal p
    da_int p_illegal = 0;
    EXPECT_EQ(da_quantile(da_axis_all, n, p_illegal, x.data(), ldx, q, dummy1.data(),
                          da_quantile_type_1),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_five_point_summary(da_axis_all, n, p_illegal, x.data(), ldx,
                                    dummy1.data(), dummy2.data(), dummy3.data(),
                                    dummy4.data(), dummy5.data()),
              da_status_invalid_array_dimension);

    // Test with illegal n
    da_int n_illegal = 0;
    EXPECT_EQ(da_quantile(da_axis_all, n_illegal, p, x.data(), ldx, q, dummy1.data(),
                          da_quantile_type_1),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_five_point_summary(da_axis_all, n_illegal, p, x.data(), ldx,
                                    dummy1.data(), dummy2.data(), dummy3.data(),
                                    dummy4.data(), dummy5.data()),
              da_status_invalid_array_dimension);

    // Test illegal q
    TypeParam q_illegal = (TypeParam)-0.1;
    EXPECT_EQ(da_quantile(da_axis_all, n, p, x.data(), ldx, q_illegal, dummy1.data(),
                          da_quantile_type_1),
              da_status_invalid_input);

    // Test illegal pointers
    TypeParam *x_null = nullptr;
    EXPECT_EQ(
        da_quantile(da_axis_all, n, p, x_null, ldx, q, dummy1.data(), da_quantile_type_1),
        da_status_invalid_pointer);
    EXPECT_EQ(da_five_point_summary(da_axis_all, n, p, x_null, ldx, dummy1.data(),
                                    dummy2.data(), dummy3.data(), dummy4.data(),
                                    dummy5.data()),
              da_status_invalid_pointer);
}