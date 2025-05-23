/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <limits>
#include <list>
#include <string>

namespace {

template <typename T> class UtilitiesTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UtilitiesTest, FloatTypes);

TYPED_TEST(UtilitiesTest, CheckData) {

    std::vector<double> x_d{4.7,  1.2, std::numeric_limits<double>::quiet_NaN(),
                            -0.3, 4.5, 0.0};
    std::vector<TypeParam> x = convert_vector<double, TypeParam>(x_d);
    da_int n_cols = 2, n_rows = 2, ldx = 2;

    // Test with illegal value of ldx
    da_int ldx_illegal = 1;
    EXPECT_EQ(da_check_data(column_major, n_rows, n_cols, x.data(), ldx_illegal),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_check_data(row_major, n_rows, n_cols, x.data(), ldx_illegal),
              da_status_invalid_leading_dimension);

    // Test with illegal array dimension
    da_int n_cols_illegal = 0;
    EXPECT_EQ(da_check_data(column_major, n_rows, n_cols_illegal, x.data(), ldx),
              da_status_invalid_array_dimension);

    // Test illegal pointers
    TypeParam *x_null = nullptr;
    EXPECT_EQ(da_check_data(column_major, n_rows, n_cols, x_null, ldx),
              da_status_invalid_pointer);

    // Test functionality
    EXPECT_EQ(da_check_data(column_major, n_rows, n_cols, x.data(), ldx),
              da_status_invalid_input);
    EXPECT_EQ(da_check_data(row_major, n_rows, n_cols, x.data(), ldx),
              da_status_invalid_input);
    ldx = 3;
    EXPECT_EQ(da_check_data(column_major, n_rows, n_cols, x.data(), ldx),
              da_status_success);
    EXPECT_EQ(da_check_data(row_major, n_rows, n_cols, x.data(), ldx), da_status_success);
}

TYPED_TEST(UtilitiesTest, SwitchOrder) {
    std::vector<double> x_col_d{1.0, 4.0, 7.0, 10.0, 0.0, 2.0, 5.0, 8.0, 11.0, 0.0,
                                3.0, 6.0, 9.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0};
    std::vector<double> x_row_d{1.0, 2.0, 3.0,  0.0,  4.0,  5.0, 6.0, 0.0, 7.0, 8.0,
                                9.0, 0.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> y_d{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<TypeParam> x_col = convert_vector<double, TypeParam>(x_col_d);
    std::vector<TypeParam> x_row = convert_vector<double, TypeParam>(x_row_d);
    std::vector<TypeParam> y = convert_vector<double, TypeParam>(y_d);
    std::vector<double> x_row_copy_d{1.0, 2.0, 3.0,  10.0, 4.0,  5.0, 6.0, 8.0, 7.0, 8.0,
                                     9.0, 6.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<TypeParam> x_row_copy = convert_vector<double, TypeParam>(x_row_copy_d);
    std::vector<double> x_col_copy_d{1.0, 4.0, 7.0, 10.0, 4.0,  2.0, 5.0, 8.0, 11.0, 8.0,
                                     3.0, 6.0, 9.0, 12.0, 12.0, 0.0, 0.0, 0.0, 0.0,  0.0};
    std::vector<TypeParam> x_col_copy = convert_vector<double, TypeParam>(x_col_copy_d);

    da_int n_cols = 3, n_rows = 4, ldx_col = 5, ldx_row = 4, ldy_col = 5, ldy_row = 4,
           total_size = 20;

    // Test with illegal value of ldx
    da_int ldx_illegal = 1;
    EXPECT_EQ(da_switch_order_copy(column_major, n_rows, n_cols, x_col.data(),
                                   ldx_illegal, y.data(), ldy_row),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_switch_order_copy(row_major, n_rows, n_cols, x_row.data(), ldx_illegal,
                                   y.data(), ldy_col),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_switch_order_in_place(column_major, n_rows, n_cols, x_col.data(),
                                       ldx_illegal, ldx_row),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_switch_order_in_place(row_major, n_rows, n_cols, x_col.data(),
                                       ldx_illegal, ldx_col),
              da_status_invalid_leading_dimension);

    // Test with illegal pointers
    TypeParam *x_null = nullptr;
    EXPECT_EQ(da_switch_order_copy(column_major, n_rows, n_cols, x_null, ldx_col,
                                   y.data(), ldy_row),
              da_status_invalid_pointer);
    EXPECT_EQ(
        da_switch_order_in_place(row_major, n_rows, n_cols, x_null, ldx_row, ldx_col),
        da_status_invalid_pointer);

    // Test with illegal array dimensions
    da_int n_rows_illegal = 0;
    EXPECT_EQ(da_switch_order_copy(column_major, n_rows_illegal, n_cols, x_col.data(),
                                   ldx_col, y.data(), ldy_row),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_switch_order_in_place(row_major, n_rows_illegal, n_cols, x_row.data(),
                                       ldx_row, ldx_col),
              da_status_invalid_array_dimension);

    // Functionality tests
    EXPECT_EQ(da_switch_order_copy(column_major, n_rows, n_cols, x_col.data(), ldx_col,
                                   y.data(), ldy_row),
              da_status_success);
    EXPECT_ARR_NEAR(total_size, y.data(), x_row.data(),
                    10 * std::numeric_limits<TypeParam>::epsilon());
    y = convert_vector<double, TypeParam>(y_d);
    EXPECT_EQ(da_switch_order_copy(row_major, n_rows, n_cols, x_row.data(), ldx_row,
                                   y.data(), ldy_col),
              da_status_success);
    EXPECT_ARR_NEAR(total_size, y.data(), x_col.data(),
                    10 * std::numeric_limits<TypeParam>::epsilon());
    EXPECT_EQ(da_switch_order_in_place(column_major, n_rows, n_cols, x_col.data(),
                                       ldx_col, ldx_row),
              da_status_success);
    EXPECT_ARR_NEAR(total_size, x_col.data(), x_row_copy.data(),
                    10 * std::numeric_limits<TypeParam>::epsilon());
    EXPECT_EQ(da_switch_order_in_place(row_major, n_rows, n_cols, x_row.data(), ldx_row,
                                       ldx_col),
              da_status_success);
    EXPECT_ARR_NEAR(total_size, x_row.data(), x_col_copy.data(),
                    10 * std::numeric_limits<TypeParam>::epsilon());
}

} // namespace