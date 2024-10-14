/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AUTO_DETECT_CSV_HPP
#define AUTO_DETECT_CSV_HPP

#include "aoclda.h"
#include "read_csv.hpp"
#include <iostream>
#include <tuple>
#include <variant>

namespace da_auto_detect {
/* We convert CSV char* data to a vector of variants, one variant for each column.
   Each variant is a vector of da_int/float/double/uint8_t, with char** to catch data that can't be handled any other way */
using CSVColumnsType =
    std::vector<std::variant<std::vector<da_int>, std::vector<float>, std::vector<double>,
                             std::vector<uint8_t>, std::vector<char **>>>;

/* Store the individual scalar elements returned from parsing the character data into a numeric type */
using CSVElementType = std::variant<da_int, float, double, uint8_t>;

/* Convert column i of the CSV data to char, when we have already read the first j elements */
inline void convert_col_to_char(CSVColumnsType &columns, da_int i, da_int j, char **data,
                                da_int nrows, da_int ncols, da_order order) {
    std::vector<char **> char_col;

    for (da_int k = 0; k <= j; k++) {
        switch (order) {
        case row_major:
            char_col.push_back(&data[i + ncols * k]);
            break;
        case column_major:
            char_col.push_back(&data[k + nrows * i]);
            break;
        }
    }
    // Replace the ith column with this new vector
    columns[i] = char_col;
}

/* Add the item elem to the ith column of the data, where j entries have already been dealt with */
template <class T>
inline void update_column(T elem, CSVColumnsType &columns, da_int i, da_int j,
                          char **data, da_int nrows, da_int ncols, da_order order) {
    // T can be da_int, float, double or uint8_t, but char** will have been caught earlier
    // If columns[i] is also of type T then simply push_back
    if (std::vector<T> *T_col = std::get_if<std::vector<T>>(&(columns[i]))) {
        // This column is already type T so we only need to push_back
        T_col->push_back(elem);
    } else if (j == 0) {
        // Special edge case of first item in column which may need to be cast to uint8_t and isn't caught elsewhere
        std::vector<T> T_col;
        T_col.push_back(elem);
        columns[i] = T_col;
    } else {
        // Type mismatch so convert the whole column up to char**
        convert_col_to_char(columns, i, j, data, nrows, ncols, order);
    }
}

/* Overload of previous function to deal with da_int data which can be cast to float or double */
inline void update_column(da_int elem, CSVColumnsType &columns, da_int i, da_int j,
                          char **data, da_int nrows, da_int ncols, da_order order) {
    if (std::vector<da_int> *int_col = std::get_if<std::vector<da_int>>(&(columns[i]))) {
        // This column already contains da_int data so we only need to push_back
        int_col->push_back(elem);
    } else if (std::vector<float> *float_col =
                   std::get_if<std::vector<float>>(&(columns[i]))) {
        float_col->push_back((float)elem);
    } else if (std::vector<double> *double_col =
                   std::get_if<std::vector<double>>(&(columns[i]))) {
        double_col->push_back((double)elem);
    } else {
        // Type mismatch so convert the whole column to char**
        convert_col_to_char(columns, i, j, data, nrows, ncols, order);
    }
}

/* Overload of previous function to deal with float data */
inline void update_column(float elem, CSVColumnsType &columns, da_int i, da_int j,
                          char **data, da_int nrows, da_int ncols, da_order order) {
    if (std::vector<float> *float_col = std::get_if<std::vector<float>>(&(columns[i]))) {
        // This column already contains float data so we only need to push_back
        float_col->push_back(elem);
    } else if (std::vector<da_int> *int_col =
                   std::get_if<std::vector<da_int>>(&(columns[i]))) {
        // Convert integer column to a float vector
        std::vector<float> float_col;
        for (da_int k = 0; k < j; k++) {
            float_col.push_back((float)((*int_col)[k]));
        }
        float_col.push_back(elem);
        columns[i] = float_col;
    } else {
        // Type mismatch so convert the whole column up to char**
        convert_col_to_char(columns, i, j, data, nrows, ncols, order);
    }
}

/* Overload of previous function to deal with double data */
inline void update_column(double elem, CSVColumnsType &columns, da_int i, da_int j,
                          char **data, da_int nrows, da_int ncols, da_order order) {
    if (std::vector<double> *double_col =
            std::get_if<std::vector<double>>(&(columns[i]))) {
        // This column already contains double data so we only need to push_back
        double_col->push_back(elem);
    } else if (std::vector<da_int> *int_col =
                   std::get_if<std::vector<da_int>>(&(columns[i]))) {
        // Convert integer column to a double vector
        std::vector<double> double_col;
        for (da_int k = 0; k < j; k++) {
            double_col.push_back((double)((*int_col)[k]));
        }
        double_col.push_back(elem);
        columns[i] = double_col;
    } else {
        // Type mismatch so convert the whole column to char**
        convert_col_to_char(columns, i, j, data, nrows, ncols, order);
    }
}

/* These next two functions use variadic templates to recursively call char_to_num
   until the string is successfully parsed as a number, or return an error if the
   string needs to remain as a string */
template <class T>
inline da_status get_number(parser_t *parser, const char *str, CSVElementType &output,
                            T number) {
    int *maybe_int = nullptr;
    char **endptr = nullptr;
    da_status error = da_csv::char_to_num(parser, str, endptr, &number, maybe_int);
    output = number;
    return error;
}

template <class T, class... Rest>
inline da_status get_number(parser_t *parser, const char *str, CSVElementType &output,
                            T number, Rest... rest) {
    int *maybe_int = nullptr;
    char **endptr = nullptr;
    if (da_csv::char_to_num(parser, str, endptr, &number, maybe_int) ==
        da_status_success) {
        output = number;
        return da_status_success;
    } else {
        return get_number(parser, str, output, rest...);
    }
}

/* Given char** data array from a CSV, detect the datatype of the columns and create std::variant vectors accordingly */
inline da_status detect_columns(da_csv::csv_reader *csv, CSVColumnsType &columns,
                                char **data, da_int nrows, da_int ncols) {

    da_status error = da_status_success, tmp_error = da_status_success;
    parser_t *parser = csv->parser;

    da_int tmp_int = 0;
    double tmp_double = 0.0;
    float tmp_float = 0.0f;
    uint8_t tmp_uint8 = 0;

    CSVElementType output;

    // Place ncols da_int vectors into the columns as that's our starting datatype (we will deal with uint8_t special case later)
    for (da_int i = 0; i < ncols; i++) {
        std::vector<da_int> vec;
        columns.push_back(vec);
    }

    da_int data_index = 0;

    for (da_int j = 0; j < nrows; j++) {

        for (da_int i = 0; i < ncols; i++) {

            // Index into data array depends on whether we have stored data in row major or column major order
            switch (csv->order) {
            case row_major:
                data_index = i + ncols * j;
                break;
            case column_major:
                data_index = j + nrows * i;
                break;
            }

            if (std::vector<char **> *char_col =
                    std::get_if<std::vector<char **>>(&(columns[i]))) {
                // This column already contains char data so we only need to store the pointer to the word
                char_col->push_back(&data[data_index]);
            }

            // Call get_number, which recursively calls char_to_num until the appropriate datatype is found
            // If statements needed to account for possible options
            if (csv->integers_as_fp) {
                if (csv->precision) {
                    tmp_error = get_number(parser, data[data_index], output, tmp_double,
                                           tmp_uint8);
                } else {
                    tmp_error = get_number(parser, data[data_index], output, tmp_float,
                                           tmp_uint8);
                }
            } else {
                if (csv->precision) {
                    tmp_error = get_number(parser, data[data_index], output, tmp_int,
                                           tmp_double, tmp_uint8);
                } else {
                    tmp_error = get_number(parser, data[data_index], output, tmp_int,
                                           tmp_float, tmp_uint8);
                }
            }

            // Append the element to the column if possible
            if (tmp_error == da_status_success) {
                std::visit(
                    [&columns, &i, &j, &data, &nrows, &ncols, &csv](const auto &elem) {
                        update_column(elem, columns, i, j, data, nrows, ncols,
                                      csv->order);
                    },
                    output);

            } else {
                // Replace this column with chars
                convert_col_to_char(columns, i, j, data, nrows, ncols, csv->order);
            }
        }
    }
    return error;
}

} // namespace da_auto_detect
#endif
