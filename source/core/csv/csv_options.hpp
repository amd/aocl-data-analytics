/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef CSV_OPTIONS_HPP
#define CSV_OPTIONS_HPP

#include "options.hpp"
#include <limits>

inline da_status register_csv_options(da_options::OptionRegistry &opts) {
    using namespace da_options;

    try {

        std::shared_ptr<OptionNumeric<da_int>> oi;
        std::shared_ptr<OptionString> os;
        std::map<string, da_int> dummy;

        os = std::make_shared<OptionString>(
            OptionString("CSV delimiter", "Select CSV delimiter", dummy, ","));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "CSV thousands", "Select which character separates thousands in CSV files",
            dummy, "\0"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "CSV decimal", "Select which character denotes a decimal point in CSV files",
            dummy, "."));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(
            OptionString("CSV comment",
                         "Select which character is used to denote comments in CSV files",
                         dummy, "#"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "CSV quote character",
            "Select which character is used to denote quotations in CSV files", dummy,
            "\""));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(
            OptionString("CSV escape character",
                         "Select the escape character in CSV files", dummy, "\0"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "CSV line terminator",
            "Select which character is used to denote line termination in CSV files",
            dummy, "\0"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "CSV scientific notation character",
            "Select which character is used to denote powers of 10 in CSV files", dummy,
            "e"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "CSV skip rows",
            "Comma or space separated list of rows to ignore in CSV files", dummy, "\0"));
        opts.register_opt(os);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "CSV double quote",
            "Whether or not to interpret two consecutive quotechar characters within a "
            "field as a single quotechar character",
            0, da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal,
            0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "CSV whitespace delimiter",
            "Whether or not to use whitespace as the delimiter when reaing CSV files", 0,
            da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "CSV skip first rows",
            "Ignore the specified number of rows from the top of the CSV fle", 0,
            da_options::lbound_t::greaterequal, DA_INT_MAX,
            da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "CSV skip empty lines", "Whether or not to ignore empty lines in CSV files",
            0, da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal,
            0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "CSV skip initial space",
            "Whether or not to ignore initial spaces in CSV file lines", 0,
            da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "CSV skip footer", "Whether or not to ignore the last line of the CSV file",
            0, da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal,
            0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "CSV warn for missing data",
            "Whether or not to warn or throw an error when missing data is encountered",
            0, da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal,
            0));
        opts.register_opt(oi);

    } catch (std::bad_alloc &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    } catch (...) {                    // LCOV_EXCL_LINE
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

#endif //CSV_OPTIONS_HPP