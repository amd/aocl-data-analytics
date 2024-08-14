/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "csv_types.hpp"
#include "options.hpp"

#include <limits>

namespace da_csv {

inline da_status register_csv_options(da_options::OptionRegistry &opts) {
    using namespace da_options;

    std::shared_ptr<OptionNumeric<da_int>> oi;
    std::shared_ptr<OptionString> os;
    std::map<string, da_int> dummy;

    os = std::make_shared<OptionString>(OptionString(
        "CSV delimiter", "The delimiter used when reading CSV files.", dummy, ","));
    opts.register_opt(os);

    os = std::make_shared<OptionString>(
        OptionString("CSV thousands",
                     "The character used to separate thousands when reading numeric "
                     "values in CSV files",
                     dummy, ""));
    opts.register_opt(os);

    os = std::make_shared<OptionString>(OptionString(
        "CSV decimal", "The character used to denote a decimal point in CSV files", dummy,
        "."));
    opts.register_opt(os);

    os = std::make_shared<OptionString>(OptionString(
        "CSV comment",
        "The character used to denote comments in CSV files (note, if a line in a CSV "
        "file is to be interpreted as only containing a comment, the comment character "
        "should be the first character on the line)",
        dummy, "#"));
    opts.register_opt(os);

    os = std::make_shared<OptionString>(OptionString(
        "CSV quote character", "The character used to denote quotations in CSV files",
        dummy, "\""));
    opts.register_opt(os);

    os = std::make_shared<OptionString>(OptionString(
        "CSV escape character", "The escape character in CSV files", dummy, "\\"));
    opts.register_opt(os);

    os = std::make_shared<OptionString>(
        OptionString("CSV line terminator",
                     "The character used to denote line termination in CSV files (leave "
                     "this empty to use the default)",
                     dummy, ""));
    opts.register_opt(os);

    os = std::make_shared<OptionString>(OptionString(
        "CSV scientific notation character",
        "The character used to denote powers of 10 in floating point values in CSV files",
        dummy, "e"));
    opts.register_opt(os);

    os = std::make_shared<OptionString>(OptionString(
        "CSV skip rows",
        "A comma- or space-separated list of rows to ignore in CSV files", dummy, "\0"));
    opts.register_opt(os);

    os = std::make_shared<OptionString>(OptionString(
        "CSV data storage",
        "Whether to store data from CSV files in row or column major format",
        {{"row major", row_major}, {"column major", col_major}}, "column major"));
    opts.register_opt(os);

    oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
        "CSV double quote",
        "Whether or not to interpret two consecutive quotechar characters within a "
        "field as a single quotechar character",
        0, da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
    opts.register_opt(oi);

    oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
        "CSV whitespace delimiter",
        "Whether or not to use whitespace as the delimiter when reading CSV files", 0,
        da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
    opts.register_opt(oi);

    oi = std::make_shared<OptionNumeric<da_int>>(
        OptionNumeric<da_int>("CSV row start",
                              "Ignore the specified number of lines from the top of the "
                              "file (note that line numbers in CSV files start at 1)",
                              0, da_options::lbound_t::greaterequal, DA_INT_MAX,
                              da_options::ubound_t::p_inf, 0));
    opts.register_opt(oi);

    oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
        "CSV skip empty lines",
        "Whether or not to ignore empty lines in CSV files (note that caution should be "
        "used when using this in conjunction with options such as CSV skip rows since "
        "line numbers may no longer correspond to the original line numbers in the CSV "
        "file)",
        0, da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
    opts.register_opt(oi);

    oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
        "CSV skip initial space",
        "Whether or not to ignore initial spaces in CSV file lines", 0,
        da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
    opts.register_opt(oi);

    oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
        "CSV skip footer",
        "Whether or not to ignore the last line when reading a CSV file", 0,
        da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
    opts.register_opt(oi);

    oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
        "CSV warn for missing data",
        "If set to 0, return error if missing data is encountered; if set to, 1 issue a "
        "warning and store "
        "missing data as either a NaN (for floating point data) or the maximum value of "
        "the integer type being used",
        0, da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
    opts.register_opt(oi);

    oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
        "CSV use header row", "Whether or not to interpret the first row as a header", 0,
        da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
    opts.register_opt(oi);

    // These additional options only apply when reading CSV files into a datastore object

    os = std::make_shared<OptionString>(OptionString(
        "CSV datatype",
        "If a CSV file is known to be of a single datatype, set this option to "
        "disable autodetection and make reading the file quicker",
        {{"auto", csv_auto},
         {"float", csv_float},
         {"double", csv_double},
         {"integer", csv_integer},
         {"string", csv_char},
         {"boolean", csv_boolean}},
        "auto"));
    opts.register_opt(os);

    os = std::make_shared<OptionString>(
        OptionString("CSV datastore precision",
                     "The precision used when reading floating point numbers "
                     "using autodetection",
                     {{"single", 0}, {"double", 1}}, "double"));
    opts.register_opt(os);

    oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
        "CSV integers as floats",
        "Whether or not to interpret integers as floating point numbers when "
        "using autodetection",
        0, da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
    opts.register_opt(oi);

    return da_status_success;
}

} //namespace da_csv

#endif //CSV_OPTIONS_HPP