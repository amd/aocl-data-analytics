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

#ifndef CSV_READER_HPP
#define CSV_READER_HPP

#include "aoclda.h"
#include "csv_options.hpp"
#include "csv_types.hpp"
#include "da_error.hpp"
#include "options.hpp"
#include "tokenizer.h"
#include <sstream>

namespace da_csv {

da_status da_parser_init(parser_t **parser);
void da_parser_destroy(parser_t **parser);

class csv_reader {
  public:
    // parser points to the struct used by the original open source C tokenizer code
    parser_t *parser;
    da_options::OptionRegistry *opts;

    // But to deal with datastore objects and autodetection we need some additional machinery;
    // For example, these attributes are used when reading a CSV file straight into a datastore object
    da_int precision;
    da_int integers_as_fp;
    da_int first_row_header;
    csv_datatype datatype;

    da_order order;

    da_errors::da_error_t *err = nullptr;

    csv_reader(da_options::OptionRegistry &opts, da_errors::da_error_t &err) {
        da_status error = da_parser_init(&parser);
        if (error != da_status_success) {
            std::bad_alloc exception; // LCOV_EXCL_LINE
            throw exception;          // LCOV_EXCL_LINE
        }
        this->opts = &opts;
        this->err = &err;
        register_csv_options(opts);
    }
    ~csv_reader() { da_parser_destroy(&parser); }

    da_status read_options() {
        da_int iopt;
        std::string sopt;

        opts->get("delimiter", sopt);
        parser->delimiter = sopt[0];

        opts->get("thousands", sopt);
        parser->thousands = sopt[0];

        opts->get("decimal", sopt);
        parser->decimal = sopt[0];

        opts->get("comment", sopt);
        parser->commentchar = sopt[0];

        opts->get("quote character", sopt);
        parser->quotechar = sopt[0];

        opts->get("escape character", sopt);
        parser->escapechar = sopt[0];

        opts->get("line terminator", sopt);
        parser->lineterminator = sopt[0];

        opts->get("scientific notation character", sopt);
        parser->sci = sopt[0];

        opts->get("skip rows", sopt);
        if (parser->skipset != NULL) {
            kh_destroy_int64((kh_int64_t *)parser->skipset);
            parser->skipset = NULL;
        }
        std::stringstream ss(sopt);
        std::string item;
        char delimiter[] = ", ";
        while (ss >> item) {
            std::replace_if(
                item.begin(), item.end(),
                [&](char c) {
                    for (int i = 0; i < 2; i++) {
                        if (c == delimiter[i])
                            return true;
                    }
                    return false;
                },
                ' ');
            std::stringstream ss_item(item);
            while (ss_item >> item) {
                try {
                    int64_t num = (int64_t)std::stoi(item);
                    parser_add_skiprow(parser, num);
                } catch (std::exception const &) {
                    return da_status_option_invalid_value; // LCOV_EXCL_LINE
                }
            }
        }

        opts->get("storage order", sopt, iopt);
        order = static_cast<da_order>(iopt);

        opts->get("double quote", iopt);
        parser->doublequote = (int)iopt;

        opts->get("whitespace delimiter", iopt);
        parser->delim_whitespace = (int)iopt;

        opts->get("row start", iopt);
        parser_set_skipfirstnrows(parser, (int64_t)iopt);

        opts->get("skip empty lines", iopt);
        parser->skip_empty_lines = (int)iopt;

        opts->get("skip initial space", iopt);
        parser->skipinitialspace = (int)iopt;

        opts->get("skip footer", iopt);
        parser->skip_footer = (int)iopt;

        opts->get("warn for missing data", iopt);
        parser->warn_for_missing_data = (int)iopt;

        // Additional options only used for reading CSV files into datastore

        opts->get("datatype", sopt, iopt);
        datatype = static_cast<csv_datatype>(iopt);

        opts->get("datastore precision", sopt, iopt);
        precision = iopt;

        opts->get("integers as floats", iopt);
        integers_as_fp = iopt;

        opts->get("use header row", iopt);
        first_row_header = iopt;

        return da_status_success;
    }
};

} //namespace da_csv

#endif //CSV_READER_HPP