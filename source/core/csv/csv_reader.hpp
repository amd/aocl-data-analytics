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

#ifndef CSV_READER_HPP
#define CSV_READER_HPP

#include "aoclda.h"
#include "csv_options.hpp"
#include "options.hpp"
#include "tokenizer.h"
#include <sstream>

da_status da_parser_init(parser_t **parser);
void da_parser_destroy(parser_t **parser);

class csv_reader {
  public:
    parser_t *parser;
    da_options::OptionRegistry opts;

    csv_reader() {
        da_status error = da_parser_init(&parser);
        if (error != da_status_success) {
            std::bad_alloc exception;
            throw exception;
        }
        register_csv_options(opts);
    }
    ~csv_reader() { da_parser_destroy(&parser); }

    da_status read_options() {
        da_int iopt;
        std::string sopt;

        opts.get("CSV delimiter", sopt);
        parser->delimiter = sopt[0];

        opts.get("CSV thousands", sopt);
        parser->thousands = sopt[0];

        opts.get("CSV decimal", sopt);
        parser->decimal = sopt[0];

        opts.get("CSV comment", sopt);
        parser->commentchar = sopt[0];

        opts.get("CSV quote character", sopt);
        parser->quotechar = sopt[0];

        opts.get("CSV escape character", sopt);
        parser->escapechar = sopt[0];

        opts.get("CSV line terminator", sopt);
        parser->lineterminator = sopt[0];

        opts.get("CSV scientific notation character", sopt);
        parser->sci = sopt[0];

        opts.get("CSV skip rows", sopt);
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
                    return da_status_option_invalid_value;
                }
            }
        }

        opts.get("CSV double quote", &iopt);
        parser->doublequote = (int)iopt;

        opts.get("CSV whitespace delimiter", &iopt);
        parser->delim_whitespace = (int)iopt;

        opts.get("CSV skip first rows", &iopt);
        parser_set_skipfirstnrows(parser, (int64_t)iopt);

        opts.get("CSV skip empty lines", &iopt);
        parser->skip_empty_lines = (int)iopt;

        opts.get("CSV skip initial space", &iopt);
        parser->skipinitialspace = (int)iopt;

        opts.get("CSV skip footer", &iopt);
        parser->skip_footer = (int)iopt;

        opts.get("CSV warn for missing data", &iopt);
        parser->warn_for_missing_data = (int)iopt;

        return da_status_success;
    }
};

#endif //CSV_READER_HPP