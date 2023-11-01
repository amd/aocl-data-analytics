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

#ifndef READ_CSV_HPP
#define READ_CSV_HPP

#include <inttypes.h>
#include <new>
#include <sstream>
#include <stdio.h>
#include <string.h>

#include "aoclda.h"
#include "parser.hpp"
#include "tokenizer.h"

/* Contains routines for parsing a csv file */
namespace da_csv {

template <typename T> inline void free_data(T **arr, [[maybe_unused]] da_int n) {
    if (*arr)
        free(*arr);
}

inline void free_data(char ***arr, da_int n) {
    if (arr && *arr) {
        for (da_int i = 0; i < n; i++) {
            if ((*arr)[i]) {
                free((*arr)[i]);
                (*arr)[i] = nullptr;
            }
        }
        free(*arr);
        *arr = nullptr;
    }
}

inline da_status parse_file(csv_reader *csv, const char *filename) {

    da_status status = da_status_success;
    parser_t *parser = csv->parser;
    int istatus;

    FILE *fp = nullptr;

/* Most of the time MSVC compiler can automatically replace CRT functions with _s versions, but not this one */
#if defined(_MSC_VER)
    if (fopen_s(&fp, filename, "r") != 0) {
#else
    fp = fopen(filename, "r");
    if (fp == nullptr) {
#endif
        return da_error(csv->err, da_status_file_reading_error, "File not found");
    }

    parser->source = (void *)fp;
    char *encoding_errors = NULL;

    istatus = tokenize_all_rows(parser, encoding_errors);
    if (istatus != 0) {
        da_error(csv->err, da_status_memory_error,
                 "Memory allocation failure"); // LCOV_EXCL_LINE
        goto exit;                             // LCOV_EXCL_LINE
    } else if (parser->skipped_lines != nullptr) {
        std::string buff;
        buff = "The following lines of the CSV file were ignored:\n";
        // Get the list of ignored lines from the parser's hash table and sort them
        std::vector<khint64_t> keys;
        for (khint64_t it = kh_begin((kh_int64_t *)parser->skipped_lines);
             it != kh_end((kh_int64_t *)parser->skipped_lines); ++it) {
            if (kh_exist((kh_int64_t *)parser->skipped_lines, it))
                keys.push_back(
                    (khint64_t)kh_key(((kh_int64_t *)parser->skipped_lines), it));
        };
        std::sort(keys.begin(), keys.end());
        for (const khint64_t &key : keys) {
            buff += std::to_string(key) + " ";
        }
        da_warn(csv->err, da_status_success, buff);
        goto exit;
    }

exit:
    fclose(fp);
    parser->source = nullptr;

    return status;
}

template <typename T>
inline da_status populate_data_array(csv_reader *csv, T **a, da_int *nrows, da_int *ncols,
                                     da_int first_line) {

    da_status tmp_error = da_status_success, status = da_status_success;
    parser_t *parser = csv->parser;

    uint64_t lines = parser->lines;
    uint64_t words_len = parser->words_len;

    // Guard against empty csv file
    if (lines == 0 || (parser->skip_footer && lines == 1)) {
        *nrows = 0;
        *ncols = 0;
        *a = nullptr;
        return da_warn(csv->err, da_status_parsing_error,
                       "No data was found in the CSV file.");
    }

    if (parser->skip_footer) {
        lines--;
        words_len -= parser->line_fields[lines];
    }

    //Guard against header-only csv file
    if (lines == (uint64_t)first_line) {
        *nrows = 0;
        *ncols = (da_int)words_len;
        *a = nullptr;
        return da_warn(csv->err, da_status_parsing_error,
                       "No data was found in the CSV file");
    }

    //The parser has some hard coded int64 and uint64 values here so care is needed when casting to da_int at the end
    uint64_t fields_per_line = words_len / lines;
    int64_t fields_per_line_signed;
    if (fields_per_line > DA_INT_MAX || lines > DA_INT_MAX) {
        return da_error(csv->err, da_status_overflow,                   // LCOV_EXCL_LINE
                        "Too many fields were found in the CSV file."); // LCOV_EXCL_LINE
    } else {
        fields_per_line_signed = (int64_t)(fields_per_line);
    }

    T *data;
    uint64_t n = fields_per_line * (lines - first_line);
    //Need calloc rather than new as this could be called from C code
    data = (T *)calloc(n, sizeof(T));

    if (data == NULL) {
        return da_error(csv->err, da_status_memory_error,
                        "Memory allocation failure"); // LCOV_EXCL_LINE
    }

    char *p_end = NULL;

    for (uint64_t i = (uint64_t)first_line; i < lines; i++) {
        // check for ragged matrix
        if (parser->line_fields[i] != fields_per_line_signed) {
            std::string buff;
            buff = "In the lines read from the CSV file,";
            buff += " line " + std::to_string(i + 1);
            buff += " had an unexpected number of fields (fields " +
                    std::to_string(parser->line_fields[i]);
            buff += ", expected " + std::to_string(fields_per_line_signed) + ").";
            free_data(&data, (da_int)n);
            return da_error(csv->err, da_status_parsing_error, buff);
        }

        for (int64_t j = parser->line_start[i];
             j < (parser->line_start[i] + parser->line_fields[i]); j++) {

            tmp_error = char_to_num(
                parser, parser->words[j], &p_end,
                &data[j - (int64_t)first_line * fields_per_line_signed], NULL);
            if (tmp_error != da_status_success) {
                std::string buff;
                if (parser->warn_for_missing_data) {
                    missing_data(&data[j - (int64_t)first_line * fields_per_line_signed]);
                    buff = "Missing data on line " + std::to_string(i) + ", entry " +
                           std::to_string(j);
                    da_warn(csv->err, da_status_missing_data, buff);
                    status = da_status_missing_data;
                } else {
                    buff = "Unable to parse data on line " + std::to_string(i) +
                           " entry " + std::to_string(j) + ".";
                    *a = nullptr;
                    free_data(&data, (da_int)n);
                    return da_error(csv->err, tmp_error, buff);
                }
            }
        }
    }

    *nrows = (da_int)lines - first_line;
    *ncols = (da_int)fields_per_line;
    *a = data;
    return status;
}

inline da_status parse_headings(csv_reader *csv, da_int ncols, char ***headings) {

    da_status status = da_status_success;
    parser_t *parser = csv->parser;

    if (ncols == 0) {
        parser_reset(parser);
        return da_status_success;
    }

    if (!(ncols == parser->line_fields[0])) {
        // This error exit should be caught earlier
        std::string buff;                                             // LCOV_EXCL_LINE
        buff = "An unexpected number of headings was found (found " + // LCOV_EXCL_LINE
               std::to_string(parser->line_fields[0]);                // LCOV_EXCL_LINE
        buff += ", expected " + std::to_string(ncols) + ").";         // LCOV_EXCL_LINE
        return da_error(csv->err, da_status_parsing_error, buff);     // LCOV_EXCL_LINE
    }

    // Calloc rather than new as can be called from C code amd want char pointers set to null
    *headings = (char **)calloc(ncols, sizeof(char *));
    if (*headings == nullptr) {
        return da_error(csv->err, da_status_memory_error,
                        "Memory allocation failure"); // LCOV_EXCL_LINE
    }

    char *p = nullptr;
    char *p_end = nullptr;

    for (da_int i = 0; i < ncols; i++) {
        p = parser->words[i];

        status = char_to_num(parser, p, &p_end, &(*headings)[i], nullptr);
        if (status != da_status_success) {
            std::string buff =
                "Unable to parse header " + std::to_string(i) + "."; // LCOV_EXCL_LINE
            free_data(headings, ncols);                              // LCOV_EXCL_LINE
            return da_error(csv->err, status, buff);                 // LCOV_EXCL_LINE
        }
    }

    return status;
}

template <typename T>
inline da_status parse_and_process(csv_reader *csv, const char *filename, T **a,
                                   da_int *nrows, da_int *ncols, da_int get_headings,
                                   char ***headings) {

    da_status error = da_status_success, tmp_error = da_status_success;

    error = parse_file(csv, filename);

    if (error != da_status_success) {
        parser_reset(csv->parser);
        return da_error_trace(csv->err, error, "Error parsing the file");
    }

    tmp_error = populate_data_array(csv, a, nrows, ncols, get_headings);

    if (tmp_error != da_status_success) {
        error = tmp_error;
    }

    // now deal with headings

    if (get_headings) {

        tmp_error = parse_headings(csv, *ncols, headings);
        if (tmp_error != da_status_success) {
            free_data(a, (*ncols) * (*nrows)); // LCOV_EXCL_LINE
            parser_reset(csv->parser);         // LCOV_EXCL_LINE
            return da_error_trace(csv->err, tmp_error,
                                  "Error parsing headings"); // LCOV_EXCL_LINE
        }
    }

    int istatus = parser_reset(csv->parser);
    if (istatus != 0) {
        return da_error(
            csv->err, da_status_memory_error, // LCOV_EXCL_LINE
            "A memory allocation error occurred while resetting the parser."); // LCOV_EXCL_LINE
    }

    return error;
}

template <typename T>
inline da_status read_csv(csv_reader *csv, const char *filename, T **a, da_int *nrows,
                          da_int *ncols, char ***headings) {

    da_status error;
    error = csv->read_options();
    if (error != da_status_success) {
        return da_error_trace(csv->err, da_status_internal_error, "Option reading error");
    }

    da_int get_headings = csv->first_row_header;

    error = parse_and_process(csv, filename, a, nrows, ncols, get_headings, headings);
    if (error != da_status_success)
        return da_error_trace(csv->err, error, "Error parsing CSV");

    return da_status_success;
}

} //namespace da_csv

#endif
