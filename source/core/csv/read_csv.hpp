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
        return da_error(csv->err, da_status_file_not_found, "File not found");
    }

    parser->source = (void *)fp;
    char *encoding_errors = NULL;

    istatus = tokenize_all_rows(parser, encoding_errors);
    if (istatus != 0 || parser->file_lines != parser->lines) {
        return da_warn(csv->err, da_status_bad_lines,
                       "Some lines were ignored - this may be because they were empty.");
    }

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
        return da_warn(csv->err, da_status_no_data, "No data was found in the csv file.");
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
        return da_warn(csv->err, da_status_no_data, "No data was found in the CSV file");
    }

    //The parser has some hard coded int64 and uint64 values here so care is needed when casting to da_int at the end
    uint64_t fields_per_line = words_len / lines;
    int64_t fields_per_line_signed;
    if (fields_per_line > DA_INT_MAX || lines > DA_INT_MAX) {
        return da_error(csv->err, da_status_overflow,
                        "Too many fields were found in the CSV file.");
    } else {
        fields_per_line_signed = (int64_t)(fields_per_line);
    }

    T *data;
    uint64_t n = fields_per_line * (lines - first_line);
    //Need calloc rather than new as this could be called from C code
    data = (T *)calloc(n, sizeof(T));

    if (data == NULL) {
        return da_error(csv->err, da_status_memory_error, "Memory allocation failure");
    }

    char *p_end = NULL;

    for (uint64_t i = (uint64_t)first_line; i < lines; i++) {
        // check for ragged matrix
        if (parser->line_fields[i] != fields_per_line_signed) {
            std::string buff;
            buff = "Line " + std::to_string(i);
            buff += " had an unexpected number of fields (fields " +
                    std::to_string(parser->line_fields[i]);
            buff += ", expected " + std::to_string(fields_per_line_signed) + ").";
            free_data(&data, (da_int)n);
            return da_error(csv->err, da_status_ragged_csv, buff);
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
                    buff = "Unable to parse entry on line " + std::to_string(i) +
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
        return da_error(csv->err, da_status_ragged_csv, "Ragged CSV");
    }

    // Calloc rather than new as can be called from C code amd want char pointers set to null
    *headings = (char **)calloc(ncols, sizeof(char *));
    if (*headings == nullptr) {
        return da_error(csv->err, da_status_memory_error, "Memory allocation failure");
    }

    char *p = nullptr;
    char *p_end = nullptr;

    for (da_int i = 0; i < ncols; i++) {
        p = parser->words[i];

        status = char_to_num(parser, p, &p_end, &(*headings)[i], nullptr);
        if (status != da_status_success) {
            std::string buff = "Unable to parse header " + std::to_string(i) + ".";
            free_data(headings, ncols);
            return da_error(csv->err, status, buff);
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

    if (!((error == da_status_success) || (error == da_status_bad_lines))) {
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
            free_data(a, (*ncols) * (*nrows));
            parser_reset(csv->parser);
            return da_error_trace(csv->err, tmp_error, "Error parsing Headings");
        }
    }

    int istatus = parser_reset(csv->parser);
    if (istatus != 0) {
        return da_error(csv->err, da_status_parsing_error,
                        "An error occurred while resetting the parser.");
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
