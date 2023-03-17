#ifndef READ_CSV_HPP
#define READ_CSV_HPP

#include <inttypes.h>
#include <new>
#include <stdio.h>
#include <string.h>

#include "aoclda.h"
#include "da_handle.hpp"
#include "parser.hpp"
#include "tokenizer.h"

/* Contains routines for parsing a csv file */

da_status parse_file(da_handle handle, const char *filename) {

    da_status error = da_status_success;
    parser_t *parser = handle->csv_parser->parser;
    int err;

    FILE *fp = nullptr;
    fp = fopen(filename, "r");

    if (fp == nullptr) {
        return da_status_file_not_found;
    }

    parser->source = (void *)fp;
    char *encoding_errors = NULL;

    err = tokenize_all_rows(parser, encoding_errors);
    error = convert_tokenizer_errors(err);
    if (parser->file_lines != parser->lines) {
        error = da_status_warn_bad_lines;
        snprintf(handle->error_message, ERR_MSG_LEN,
                 "Some lines were ignored - this may be because they were empty.");
    }

    fclose(fp);
    parser->source = nullptr;

    return error;
}

template <typename T>
da_status populate_data_array(da_handle handle, T **a, da_int *nrows, da_int *ncols,
                              da_int first_line) {

    da_status tmp_error = da_status_success, error = da_status_success;
    parser_t *parser = handle->csv_parser->parser;

    uint64_t lines = parser->lines;
    uint64_t words_len = parser->words_len;

    // Guard against empty csv file
    if (lines == 0 || (parser->skip_footer && lines == 1)) {
        *nrows = 0;
        *ncols = 0;
        *a = nullptr;
        snprintf(handle->error_message, ERR_MSG_LEN,
                 "No data was found in the csv file.");
        return da_status_warn_no_data;
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
        return da_status_success;
    }

    //The parser has some hard coded int64 and uint64 values here so care is needed when casting to da_int at the end
    uint64_t fields_per_line = words_len / lines;
    int64_t fields_per_line_signed;
    if (fields_per_line > DA_INT_MAX || lines > DA_INT_MAX) {
        return da_status_overflow;
    } else {
        fields_per_line_signed = (int64_t)(fields_per_line);
    }

    T *data;
    data = (T *)malloc(sizeof(T) * fields_per_line * (lines - first_line));

    if (data == NULL) {
        return da_status_memory_error;
    }

    char *p_end = NULL;

    for (uint64_t i = (uint64_t)first_line; i < lines; i++) {
        // check for ragged matrix
        if (parser->line_fields[i] != fields_per_line_signed) {
            if (data)
                free(data);
            snprintf(handle->error_message, ERR_MSG_LEN,
                     "Line %" PRIu64
                     " had an unexpected number of fields (fields %" PRId64
                     ", expected %" PRId64 ").",
                     i, parser->line_fields[i], fields_per_line_signed);
            return da_status_ragged_csv;
        }
        for (int64_t j = parser->line_start[i];
             j < (parser->line_start[i] + parser->line_fields[i]); j++) {

            tmp_error = char_to_num(
                parser, parser->words[j], &p_end,
                &data[j - (int64_t)first_line * fields_per_line_signed], NULL);
            if (tmp_error != da_status_success) {
                if (parser->warn_for_missing_data) {
                    missing_data(&data[j - (int64_t)first_line * fields_per_line_signed]);
                    snprintf(handle->error_message, ERR_MSG_LEN,
                             "Missing data on line %" PRIu64 ", entry %" PRId64 ".", i,
                             j);
                    error = da_status_warn_missing_data;
                } else {
                    snprintf(handle->error_message, ERR_MSG_LEN,
                             "Unable to parse entry on line %" PRIu64 " entry %" PRId64
                             ".",
                             i, j);
                    *a = nullptr;
                    if (data)
                        free(data);
                    return tmp_error;
                }
            }
        }
    }

    *nrows = (da_int)lines - first_line;
    *ncols = (da_int)fields_per_line;
    *a = data;
    return error;
}

template <typename T>
da_status da_read_csv(da_handle handle, const char *filename, T **a, da_int *nrows,
                      da_int *ncols) {

    da_status error = da_status_success, tmp_error = da_status_success;

    error = da_check_handle_type(handle, da_handle_csv_opts);
    if (error != da_status_success) {
        return error;
    }

    error = handle->csv_parser->read_options();
    if (error != da_status_success) {
        return error;
    }

    error = parse_file(handle, filename);

    if (!(error == da_status_success) && !(error == da_status_warn_bad_lines)) {
        parser_reset(handle->csv_parser->parser);
        return error;
    }

    tmp_error = populate_data_array(handle, a, nrows, ncols, 0);
    if (!(tmp_error == da_status_success)) {
        error = tmp_error;
    }

    tmp_error = convert_tokenizer_errors(parser_reset(handle->csv_parser->parser));
    if (!(tmp_error == da_status_success)) {
        error = tmp_error;
    }

    return error;
}

template <typename T>
da_status da_read_csv(da_handle handle, const char *filename, T **a, da_int *nrows,
                      da_int *ncols, char ***headings) {

    da_status error = da_status_success, tmp_error = da_status_success;

    error = da_check_handle_type(handle, da_handle_csv_opts);
    if (error != da_status_success) {
        return error;
    }

    error = handle->csv_parser->read_options();
    if (error != da_status_success) {
        return error;
    }

    error = parse_file(handle, filename);

    if (!(error == da_status_success) && !(error == da_status_warn_bad_lines)) {
        parser_reset(handle->csv_parser->parser);
        return error;
    }

    tmp_error = populate_data_array(handle, a, nrows, ncols, 1);

    if (!(tmp_error == da_status_success)) {
        error = tmp_error;
    }

    // now deal with headings
    if (*ncols == 0) {
        parser_reset(handle->csv_parser->parser);
        return tmp_error;
    }

    if (!(*ncols == handle->csv_parser->parser->line_fields[0])) {
        if (*a)
            free(*a);
        parser_reset(handle->csv_parser->parser);
        return da_status_ragged_csv;
    }

    *headings = (char **)malloc(sizeof(char *) * (*ncols));
    if (*headings == NULL) {
        if (*a)
            free(*a);
        parser_reset(handle->csv_parser->parser);
        return da_status_memory_error;
    }

    char *p = NULL;
    for (da_int i = 0; i < *ncols; i++) {
        p = handle->csv_parser->parser->words[i];
        // Skip leading whitespace.
        if (handle->csv_parser->parser->skipinitialspace) {
            while (isspace_ascii(*p))
                p++;
        }

        (*headings)[i] = (char *)malloc(sizeof(char) * (1 + strlen(p)));
        if ((*headings)[i] == NULL) {
            if (*a)
                free(*a);
            for (da_int j = 0; j < i; j++) {
                if ((*headings)[j])
                    free((*headings)[j]);
            }
            if ((*headings))
                free((*headings));
            parser_reset(handle->csv_parser->parser);
            return da_status_memory_error;
        }
        strcpy((*headings)[i], p);
    }

    tmp_error = convert_tokenizer_errors(parser_reset(handle->csv_parser->parser));
    if (!(tmp_error == da_status_success)) {
        error = tmp_error;
    }

    return error;
}

#endif