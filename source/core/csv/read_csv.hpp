#ifndef READ_CSV_HPP
#define READ_CSV_HPP

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
    parser_t *parser = handle->parser;
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
    }

    fclose(fp);
    parser->source = nullptr;

    return error;
}

template <typename T>
da_status populate_data_array(da_handle handle, T **a, da_int *nrows, da_int *ncols,
                              da_int first_line) {

    da_status tmp_error = da_status_success, error = da_status_success;
    parser_t *parser = handle->parser;

    //the parser has some hard coded int64 and uint64 values here so care is needed when casting to da_int at the end

    uint64_t fields_per_line = parser->words_len / parser->lines;
    int64_t fields_per_line_signed;
    if (fields_per_line > DA_INT_MAX || parser->lines > DA_INT_MAX) {
        return da_status_overflow;
    } else {
        fields_per_line_signed = (int64_t)(fields_per_line);
    }

    T *data;
    data = (T *)malloc(sizeof(T) * fields_per_line * (parser->lines - first_line));

    if (data == NULL) {
        return da_status_memory_error;
    }

    char *p_end = NULL;

    for (uint64_t i = (uint64_t)first_line; i < parser->lines; i++) {
        // check for ragged matrix
        if (parser->line_fields[i] != fields_per_line_signed) {
            if (data)
                free(data);
            snprintf(
                handle->error_message, ERR_MSG_LEN,
                "Line %i had an unexpected number of fields (fields %i, expected %i).", i,
                parser->line_fields[i], fields_per_line_signed);
            return da_status_ragged_csv;
        }
        for (int64_t j = parser->line_start[i];
             j < (parser->line_start[i] + parser->line_fields[i]); j++) {
            tmp_error = char_to_num(
                parser, parser->words[j], &p_end,
                &data[j - (int64_t)first_line * fields_per_line_signed], NULL);
            if (tmp_error != da_status_success) {
                if (parser->warn_for_missing_data) {
                    missing_data(
                        &data[j - (int64_t)first_line * fields_per_line_signed]);
                    snprintf(handle->error_message, ERR_MSG_LEN,
                             "Missing data on line %i.", j);
                    error = da_status_warn_missing_data;
                } else {
                    if (data)
                        free(data);
                    return tmp_error;
                }
            }
        }
    }

    *nrows = (da_int)parser->lines - first_line;
    *ncols = fields_per_line;
    *a = data;
    return error;
}

template <typename T>
da_status da_read_csv(da_handle handle, const char *filename, T **a, da_int *nrows,
                      da_int *ncols) {

    da_status error = da_status_success, tmp_error = da_status_success;

    error = da_check_handle_type(handle, da_handle_csv_opts);
    if (error != da_status_success)
        return error;

    error = parse_file(handle, filename);

    if (!(error == da_status_success) && !(error == da_status_warn_bad_lines)) {
        return error;
    }

    tmp_error = populate_data_array(handle, a, nrows, ncols, 0);
    if (!(tmp_error == da_status_success)) {
        error = tmp_error;
    }

    tmp_error = convert_tokenizer_errors(parser_reset(handle->parser));
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
    if (error != da_status_success)
        return error;

    error = parse_file(handle, filename);

    if (!(error == da_status_success) && !(error == da_status_warn_bad_lines)) {
        return error;
    }

    tmp_error = populate_data_array(handle, a, nrows, ncols, 1);

    if (!(tmp_error == da_status_success)) {
        error = tmp_error;
    }

    // now deal with headings
    if (!(*ncols == handle->parser->line_fields[0])) {
        if (*a)
            free(*a);
        parser_reset(handle->parser);
        return da_status_ragged_csv;
    }

    *headings = (char **)malloc(sizeof(char *) * (*ncols));
    if (*headings == NULL) {
        if (*a)
            free(*a);
        parser_reset(handle->parser);
        return da_status_memory_error;
    }

    char *p = NULL;
    for (da_int i = 0; i < *ncols; i++) {
        p = handle->parser->words[i];
        // Skip leading whitespace.
        if (handle->parser->skipinitialspace) {
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
            parser_reset(handle->parser);
            return da_status_memory_error;
        }
        strcpy((*headings)[i], p);
    }

    tmp_error = convert_tokenizer_errors(parser_reset(handle->parser));
    if (!(tmp_error == da_status_success)) {
        error = tmp_error;
    }

    return error;
}

#endif