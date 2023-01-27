#ifndef READ_CSV_HPP
#define READ_CSV_HPP

#include <new>
#include <stdio.h>
#include <string.h>

#include "aoclda.h"
#include "tokenizer.h"

/* This callback is as required by the code in tokenizer.c - it reads data from the csv
 * file*/
void *read_bytes(void *source, size_t nbytes, size_t *bytes_read, int *status,
                 const char *encoding_errors) {
    FILE *fp = (FILE *)source;
    char *buffer = (char *)malloc(nbytes);
    if (buffer == NULL && nbytes > 0) {
        *status = PARSER_OUT_OF_MEMORY;
        return NULL;
    }

    size_t read_status = fread(buffer, 1, nbytes, fp);
    *bytes_read = read_status;

    if (read_status == nbytes) {
        *status = 0;
        return (void *)buffer;
    } else { // error handling
        if (read_status == 0) {
            *status = REACHED_EOF;
            return (void *)buffer;
        } else if (ferror(fp)) {
            *status = CALLING_READ_FAILED;
            free(buffer);
            return NULL;
        } else {
            return (void *)buffer;
        }
    }

    free(buffer);
    return NULL;
}

int cleanup(void *source) {
    if (source) {
        FILE *fp = (FILE *)source;
        fclose(fp);
        source = NULL;
    }
    return 0;
}

/* We don't want to edit tokenize.c in case it needs updating, so convert the error exits
 * here */
da_status convert_tokenizer_errors(int ierror) {
    da_status error;

    switch (ierror) {
    case 0:
        error = da_status_success;
        break;
    case CALLING_READ_FAILED:
        error = da_status_file_reading_error;
        break;
    case PARSER_OUT_OF_MEMORY:
        error = da_status_memory_error;
        break;
    case -1:
        error = da_status_parsing_error;
        break;
    }

    return error;
}

da_status parse_file(da_csv_opts opts, const char *filename) {

    da_status error = da_status_success;
    int err;

    FILE *fp = nullptr;
    fp = fopen(filename, "r");

    if (fp == nullptr){
        return da_status_file_not_found;
    }

    opts->source = (void *)fp;
    char *encoding_errors = NULL;

    err = tokenize_all_rows(opts, encoding_errors);
    error = convert_tokenizer_errors(err);
    if (opts->file_lines != opts->lines) {
        error = da_status_warn_bad_lines;
    }

    return error;
}

template <typename T>
da_status populate_data_array(da_csv_opts opts, T **a, da_int *nrows, da_int *ncols,
                              da_int first_line) {

    da_status tmp_error = da_status_success, error = da_status_success;

    //the parser has some hard coded int64 and uint64 values here so care is needed when casting to da_int at the end

    uint64_t fields_per_line = opts->words_len / opts->lines;
    int64_t fields_per_line_signed;
    if (fields_per_line > DA_INT_MAX || opts->lines > DA_INT_MAX) {
        return da_status_overflow;
    } else {
        fields_per_line_signed = (int64_t)(fields_per_line);
    }

    T *data;
    data = (T *)malloc(sizeof(T) * fields_per_line * (opts->lines - first_line));

    if (data == NULL) {
        return da_status_memory_error;
    }

    char *p_end = NULL;

    for (uint64_t i = (uint64_t)first_line; i < opts->lines; i++) {
        // check for ragged matrix
        if (opts->line_fields[i] != fields_per_line_signed) {
            if (data)
                free(data);
            return da_status_ragged_csv;
        }
        for (int64_t j = opts->line_start[i];
             j < (opts->line_start[i] + opts->line_fields[i]); j++) {
            tmp_error = char_to_num(opts, opts->words[j], &p_end,
                                    &data[j - (int64_t)first_line * fields_per_line_signed], NULL);
            if (tmp_error != da_status_success) {
                if (opts->warn_for_missing_data) {
                    missing_data(&data[j - (int64_t)first_line * fields_per_line_signed]);
                    error = da_status_warn_missing_data;
                } else {
                    if (data)
                        free(data);
                    return tmp_error;
                }
            }
        }
    }

    *nrows = (da_int)opts->lines - first_line;
    *ncols = fields_per_line;
    *a = data;
    return error;
}

template <typename T>
da_status da_read_csv(da_csv_opts opts, const char *filename, T **a, da_int *nrows,
                      da_int *ncols) {

    da_status error = da_status_success, tmp_error = da_status_success;

    error = parse_file(opts, filename);

    if (!(error == da_status_success) && !(error == da_status_warn_bad_lines)) {
        return error;
    }

    tmp_error = populate_data_array(opts, a, nrows, ncols, 0);

    if (!(tmp_error == da_status_success)) {
        error = tmp_error;
    }

    return error;
}

template <typename T>
da_status da_read_csv(da_csv_opts opts, const char *filename, T **a, da_int *nrows,
                      da_int *ncols, char ***headings) {
    da_status error = da_status_success, tmp_error = da_status_success;

    error = parse_file(opts, filename);

    if (!(error == da_status_success) && !(error == da_status_warn_bad_lines)) {
        return error;
    }

    tmp_error = populate_data_array(opts, a, nrows, ncols, 1);

    if (!(tmp_error == da_status_success)) {
        error = tmp_error;
    }

    // now deal with headings
    if (!(*ncols == opts->line_fields[0])) {
        if (*a)
            free(*a);
        return da_status_ragged_csv;
    }

    *headings = (char **)malloc(sizeof(char *) * (*ncols));
    if (*headings == NULL) {
        if (*a)
            free(*a);
        return da_status_memory_error;
    }

    char *p = NULL;
    for (da_int i = 0; i < *ncols; i++) {
        p = opts->words[i];
        // Skip leading whitespace.
        if (opts->skipinitialspace){
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
            return da_status_memory_error;
        }
        strcpy((*headings)[i], p);
    }

    return error;
}

#endif