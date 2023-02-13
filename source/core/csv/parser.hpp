#ifndef PARSER_HPP
#define PARSER_HPP

#include <new>
#include <stdio.h>

#include "aoclda.h"
#include "char_to_num.hpp"
#include "da_handle.hpp"
#include "tokenizer.h"

/* Contains routines for creating and destroying the parser_t struct, separate from those in tokenize.h */

/* We don't want to edit tokenize.c much in case it needs updating, so convert the error exits
 * here */
inline da_status convert_tokenizer_errors(int ierror) {
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

/* This callback is as required by the code in tokenizer.c - it reads data from the csv
 * file */
inline void *read_bytes(void *source, size_t nbytes, size_t *bytes_read, int *status,
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

inline int cleanup(void *source) {
    if (source) {
        FILE *fp = (FILE *)source;
        fclose(fp);
        source = NULL;
    }
    return 0;
}

/* Destroy the da_csv_opts struct */
inline void da_parser_destroy(parser_t **parser) {
    if (parser) {
        if (*parser) {
            parser_free(*parser);
            if (*parser)
                delete (*parser);
            *parser = nullptr;
        }
    }
}

/* Create (and populate with defaults) */
inline da_status da_parser_init(parser_t **parser) {
    try {
        *parser = new parser_t;
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }

    int err = parser_init(*parser);

    da_status error = convert_tokenizer_errors(err);

    if (!(error == da_status_success)) {
        da_parser_destroy(parser);
        return error;
    }

    parser_set_default_options(*parser);

    /* Need these callbacks to read from files and clean things up*/
    (*parser)->cb_io = read_bytes;
    (*parser)->cb_cleanup = cleanup;

    return da_status_success;
}

/* Option setting routine */
inline da_status da_parser_set_option(da_handle handle, da_handle_option option,
                                      char *str) {

    da_status error = da_status_success;
    parser_t *parser = handle->parser;

    char *p_end;
    int64_t i64temp;
    uint64_t ui64temp;

    switch (option) {
    case csv_option_delimiter:
        parser->delimiter = str[0];
        break;
    case csv_option_thousands:
        parser->thousands = str[0];
        break;

    case csv_option_decimal:
        parser->decimal = str[0];
        break;

    case csv_option_comment:
        parser->commentchar = str[0];
        break;

    case csv_option_doublequote:
        error = char_to_num(parser, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            parser->doublequote = (int)i64temp;
        }
        break;

    case csv_option_delim_whitespace:
        error = char_to_num(parser, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            parser->delim_whitespace = (int)i64temp;
        }
        break;

    case csv_option_quotechar:
        parser->quotechar = str[0];
        break;

    case csv_option_escapechar:
        parser->escapechar = str[0];
        break;

    case csv_option_lineterminator:
        parser->lineterminator = str[0];
        break;

    case csv_option_quoting:
        error = char_to_num(parser, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            parser->quoting = (int)i64temp;
        }
        break;

    case csv_option_sci:
        parser->sci = str[0];
        break;

    case csv_option_skip_first_N_rows:
        error = char_to_num(parser, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            parser_set_skipfirstnrows(parser, i64temp);
        }
        break;

    case csv_option_skip_empty_lines:
        error = char_to_num(parser, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            parser->skip_empty_lines = (int)i64temp;
        }
        break;

    case csv_option_skip_initial_space:
        error = char_to_num(parser, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            parser->skipinitialspace = (int)i64temp;
        }
        break;

    case csv_option_skip_footer:
        error = char_to_num(parser, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            parser->skip_footer = (int)i64temp;
        }
        break;

    case csv_option_add_skiprow:
        error = char_to_num(parser, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            parser_add_skiprow(parser, i64temp);
        }
        break;

    case csv_option_warn_for_missing_data:
        error = char_to_num(parser, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            parser->warn_for_missing_data = (int)i64temp;
        }
        break;

    default:
        snprintf(handle->error_message, ERR_MSG_LEN,
                 "The specified option is not valid for csv handles.");
        error = da_status_invalid_option;
    }

    return error;
}

#endif