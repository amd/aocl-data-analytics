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

namespace da_csv {

inline da_status convert_tokenizer_errors(int ierror) {
    da_status status;

    switch (ierror) {
    case 0:
        status = da_status_success;
        break;
    case CALLING_READ_FAILED:
        status = da_status_file_reading_error;
        break;
    case PARSER_OUT_OF_MEMORY:
        status = da_status_memory_error;
        break;
    case -1:
        status = da_status_parsing_error;
        break;
    }

    return status;
}

/* This callback is as required by the code in tokenizer.c - it reads data from the csv
 * file */
inline void *read_bytes(void *source, size_t nbytes, size_t *bytes_read, int *status,
                        [[maybe_unused]] const char *encoding_errors) {

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
}

inline int cleanup(void *source) {
    if (source) {
        FILE *fp = (FILE *)source;
        fclose(fp);
        source = NULL;
    }
    return 0;
}

/* Destroy the parser_t struct */
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

    da_status status = convert_tokenizer_errors(err);

    if (!(status == da_status_success)) {
        da_parser_destroy(parser);
        return status;
    }

    parser_set_default_options(*parser);

    /* Need these callbacks to read from files and clean things up*/
    (*parser)->cb_io = read_bytes;
    (*parser)->cb_cleanup = cleanup;

    return da_status_success;
}

} //namespace da_csv

#endif