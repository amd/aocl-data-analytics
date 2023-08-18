#ifndef PARSER_HPP
#define PARSER_HPP

#include <new>
#include <stdio.h>

#include "aoclda.h"
#include "char_to_num.hpp"
#include "da_handle.hpp"
#include "tokenizer.h"

/* Contains routines for creating and destroying the parser_t struct, separate from those in tokenize.h */

namespace da_csv {

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

    int istatus = parser_init(*parser);

    if (istatus != 0) {
        da_parser_destroy(parser);
        return da_status_memory_error;
    }

    parser_set_default_options(*parser);

    /* Need these callbacks to read from files and clean things up*/
    (*parser)->cb_io = read_bytes;
    (*parser)->cb_cleanup = cleanup;

    return da_status_success;
}

} //namespace da_csv

#endif