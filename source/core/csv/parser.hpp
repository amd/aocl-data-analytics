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