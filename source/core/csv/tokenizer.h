// clang-format off
/*
This file was originally obtained from

https://github.com/pandas-dev/pandas/blob/d6608313e211be0a44608252a3a31cf5220963f4/pandas/_libs/src/parser/tokenizer.h
licensed under 3-clause BSD (see below)

Copyright (c) 2012, Lambda Foundry, Inc., except where noted

It incorporates components of WarrenWeckesser/textreader (https://github.com/WarrenWeckesser/textreader), also licensed under 3-clause
BSD

Copyright 2012 Warren Weckesser

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

Modifications Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.

*/

#ifndef TOKENIZER_H
#define TOKENIZER_H

#define ERROR_NO_DIGITS 1
#define ERROR_OVERFLOW 2
#define ERROR_INVALID_CHARS 3

#include "khash.h"
#include "portable.h"
#include <errno.h>
#include <stdint.h>

#define STREAM_INIT_SIZE 32

#define REACHED_EOF 1
#define CALLING_READ_FAILED 2

/*

  C flat file parsing low level code for pandas / NumPy

 */

/*
 *  Common set of error types for the read_rows() and tokenize()
 *  functions.
 */

//#define VERBOSE
#if defined(VERBOSE)
#define TRACE(X) printf X;
#else
#define TRACE(X)
#endif // VERBOSE

#define PARSER_OUT_OF_MEMORY -2

/*
 *  TODO: Might want to couple count_rows() with read_rows() to avoid
 *        duplication of some file I/O.
 */

#ifdef __cplusplus
 extern "C" {
#endif

typedef enum {
    START_RECORD,
    START_FIELD,
    ESCAPED_CHAR,
    IN_FIELD,
    IN_QUOTED_FIELD,
    ESCAPE_IN_QUOTED_FIELD,
    QUOTE_IN_QUOTED_FIELD,
    EAT_CRNL,
    EAT_CRNL_NOP,
    EAT_WHITESPACE,
    EAT_COMMENT,
    EAT_LINE_COMMENT,
    WHITESPACE_LINE,
    START_FIELD_IN_SKIP_LINE,
    IN_FIELD_IN_SKIP_LINE,
    IN_QUOTED_FIELD_IN_SKIP_LINE,
    QUOTE_IN_QUOTED_FIELD_IN_SKIP_LINE,
    FINISHED
} ParserState;

typedef enum { QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONNUMERIC, QUOTE_NONE } QuoteStyle;

#undef ERROR
#undef WARN
#undef SKIP
typedef enum { ERROR, WARN, SKIP } BadLineHandleMethod;

typedef void *(*io_callback)(void *src, size_t nbytes, size_t *bytes_read, int *status,
                             const char *encoding_errors);
typedef int (*io_cleanup)(void *src);

typedef struct parser_t {
    void *source;
    io_callback cb_io;
    io_cleanup cb_cleanup;

    int64_t chunksize; // Number of bytes to prepare for each chunk
    char *data;        // pointer to data to be processed
    int64_t datalen;   // amount of data available
    int64_t datapos;

    // where to write out tokenized data
    char *stream;
    uint64_t stream_len;
    uint64_t stream_cap;

    // Store words in (potentially ragged) matrix for now, hmm
    char **words;
    int64_t *word_starts; // where we are in the stream
    uint64_t words_len;
    uint64_t words_cap;
    uint64_t max_words_cap; // maximum word cap encountered

    char *pword_start;  // pointer to stream start of current field
    int64_t word_start; // position start of current field

    int64_t *line_start;  // position in words for start of line
    int64_t *line_fields; // Number of fields in each line
    uint64_t lines;       // Number of (good) lines observed
    uint64_t file_lines;  // Number of lines (including bad or skipped)
    uint64_t lines_cap;   // Vector capacity

    // Tokenizing stuff
    ParserState state;
    int doublequote;      /* is " represented by ""? */
    char delimiter;       /* field separator */
    int delim_whitespace; /* delimit by consuming space/tabs instead */
    char quotechar;       /* quote character */
    char escapechar;      /* escape character */
    char lineterminator;
    int skipinitialspace; /* ignore spaces following delimiter? */
    int quoting;          /* style of quoting to write */
    int skip_trailing;    /* skip trailing whitespace when converting char token to number*/

    char commentchar;
    int allow_embedded_newline;

    int usecols; // Boolean: 1: usecols provided, 0: none provided

    ssize_t expected_fields;
    BadLineHandleMethod on_bad_lines;

    // floating point options
    char decimal;
    char sci;

    // thousands separator (comma, period)
    char thousands;

    int header;           // Boolean: 1: has header, 0: no header
    int64_t header_start; // header row start
    uint64_t header_end;  // header row end

    void *skipset;
    int64_t skip_first_N_rows;
    int64_t skip_footer;

    // Maximum and minimum values when trying to parse 64-bit integers
    int64_t int_max;
    int64_t int_min;
    uint64_t uint_max;

    // If parsing fails store a NaN (or int_max) and return with a warning
    int warn_for_missing_data;

    // error handling
    char *warn_msg;
    char *error_msg;

    int skip_empty_lines;
} parser_t;

typedef struct coliter_t {
    char **words;
    int64_t *line_start;
    int64_t col;
} coliter_t;

void coliter_setup(coliter_t *self, parser_t *parser, int64_t i, int64_t start);

#define COLITER_NEXT(iter, word)                                                         \
    do {                                                                                 \
        const int64_t i = *iter.line_start++ + iter.col;                                 \
        word = i >= *iter.line_start ? "" : iter.words[i];                               \
    } while (0)

parser_t *parser_new(void);

int parser_init(parser_t *self);

int parser_consume_rows(parser_t *self, size_t nrows);

int parser_trim_buffers(parser_t *self);

int parser_add_skiprow(parser_t *self, int64_t row);

int parser_set_skipfirstnrows(parser_t *self, int64_t nrows);

void parser_free(parser_t *self);

void parser_del(parser_t *self);

int parser_reset(parser_t *self);

void parser_set_default_options(parser_t *self);

int tokenize_nrows(parser_t *self, size_t nrows, const char *encoding_errors);

int tokenize_all_rows(parser_t *self, const char *encoding_errors);

#ifdef __cplusplus
}
#endif

#endif