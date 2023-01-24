#ifndef AOCLDA_CSV
#define AOCLDA_CSV

#include "aoclda_error.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum csv_option_ {
    csv_option_delimiter = 0,
    csv_option_thousands,
    csv_option_decimal,
    csv_option_comment,
    csv_option_doublequote,
    csv_option_delim_whitespace,
    csv_option_quotechar,
    csv_option_escapechar,
    csv_option_lineterminator,
    csv_option_quoting,
    csv_option_sci,
    csv_option_skip_first_N_rows,
    csv_option_skip_empty_lines,
    csv_option_skip_footer,
    csv_option_skip_initial_space,
    csv_option_header,
    csv_option_header_start,
    csv_option_header_end,
    csv_option_warn_for_missing_data,
} csv_option;

typedef struct parser_t *da_csv_opts;

/* Initialize da_csv_opts struct with default values, and destroy it */
da_status da_csv_init(da_csv_opts *opts);

void da_csv_destroy(da_csv_opts *opts);

/* Option setting routine */
da_status da_csv_set_option(da_csv_opts opts, csv_option option, char *str);

/* Routines for reading in a csv and storing an array of data, with or without a headings
 * row */
da_status da_read_csv_d(da_csv_opts opts, const char *filename, double **a, size_t *nrows,
                        size_t *ncols);

da_status da_read_csv_s(da_csv_opts opts, const char *filename, float **a, size_t *nrows,
                        size_t *ncols);

da_status da_read_csv_int64(da_csv_opts opts, const char *filename, int64_t **a, size_t *nrows,
                            size_t *ncols);

da_status da_read_csv_uint64(da_csv_opts opts, const char *filename, uint64_t **a,
                             size_t *nrows, size_t *ncols);

da_status da_read_csv_uint8(da_csv_opts opts, const char *filename, uint8_t **a, size_t *nrows,
                            size_t *ncols);

da_status da_read_csv_d_h(da_csv_opts opts, const char *filename, double **a, size_t *nrows,
                          size_t *ncols, char ***headings);

da_status da_read_csv_s_h(da_csv_opts opts, const char *filename, float **a, size_t *nrows,
                          size_t *ncols, char ***headings);

da_status da_read_csv_int64_h(da_csv_opts opts, const char *filename, int64_t **a,
                              size_t *nrows, size_t *ncols, char ***headings);

da_status da_read_csv_uint64_h(da_csv_opts opts, const char *filename, uint64_t **a,
                               size_t *nrows, size_t *ncols, char ***headings);

da_status da_read_csv_uint8_h(da_csv_opts opts, const char *filename, uint8_t **a,
                              size_t *nrows, size_t *ncols, char ***headings);

#ifdef __cplusplus
}
#endif

#endif