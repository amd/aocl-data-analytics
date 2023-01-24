#include "char_to_num.hpp"
#include "read_csv.hpp"

/* Create (and populate with defaults) */
da_status da_csv_init(da_csv_opts *opts) {
    try {
        *opts = new parser_t;
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }

    int err = parser_init(*opts);

    da_status error = convert_tokenizer_errors(err);

    if (!(error == da_status_success)) {
        da_csv_destroy(opts);
        return error;
    }

    parser_set_default_options(*opts);

    /* Need these callbacks to read from files and clean things up*/
    (*opts)->cb_io = read_bytes;
    (*opts)->cb_cleanup = cleanup;

    return da_status_success;
}

/* Destroy the da_csv_opts struct */
void da_csv_destroy(da_csv_opts *opts) {
    if (opts) {
        if (*opts) {
            parser_free(*opts);
            if (*opts)
                delete (*opts);
            *opts = nullptr;
        }
    }
}

/* Option setting routine */
da_status da_csv_set_option(da_csv_opts opts, csv_option option, char *str) {

    da_status error = da_status_success;

    char *p_end;
    int64_t i64temp;
    uint64_t ui64temp;

    switch (option) {
    case csv_option_delimiter:
        opts->delimiter = str[0];
        break;
    case csv_option_thousands:
        opts->thousands = str[0];
        break;

    case csv_option_decimal:
        opts->decimal = str[0];
        break;

    case csv_option_comment:
        opts->commentchar = str[0];
        break;

    case csv_option_doublequote:
        error = char_to_num(opts, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            opts->doublequote = (int)i64temp;
        }
        break;

    case csv_option_delim_whitespace:
        error = char_to_num(opts, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            opts->delim_whitespace = (int)i64temp;
        }
        break;

    case csv_option_quotechar:
        opts->quotechar = str[0];
        break;

    case csv_option_escapechar:
        opts->escapechar = str[0];
        break;

    case csv_option_lineterminator:
        opts->lineterminator = str[0];
        break;

    case csv_option_quoting:
        error = char_to_num(opts, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            opts->quoting = (int)i64temp;
        }
        break;

    case csv_option_sci:
        opts->sci = str[0];
        break;

    case csv_option_skip_first_N_rows:
        error = char_to_num(opts, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            opts->skip_first_N_rows = i64temp;
        }
        break;

    case csv_option_skip_empty_lines:
        error = char_to_num(opts, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            opts->skip_empty_lines = (int)i64temp;
        }
        break;

    case csv_option_skip_initial_space:
        error = char_to_num(opts, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            opts->skipinitialspace = (int)i64temp;
        }
        break;

    case csv_option_skip_footer:
        error = char_to_num(opts, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            opts->skip_footer = (int)i64temp;
        }
        break;

    case csv_option_header:
        error = char_to_num(opts, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            opts->header = (int)i64temp;
        }
        break;

    case csv_option_header_start:
        error = char_to_num(opts, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            opts->header_start = (int)i64temp;
        }
        break;

    case csv_option_header_end:
        error = char_to_num(opts, str, &p_end, &ui64temp, NULL);
        if (error == da_status_success) {
            opts->header_end = ui64temp;
        }
        break;

    case csv_option_warn_for_missing_data:
        error = char_to_num(opts, str, &p_end, &i64temp, NULL);
        if (error == da_status_success) {
            opts->warn_for_missing_data = (int)i64temp;
        }
        break;

    default:
        error = da_status_invalid_option;
    }

    return error;
}

/* Rountines for reading in a csv and storing an array of data, with or without a headings
 * row*/
da_status da_read_csv_d(da_csv_opts opts, const char *filename, double **a, size_t *nrows,
                        size_t *ncols) {
    return da_read_csv(opts, filename, a, nrows, ncols);
}

da_status da_read_csv_s(da_csv_opts opts, const char *filename, float **a, size_t *nrows,
                        size_t *ncols) {
    return da_read_csv(opts, filename, a, nrows, ncols);
}

da_status da_read_csv_int64(da_csv_opts opts, const char *filename, int64_t **a, size_t *nrows,
                            size_t *ncols) {
    return da_read_csv(opts, filename, a, nrows, ncols);
}

da_status da_read_csv_uint64(da_csv_opts opts, const char *filename, uint64_t **a,
                             size_t *nrows, size_t *ncols) {
    return da_read_csv(opts, filename, a, nrows, ncols);
}

da_status da_read_csv_uint8(da_csv_opts opts, const char *filename, uint8_t **a, size_t *nrows,
                            size_t *ncols) {
    return da_read_csv(opts, filename, a, nrows, ncols);
}

da_status da_read_csv_d_h(da_csv_opts opts, const char *filename, double **a, size_t *nrows,
                          size_t *ncols, char ***headings) {
    return da_read_csv(opts, filename, a, nrows, ncols, headings);
}

da_status da_read_csv_s_h(da_csv_opts opts, const char *filename, float **a, size_t *nrows,
                          size_t *ncols, char ***headings) {
    return da_read_csv(opts, filename, a, nrows, ncols, headings);
}

da_status da_read_csv_int64_h(da_csv_opts opts, const char *filename, int64_t **a,
                              size_t *nrows, size_t *ncols, char ***headings) {
    return da_read_csv(opts, filename, a, nrows, ncols, headings);
}

da_status da_read_csv_uint64_h(da_csv_opts opts, const char *filename, uint64_t **a,
                               size_t *nrows, size_t *ncols, char ***headings) {
    return da_read_csv(opts, filename, a, nrows, ncols, headings);
}

da_status da_read_csv_uint8_h(da_csv_opts opts, const char *filename, uint8_t **a,
                              size_t *nrows, size_t *ncols, char ***headings) {
    return da_read_csv(opts, filename, a, nrows, ncols, headings);
}