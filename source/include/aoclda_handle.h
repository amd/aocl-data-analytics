#ifndef AOCLDA_HANDLE
#define AOCLDA_HANDLE

#include "aoclda_error.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum da_handle_option_ {
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
    csv_option_add_skiprow,
    csv_option_warn_for_missing_data,
} da_handle_option;

typedef enum da_handle_type_ {
    da_handle_uninitialized,
    da_handle_csv_opts,
    da_handle_linreg,
} da_handle_type;

typedef struct _da_handle *da_handle;

/* Initialize da_handle struct with default values */
da_status da_handle_init_d(da_handle *handle, da_handle_type handle_type);

da_status da_handle_init_s(da_handle *handle, da_handle_type handle_type);

/* Print error information stored in the struct */
void da_handle_print_error_message(da_handle handle);

/* Check whether handle is of the correct type */
da_status da_check_handle_type(da_handle handle, da_handle_type expected_handle_type);

/* Generic option setting routine */
da_status da_handle_set_option(da_handle handle, da_handle_option option, char *str);

/* Destroy the da_handle struct */
void da_handle_destroy(da_handle *handle);

#ifdef __cplusplus
}
#endif

#endif