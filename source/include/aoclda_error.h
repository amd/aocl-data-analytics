#ifndef AOCLDA_ERROR_H
#define AOCLDA_ERROR_H

typedef enum da_status_ {
    // Common errors 0-99
    da_status_success = 0,
    da_status_memory_error,
    da_status_invalid_pointer,
    da_status_invalid_input,
    da_status_not_implemented,
    da_status_out_of_date,
    da_status_wrong_type,
    da_status_overflow,
    da_status_invalid_handle_type,
    da_status_handle_not_initialized,
    da_status_invalid_option,

    // CSV errors 100-199
    da_status_file_not_found = 100,
    da_status_range_error,
    da_status_no_digits,
    da_status_invalid_chars,
    da_status_invalid_boolean,
    da_status_sign_error,
    da_status_file_reading_error,
    da_status_parsing_error,
    da_status_ragged_csv,
    da_status_warn_bad_lines,
    da_status_warn_missing_data,
    da_status_warn_no_data,

    // linreg errors 200-299
    // PCA errors 300-399
} da_status;
#endif