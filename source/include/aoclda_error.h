#ifndef AOCLDA_ERROR_H
#define AOCLDA_ERROR_H

/**
 * \file
 * \anchor apx_e
 * \brief Appendix E - Error codes
 */

/**
 * \brief This enumeration describes all the possible return values from the exposed APIs.
 */
typedef enum da_status_ {
    // Common errors 0-99
    da_status_success = 0,         ///< Operation was successful
    da_status_internal_error,      ///< Unexpected error occurred
    da_status_memory_error,        ///< Memory could not be allocated
    da_status_invalid_pointer,     ///< A provided pointer was invalid
    da_status_invalid_input,       ///< An input parameter had an invalid value
    da_status_not_implemented,     ///< Feature not implemented
    da_status_out_of_date,         ///< ???
    da_status_wrong_type,          ///< A function called with the wrong data type
    da_status_overflow,            ///< Numerical Overflow detected
    da_status_invalid_handle_type, ///< Invalid handle type provided
    da_status_handle_not_initialized, ///< Handle was not initialized properly or is corrupted
    da_status_invalid_option,         ///< Invalid option detected
    da_status_incompatible_options, ///< Incompatible option detected

    // CSV errors 100-199
    da_status_file_not_found = 100, ///< ?
    da_status_range_error,          ///< ?
    da_status_no_digits,            ///< ?
    da_status_invalid_chars,        ///< ?
    da_status_invalid_boolean,      ///< ?
    da_status_sign_error,           ///< ?
    da_status_file_reading_error,   ///< ?
    da_status_parsing_error,        ///< ?
    da_status_ragged_csv,           ///< ?
    da_status_warn_bad_lines,       ///< ?
    da_status_warn_missing_data,    ///< ?
    da_status_warn_no_data,         ///< ?

    // linreg errors 200-299
    // PCA errors 300-399
    // Options errors 400-499
    da_status_option_not_found = 400, ///< Option not found
    da_status_option_locked,          ///< Cannot change option value at this point
    da_status_option_wrong_type,      ///< Wrong option type passed
    da_status_option_invalid_bounds,  ///< Option value is out-of-bounds
    da_status_option_invalid_value,   ///< Cannot set option to an invalid value
} da_status;
#endif
