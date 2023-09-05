#ifndef AOCLDA_ERROR_H
#define AOCLDA_ERROR_H

/**
 * \file
 * \anchor apx_e
 * \brief Error codes
 *
 * The following error codes can be returned by any of the routines in this library:
 * - \ref da_status_success = 0 - operation was successful
 * - \ref da_status_internal_error - unexpected error occurred
 * - \ref da_status_memory_error - memory could not be allocated
 * - \ref da_status_invalid_pointer - a provided pointer was invalid
 * - \ref da_status_invalid_input - an input parameter had an invalid value
 * - \ref da_status_not_implemented - this feature has not been implemented
 * - \ref da_status_out_of_date - the data is out of date
 * - \ref da_status_wrong_type - a function was called with the wrong data type
 * - \ref da_status_overflow - numerical overflow detected
 * - \ref da_status_invalid_handle_type - invalid handle type provided
 * - \ref da_status_handle_not_initialized - handle was not initialized properly or is corrupted
 * - \ref da_status_invalid_option - invalid option detected
 * - \ref da_status_incompatible_options - incompatible option detected
 *
 * Additional error exits specific to particular routines are detailed in the individual routine documentation.

 */

/**
 * \brief This enumeration describes all the possible return values from the exposed APIs.
 */
typedef enum da_status_ {
    // Common errors 0-99
    da_status_success = 0,               ///< Operation was successful
    da_status_internal_error,            ///< Unexpected error occurred
    da_status_memory_error,              ///< Memory could not be allocated
    da_status_invalid_pointer,           ///< A provided pointer was invalid
    da_status_invalid_input,             ///< An input parameter had an invalid value
    da_status_not_implemented,           ///< Feature not implemented
    da_status_out_of_date,               ///< The data is out of date
    da_status_wrong_type,                ///< A function called with the wrong data type
    da_status_overflow,                  ///< Numerical Overflow detected
    da_status_invalid_handle_type,       ///< Invalid handle type provided
    da_status_handle_not_initialized,    ///< Handle was not initialized properly or is corrupted
    da_status_store_not_initialized,     ///< Store was not initialized properly or is corrupted
    da_status_invalid_option,            ///< Invalid option detected
    da_status_incompatible_options,      ///< Incompatible option detected
    da_status_operation_failed,          ///< Signal that an internal operation failed (could not solve a hard linear system as part of a larger process)
    da_status_invalid_leading_dimension, ///< Invalid leading dimension for a 2D array
    da_status_negative_data,             ///< The function expected positive data but a negative entry was found
    da_status_invalid_array_dimension,   ///< The size of an array was too small
    da_status_unknown_query,             ///< The result queried cannot be found in the handle

    // CSV errors 100-199
    da_status_file_reading_error = 100, ///< An error occured when reading a CSV file
    da_status_parsing_error,      ///< An error occured when parsing a CSV file into numeric data
    da_status_missing_data, ///< The array returned from reading a CSV file contains missing data

    // linreg errors 200-299
    // PCA errors 300-399
    da_status_no_data = 300,

    // Options errors 400-499
    da_status_option_not_found = 400, ///< Option not found
    da_status_option_locked,          ///< Cannot change option value at this point
    da_status_option_wrong_type,      ///< Wrong option type passed
    da_status_option_invalid_bounds,  ///< Option value is out-of-bounds
    da_status_option_invalid_value,   ///< Cannot set option to an invalid value

    // Optimization solvers 500-599
    da_status_optimization_usrstop = 500, ///< User requested to stop optimization process
    da_status_optimization_num_difficult, ///< Numerical difficulties encoutered during the optimization process
    da_status_optimization_infeasible,    ///< The problem is infeasible, the bounds describe an empty domain
    da_status_optimization_empty_space,   ///< No variables are defined in the problem
    da_status_optimization_maxit,         ///< Iteration limit reached without converging
    da_status_optimization_maxtime,       ///< Time limit reached without converging

    // datastores 600-699
    da_status_missing_block = 600, ///< The store is missing a block, the requested operation cannot be performed
} da_status;
#endif
