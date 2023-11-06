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

#ifndef AOCLDA_ERROR_H
#define AOCLDA_ERROR_H

/**
 * \file
 */

/**
 * \brief This enumeration describes all the possible return values from AOCL-DA APIs.
 */
typedef enum da_status_ {
    // Common errors 0-99
    da_status_success = 0,         ///< Operation was successful
    da_status_internal_error,      ///< Unexpected error occurred
    da_status_memory_error,        ///< Memory could not be allocated
    da_status_invalid_pointer,     ///< A provided pointer was invalid
    da_status_invalid_input,       ///< An input parameter had an invalid value
    da_status_not_implemented,     ///< Feature not implemented
    da_status_out_of_date,         ///< The data is out of date
    da_status_wrong_type,          ///< A function called with the wrong data type
    da_status_overflow,            ///< Numerical Overflow detected
    da_status_invalid_handle_type, ///< Invalid handle type provided
    da_status_handle_not_initialized, ///< Handle was not initialized properly or is corrupted
    da_status_store_not_initialized, ///< Store was not initialized properly or is corrupted
    da_status_invalid_option,        ///< Invalid option detected
    da_status_incompatible_options, ///< Incompatible option detected
    da_status_operation_failed, ///< Signal that an internal operation failed (could not solve a hard linear system as part of a larger process)
    da_status_invalid_leading_dimension, ///< Invalid leading dimension for a 2D array
    da_status_negative_data, ///< The function expected positive data but a negative entry was found
    da_status_invalid_array_dimension, ///< The size of an array was too small
    da_status_unknown_query,    ///< The result queried cannot be found in the handle
    da_status_incorrect_output, ///< Wrong output
    da_status_no_data,          ///< No data was found

    // CSV errors 100-199
    da_status_file_reading_error = 100, ///< An error occurred when reading a CSV file
    da_status_parsing_error, ///< An error occurred when parsing a CSV file into numeric data
    da_status_missing_data, ///< The array returned from reading a CSV file contains missing data

    // linreg errors 200-299
    // PCA errors 300-399

    // Options errors 400-499
    da_status_option_not_found = 400, ///< Option not found
    da_status_option_locked,          ///< Cannot change option value at this point
    da_status_option_wrong_type,      ///< Wrong option type passed
    da_status_option_invalid_bounds,  ///< Option value is out-of-bounds
    da_status_option_invalid_value,   ///< Cannot set option to an invalid value

    // Optimization solvers 500-599
    da_status_optimization_usrstop = 500, ///< User requested to stop optimization process
    da_status_optimization_num_difficult, ///< Numerical difficulties encountered during the optimization process
    da_status_optimization_infeasible, ///< The problem is infeasible, the bounds describe an empty domain
    da_status_optimization_empty_space, ///< No variables are defined in the problem
    da_status_optimization_maxit,       ///< Iteration limit reached without converging
    da_status_optimization_maxtime,     ///< Time limit reached without converging

    // datastores 600-699
    da_status_missing_block =
        600, ///< The store is missing a block, the requested operation cannot be performed
    da_status_full_extraction, ///< No selection was defined, the full store is being extracted
} da_status;
#endif
