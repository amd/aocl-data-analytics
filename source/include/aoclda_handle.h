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

#ifndef AOCLDA_HANDLE
#define AOCLDA_HANDLE

#include "aoclda_error.h"
#include "aoclda_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file
 */

/**
 * @brief Enumeration defining the types of handles available
 */
typedef enum da_handle_type_ {
    da_handle_uninitialized, ///< handle is not initialized,
    da_handle_linmod,        ///< @rst
    ///< handle is to be used with functions from the :ref:`Linear Models chapter <chapter_linmod>`
    ///< @endrst
    da_handle_pca, ///< handle is to be used with functions from the Matrix decomposition chapter TODO ADD LINK.
    da_handle_decision_tree,  ///< handle is to be used with functions from the Decision Forests chapter TODO ADD LINK.
    da_handle_decision_forest ///< handle is to be used with functions from the Decision Forests chapter TODO ADD LINK.
} da_handle_type;

/**
 * @brief
 * @rst
 * Main handle object. For more info on the handle structure: :ref:`Handle description <intro_handle>`
 * @endrst
 */
typedef struct _da_handle *da_handle;

/** \{
 * @brief Initialize da_handle struct with default values.
 * Set up handle to be used with a specific chapter.
 * This function must be called before calling any functions that requires a
 * valid handle.
 *
 * @rst
 * For more info on the handle structure: :ref:`Handle description <intro_handle>`
 * @endrst
 *
 * @param[in,out] handle The main data structure.
 * @param[in] handle_type type of handle to intialize use @ref da_handle_type.
 * @returns @ref da_status
 * - @ref da_status_success operation was successfully completed.
 * - TODO write other error codes
 */
da_status da_handle_init_d(da_handle *handle, da_handle_type handle_type);
da_status da_handle_init_s(da_handle *handle, da_handle_type handle_type);
/** \} */

/**
 * @brief  Print error information stored in the handle.
 *
 * Print (trace of) error message(s) stored in the handle.
 * Some functions store extra information about errors and
 * this function prints (to standard output) the stored error message(s).
 *
 * @param[in,out] handle The handle structure.
 * @return \ref da_status_success on success and \ref da_status_invalid_input 
 *         if the handle pointer is invalid.
 */
da_status da_handle_print_error_message(da_handle handle);

/**
 * @brief Check whether the handle is of the correct type.
 *
 * TODO add description
 * @param[in,out] handle The main data structure.
 * @param[in] expected_handle_type type of handle to check for.
 *
 * @returns @ref da_status
 * - @ref da_status_success Handle type matches with expected type.
 * - @retval TODO add others.
 */
da_status da_check_handle_type(da_handle handle, da_handle_type expected_handle_type);

/**
 * @brief Destroy the da_handle struct.
 *
 * Free all allocated memory in handle.
 *
 * This function should always be called after finishing using the handle.
 *
 * @note memory leaks may occur if handles are not destroyed after use.
 *
 * @param[in,out] handle The main data structure.
 * @return void.
 */
void da_handle_destroy(da_handle *handle);

#ifdef __cplusplus
}
#endif

#endif
