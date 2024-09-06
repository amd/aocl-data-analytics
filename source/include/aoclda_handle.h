/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * \file
 */

/**
 * @brief Enumeration defining the types of handles available
 */

// clang-format off
enum da_handle_type_ {
    da_handle_uninitialized, ///< the handle has not yet been initialized.
    da_handle_linmod, ///< @rst
                      ///< the handle is to be used with functions from the :ref:`linear models chapter <chapter_linmod>`.
                      ///< @endrst
    da_handle_pca, ///< @rst
                   ///< the handle is to be used with functions for computing the :ref:`principal component analysis <PCA_intro>`.
                   ///< @endrst
    da_handle_kmeans, ///< @rst
                      ///< the handle is to be used with functions for computing :ref:`k-means clustering <kmeans_intro>`.
                      ///< @endrst
    da_handle_decision_tree, ///< @rst
                             ///< the handle is to be used with functions for computing :ref:`decision trees <decision_forest_intro>`.
                             ///< @endrst
    da_handle_decision_forest, ///< @rst
                               ///< the handle is to be used with functions for computing :ref:`decision forests <decision_forest_intro>`.
                               ///< @endrst
    da_handle_nlls , ///< @rst
                     ///< the handle is to be used with functions from the :ref:`nonlinear data fitting chapter <chapter_nlls>`.
                     ///< @endrst
    da_handle_knn,  ///< @rst
                    ///< the handle is to be used with functions from the :ref:`k-nearest neighbors for classification <knn_intro>`.
                    ///< @endrst
};
// clang-format on

/** @brief Alias for the \ref da_handle_type_ enum. */
typedef enum da_handle_type_ da_handle_type;

/**
 * @brief
 * @rst
 * The main handle object.
 *
 * For more information on the handle structure, see the :ref:`higher-level handle description <intro_handle>`.
 * @endrst
 */
typedef struct _da_handle *da_handle;

/** \{
 * @brief Initialize a \ref da_handle with default values.
 *
 * Set up a \ref da_handle to be used with a specific chapter.
 * This function must be called before calling any functions that require a
 * valid handle.
 *
 * @rst
 * For more info on the handle structure: :ref:`higher-level handle description <intro_handle>`.
 * @endrst
 *
 * @param[in,out] handle the main data structure.
 * @param[in] handle_type the type of handle to initialize (see @ref da_handle_type).
 * @returns @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_memory_error - a memory allocation error occurred.
 * - @ref da_status_internal_error - this should not occur and indicates a memory corruption issue.
 */
da_status da_handle_init_d(da_handle *handle, da_handle_type handle_type);
da_status da_handle_init_s(da_handle *handle, da_handle_type handle_type);
/** \} */

/**
 * @brief  Print error information stored in the handle.
 *
 * Some functions store extra information about errors and
 * this function prints (to standard output) the stored error message(s).
 *
 * @param[in,out] handle the @ref da_handle structure.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_invalid_input - the handle pointer is invalid.
 */
da_status da_handle_print_error_message(da_handle handle);

/**
 * @brief Destroy the da_handle struct.
 *
 * Free all allocated memory in handle.
 *
 * This function should always be called after finishing using the handle.
 *
 * @note Memory leaks may occur if handles are not destroyed after use.
 *
 * @param[in,out] handle the main \ref da_handle structure.
 */
void da_handle_destroy(da_handle *handle);

/* The following routines are undocumented and are used internally to help the Python interfaces */
da_status da_handle_get_error_message(da_handle handle, char **message);
da_status da_handle_get_error_severity(da_handle handle, da_severity *severity);

#endif
