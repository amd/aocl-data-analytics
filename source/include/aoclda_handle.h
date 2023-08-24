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
 * @brief Enumeration defining the types of handles available
 */
typedef enum da_handle_type_ {
    da_handle_uninitialized, ///< handle is not initialized,
    da_handle_linmod, ///< @rst
                      ///< handle is to be used with functions from the :ref:`Linear Models chapter <chapter_linmod>`
                      ///< @endrst
    da_handle_pca, ///< handle is to be used with functions from the Matrix decomposition chapter TODO ADD LINK.
    da_handle_decision_tree ///< handle is to be used with functions from the Decision Forests chapter TODO ADD LINK.
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
 * @brief  Print error information stored in the struct.
 *
 * Print the last error message stored in the handle.
 * Some functions store extra information about the last error.
 * The function prints (to standard output) the stored error message.
 *
 * @param[in,out] handle The main data structure.
 * @return void.
 **/
void da_handle_print_error_message(da_handle handle);

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
