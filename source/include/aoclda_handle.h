#ifndef AOCLDA_HANDLE
#define AOCLDA_HANDLE

#include "aoclda_error.h"
#include "aoclda_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \file
 *  \anchor apx_a
 *  \brief Appendix A - Handles
 *
 *  Description about the handle system used in the library TODO
 *  \section handle_system Handle system?
 *  TODO
 *  \section handle_details Details
 *  TODO
 */

/**
 * \brief Enumaration defining the types of handles available
 */
typedef enum da_handle_type_ {
    da_handle_uninitialized, ///< handle is not initialized,
    da_handle_csv_opts,      ///< handle is to be used with data import functions,
    da_handle_linmod, ///< handle is to be used with functions from the Linear Models chapter TODO ADD LINK.
    da_handle_pca ///< handle is to be used with functions from the Linear Models chapter TODO ADD LINK.
} da_handle_type;

/**
 * \brief Main handle object
 */
typedef struct _da_handle *da_handle;

/** \{
 * \anchor da_handle_init
 * \brief Initialize da_handle struct with default values.
 *
 * Set up handle to be used with a specific chapter, see \ref apx_a.
 *
 * This function must be called before calling any functions that require a
 * valid handle.
 *
 * \param[in,out] handle a handle, see \ref da_handle.
 * \param[in] handle_type type of handle to intialize use \ref da_handle_type.
 * \return \ref da_status_.
 *
 * \section da_handle_init_return Return
 * The function returns
 *
 * * \ref da_status_success - operation was successfully completed,
 * * da_status_        - on failure TODO complete.
 */
da_status da_handle_init_d(da_handle *handle, da_handle_type handle_type);
da_status da_handle_init_s(da_handle *handle, da_handle_type handle_type);
/** \} */

/**
 * \brief  Print error information stored in the struct.
 *
 * Print the last error message stored in the handle.
 * Some functions store extra information about the last error.
 * The function prints (to standard output) the stored error message.
 *
 * \param[in,out] handle a handle, see \ref da_handle.
 * \return void.
 *
 * \section da_handle_print_error_return Return
 * This function does not return any value.
 *
 **/
void da_handle_print_error_message(da_handle handle);

/**
 * \brief Check whether the handle is of the correct type.
 *
 * TODO add description
 * \param[in,out] handle a handle, see \ref da_handle.
 * \param[in] expected_handle_type type of handle to check for, see \ref da_handle_type.
 * \return \ref da_status_.
 *
 * \section da_check_handle_return Return
 * * \ref da_status_success - Handle type matches with expected type,
 * * TODO add others.
 * */
da_status da_check_handle_type(da_handle handle, da_handle_type expected_handle_type);

/**
 * \brief Destroy the da_handle struct.
 *
 * Free all allocated memory in handle.
 *
 * This function should always be called after finishing using the handle.
 *
 * \note memory leaks may occur if handles are not destroyed after use.
 *
 * \param[in,out] handle a handle, see \ref da_handle.
 * \return void.
 * \section da_handle_destroy_return Return
 * This function does not return any value.
 * */
void da_handle_destroy(da_handle *handle);

#ifdef __cplusplus
}
#endif

#endif
