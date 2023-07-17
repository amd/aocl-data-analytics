#ifndef AOCLDA_RESULT
#define AOCLDA_RESULT

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

#ifdef __cplusplus
extern "C" {
#endif
/* header file for public API to extract data from a handle 
 */

/** \file
 *  \anchor getresult
 *  \brief Extracting results from \ref handles 
 *
 *  These functions extract data stored in the handle by any given solver.
 * 
 *  Some functions in the library store in the handle relevant data produced during the
 *  solving phase of a given problem. For example, while performing a linear regression,
 *  the solver will store in the handle the optimal fitting coefficients as well as other
 *  data of interest. 
 *
 */

/**
 * \brief Enumeration defining the type object (result)
 * to query or extract from a handle
 */
enum da_result {
    // General purpose data 1..100
    da_rinfo =
        1, ///< General information array, containing a varierty of metrics, see each solver's details for further information. Each solver stores different information into this array.
    // Linear models 101..200
    da_linmod_coeff =
        101, ///< Optimal fitted coefficients produced by the last call to a linear regression solver.
    // PCA 201..300
    da_pca_scores = 201, ///< PCA scores
    // Nonlinear Optimization 301..400
    // Random Forrests 401..500
    // ...
};

/** \{
 * @brief Get results stored in handle
 * 
 * Some solvers will store relevant data in the handle. These function provide means to extract it.
 * To check the available data stored by a given API check its associated documentation.
 * 
 * @param handle - a valid handle used to call any of the solvers.
 * @param query  - the data of interest, see \ref da_result, if the queried data is not avaible or not found in the \p handle then the function returns \ref da_status_unknown_query.
 * @param dim - the size of the array \p result, in the case where the provided size is too small, after the call to this API, \p dim will contain the correct size and the return status will be \ref da_status_invalid_array_dimension.
 * @param result - location of an array where to store the queried data.
 * 
 * @return da_status flag indicating the failure or success of the call.
 * 
 * @retval aoclsparse_status_success The operation completed successfully.
 * @retval aoclsparse_status_unknown_query Indicates that the requested \p query 
 *         result is either not available or not found in the \p handle. 
 *         This can happen in the following scenarios.
 *         Querying a result before actually performing the operation, for instance 
 *         when querying the coefficient of a linear regression before actually performing a 
 *         sucessful fit. Also when, e.g. querying the principal components to a handle 
 *         initialized for linear models. In this case the handle only contains information 
 *         relevan to linear regression solvers.
 * @retval da_status_wrong_type Indicates that the data types used to initialize the 
 *         \p handle and \p result don't match.
 * @retval da_status_handle_not_initialized Indicates that the \p handle has not been initialized or is corrupted.
 * @retval da_status_invalid_array_dimension Indicates that the size \p dim of the \p result 
 *         vector is too small. After the call \p dim contains the correct size. 
 *         Call the function again with this new size.
 * @retval da_status_invalid_pointer Indicates that the pointer to \p handle is invalid.
 */
da_status da_handle_get_result_d(const da_handle handle, da_result query, da_int *dim,
                                 double *result);

da_status da_handle_get_result_s(const da_handle handle, da_result query, da_int *dim,
                                 float *result);

da_status da_handle_get_result_int(const da_handle handle, da_result query, da_int *dim,
                                   da_int *result);
/* \} */

#ifdef __cplusplus
}
#endif

#endif