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

#ifndef AOCLDA_LINREG
#define AOCLDA_LINREG

/**
 * \file
 */

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Defines which linear model is computed
 **/
typedef enum linmod_model_ {
    linmod_model_undefined = 0, ///< No linear model set.
    linmod_model_mse,           ///< \f$L_2\f$ norm linear regression.
    linmod_model_logistic,      ///< Logistic regression.
} linmod_model;

/** \{
 * @brief Select which linear model to compute.
 * @rst
 * The last suffix of the function name marks the floating point precision on which the handle operates (see :ref:`precision section <da_real_prec>`).
 * @endrst
 *
 * @rst
 * The model definition can be further enhanced with elements such as a reguarization term by setting up optional parameters.
 * See the :ref:`linear model options section <linmod_options>` for more information.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_linmod.
 * @param[in] mod a @ref linmod_model enum type to select the linear model.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 */
da_status da_linmod_select_model_s(da_handle handle, linmod_model mod);
da_status da_linmod_select_model_d(da_handle handle, linmod_model mod);
/** \} */

/** \{
 * @brief Define the data to train a linear model.
 * @rst
 * The last suffix of the function name marks the floating point precision on which the handle operates (see :ref:`precision section <da_real_prec>`).
 * @endrst
 *
 * Pass pointers to a data matrix @p X containing @p n_samples observations (rows) over @p n_feat features (columns)
 * and a response vector @p y of size @p n_samples.
 *
 * Only the pointers to @p X and @p y are stored; no internal copy is made.
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_linmod.
 * @param[in] n_samples the number of observations (rows) of the data matrix @p X. Constraint: @p n_samples @f$\ge@f$ 1.
 * @param[in] n_feat the number of features (columns) of the data matrix, @p X. Constraint: @p n_feat @f$\ge@f$ 1.
 * @param[in] X the @p n_samples @f$\times@f$ @p n_feat data matrix, in column major format.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using @ref da_handle_print_error_message.
 */
da_status da_linmod_define_features_d(da_handle handle, da_int n_samples, da_int n_feat,
                                      double *X, double *y);
da_status da_linmod_define_features_s(da_handle handle, da_int n_samples, da_int n_feat,
                                      float *X, float *y);
/** \} */

/** \{
 * @brief Fit the linear model defined in the @p handle.
 *
 * Compute the linear model defined by @a da_linmod_select_model_X on the data passed by the last call to the function @a da_linmod_define_features_X.
 * @rst
 * Note that you can customize the model before using the fit function through the use of optional parameters,
 * see :ref:`this section <linmod_options>` for a list of avaliable options (e.g., the regularization terms).
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_linmod.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_incompatible_options - some of the options set are incompatible with the model defined in \p handle.
 *        You can obtain further information using @ref da_handle_print_error_message.
 * - @ref da_status_memory_error - internal memory allocation encountered a problem.
 * - @ref da_status_internal_error - an unexpected error occurred.
 */
da_status da_linmod_fit_d(da_handle handle);
da_status da_linmod_fit_s(da_handle handle);
/** \} */

/**
 * @brief Fit the linear model defined in the @p handle using a custom starting estimate for the model coefficients.
 *
 * Compute the same model as @ref da_linmod_fit_d and @ref da_linmod_fit_s, starting the fitting process with the custom values defined in \p coefs.
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_linmod.
 * @param[in] ncoefs the number of coeficients provided in coefs. It must match the number of expected coeficients for the model defined in \p handle to be taken into account.
 * @param[in] coefs the initial coefficients
 * @return da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_incompatible_options - some of the options set are incompatible with the model defined in \p handle.
 *        You can obtain further information using @ref da_handle_print_error_message.
 * - @ref da_status_memory_error - internal memory allocation encountered a problem.
 * - @ref da_status_internal_error - an unexpected error occurred.
 * \{
 */
da_status da_linmod_fit_start_d(da_handle handle, da_int ncoefs, double *coefs);
da_status da_linmod_fit_start_s(da_handle handle, da_int ncoefs, float *coefs);
/** \} */

/**
 * @brief Evaluate the model previously computed on a new set of data @p Xt.
 *
 * After a model has been fit using @ref da_linmod_fit_d or @ref da_linmod_fit_s, it can be evaluated on  a new set of data.
 * This function returns the model evaluation in the array @p predictions.
 *
 * In the case where the model chosen solves a classification problem (e.g., logistic regression), the predictions computed will be categorical.
 * For each data point i, prediction[i] will contain the index of the most likely class according to the model.
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_linmod.
 * @param nt_feat number of columns of \p Xt: the number of features of the test data. It must match the number features of the data defined in the \p handle.
 * @param nt_samples number of rows of \p Xt: the number of samples to estimate the model on.
 * @param Xt the @p nt_samples @f$\times@f$ @p nt_feat data matrix to evaluate the model on, in column major format.
 * @param predictions
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 *
 * \{
 */
da_status da_linmod_evaluate_model_d(da_handle handle, da_int nt_samples, da_int nt_feat,
                                     double *Xt, double *predictions);
da_status da_linmod_evaluate_model_s(da_handle handle, da_int nt_samples, da_int nt_feat,
                                     float *Xt, float *predictions);
/** \} */

#ifdef __cplusplus
}
#endif

#endif
