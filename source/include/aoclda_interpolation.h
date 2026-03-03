/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AOCLDA_INTERPOLATION
#define AOCLDA_INTERPOLATION

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

/**
 * \file
 */

/**
 * \brief Interpolation model type
 **/
enum da_interpolation_model_ {
    interpolation_unset = 0,    ///< Interpolation model not set.
    interpolation_cubic_spline, ///< Cubic spline interpolation.
};

/** @brief Alias for the \ref interpolation_model_ enum. */
typedef enum da_interpolation_model_ da_interpolation_model;

/** \{
 * \brief Select the interpolation model.
 *
 * Choose the type of interpolation model to use.
 * This function must be called before setting interpolation sites. At the moment, only cubic spline is supported ( \ref interpolation_cubic_spline).
 *
 * The interpolation model can only be set once. After calling this function, the model cannot be changed.
 *
 * \param[inout] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?" with type \ref da_handle_interpolation.
 * \param[in] model an @ref da_interpolation_model enum type to select the interpolation model (only \ref da_interpolation_cubic_spline is supported).
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_handle_type - the handle was not initialized with da_handle_interpolation.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_handle_not_initialized - the handle has not been initialized.
 * - \ref da_status_invalid_input - the model has already been set.
 */
da_status da_interpolation_select_model_d(da_handle handle, da_interpolation_model model);
da_status da_interpolation_select_model_s(da_handle handle, da_interpolation_model model);
/** \} */

/** \{
 * \brief Set interpolation sites with custom @f$x@f$ coordinates.
 *
 * Define the sites \p x[i] where the function is sampled.
 * The @f$x@f$ coordinates must be sorted in increasing order, with no repeats.
 *
 * The pointer to the @f$x@f$ coordinates array is not copied internally, the memory must remain valid during the call to the interpolation functions.
 *
 * \param[inout] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?" with type \ref da_handle_interpolation.
 * \param[in] n_sites the number of interpolation sites. Constraint: \p n_sites @f$\ge@f$ 2.
 * \param[in] x array of x-coordinates of the interpolation sites, of size \p n_sites.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_handle_type - the handle was not initialized with da_handle_interpolation.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or \p x is null.
 * - \ref da_status_invalid_input - the interpolation model has not been selected, \p n_sites < 2, or \p x is not sorted in increasing order.
 */
da_status da_interpolation_set_sites_d(da_handle handle, da_int n_sites, const double *x);

da_status da_interpolation_set_sites_s(da_handle handle, da_int n_sites, const float *x);
/** \} */

/** \{
 * \brief Set interpolation sites with uniform spacing.
 *
 * Define the interpolation sites with uniformly spaced @f$x@f$ coordinates in the interval [ \p x_start, \p x_end].
 *
 * \param[inout] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?" with type \ref da_handle_interpolation.
 * \param[in] n_sites the number of interpolation sites. Constraint: \p n_sites @f$\ge@f$ 2.
 * \param[in] x_start the starting @f$x@f$-coordinate.
 * \param[in] x_end the ending @f$x@f$-coordinate. Constraint: \p x_end @f$>@f$ \p x_start.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_handle_type - the handle was not initialized with the correct type.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized.
 * - \ref da_status_invalid_input - the interpolation model has not been selected, \p n_sites < 2, or \p x_start >= \p x_end.
 */
da_status da_interpolation_set_sites_uniform_d(da_handle handle, da_int n_sites,
                                               double x_start, double x_end);

da_status da_interpolation_set_sites_uniform_s(da_handle handle, da_int n_sites,
                                               float x_start, float x_end);
/** \} */

/** \{
 * \brief Set interpolation data (y-values or derivatives).
 *
 * Define the y-values or first derivatives at the interpolation sites previously set using
 * \ref da_interpolation_set_sites_s "da_interpolation_set_sites_?" or
 * \ref da_interpolation_set_sites_uniform_s "da_interpolation_set_sites_uniform_?".
 *
 * Multiple dimensions can be set at once by specifying \p dim > 1.
 * The values are provided in a 2D array \p y_data of size ( \p n x \p dim), stored in the memory layout
 * defined by the optional parameter "storage order" (default is column-major).
 *
 * \param[inout] handle a \ref da_handle object with interpolation sites already set.
 * \param[in] n the number of values. Must match n_sites previously defined by a call to \ref da_interpolation_set_sites_s "da_interpolation_set_sites_?" or \ref da_interpolation_set_sites_uniform_s "da_interpolation_set_sites_uniform_?".
 * \param[in] dim the dimensionality of the values.
 * \param[in] y_data array of values (function or derivative) at the interpolation sites, of size ( \p n x \p dim). By default, it should be stored in column-major order, unless you have set the storage order option to row-major.
 * \param[in] ldy_data the leading dimension of the values array. Constraint: \p ldy_data @f$\ge@f$ \p n if stored in column-major order, or \p ldy_data @f$\ge@f$ \p dim if stored in row-major order.
 * \param[in] order the order of derivative: 0 for function values (default), 1 for first derivatives.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_handle_type - the handle was not initialized with the correct type.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_leading_dimension - invalid leading dimension \p ldy_data.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or \p y_data is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value.
 */
da_status da_interpolation_set_values_d(da_handle handle, da_int n, da_int dim,
                                        const double *y_data, da_int ldy_data,
                                        da_int order);

da_status da_interpolation_set_values_s(da_handle handle, da_int n, da_int dim,
                                        const float *y_data, da_int ldy_data,
                                        da_int order);
/** \} */

/** \{
 * \brief Search for cells containing query points.
 *
 * For each evaluation point \p x_eval[i], find the cell index j such that
 * \p x_eval[i] is in the interval [x_sites[j], x_sites[j+1]].
 *
 * \param[inout] handle a \ref da_handle object with interpolation sites already set.
 * \param[in] n_eval the number of evaluation points. Constraint: \p n_eval @f$\ge@f$ 1.
 * \param[in] x_eval array of @f$x@f$ coordinates to evaluate, of size \p n_eval.
 * \param[out] cells array to store cell indices, of size \p n_eval.
 *              For each i, \p cells[i] is the index j such that \p x_eval[i] is in [x_sites[j], x_sites[j+1]].
 *              Out-of-bounds points are clamped to the first (0) or last (n_sites-2) cell.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_handle_type - the handle was not initialized with the correct type.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or \p x_eval or \p cells is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value.
 */
da_status da_interpolation_search_cells_d(da_handle handle, da_int n_eval,
                                          const double *x_eval, da_int *cells);

da_status da_interpolation_search_cells_s(da_handle handle, da_int n_eval,
                                          const float *x_eval, da_int *cells);
/** \} */

/** \{
 * \brief Compute cubic spline interpolation.
 *
 * Compute the selected interpolation model using the sites and values previously set.
 * The spline type is determined by the "cubic spline type" option.
 *
 * \param[inout] handle a \ref da_handle object with interpolation sites and values already set.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_handle_type - the handle was not initialized with the correct type.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized.
 * - \ref da_status_out_of_date - one of the required data (sites or values) has not been set.
 * - \ref da_status_internal_error - an unexpected error occurred.
 */
da_status da_interpolation_interpolate_d(da_handle handle);

da_status da_interpolation_interpolate_s(da_handle handle);
/** \} */

/** \{
 * \brief Set boundary conditions for cubic spline interpolation.
 *
 * Define custom boundary conditions for the cubic spline interpolation.
 * This function sets the derivative orders and values at the left and right boundaries.
 *
 * The boundary condition orders can be 1 (first derivative) or 2 (second derivative).
 * The same order is applied to all dimensions, but different values can be provided for each dimension.
 *
 * \param[inout] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?" with type \ref da_handle_interpolation.
 * \param[in] dim the number of dimensions. Must match the dimension set with \ref da_interpolation_set_values_s "da_interpolation_set_values_?".
 * \param[in] left_order the order of the derivative at the left boundary (1 for first derivative or 2 for second derivative). Constraint: \p left_order must be 1 or 2.
 * \param[in] left_values array of derivative values at the left boundary, of size \p dim.
 * \param[in] right_order the order of the derivative at the right boundary (1 for first derivative or 2 for second derivative). Constraint: \p right_order must be 1 or 2.
 * \param[in] right_values array of derivative values at the right boundary, of size \p dim.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_handle_type - the handle was not initialized with the correct type.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or \p left_values or \p right_values is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value.
 */
da_status da_interpolation_set_boundary_conditions_d(da_handle handle, da_int dim,
                                                     da_int left_order,
                                                     const double *left_values,
                                                     da_int right_order,
                                                     const double *right_values);

da_status da_interpolation_set_boundary_conditions_s(da_handle handle, da_int dim,
                                                     da_int left_order,
                                                     const float *left_values,
                                                     da_int right_order,
                                                     const float *right_values);
/** \} */

/** \{
 * \brief Evaluate the cubic spline at given points.
 *
 * Evaluate the computed cubic spline (or its derivatives) at a set of query points.
 * The spline must be computed using \ref da_interpolation_interpolate_d "da_interpolation_interpolate_?"
 * before calling this function.
 *
 * The \p order parameter specifies which derivative to evaluate:
 * - 0: function value (default)
 * - 1: first derivative
 * - 2: second derivative
 * - 3: third derivative
 *
 * Multiple orders can be evaluated at once by providing an array of orders of size > 1.
 *
 * \param[inout] handle a \ref da_handle object with a computed spline.
 * \param[in] n_eval the number of evaluation points. Constraint: \p n_eval @f$\ge@f$ 1.
 * \param[in] x_eval array of @f$x@f$ coordinates where to evaluate the spline, of size \p n_eval.
 * \param[out] y_eval array to store the evaluated values, of size \p n_eval * \p n_orders * \p dim.
 * \param[in] n_orders the number of derivative orders to evaluate. Constraint: \p n_orders @f$\ge@f$ 1.
 * \param[in] orders array of derivative orders to evaluate (0=function, 1=1st derivative, 2=2nd derivative, 3=3rd derivative), of size \p n_orders.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_handle_type - the handle was not initialized with the correct type.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or \p x_eval, \p y_eval, or \p orders is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value.
 * - \ref da_status_out_of_date - the spline has not been computed yet.
 */
da_status da_interpolation_evaluate_d(da_handle handle, da_int n_eval,
                                      const double *x_eval, double *y_eval,
                                      da_int n_orders, da_int *orders);

da_status da_interpolation_evaluate_s(da_handle handle, da_int n_eval,
                                      const float *x_eval, float *y_eval, da_int n_orders,
                                      da_int *orders);
/** \} */

#endif // AOCLDA_INTERPOLATION
