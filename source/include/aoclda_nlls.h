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

#ifndef AOCLDA_NLLS
#define AOCLDA_NLLS

/**
 * \file
 */

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

// Function callbacks

/**
 * \{
 * \brief Nonlinear data fitting call-back. Residual function signature
 * \details
 * This function evaluates the nonlinear model at the point \p x
 * and returns in \p res (of length \p n_res) the model's error residual:
 * @rst
 * .. math::
 *
 *      r_i(x) = \theta(t_i,x) - y_i,
 *
 * where :math:`\theta` is the model and the pair :math:`(t_i, y_i), i = 1,
 * \ldots, n_{res}` are the  model parameters and observations.
 * @endrst
 *
 * \param[in] n_coef number of coefficients in the model.
 * \param[in] n_res number of residuals declared.
 * \param[inout] data user data pointer, the solver does not touch this pointer and
 *            passes it along to the call-back.
 * \param[in] x the vector of coefficients (at the current iterate) and of size \p n_coef.
 * \param[out] res residual vector of size \p n_res for the model evaluated at \p x.
 * \return flag - indicating if evaluation of the model was successful, zero to
 *                inform success; nonzero informs failure, in this case the solver will
 *                terminate with \ref da_status_optimization_usrstop.
 */
typedef da_int da_resfun_t_s(da_int n_coef, da_int n_res, void *data, const float *x,
                             float *res);
typedef da_int da_resfun_t_d(da_int n_coef, da_int n_res, void *data, const double *x,
                             double *res);
/** \} */

/**
 * \{
 * \brief Nonlinear data fitting call-back. Residual Jacobian function signature
 * \details
 * This function evaluates the nonlinear model's residual first
 * derivatives (gradients) at the point \p x
 * and returns in \p jac (of size \p n_res by \p n_coef) the model's Jacobian matrix:
 * @rst
 * .. math::
 *
 *      \nabla r(x) = \left [ \nabla r_1(x), \nabla r_2(x), \ldots, \nabla r_{n_{res}}(x)\right ]^{\text{T}},
 *      \text { with }
 *      \nabla r_i(x) = \left [ \frac{\partial r_i(x)}{\partial x_1},
 *                \frac{\partial r_i(x)}{\partial x_2}, \ldots,
 *                \frac{\partial r_i(x)}{\partial x_{n_{coef}}} \right].
 *
 * @endrst
 *
 * \param[in] n_coef number of coefficients in the model.
 * \param[in] n_res number of residuals declared.
 * \param[inout] data user data pointer, the solver does not touch this pointer and
 *            passes it along to the call-back.
 * \param[in] x the vector of coefficients (at the current iterate) and of size \p n_coef.
 * \param[out] jac Jacobian matrix (\p n_res by \p n_coef), first derivatives of the residual function.
 *             evaluated at \p x. This matrix expects to be stored in the format
 *             defined by the optional parameter \p storage_scheme and defaults to
 *             row-major, this can be changed to column-major (Fortran format as well).
 * \return flag - indicating if evaluation of the model was successful, zero to
 *                inform success; nonzero informs failure, in this case the solver will
 *                terminate with \ref da_status_optimization_usrstop.
 */
typedef da_int da_resgrd_t_s(da_int n_coef, da_int n_res, void *data, float const *x,
                             float *jac);
typedef da_int da_resgrd_t_d(da_int n_coef, da_int n_res, void *data, double const *x,
                             double *jac);
/** \} */

/**
 * \{
 * \brief Nonlinear data fitting call-back. Residual Hessians function signature
 * \details
 * This function evaluates the nonlinear model's residual second
 * derivatives (Hessians) at the point \p x
 * and returns in \p hes (of size \p n_coef by \p n_coef) the matrix:
 * @rst
 * .. math::
 *
 *      H(x) = \sum_{i=1}^{n_{res}} w_{r_i}\nabla^2 r_i(x) \text { with }
 *      \nabla^2 r_i(x) = \left [ \begin{array}{ccc}
 *      \frac{\partial^2 r_i(x)}{\partial x_1 x_1} &\ldots & \frac{\partial^2 r_i(x)}{\partial x_1 x_{n_{coef}}}\\
 *      \vdots                                     & \ddots&\vdots\\
 *      \frac{\partial^2 r_i(x)}{\partial x_{n_{coef}} x_1} &\ldots & \frac{\partial^2 r_i(x)}{\partial x_{n_{coef}} x_{n_{coef}}}
 *      \end{array} \right],
 *
 *
 * where
 * :math:`w_{r_i}` is the :math:`i`-th weighted residual.
 * @endrst
 *
 * \param[in] n_coef number of coefficients in the model.
 * \param[in] n_res number of residuals declared.
 * \param[inout] data user data pointer, the solver does not touch this pointer and
 *            passes it along to the call-back.
 * \param[in] x the vector of coefficients (at the current iterate) and of size \p n_coef.
 * \param[in] wr a scaled (weighted) version of the residual vector evaluated at \p x and of size \p n_res.
 * \param[out] hes Hessian matrix (size of \p n_coef by \p n_coef), second derivatives of the residual function.
 *             evaluated at \p x. This symmetric matrix is expected to be stored
 *             in the format defined by the optional parameter \p storage_scheme
 *             and defaults to row-major, this can be changed to column-major
 *             (Fortran format as well).
 * \return flag - indicating if evaluation of the model was successful, zero to
 *                inform success; nonzero informs failure, in this case the solver will
 *                terminate with \ref da_status_optimization_usrstop.
 */
typedef da_int da_reshes_t_s(da_int n_coef, da_int n_res, void *data, float const *x,
                             float const *wr, float *hes);
typedef da_int da_reshes_t_d(da_int n_coef, da_int n_res, void *data, double const *x,
                             double const *wr, double *hes);
/** \} */

/**
 * \{
 * \brief Nonlinear data fitting call-back. Residual Hessians-vector product function signature
 * \details
 * This function evaluates the nonlinear model's residual second
 * derivatives (Hessians) at the point \p x,
 * performs a matrix-vector product with \p y and returns in \p hp (of size \p n_coef by \p n_res) the matrix:
 * @rst
 * .. math::
 *
 *      H_P(x) = \left [ \nabla^2 r_1(x)\,y,\; \nabla^2 r_2(x)\,y,\;\ldots,\;\nabla^2 r_{n_{res}}(x)\,y\right].
 *
 * @endrst
 *
 * \param[in] n_coef number of coefficients in the model.
 * \param[in] n_res number of residuals declared.
 * \param[inout] data user data pointer, the solver does not touch this pointer and
 *            passes it along to the call-back.
 * \param[in] x the vector of coefficients (at the current iterate) and of size \p n_coef.
 * \param[in] y an arbitrary vector of size \p n_coef.
 * \param[out] hp Hessians matrix-vector product with \p y.
 *             This dense matrix of size \p n_coef by \p n_res is expected to be stored
 *             in the format defined by the optional parameter \p storage_scheme
 *             and defaults to row-major, this can be changed to column-major
 *             (Fortran format as well).
 * \return flag - indicating if evaluation of the model was successful, zero to
 *                inform success; nonzero informs failure, in this case the solver will
 *                terminate with \ref da_status_optimization_usrstop.
 */
typedef da_int da_reshp_t_s(da_int n_coef, da_int n_res, const float *x, const float *y,
                            float *hp, void *data);
typedef da_int da_reshp_t_d(da_int n_coef, da_int n_res, const double *x, const double *y,
                            double *hp, void *data);
/** \} */

/**
 * \{
 * \brief Nonlinear data fitting function call-backs registration.
 * \details
 * This function registers in the nonlinear data fitting handle the residual
 * function call-backs that define the nonlinear model to train.
 * \p resfun (residual function) and \p resgrd (residual Jacobian matrix) are mandatory
 * while \p reshes (residual Hessians) and \p reshp (residual Hessians matrix-vector products) are only
 * required if the solver for the model chosen requires higher order derivatives, if these
 * are not provided and the solver requires them an error will be returned.
 *
 * @rst
 * For details on the function call-backs, see :ref:`nonlinear data fitting
 * call-back declarations<da_nlls_callbacks>`.
 * @endrst
 *
 * \param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_nlls.
 * \param[in] n_coef number of coefficients of the model.
 * \param[in] n_res number of residuals.
 * \param[in] resfun function callback to provide residual
 *             vector for the model evaluated at \p x.
 * \param[in] resgrd function callback to provide the Jacobian matrix (first derivatives) of
 *             the residual function evaluated at \p x.
 * \param[in] reshes function callback to evaluate residual Hessian matrices (second derivatives of
 *             the residual function) evaluated at \p x. Optionally can be passed as \p NULL.
 * \param[in] reshp function callback to evaluate the residual Hessians evaluated at \p x and perform
 *             for each Hessian matrix-vector product and store the resulting products into a
 *             dense matrix. Optionally can be passed as \p NULL.
 * \return \ref da_status. The function returns:
 *  - @ref da_status_success - the operation was successfully completed.
 *  - @ref da_status_handle_not_initialized - handle was not initialized properly (with @ref da_handle_nlls) or has been corrupted.
 *  - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 *  - @ref da_status_invalid_input - one or more of the input arguments are invalid.
 *
 */
da_status da_nlls_define_residuals_d(da_handle handle, da_int n_coef, da_int n_res,
                                     da_resfun_t_d *resfun, da_resgrd_t_d *resgrd,
                                     da_reshes_t_d *reshes, da_reshp_t_d *reshp);
da_status da_nlls_define_residuals_s(da_handle handle, da_int n_coef, da_int n_res,
                                     da_resfun_t_s *resfun, da_resgrd_t_s *resgrd,
                                     da_reshes_t_s *reshes, da_reshp_t_s *reshp);
/** \} */

da_status da_nlls_define_bounds_d(da_handle handle, da_int n_coef, double *lower,
                                  double *upper);
da_status da_nlls_define_bounds_s(da_handle handle, da_int n_coef, float *lower,
                                  float *upper);

da_status da_nlls_define_weights_d(da_handle handle, da_int n_res, double *weights);
da_status da_nlls_define_weights_s(da_handle handle, da_int n_res, float *weights);

da_status da_nlls_fit_d(da_handle handle, da_int n_coef, double *coef, void *udata);
da_status da_nlls_fit_s(da_handle handle, da_int n_coef, float *coef, void *udata);

/**
 * @brief Information vector containing metrics from optimization solvers
 *
 * @details
 * @rst
 * The information vector can be retrieved after a successfull return from the
 * fit function :cpp:func:`da_nlls_fit_? <da_nlls_fit_d>` by querying the handle,
 * using :cpp:func:`da_handle_get_result_? <da_handle_get_result_d>` and passing
 * the :ref:`query <extracting-results>` :cpp:enum:`da_result::da_rinfo`.
 * @endrst
 * \{
 */
typedef enum da_optim_info_t_ {
    info_objective = 0,      ///< objective value
    info_grad_norm = 1,      ///< norm of the objective gradient
    info_iter = 2,           ///< number of iterations
    info_time = 3,           ///< current time
    info_nevalf = 4,         ///< number of objective function callback evaluations
    info_inorm = 5,          ///< infinity norm of gradient
    info_inorm_init = 6,     ///< infinity norm of gradient at the initial iterate
    info_ncheap = 7,         ///< number of objective function callback
                             ///< evaluations requesting "cheap" update
    info_nevalg = 8,         ///< number of gradient callback evaluations
    info_nevalh = 9,         ///< number of Hessian callback evaluations
    info_nevalhp = 10,       ///< number of Hessian-vector callback evaluations
    info_scl_grad_norm = 11, ///< scaled gradient norm of objective

    info_number ///< for internal use
} da_optim_info_t;
/** \} */

#endif
