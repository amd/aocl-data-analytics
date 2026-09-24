# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#' Nonlinear data fitting.
#'
#' This class defines a model and solves the problem
#'
#' minimize F(x) = 1/2 sum_{i=0}^{n_res-1} r_i(x)^2_W + sigma/p ||x||_2^p
#' subject to x in R^{n_coef}
#'
#' where
#' \itemize{
#'   \item x is the n_coef-vector of coefficients (parameters) to be trained,
#'   \item r_i(x) is the n_res-vector of model residuals evaluated at x,
#'   \item sigma > 0, and p=2,3 are the regularization hyperparameters, and
#'   \item W is a vector of the diagonal elements of the weight norm matrix.
#' }
#'
#' @param n_coef Number of coefficients in the model. Must be positive.
#' @param n_res Number of residuals in the model. Must be positive.
#' @param weights Vector containing the values of the diagonal weight
#'   matrix W. It is expected that the values are all non-negative and normalized.
#'   Default = NULL.
#' @param lower_bounds Vector of lower bounds of the coefficient vector.
#'   Default = NULL.
#' @param upper_bounds Vector of upper bounds of the coefficient vector.
#'   Default = NULL.
#' @param model Choose the nonlinear model to use. Default = 'hybrid'.
#'   'gauss-newton' Gauss-Newton method,
#'   'quasi-newton' Quasi-Newton method,
#'   'hybrid' uses a strategy that enables switching between the Gauss-Newton
#'   and Quasi-Newton methods,
#'   'tensor-newton' higher order method that uses Hessians. This option
#'   requires call-backs for second order derivative Hessians.
#' @param method Optimization solver for the subproblem. Default = 'galahad'.
#'   Valid option values for when glob_strategy = 'tr':
#'   'galahad' GALAHAD's DTRS,
#'   'more-sorensen' variant of More-Sorensen's method,
#'   'powell-dogleg' Powell's dog-leg method,
#'   'aint' Generalized eigen value method.
#'   Valid option values for when glob_strategy = 'reg':
#'   'galahad' GALAHAD's DRQS,
#'   'linear solver' solve step system using a linear solver.
#'   Refer to da_ralfit for details on these subproblem solvers.
#' @param glob_strategy Globalization strategy. Only relevant when model =
#'   'hybrid', 'gauss-newton', or 'quasi-newton'. Default = 'tr'.
#'   'tr' uses trust-region methods,
#'   'reg' uses regularization techniques.
#' @param reg_power Regularization power (p). Default = 'quadratic'.
#'   This option is only relevant if the regularization term (parameter reg_term
#'   in \code{$fit()}) is positive.
#'   'quadratic' uses second order regularization,
#'   'cubic' uses third order regularization.
#' @param check_derivatives Verify the user-supplied Jacobian derivatives using
#'   finite-differences method.
#'   'no' will not perform any verifications.
#'   'yes' will trigger the verification on each element of the Jacobian matrix.
#'   The verification requires one call to the residual function for each column
#'   of the Jacobian. The verification process will print one line for each
#'   Jacobian entry that is deemed to be incorrect in a similar format to
#'   Jac[2,6] = 1.28402 ~ 1.582 [2.097E-01], (0.250E-06) XT
#'   informing with an X that the entry is likely to be wrong. It also informs
#'   with a T that the entry at Jac[2,6] matches the correct value in the
#'   transposed Jacobian, indicating that the storage scheme should be reviewed.
#'   If the verbose level is high then one line for each entry of the whole
#'   Jacobian is printed regardless of its correctness.
#'   The finite-difference and derivative checker machinery is controlled via
#'   two arguments of the fit function: fd_step which defines the overall step
#'   size for the finite-difference approximation, and fd_ttol the derivative
#'   test tolerance which sets the threshold for deciding if a user-supplied
#'   derivative is close to the finite-difference approximation.
#'   If false positives are reported by the derivative checker or if the
#'   optimization process fails, then trialing different values for this
#'   parameter may help.
#' @param verbose Set verbosity level (0 to 3) of the solver. Default = 0.
#' @param check_data Whether to check the data for NaNs. Default = FALSE.
#' @export
#' @examples
#' # Fit an exponential model: y = a * exp(b * t) + noise
#' set.seed(42)
#' t_obs <- seq(0, 1, length.out = 20)
#' a_true <- 2.0; b_true <- -1.5
#' y_obs <- a_true * exp(b_true * t_obs) + rnorm(20, sd = 0.05)
#'
#' # Residual function: returns r_i = model(x, t_i) - y_i
#' res_fun <- function(x, data) {
#'   data$y - x[1] * exp(x[2] * data$t)
#' }
#'
#' # Jacobian function: returns n_res x n_coef matrix
#' jac_fun <- function(x, data) {
#'   J <- matrix(0, nrow = length(data$t), ncol = 2)
#'   J[, 1] <- -exp(x[2] * data$t)
#'   J[, 2] <- -x[1] * data$t * exp(x[2] * data$t)
#'   J
#' }
#'
#' model <- nlls$new(n_coef = 2, n_res = 20)
#' x0 <- c(1.0, -1.0)
#' data_obj <- list(t = t_obs, y = y_obs)
#' result <- model$fit(x0, fun = res_fun, jac = jac_fun, data = data_obj)
#' cat("Fitted coefficients:", result, "\n")
nlls <- R6::R6Class("nlls",
    private = list(
        .nlls = NULL,
        .n_coef = NULL,
        .n_res = NULL,
        .weights = NULL,
        .lower_bounds = NULL,
        .upper_bounds = NULL,
        .model = NULL,
        .method = NULL,
        .glob_strategy = NULL,
        .reg_power = NULL,
        .check_derivatives = NULL,
        .verbose = NULL,
        .check_data = NULL,
        .coef = NULL
    ),
    public = list(

        #' @description Create a new nlls object.
        initialize = function(n_coef, n_res,
                              weights = NULL, lower_bounds = NULL, upper_bounds = NULL,
                              model = "hybrid", method = "galahad",
                              glob_strategy = "tr", reg_power = "quadratic",
                              check_derivatives = "no", verbose = 0L,
                              check_data = FALSE) {
            private$.n_coef <- as.integer(n_coef)
            private$.n_res <- as.integer(n_res)

            if (!is.null(weights)) {
                if (is.matrix(weights)) stop("weights must be a 1D vector, not a matrix")
                if (length(weights) == 0) stop("weights must not be an empty vector")
                private$.weights <- check_convert_data(weights, "float", "vector")
            }
            if (!is.null(lower_bounds)) {
                if (is.matrix(lower_bounds)) stop("lower_bounds must be a 1D vector, not a matrix")
                if (length(lower_bounds) == 0) stop("lower_bounds must not be an empty vector")
                private$.lower_bounds <- check_convert_data(lower_bounds, "float", "vector")
            }
            if (!is.null(upper_bounds)) {
                if (is.matrix(upper_bounds)) stop("upper_bounds must be a 1D vector, not a matrix")
                if (length(upper_bounds) == 0) stop("upper_bounds must not be an empty vector")
                private$.upper_bounds <- check_convert_data(upper_bounds, "float", "vector")
            }
            if (!is.null(lower_bounds) && !is.null(upper_bounds) &&
                length(lower_bounds) != length(upper_bounds)) {
                stop("lower_bounds and upper_bounds must be the same size")
            }

            private$.model <- model
            private$.method <- method
            private$.glob_strategy <- glob_strategy
            private$.reg_power <- reg_power
            private$.check_derivatives <- check_derivatives
            private$.verbose <- as.integer(verbose)
            private$.check_data <- check_data
        },

        #' @description
        #' Fit data to a nonlinear model using regularized least-squares.
        #'
        #' @param x Initial guess to start optimizing from.
        #' @param fun Function that calculates the n_res residuals. This function
        #'   must return the residual vector of the model evaluated at the point x.
        #'   It has the interface \code{fun(x, data)}.
        #'   Example: \code{fun <- function(x, data) \{ data$y - x[1] * exp(x[2] * data$t) \}}
        #' @param jac Function that calculates the residual Jacobian matrix of size
        #'   n_res by n_coef of the model evaluated at the point x. It has the
        #'   interface \code{jac(x, data)}. If NULL, finite differences are used to
        #'   approximate the Jacobian.
        #'   Example: \code{jac <- function(x, data) \{ J <- matrix(0, n_res, n_coef); ... ; J \}}
        #' @param hes Function that calculates the n_coef by n_coef symmetric
        #'   residual Hessian matrix: H = sum_i r_i H_i.
        #'   It has the interface \code{hes(x, wr, data)}.
        #'   Required only when model = 'tensor-newton'. Default = NULL.
        #'   Example: \code{hes <- function(x, wr, data) \{ H <- matrix(0, n_coef, n_coef); ... ; H \}}
        #' @param hp Function that calculates the matrix-vector product of the
        #'   symmetric residual Hessian matrices with a vector y:
        #'   hp = [H1(x)*y, H2(x)*y, ..., Hn(x)*y]. This resulting
        #'   matrix is of size number of coefficients by number of residuals.
        #'   It has the interface \code{hp(x, y, data)}.
        #'   Required only when model = 'tensor-newton'. Default = NULL.
        #'   Example: \code{hp <- function(x, y, data) \{ HP <- matrix(0, n_coef, n_res); ... ; HP \}}
        #' @param data User data object to pass to the user functions. The solver
        #'   does not read or write to this object. Default = NULL.
        #' @param ftol Defines the relative tolerance for the residual norm to
        #'   declare convergence. Default = 1.0e-8.
        #' @param abs_ftol Defines the absolute tolerance for the residual norm to
        #'   declare convergence. Default = 1.0e-8.
        #' @param gtol Defines the relative tolerance of the gradient norm to
        #'   declare convergence. Default = 1.0e-8.
        #' @param abs_gtol Defines the absolute tolerance of the gradient norm to
        #'   declare convergence. Default = 1.0e-5.
        #' @param xtol Defines the tolerance of the step length of two consecutive
        #'   iterates to declare convergence. Default = 2.22e-16.
        #' @param reg_term Defines the regularization penalty term (sigma).
        #'   Default = 0.
        #' @param maxit Defines the iteration limit. Default = 100.
        #' @param fd_step Defines the overall step size to use in the
        #'   finite-difference approximation. If the optimization process stagnates
        #'   or fails, then trialing different values for this parameter may help.
        #'   Default = 1.0e-7.
        #' @param fd_ttol Derivative test tolerance, sets the threshold for deciding
        #'   if a user-supplied derivative is acceptably close to the finite-difference
        #'   approximation. Default = 1.0e-4.
        #' @param maxtime Defines the time limit. Solver will stop when this limit
        #'   is surpassed. Default = 1.0e6.
        #' @return Returns the fitted model, instance itself.
        fit = function(x, fun, jac = NULL, hes = NULL, hp = NULL, data = NULL,
                       ftol = 1e-8, abs_ftol = 1e-8, gtol = 1e-8, abs_gtol = 1e-5,
                       xtol = 2.22e-16, reg_term = 0.0, maxit = 100L,
                       fd_step = 1e-7, fd_ttol = 1e-4, maxtime = 1e6) {
            if (is.null(private$.nlls)) {
                private$.nlls <- aoclda:::rcpp_NLLSModel$new(
                    private$.n_coef,
                    private$.n_res,
                    private$.weights,
                    private$.lower_bounds,
                    private$.upper_bounds,
                    private$.model,
                    private$.method,
                    private$.glob_strategy,
                    private$.reg_power,
                    private$.check_derivatives,
                    private$.verbose,
                    private$.check_data
                )
            }
            if (is.matrix(x)) stop("x must be a 1D vector, not a matrix")
            x <- check_convert_data(x, "float", "vector")
            private$.nlls$fit(
                x, fun, jac, hes, hp,
                data,
                ftol, abs_ftol, gtol, abs_gtol, xtol,
                reg_term, as.integer(maxit), fd_step, fd_ttol, maxtime
            )
            private$.coef <- x
            invisible(self)
        }

    ),
    active = list(
        #' @field coef Numeric vector: Fitted coefficients after \code{$fit()}.
        coef = function() {
            if (is.null(private$.coef)) {
                stop("coef requires fit() to be called first.", call. = FALSE)
            }
            private$.coef
        },
        #' @field n_iter Integer: Number of iterations the solver made.
        n_iter = function() {
            if (is.null(private$.nlls)) {
                stop("n_iter requires fit() to be called first.", call. = FALSE)
            }
            private$.nlls$get_info_iter()
        },
        #' @field n_eval Named list with the number of function calls for:
        #'   f: Residual function fun (includes fd_f count as well).
        #'   j: Gradient function jac.
        #'   h: Hessian function hes.
        #'   hp: Hessian-vector product function hep.
        #'   fd_f: Residual function fun due to finite-differences or derivative checker.
        n_eval = function() {
            if (is.null(private$.nlls)) {
                stop("n_eval requires fit() to be called first.", call. = FALSE)
            }
            private$.nlls$get_info_evals()
        },
        #' @field metrics Named list with the metrics:
        #'   obj: Objective value at iterate x.
        #'   norm_g: Norm of the residual gradient at iterate x.
        #'   scl_norm_g: Norm of the scaled residual gradient at x.
        metrics = function() {
            if (is.null(private$.nlls)) {
                stop("metrics requires fit() to be called first.", call. = FALSE)
            }
            private$.nlls$get_info_optim()
        }
    )
)
