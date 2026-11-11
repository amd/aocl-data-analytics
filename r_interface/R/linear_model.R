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

#' Linear models.
#'
#' @param mod Which linear model to compute.
#'   If mod='mse' then l2 norm linear regression is calculated.
#'   If mod='logistic' then logistic regression is calculated.
#' @param intercept Controls whether to add an intercept variable to the model.
#'   Default=FALSE.
#' @param solver Which solver to use to compute the coefficients. It can
#'   take values 'auto', 'svd', 'cholesky', 'sparse_cg', 'qr', 'coord', 'lbfgs'.
#'   Some solvers are not suitable for some regularization types.
#'   'auto' chooses the best solver based on regression type and data.
#'   'svd' works with normal and ridge regression. The most robust solver but the
#'   least efficient.
#'   'cholesky' works with normal and ridge regression. Will return an error when a
#'   singular matrix is encountered.
#'   'sparse_cg' works with normal and ridge regression. You may need to set a
#'   smaller tol if a badly conditioned matrix is encountered.
#'   'qr' works with normal linear regression only. Will return an error when an
#'   underdetermined system is encountered.
#'   'coord' works with all regression types. Requires data to have variance of 1
#'   column-wise (this can be achieved with the scaling option set to 'scale only').
#'   In the case of normal linear regression and an underdetermined system it
#'   will converge to a solution that is not necessarily a minimum norm solution.
#'   'lbfgs' works with normal and ridge regression. In the case of normal linear
#'   regression and an underdetermined system it will converge to a solution that is
#'   not necessarily a minimum norm solution.
#' @param scaling What type of preprocessing you want to apply on the dataset.
#'   Available options are: 'none', 'centering', 'scale_only', 'standardize'.
#' @param max_iter Maximum number of iterations. Applies only to iterative
#'   solvers: 'sparse_cg', 'coord', 'lbfgs'. The default value depends on the solver. For
#'   'sparse_cg' it is 500, for 'lbfgs' and 'coord' it is 10000.
#' @param constraint Affects only multinomial logistic regression.
#'   The type of constraint put on coefficients. This will affect the number of
#'   coefficients returned.
#'   'rsc' means we choose a reference category whose coefficients
#'   will be set to all 0. This results in K-1 class coefficients for K
#'   class problems.
#'   'ssc' means the sum of coefficients class-wise for each feature
#'   is 0. It will result in K class coefficients for K class problems.
#' @param reg_lambda Lambda, the magnitude of the regularization term.
#'   Default=0.0.
#' @param reg_alpha Alpha, the share of the l1 term in the
#'   regularization. Default=0.0.
#' @param warm_start Reuse coefficients from the previous run when using the
#'   same object to compute coefficients on the new data. Applies only to iterative solvers.
#'   For more details refer to initial coefficients section of linear models documentation.
#' @param tol Convergence tolerance for iterative solvers. Applies only to
#'   iterative solvers: 'sparse_cg', 'coord', 'lbfgs'. Default=1.0e-4.
#' @param progress_factor Applies only to 'lbfgs' and 'coord' solver. Factor used
#'   to detect convergence of the iterative optimization step. Default=NULL.
#' @param mixed_precision Whether to use mixed precision iterative refinement,
#'   in which lower precision arithmetic is used before switching to the working precision
#'   for the final iterations. Default = FALSE.
#' @param low_precision_max_iter If mixed precision iterative refinement is enabled,
#'   maximum number of iterations for the low precision phase. Default = 5000.
#' @param low_precision_tol If mixed precision iterative refinement is enabled,
#'   convergence tolerance for the low precision phase. Default = 1e-3.
#' @param check_data Whether to check the data for NaNs. Default = FALSE.
#' @param optim_coord_skip_min Applies only to 'coord' solver. Minimum number of
#'   iterations before coordinate optimization can be skipped. Default = 2.
#' @param optim_coord_skip_max Applies only to 'coord' solver. Maximum number of
#'   iterations that can be skipped during coordinate optimization. Default = 100.
#' @param dual_tol Tolerance for the dual gap convergence criterion. Default = 1.0e-4.
#' @param optim_coord_skip_tol Applies only to 'coord' solver. Tolerance for
#'   skipping coordinate optimization iterations. Default = 1.0e-4.
#' @param print_level Controls the amount of information printed during the fitting
#'   process. Default = 0 (no output). Set to 1 for basic information, 2 for more
#'   detailed information, and 3..5 for even more verbose output.
#' @export
#' @examples
#' # Basic Linear Regression
#' data <- matrix(rnorm(200), nrow = 100, ncol = 2)
#' target <- rnorm(100)
#' linear_model <- linmod$new('mse')
#' linear_model$fit(data, target)
#' 
#' # Predict on new data
#' predictions <- linear_model$predict(data)
linmod <- R6::R6Class("linmod",
    private = list(
        .mod = NULL, .reg_lambda = NULL, .reg_alpha = NULL,
        .low_precision_tol = NULL, .tol = NULL,
        .progress_factor = NULL, .linmod = NULL, .x0 = NULL,
        .dual_tol = NULL, .optim_coord_skip_tol = NULL
    ),
    public = list(
        
        #' @description Create a new linmod object.
        initialize = function(mod = "mse", intercept = FALSE, solver = 'auto', scaling = 'auto',
                            max_iter = NULL, constraint = 'ssc', reg_lambda = 0.0, reg_alpha = 0.0,
                    warm_start = FALSE, tol = 1.0e-4, progress_factor = NULL,
                    mixed_precision = FALSE, low_precision_max_iter = 5000L,
                    low_precision_tol = 1e-3, check_data = FALSE,
                    optim_coord_skip_min = 2L, optim_coord_skip_max = 100L,
                    dual_tol = 1.0e-4, optim_coord_skip_tol = 1.0e-4, print_level = 0L) {
            
            private$.linmod <- aoclda:::rccp_LinearModel$new(
                mod, max_iter, intercept, solver, scaling, constraint,
                warm_start, mixed_precision, low_precision_max_iter, check_data,
                optim_coord_skip_min, optim_coord_skip_max, print_level)
            
            private$.reg_lambda <- reg_lambda
            private$.reg_alpha <- reg_alpha
            private$.x0 <- NULL
            private$.tol <- tol
            private$.low_precision_tol <- low_precision_tol
            private$.progress_factor <- progress_factor
            private$.dual_tol <- dual_tol
            private$.optim_coord_skip_tol <- optim_coord_skip_tol
            private$.mod <- mod
        },
        
        #' @description Computes the chosen linear model on the feature matrix X and response vector y.
        #' @param X The feature matrix on which to compute the model.
        #'   Its shape is (n_samples, n_features).
        #' @param y The response vector. Its shape is (n_samples).
        #' @param x0 Initial guess for solution. Applies only to iterative solvers.
        #'   The required shape depends on the problem that is being solved (look
        #'   at \code{coef} property). If NULL then x0 is set to a vector of 0. Default=NULL.
        #' @return Returns the instance itself.
        fit = function(X, y, x0 = NULL) {
            X <- check_convert_data(X, "float", "matrix")
            y <- check_convert_data(y, "float", "vector")
            
            if (!is.null(x0)) {
                private$.x0 <- check_convert_data(x0, "float", "vector")
            }
            
            private$.linmod$fit(
                X, y, x0 = private$.x0, progress_factor = private$.progress_factor,
                reg_lambda = private$.reg_lambda, reg_alpha = private$.reg_alpha,
                tol = private$.tol, low_precision_tol = private$.low_precision_tol,
                dual_tol = private$.dual_tol,
                optim_coord_skip_tol = private$.optim_coord_skip_tol)
            
            invisible(self)
        },
        
        #' @description Evaluate the model on a data set X.
        #' @param X The feature matrix to evaluate the model on.
        #'   It must have n_features columns.
        #' @return Numeric vector of length n_samples: The prediction vector, where
        #'   n_samples is the number of rows of X.
        predict = function(X) {
            X <- check_convert_data(X, "float", "matrix")
            return(private$.linmod$predict(X, NULL)$predictions)
        },

        #' @description Compute the score of the model on a data set X with response vector y.
        #' @param X The feature matrix to evaluate the model on.
        #'   It must have n_features columns.
        #' @param y The response vector. Its shape is (n_samples).
        #' @return Float: The score of the model on the given data.
        score = function(X, y) {
            X <- check_convert_data(X, "float", "matrix")
            y <- check_convert_data(y, "float", "vector")
            return(private$.linmod$predict(X, y)$loss)
        },

        #' @description Save the model to an RDS file. Not intended for direct use;
        #' use \code{save_model()} from utils instead.
        #' @param path File path.
        save = function(path) {
            saveRDS(list(
                class  = "linmod",
                wrapper_state = list(
                    mod = private$.mod,
                    reg_lambda = private$.reg_lambda,
                    reg_alpha = private$.reg_alpha,
                    tol = private$.tol,
                    progress_factor = private$.progress_factor,
                    x0 = private$.x0,
                    low_precision_tol = private$.low_precision_tol,
                    dual_tol = private$.dual_tol,
                    optim_coord_skip_tol = private$.optim_coord_skip_tol
                ),
                cpp_state = private$.linmod$save_model()
            ), path)
            invisible(self)
        },

        #' @description Load previously saved state into the current object. Not intended
        #' for direct use; use \code{load_model()} from utils instead.
        #' @param cpp_state Raw list returned by save_model() at the C++ layer.
        #' @param wrapper_state Optional named list produced by \\code{$save()}.
        load_state = function(cpp_state, wrapper_state = NULL) {
            private$.linmod$load_model(cpp_state)
            if (!is.null(wrapper_state)) {
                private$.mod <- wrapper_state$mod
                private$.reg_lambda <- wrapper_state$reg_lambda
                private$.reg_alpha <- wrapper_state$reg_alpha
                private$.tol <- wrapper_state$tol
                private$.progress_factor <- wrapper_state$progress_factor
                private$.x0 <- wrapper_state$x0
                private$.low_precision_tol <- wrapper_state$low_precision_tol
                private$.dual_tol <- wrapper_state$dual_tol
                private$.optim_coord_skip_tol <- wrapper_state$optim_coord_skip_tol
            }
            invisible(self)
        },

        #' @description Print model version metadata.
        print_model_versions = function() {
            private$.linmod$print_model_versions()
            invisible(self)
        }
    ),
    
    active = list(
        #' @field trained Logical: TRUE if the linear model has been trained, FALSE otherwise.
        trained = function() {
            return(private$.linmod$get_model_trained())
        },

        #' @field coef Numeric vector or matrix: Contains the output coefficients of the
        #'   model. Its shape depends on the problem being solved. When mod='mse' or when
        #'   mod='logistic' but the data has 2 classes, it is a vector of shape (, ncoef)
        #'   where ncoef is nfeat+intercept. Otherwise, for K-class problem, it is a matrix
        #'   of shape (nrows, ncoef) where nrows is K-1 if constraint='rsc' and K otherwise.
        coef = function() {
            return(private$.linmod$get_coef())
        },
        
        #' @field dual_coef Numeric vector or matrix: Contains the dual coefficients of the
        #'   model. Only valid for CG solver and underdetermined problems.
        dual_coef = function() {
            return(private$.linmod$get_dual_coef())
        },
        
        #' @field loss Numeric: The value of loss function L(beta_0, beta).
        loss = function() {
            return(private$.linmod$get_loss())
        },
        
        #' @field nrm_gradient_loss Numeric: The norm of the gradient of the loss function.
        #'   Only valid for iterative solvers.
        nrm_gradient_loss = function() {
            return(private$.linmod$get_norm_gradient_loss())
        },
        
        #' @field n_iter Integer: The number iterations performed to find the solution.
        #'   Only valid for iterative solvers.
        n_iter = function() {
            return(private$.linmod$get_n_iter())
        },
        
        #' @field lp_n_iter Integer: The number of low precision iterations performed if
        #'   mixed precision was enabled.
        lp_n_iter = function() {
            return(private$.linmod$get_lp_n_iter())
        },
        
        #' @field time Numeric: Compute time (wall clock time in seconds).
        time = function() {
            return(private$.linmod$get_time())
        }
    )
)
