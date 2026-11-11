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

#' Base class for Support Vector Machine models
#'
#' This class provides common functionality for all SVM variants including
#' data handling, prediction, and property access.
BaseSVM <- R6::R6Class("BaseSVM",
  private = list(
    .kernel = NULL, .degree = NULL, .gamma = NULL, .coef0 = NULL,
    .tol = NULL, .cache_size = NULL, .max_iter = NULL, .tau = NULL,
    .max_ws_size = NULL, .mixed_precision = NULL, .low_precision_max_iter = NULL,
    .low_precision_tol = NULL, .check_data = NULL,
    .model = NULL
  ),
  active = list(
    #' @field n_samples Integer: The number of training samples used to fit the model.
    n_samples = function() private$.model$get_n_samples(),
    #' @field n_features Integer: The number of features in the training data.
    n_features = function() private$.model$get_n_features(),
    #' @field n_iter Integer vector of shape (n_classifiers): The number of iterations
    #'   performed for each binary classifier. In this context it counts the number
    #'   of SMO subproblems solved.
    n_iter = function() private$.model$get_n_iterations(),
    #' @field lp_n_iter Integer vector of shape (n_classifiers): The number of low
    #'   precision iterations performed for each binary classifier when mixed precision
    #'   is enabled. Contains zeros when mixed precision is not used.
    lp_n_iter = function() private$.model$get_lp_n_iterations(),
    #' @field n_support Integer: The total number of support vectors.
    n_support = function() private$.model$get_n_sv(),
    #' @field n_support_per_class Integer vector of shape (n_classes): The number of
    #'   support vectors for each class.
    n_support_per_class = function() private$.model$get_n_sv_per_class(),
    #' @field dual_coef Matrix of shape (n_classes-1, n_support): The dual coefficients
    #'   of the support vectors.
    dual_coef = function() private$.model$get_dual_coefs(),
    #' @field support_vectors_idx Integer vector of shape (n_support): The indices of
    #'   the support vectors.
    support_vectors_idx = function() private$.model$get_support_vectors_idx(),
    #' @field support_vectors Matrix of shape (n_support, n_features): The support
    #'   vectors used by the model.
    support_vectors = function() private$.model$get_sv(),
    #' @field bias Numeric vector or scalar: The bias term(s) of the model (also known
    #'   as the intercept).
    bias = function() private$.model$get_bias()
  ),
  public = list(
    #' @description Create a new BaseSVM object.
    initialize = function(kernel = "rbf", degree = 3, gamma = -1.0, coef0 = 0.0,
                          tol = 0.001, cache_size = 200.0, max_iter = 0, tau = NULL,
                          max_ws_size = -1, mixed_precision = FALSE,
                          low_precision_max_iter = 0L, low_precision_tol = 0.01,
                          check_data = FALSE) {
      if (max_iter == -1) max_iter <- 0
      if (low_precision_max_iter == -1) low_precision_max_iter <- 0

      # Store parameters
      private$.kernel <- kernel
      private$.degree <- as.integer(degree)
      private$.gamma <- gamma
      private$.coef0 <- coef0
      private$.tol <- tol
      private$.cache_size <- cache_size
      private$.max_iter <- as.integer(max_iter)
      private$.tau <- tau
      private$.max_ws_size <- as.integer(max_ws_size)
      private$.mixed_precision <- mixed_precision
      private$.low_precision_max_iter <- as.integer(low_precision_max_iter)
      private$.low_precision_tol <- low_precision_tol
      private$.check_data <- check_data

      private$.model <- NULL
    },

    #' @description Save the model to an RDS file. Not intended for direct use;
    #'   use \code{save_model()} from utils instead.
    #' @param path File path.
    #' @param class_name Class name to persist.
    #' @param subclass_state Named list of subclass-specific state to include in wrapper_state.
    save = function(path, class_name, subclass_state = list()) {
      base_state <- list(
        max_iter = private$.max_iter,
        kernel = private$.kernel,
        degree = private$.degree,
        gamma = private$.gamma,
        coef0 = private$.coef0,
        tol = private$.tol,
        low_precision_tol = private$.low_precision_tol,
        mixed_precision = private$.mixed_precision,
        low_precision_max_iter = private$.low_precision_max_iter,
        cache_size = private$.cache_size,
        tau = private$.tau,
        max_ws_size = private$.max_ws_size,
        check_data = private$.check_data
      )

      saveRDS(list(
        class = class_name,
        wrapper_state = c(subclass_state, base_state),
        cpp_state = private$.model$save_model()
      ), path)
      invisible(self)
    },

    #' @description Load previously saved state into the current object. Not intended
    #'   for direct use; use \code{load_model()} from utils instead.
    #' @param cpp_state Raw list returned by save_model() at the C++ layer.
    #' @param wrapper_state Optional named list produced by \code{$save()}.
    load_state = function(cpp_state, wrapper_state = NULL) {
      private$.model$load_model(cpp_state)
      if (!is.null(wrapper_state)) {
        private$.max_iter <- as.integer(wrapper_state$max_iter)
        private$.kernel <- wrapper_state$kernel
        private$.degree <- as.integer(wrapper_state$degree)
        private$.gamma <- wrapper_state$gamma
        private$.coef0 <- wrapper_state$coef0
        private$.tol <- wrapper_state$tol
        private$.low_precision_tol <- wrapper_state$low_precision_tol
        private$.mixed_precision <- wrapper_state$mixed_precision
        private$.low_precision_max_iter <- as.integer(wrapper_state$low_precision_max_iter)
        private$.cache_size <- wrapper_state$cache_size
        private$.tau <- wrapper_state$tau
        private$.max_ws_size <- as.integer(wrapper_state$max_ws_size)
        private$.check_data <- wrapper_state$check_data
      }
      invisible(self)
    },

    #' @description Fit the SVM model.
    #' @param X Training data.
    #' @param y Target values.
    #' @param y_dtype Data type for target conversion ('float' or 'integer'). Default = 'float'.
    #' @param ... Model-specific fit parameters forwarded to the backend.
    #' @return Returns the instance itself.
    fit = function(X, y, y_dtype = "float", ...) {
      X <- check_convert_data(X, "float", "matrix")
      y <- check_convert_data(y, y_dtype, "vector")
      private$.model$fit(X, y, ...)

      invisible(self)
    },

    #' @description Predict using the SVM model.
    #' @param X Input data used to perform the regression/classification.
    #' @return Numeric vector: Predicted values.
    predict = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.model$predict(X))
    },

    #' @description Compute a model performance score.
    #'   The score metric depends on the specific SVM variant:
    #'   For classification models (SVC, NuSVC): returns mean accuracy.
    #'   For regression models (SVR, NuSVR): returns coefficient of determination (R²).
    #' @param X Test samples of shape (n_samples, n_features).
    #' @param y True target values of shape (n_samples).
    #' @param y_dtype Data type for target conversion ('float' or 'integer'). Default = 'float'.
    #' @return Float: Performance score. Higher values indicate better model performance.
    score = function(X, y, y_dtype = "float") {
      X <- check_convert_data(X, "float", "matrix")
      y <- check_convert_data(y, y_dtype, "vector")
      return(private$.model$score(X, y))
    },

    #' @description Print model version metadata.
    print_model_versions = function() {
      private$.model$print_model_versions()
      invisible(self)
    }
  )
)

#' Support Vector Classification
#'
#' Train a C-Support Vector Classification model.
#'
#' Note: probability estimation is expensive as it internally uses 5-fold cross validation.
#'
#' @param C Regularization parameter. Controls the trade-off between maximizing the margin
#'   between classes and minimizing classification errors. A larger value means higher penalty
#'   to the loss function on misclassified observations. Must be strictly positive. Default=1.0.
#' @param kernel Kernel type to use in the algorithm. Possible values: 'linear', 'poly', 'rbf',
#'   'sigmoid'. Default='rbf'.
#' @param degree Degree of the polynomial kernel function. Ignored by all other kernels. Default=3.
#' @param gamma Kernel coefficient. If set to -1, it is calculated as 1/(Var(X) * n_features).
#'   Default=-1.0.
#' @param coef0 Independent term in kernel function (check kernel functions for
#'   more details). It is only used in 'poly' and 'sigmoid'
#'   kernel functions. Default=0.0.
#' @param probability Whether to enable probability estimates. Default=FALSE.
#' @param tol Tolerance for stopping criterion. Default=0.001.
#' @param cache_size Size of the kernel cache (in MB). Default=200.0.
#' @param max_iter Hard limit on iterations within solver, or 0 for no limit. Default=0.
#' @param random_state Random seed used for probability estimates. Use -1 for non deterministic
#'   behavior. Default=0.
#' @param tau Numerical stability parameter. If it is NULL then machine epsilon is used.
#'   Default=NULL.
#' @param max_ws_size Maximum working set size. A value divisible by 64 is
#'   recommended for optimal performance. Setting -1 automatically selects the
#'   optimal size based on the input data. Default=-1.
#' @param mixed_precision Whether to use mixed precision iterative refinement,
#'   in which lower precision arithmetic is used before switching to the working precision
#'   for the final iterations. Default=FALSE.
#' @param low_precision_max_iter If mixed precision iterative refinement is enabled,
#'   maximum number of iterations for the low precision phase. Default=0.
#' @param low_precision_tol If mixed precision iterative refinement is enabled,
#'   convergence tolerance for the low precision phase. Default=0.01.
#' @param check_data Whether to check data for NaNs. Default=FALSE.
SVC <- R6::R6Class("SVC",
  inherit = BaseSVM,
  private = list(.C = NULL, .probability = NULL, .random_state = NULL),
  public = list(
    #' @description Create a new SVC object
    initialize = function(C = 1.0, kernel = "rbf", degree = 3, gamma = -1.0, coef0 = 0.0,
                          probability = FALSE, tol = 0.001, cache_size = 200.0, max_iter = 0,
                          random_state = 0, tau = NULL, max_ws_size = -1,
                          mixed_precision = FALSE, low_precision_max_iter = 0L,
                          low_precision_tol = 0.01, check_data = FALSE) {
      # Call parent constructor
      super$initialize(
        kernel = kernel,
        degree = degree,
        gamma = gamma,
        coef0 = coef0,
        tol = tol,
        cache_size = cache_size,
        max_iter = max_iter,
        tau = tau,
        max_ws_size = max_ws_size,
        mixed_precision = mixed_precision,
        low_precision_max_iter = low_precision_max_iter,
        low_precision_tol = low_precision_tol,
        check_data = check_data
      )

      # Store SVC-specific parameters
      private$.C <- C
      private$.probability <- probability
      private$.random_state <- as.integer(random_state)

      # Create backend objects
      private$.model <- aoclda:::rccp_SVC$new(
        kernel = as.character(private$.kernel),
        degree = as.integer(private$.degree),
        max_iter = as.integer(private$.max_iter),
        probability = as.integer(private$.probability),
        seed = as.integer(private$.random_state),
        max_ws_size = as.integer(private$.max_ws_size),
        mixed_precision = private$.mixed_precision,
        low_precision_max_iter = as.integer(private$.low_precision_max_iter),
        check_data = private$.check_data
      )
    },

    #' @description Fit the SVC model according to the given training data
    #' @param X Training vectors of shape (n_samples, n_features)
    #' @param y Target values of shape (n_samples). They are expected to range from 0 to n_class - 1
    #' @return Returns the instance itself
    fit = function(X, y) {
      super$fit(
        X, y,
        y_dtype = "integer",
        private$.tau, private$.C, private$.gamma, private$.coef0,
        private$.tol, private$.low_precision_tol, private$.cache_size
      )
    },

    #' @description Perform classification on samples in X
    #' @param X Input vectors of shape (n_samples, n_features)
    #' @return Predicted class labels for samples in X
    predict = function(X) {
      return(super$predict(X))
    },

    #' @description Evaluate the decision function for the samples in X.
    #'   In multi-class problems, you can use the 'shape' parameter to choose
    #'   between one-vs-rest ('ovr') and one-vs-one ('ovo') decision function shape.
    #'   Note that in our case, OvR decision values are derived from OvO.
    #'   For binary problems, this parameter is ignored.
    #' @param X Input vectors of shape (n_samples, n_features)
    #' @param shape Whether to return a one-vs-rest ('ovr') decision function or the original one-vs-one ('ovo'). Default='ovr'
    #' @return Decision function values for each sample
    decision_function = function(X, shape = "ovr") {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.model$decision_function(X, shape))
    },

    #' @description Return the mean accuracy on the given test data and labels
    #' @param X Test samples of shape (n_samples, n_features)
    #' @param y True labels for X
    #' @return Float: Mean accuracy of self$predict(X) wrt. y
    score = function(X, y) {
      return(super$score(X, y, y_dtype = "integer"))
    },

    #' @description Predict class probabilities for samples in X
    #' @param X Input data of shape (n_samples, n_features)
    #' @return Array of shape (n_samples, n_classes) with class probability estimates
    predict_proba = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.model$predict_proba(X))
    },

    #' @description Predict class log probabilities for samples in X
    #' @param X Input data of shape (n_samples, n_features)
    #' @return Array of shape (n_samples, n_classes) with class log probability estimates
    predict_log_proba = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.model$predict_log_proba(X))
    },

    #' @description Save the model to an RDS file. Not intended for direct use;
    #'   use \code{save_model()} from utils instead.
    #' @param path File path.
    save = function(path) {
      super$save(
        path,
        class_name = "SVC",
        subclass_state = list(
          C = private$.C,
          probability = private$.probability,
          random_state = private$.random_state
        )
      )
    },

    #' @description Load previously saved state into the current object. Not intended
    #'   for direct use; use \code{load_model()} from utils instead.
    #' @param cpp_state Raw list returned by save_model() at the C++ layer.
    #' @param wrapper_state Optional named list produced by \code{$save()}.
    load_state = function(cpp_state, wrapper_state = NULL) {
      super$load_state(cpp_state, wrapper_state)
      if (!is.null(wrapper_state)) {
        private$.C <- wrapper_state$C
        private$.probability <- wrapper_state$probability
        private$.random_state <- as.integer(wrapper_state$random_state)
      }
      invisible(self)
    }
  ),
  active = list(
    #' @field n_classes Integer: The number of classes in the classification problem.
    n_classes = function() private$.model$get_n_classes(),

    #' @field probA Numeric vector or scalar: The probability parameter A of the model,
    #'   used in probability estimates.
    probA = function() private$.model$get_probA(),

    #' @field probB Numeric vector or scalar: The probability parameter B of the model,
    #'   used in probability estimates.
    probB = function() private$.model$get_probB()
  )
)

#' Support Vector Regression
#'
#' Train an epsilon-Support Vector Regression model.
#'
#' @param C Regularization parameter. Controls the trade-off between maximizing the margin
#'   between classes and minimizing classification errors. A larger value means higher penalty
#'   to the loss function on misclassified observations. Must be strictly positive. Default=1.0.
#' @param epsilon Epsilon in the SVR model. Defines the tolerance for errors in predictions by
#'   creating an acceptable margin (tube) within which errors are not penalized. Default=0.1.
#' @param kernel Kernel type to use in the algorithm. Possible values: 'linear', 'poly', 'rbf',
#'   'sigmoid'. Default='rbf'.
#' @param degree Degree of the polynomial kernel function. Ignored by all other kernels. Default=3.
#' @param gamma Kernel coefficient. If set to -1, it is calculated as 1/(Var(X) * n_features).
#'   Default=-1.0.
#' @param coef0 Independent term in kernel function (check kernel functions for
#'   more details). It is only used in 'poly' and 'sigmoid'
#'   kernel functions. Default=0.0.
#' @param tol Tolerance for stopping criterion. Default=0.001.
#' @param cache_size Size of the kernel cache (in MB). Default=200.0.
#' @param max_iter Hard limit on iterations within solver, or 0 for no limit. Default=0.
#' @param tau Numerical stability parameter. If it is NULL then machine epsilon is used.
#'   Default=NULL.
#' @param max_ws_size Maximum working set size. A value divisible by 64 is
#'   recommended for optimal performance. Setting -1 automatically selects the
#'   optimal size based on the input data. Default=-1.
#' @param mixed_precision Whether to use mixed precision iterative refinement,
#'   in which lower precision arithmetic is used before switching to the working precision
#'   for the final iterations. Default=FALSE.
#' @param low_precision_max_iter If mixed precision iterative refinement is enabled,
#'   maximum number of iterations for the low precision phase. Default=0.
#' @param low_precision_tol If mixed precision iterative refinement is enabled,
#'   convergence tolerance for the low precision phase. Default=0.01.
#' @param check_data Whether to check data for NaNs. Default=FALSE.
SVR <- R6::R6Class("SVR",
  inherit = BaseSVM,
  private = list(.C = NULL, .epsilon = NULL),
  public = list(
    #' @description Create a new SVR object
    initialize = function(C = 1.0, epsilon = 0.1, kernel = "rbf", degree = 3, gamma = -1.0,
                          coef0 = 0.0, tol = 0.001, cache_size = 200.0, max_iter = 0,
                          max_ws_size = -1, tau = NULL, check_data = FALSE,
                          mixed_precision = FALSE, low_precision_max_iter = 0L,
                          low_precision_tol = 0.01) {
      # Call parent constructor
      super$initialize(
        kernel = kernel,
        degree = degree,
        gamma = gamma,
        coef0 = coef0,
        tol = tol,
        cache_size = cache_size,
        max_iter = max_iter,
        max_ws_size = max_ws_size,
        tau = tau,
        mixed_precision = mixed_precision,
        low_precision_max_iter = low_precision_max_iter,
        low_precision_tol = low_precision_tol,
        check_data = check_data
      )

      # Create backend objects
      private$.model <- aoclda:::rccp_SVR$new(
        kernel = as.character(private$.kernel),
        degree = as.integer(private$.degree),
        max_iter = as.integer(private$.max_iter),
        max_ws_size = as.integer(private$.max_ws_size),
        mixed_precision = private$.mixed_precision,
        low_precision_max_iter = as.integer(private$.low_precision_max_iter),
        check_data = private$.check_data
      )

      # Store SVR-specific parameters
      private$.C <- C
      private$.epsilon <- epsilon
    },

    #' @description Fit the SVR model according to the given training data
    #' @param X Training vectors of shape (n_samples, n_features)
    #' @param y Target values of shape (n_samples)
    #' @return Returns the instance itself
    fit = function(X, y) {
      super$fit(
        X, y,
        y_dtype = "float",
        private$.tau, private$.C, private$.epsilon, private$.gamma,
        private$.coef0, private$.tol, private$.low_precision_tol, private$.cache_size
      )
    },

    #' @description Predict regression values for samples in X
    #' @param X Input vectors of shape (n_samples, n_features)
    #' @return Predicted values
    predict = function(X) {
      return(super$predict(X))
    },

    #' @description Return the coefficient of determination R² of the prediction
    #' @param X Test samples of shape (n_samples, n_features)
    #' @param y True values for X
    #' @return Float: R² of self$predict(X) wrt. y
    score = function(X, y) {
      return(super$score(X, y))
    },

    #' @description Save the model to an RDS file. Not intended for direct use;
    #'   use \code{save_model()} from utils instead.
    #' @param path File path.
    save = function(path) {
      super$save(
        path,
        class_name = "SVR",
        subclass_state = list(
          C = private$.C,
          epsilon = private$.epsilon
        )
      )
    },

    #' @description Load previously saved state into the current object. Not intended
    #'   for direct use; use \code{load_model()} from utils instead.
    #' @param cpp_state Raw list returned by save_model() at the C++ layer.
    #' @param wrapper_state Optional named list produced by \code{$save()}.
    load_state = function(cpp_state, wrapper_state = NULL) {
      super$load_state(cpp_state, wrapper_state)
      if (!is.null(wrapper_state)) {
        private$.C <- wrapper_state$C
        private$.epsilon <- wrapper_state$epsilon
      }
      invisible(self)
    }
  )
)

#' Nu-Support Vector Classification
#'
#' Train a Nu-Support Vector Classification model.
#'
#' Note: probability estimation is expensive as it internally uses 5-fold cross validation.
#'
#' @param nu An upper bound on the fraction of training errors and a lower bound of the fraction
#'   of support vectors. Default=0.5.
#' @param kernel Kernel type to use in the algorithm. Possible values: 'linear', 'poly', 'rbf',
#'   'sigmoid'. Default='rbf'.
#' @param degree Degree of the polynomial kernel function. Ignored by all other kernels. Default=3.
#' @param gamma Kernel coefficient. If set to -1, it is calculated as 1/(Var(X) * n_features).
#'   Default=-1.0.
#' @param coef0 Independent term in kernel function (check kernel functions for
#'   more details). It is only used in 'poly' and 'sigmoid'
#'   kernel functions. Default=0.0.
#' @param probability Whether to enable probability estimates. Default=FALSE.
#' @param tol Tolerance for stopping criterion. Default=0.001.
#' @param cache_size Size of the kernel cache (in MB). Default=200.0.
#' @param max_iter Hard limit on iterations within solver, or 0 for no limit. Default=0.
#' @param random_state Random seed used for probability estimates. Use -1 for non deterministic
#'   behavior. Default=0.
#' @param tau Numerical stability parameter. If it is NULL then machine epsilon is used.
#'   Default=NULL.
#' @param max_ws_size Maximum working set size. A value divisible by 64 is
#'   recommended for optimal performance. Setting -1 automatically selects the
#'   optimal size based on the input data. Default=-1.
#' @param mixed_precision Whether to use mixed precision iterative refinement,
#'   in which lower precision arithmetic is used before switching to the working precision
#'   for the final iterations. Default=FALSE.
#' @param low_precision_max_iter If mixed precision iterative refinement is enabled,
#'   maximum number of iterations for the low precision phase. Default=0.
#' @param low_precision_tol If mixed precision iterative refinement is enabled,
#'   convergence tolerance for the low precision phase. Default=0.01.
#' @param check_data Whether to check data for NaNs. Default=FALSE.
NuSVC <- R6::R6Class("NuSVC",
  inherit = BaseSVM,
  private = list(.nu = NULL, .probability = NULL, .random_state = NULL),
  public = list(
    #' @description Create a new NuSVC object
    initialize = function(nu = 0.5, kernel = "rbf", degree = 3, gamma = -1.0, coef0 = 0.0,
                          probability = FALSE, tol = 0.001, cache_size = 200.0, max_iter = 0,
                          random_state = 0, max_ws_size = -1, tau = NULL,
                          mixed_precision = FALSE, low_precision_max_iter = 0L,
                          low_precision_tol = 0.01, check_data = FALSE) {
      # Store specific parameters first
      private$.probability <- probability
      private$.random_state <- as.integer(random_state)

      # Call parent constructor
      super$initialize(
        kernel = kernel,
        degree = degree,
        gamma = gamma,
        coef0 = coef0,
        tol = tol,
        cache_size = cache_size,
        max_iter = max_iter,
        max_ws_size = max_ws_size,
        tau = tau,
        mixed_precision = mixed_precision,
        low_precision_max_iter = low_precision_max_iter,
        low_precision_tol = low_precision_tol,
        check_data = check_data
      )

      # Create backend objects
      private$.model <- aoclda:::rccp_NuSVC$new(
        kernel = as.character(private$.kernel),
        degree = as.integer(private$.degree),
        max_iter = as.integer(private$.max_iter),
        probability = as.integer(private$.probability),
        seed = as.integer(private$.random_state),
        max_ws_size = as.integer(private$.max_ws_size),
        mixed_precision = private$.mixed_precision,
        low_precision_max_iter = as.integer(private$.low_precision_max_iter),
        check_data = private$.check_data
      )

      private$.nu <- nu
    },

    #' @description Fit the NuSVC model according to the given training data
    #' @param X Training vectors of shape (n_samples, n_features)
    #' @param y Target values of shape (n_samples). They are expected to range from 0 to n_class - 1
    #' @return Returns the instance itself
    fit = function(X, y) {
      super$fit(
        X, y,
        y_dtype = "integer",
        private$.tau, private$.nu, private$.gamma, private$.coef0,
        private$.tol, private$.low_precision_tol, private$.cache_size
      )
    },

    #' @description Perform classification on samples in X
    #' @param X Input vectors of shape (n_samples, n_features)
    #' @return Predicted class labels for samples in X
    predict = function(X) {
      return(super$predict(X))
    },

    #' @description Evaluate the decision function for the samples in X.
    #'   In multi-class problems, you can use the 'shape' parameter to choose
    #'   between one-vs-rest ('ovr') and one-vs-one ('ovo') decision function shape.
    #'   Note that in our case, OvR decision values are derived from OvO.
    #'   For binary problems, this parameter is ignored.
    #' @param X Input vectors of shape (n_samples, n_features)
    #' @param shape Whether to return a one-vs-rest ('ovr') decision function or the original one-vs-one ('ovo'). Default='ovr'
    #' @return Decision function values for each sample
    decision_function = function(X, shape = "ovr") {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.model$decision_function(X, shape))
    },

    #' @description Return the mean accuracy on the given test data and labels
    #' @param X Test samples of shape (n_samples, n_features)
    #' @param y True labels for X
    #' @return Float: Mean accuracy of self$predict(X) wrt. y
    score = function(X, y) {
      return(super$score(X, y, y_dtype = "integer"))
    },

    #' @description Predict class probabilities for samples in X
    #' @param X Input data of shape (n_samples, n_features)
    #' @return Array of shape (n_samples, n_classes) with class probability estimates
    predict_proba = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.model$predict_proba(X))
    },

    #' @description Predict class log probabilities for samples in X
    #' @param X Input data of shape (n_samples, n_features)
    #' @return Array of shape (n_samples, n_classes) with class log probability estimates
    predict_log_proba = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.model$predict_log_proba(X))
    },

    #' @description Save the model to an RDS file. Not intended for direct use;
    #'   use \code{save_model()} from utils instead.
    #' @param path File path.
    save = function(path) {
      super$save(
        path,
        class_name = "NuSVC",
        subclass_state = list(
          nu = private$.nu,
          probability = private$.probability,
          random_state = private$.random_state
        )
      )
    },

    #' @description Load previously saved state into the current object. Not intended
    #'   for direct use; use \code{load_model()} from utils instead.
    #' @param cpp_state Raw list returned by save_model() at the C++ layer.
    #' @param wrapper_state Optional named list produced by \code{$save()}.
    load_state = function(cpp_state, wrapper_state = NULL) {
      super$load_state(cpp_state, wrapper_state)
      if (!is.null(wrapper_state)) {
        private$.nu <- wrapper_state$nu
        private$.probability <- wrapper_state$probability
        private$.random_state <- as.integer(wrapper_state$random_state)
      }
      invisible(self)
    }
  ),
  active = list(
    #' @field n_classes Integer: The number of classes in the classification problem.
    n_classes = function() private$.model$get_n_classes(),

    #' @field probA Numeric vector or scalar: The probability parameter A of the model,
    #'   used in probability estimates.
    probA = function() private$.model$get_probA(),

    #' @field probB Numeric vector or scalar: The probability parameter B of the model,
    #'   used in probability estimates.
    probB = function() private$.model$get_probB()
  )
)

#' Nu-Support Vector Regression
#'
#' Train a Nu-Support Vector Regression model.
#'
#' @param nu An upper bound on the fraction of training errors and a lower bound of the fraction
#'   of support vectors. Default=0.5.
#' @param C Regularization parameter. Controls the trade-off between maximizing the margin
#'   between classes and minimizing classification errors. A larger value means higher penalty
#'   to the loss function on misclassified observations. Must be strictly positive. Default=1.0.
#' @param kernel Kernel type to use in the algorithm. Possible values: 'linear', 'poly', 'rbf',
#'   'sigmoid'. Default='rbf'.
#' @param degree Degree of the polynomial kernel function. Ignored by all other kernels. Default=3.
#' @param gamma Kernel coefficient. If set to -1, it is calculated as 1/(Var(X) * n_features).
#'   Default=-1.0.
#' @param coef0 Independent term in kernel function (check kernel functions for
#'   more details). It is only used in 'poly' and 'sigmoid'
#'   kernel functions. Default=0.0.
#' @param tol Tolerance for stopping criterion. Default=0.001.
#' @param cache_size Size of the kernel cache (in MB). Default=200.0.
#' @param max_iter Hard limit on iterations within solver, or 0 for no limit. Default=0.
#' @param tau Numerical stability parameter. If it is NULL then machine epsilon is used.
#'   Default=NULL.
#' @param max_ws_size Maximum working set size. A value divisible by 64 is
#'   recommended for optimal performance. Setting -1 automatically selects the
#'   optimal size based on the input data. Default=-1.
#' @param mixed_precision Whether to use mixed precision iterative refinement,
#'   in which lower precision arithmetic is used before switching to the working precision
#'   for the final iterations. Default=FALSE.
#' @param low_precision_max_iter If mixed precision iterative refinement is enabled,
#'   maximum number of iterations for the low precision phase. Default=0.
#' @param low_precision_tol If mixed precision iterative refinement is enabled,
#'   convergence tolerance for the low precision phase. Default=0.01.
#' @param check_data Whether to check data for NaNs. Default=FALSE.
NuSVR <- R6::R6Class("NuSVR",
  inherit = BaseSVM,
  private = list(.nu = NULL, .C = NULL),
  public = list(
    #' @description Create a new NuSVR object
    initialize = function(nu = 0.5, C = 1.0, kernel = "rbf", degree = 3, gamma = -1.0,
                          coef0 = 0.0, tol = 0.001, cache_size = 200.0, max_iter = 0,
                          max_ws_size = -1, tau = NULL, mixed_precision = FALSE,
                          low_precision_max_iter = 0L, low_precision_tol = 0.01,
                          check_data = FALSE) {
      # Call parent constructor
      super$initialize(
        kernel = kernel,
        degree = degree,
        gamma = gamma,
        coef0 = coef0,
        tol = tol,
        cache_size = cache_size,
        max_iter = max_iter,
        max_ws_size = max_ws_size,
        tau = tau,
        mixed_precision = mixed_precision,
        low_precision_max_iter = low_precision_max_iter,
        low_precision_tol = low_precision_tol,
        check_data = check_data
      )

      # Create backend objects
      private$.model <- aoclda:::rccp_NuSVR$new(
        kernel = as.character(private$.kernel),
        degree = as.integer(private$.degree),
        max_iter = as.integer(private$.max_iter),
        max_ws_size = as.integer(private$.max_ws_size),
        mixed_precision = private$.mixed_precision,
        low_precision_max_iter = as.integer(private$.low_precision_max_iter),
        check_data = private$.check_data
      )

      # Store specific parameters
      private$.nu <- nu
      private$.C <- C
    },

    #' @description Fit the NuSVR model according to the given training data
    #' @param X Training vectors of shape (n_samples, n_features)
    #' @param y Target values of shape (n_samples)
    #' @return Returns the instance itself
    fit = function(X, y) {
      super$fit(
        X, y,
        y_dtype = "float",
        private$.tau, private$.nu, private$.C, private$.gamma,
        private$.coef0, private$.tol, private$.low_precision_tol, private$.cache_size
      )
    },

    #' @description Predict regression values for samples in X
    #' @param X Input vectors of shape (n_samples, n_features)
    #' @return Predicted values
    predict = function(X) {
      return(super$predict(X))
    },

    #' @description Return the coefficient of determination R² of the prediction
    #' @param X Test samples of shape (n_samples, n_features)
    #' @param y True values for X
    #' @return Float: R² of self$predict(X) wrt. y
    score = function(X, y) {
      return(super$score(X, y))
    },

    #' @description Save the model to an RDS file. Not intended for direct use;
    #'   use \code{save_model()} from utils instead.
    #' @param path File path.
    save = function(path) {
      super$save(
        path,
        class_name = "NuSVR",
        subclass_state = list(
          nu = private$.nu,
          C = private$.C
        )
      )
    },

    #' @description Load previously saved state into the current object. Not intended
    #'   for direct use; use \code{load_model()} from utils instead.
    #' @param cpp_state Raw list returned by save_model() at the C++ layer.
    #' @param wrapper_state Optional named list produced by \code{$save()}.
    load_state = function(cpp_state, wrapper_state = NULL) {
      super$load_state(cpp_state, wrapper_state)
      if (!is.null(wrapper_state)) {
        private$.nu <- wrapper_state$nu
        private$.C <- wrapper_state$C
      }
      invisible(self)
    }
  )
)
