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

#' A decision tree classifier.
#' 
#' @param criterion Character, select scoring function to use. It can take the values
#'   'cross-entropy', 'gini', or 'misclassification'. Default = 'gini'.
#' @param seed Integer, set the random seed for the random number generator. If the
#'   value is -1, a random seed is automatically generated. In this case the resulting
#'   classification will create non-reproducible results. Default = -1.
#' @param max_depth Integer, set the maximum depth of the tree. Default = 29.
#' @param min_samples_split Integer, the minimum number of samples required to split
#'   an internal node. Default = 2.
#' @param max_features Integer, set the number of features to consider when splitting
#'   a node. 0 means take all the features. Default = 0.
#' @param min_impurity_decrease Numeric, minimum score improvement needed to consider
#'   a split from the parent node. Default = 0.0.
#' @param min_split_score Numeric, minimum score needed for a node to be considered
#'   for splitting. Default = 0.0.
#' @param feat_thresh Numeric, minimum difference in feature value required for
#'   splitting. Default = 1.0e-05.
#' @param category_split_strategy Character, the strategy to use for splitting
#'   categorical features. It can take the values 'ordered' or 'one-vs-all'.
#'   Default = 'ordered'.
#' @param histogram Logical, whether to use histograms for continuous features.
#'   Default = FALSE.
#' @param maximum_bins Integer, maximum number of bins to use for histograms.
#'   Default = 256.
#' @param predict_proba Logical, whether to predict class probabilities.
#'   Default = TRUE.
#' @param check_data Logical, whether to check the data for NaNs. Default = FALSE.
#'
#' @export
#' @examples
#' # Basic decision tree
#' data <- matrix(rnorm(200), nrow = 100, ncol = 2)
#' labels <- sample(c(0, 1), 100, replace = TRUE)
#' tree_model <- decision_tree$new()
#' tree_model$fit(data, labels)
#' predictions <- tree_model$predict(data)
decision_tree <- R6::R6Class("decision_tree",
  private = list(
    .max_features = NULL, .min_impurity_decrease = NULL,
    .min_split_score = NULL, .feat_thresh = NULL,
    .decision_tree = NULL, .model_info = NULL
  ),
  public = list(
    
    #' @description Create a new decision_tree object.
    initialize = function(criterion = "gini", seed = -1, max_depth = 29,
                         min_samples_split = 2, max_features = 0,
                         min_impurity_decrease = 0.0, min_split_score = 0.0,
                         feat_thresh = 1.0e-05,
                         category_split_strategy = "ordered", histogram = FALSE,
                         maximum_bins = 256, predict_proba = TRUE, check_data = FALSE) {

      private$.decision_tree <- aoclda:::rccp_DecisionTree$new(
        as.integer(seed),
        as.integer(max_depth),
        as.integer(max_features),
        criterion,
        as.integer(min_samples_split),
        category_split_strategy,
        histogram,
        as.integer(maximum_bins),
        predict_proba,
        check_data
      )
      
      private$.max_features <- as.integer(max_features)
      private$.min_impurity_decrease <- min_impurity_decrease
      private$.min_split_score <- min_split_score
      private$.feat_thresh <- feat_thresh
      private$.model_info <- NULL
    },
    
    #' @description Computes the decision tree on the feature matrix X and response vector y.
    #' @param X The feature matrix on which to compute the model.
    #'   Its shape is (n_samples, n_features).
    #' @param y The response vector. Its shape is (n_samples).
    #' @param categorical_features Integer vector. categorical_features[i] should be set to
    #'   a negative value if feature i is continuous or to the number of different categories
    #'   if feature i is categorical. If NULL, all features are considered continuous.
    #'   Its shape is (n_features).
    #' @return Returns the instance itself.
    fit = function(X, y, categorical_features = NULL) {
      X <- check_convert_data(X, "float", "matrix")
      y <- check_convert_data(y, "integer", "vector")
      
      if (!is.null(categorical_features)) {
        categorical_features <- check_convert_data(categorical_features, "integer", "vector")
      }

      # Reset model info
      private$.model_info <- NULL
      private$.decision_tree$fit(
        X, y,
        private$.min_impurity_decrease,
        private$.min_split_score,
        private$.feat_thresh,
        categorical_features
      )
      
      invisible(self)
    },
    
    #' @description Calculates score (prediction accuracy) by comparing predicted labels
    #'   and actual labels on a new set of data.
    #' @param X The feature matrix to evaluate the model on.
    #'   It must have n_features columns.
    #' @param y The response vector. It must have shape (n_samples).
    #' @return Numeric scalar: the mean accuracy of the model on the test data.
    score = function(X, y) {
      X <- check_convert_data(X, "float", "matrix")
      y <- check_convert_data(y, "integer", "vector")
      return(private$.decision_tree$score(X, y))
    },
    
    #' @description Generate labels using fitted decision tree on a new set of data X.
    #' @param X The feature matrix to evaluate the model on.
    #'   It must have n_features columns.
    #' @return Integer vector of length n_samples: the prediction vector, where n_samples
    #'   is the number of rows of X.
    predict = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.decision_tree$predict(X))
    },
    
    #' @description Generate class probabilities using fitted decision tree on a new set of data X.
    #' @param X The feature matrix to evaluate the model on.
    #'   It must have n_features columns.
    #' @return Matrix of length n_samples: the prediction vector, where n_samples
    #'   is the number of rows of X.
    predict_proba = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.decision_tree$predict_proba(X))
    },
    
    #' @description Generate class log probabilities using fitted decision tree on a new set of data X.
    #' @param X The feature matrix to evaluate the model on.
    #'   It must have n_features columns.
    #' @return Matrix of length n_samples: the prediction vector, where n_samples
    #'   is the number of rows of X.
    predict_log_proba = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.decision_tree$predict_log_proba(X))
    },
    
    #' @description Update the model information
    update_model_info = function() {
      private$.model_info <- private$.decision_tree$get_model_info()
    },

    #' @description Save the model to an RDS file. Not intended for direct use;
    #'   use \code{save_model()} from utils instead.
    #' @param path File path.
    save = function(path) {
      saveRDS(list(
        class  = "decision_tree",
        wrapper_state = list(
          max_features = private$.max_features,
          min_impurity_decrease = private$.min_impurity_decrease,
          min_split_score = private$.min_split_score,
          feat_thresh = private$.feat_thresh,
          model_info = private$.model_info
        ),
        cpp_state = private$.decision_tree$save_model()
      ), path)
      invisible(self)
    },

    #' @description Load previously saved state into the current object. Not intended
    #'   for direct use; use \code{load_model()} from utils instead.
    #' @param cpp_state Raw list returned by save_model() at the C++ layer.
    #' @param wrapper_state Optional named list produced by \code{$save()}.
    load_state = function(cpp_state, wrapper_state = NULL) {
      private$.decision_tree$load_model(cpp_state)
      if (!is.null(wrapper_state)) {
        private$.max_features <- as.integer(wrapper_state$max_features)
        private$.min_impurity_decrease <- wrapper_state$min_impurity_decrease
        private$.min_split_score <- wrapper_state$min_split_score
        private$.feat_thresh <- wrapper_state$feat_thresh
        private$.model_info <- wrapper_state$model_info
      }
      invisible(self)
    },

    #' @description Print model version metadata.
    print_model_versions = function() {
      private$.decision_tree$print_model_versions()
      invisible(self)
    }
  ),

  active = list(
    #' @field max_features Integer: get/set the maximum number of features to consider when splitting a node.
    max_features = function(value) {
      if (missing(value)) {
        return(private$.max_features)
      } else {
        value <- as.integer(value)
        private$.decision_tree$set_max_features_opt(max_features = value)
        private$.max_features <- value
      }
    },
    
    #' @field n_samples Integer: the number of samples used in the trained tree.
    n_samples = function() {
      if (is.null(private$.model_info)) {
        self$update_model_info()
      }
      return(private$.model_info$n_samples)
    },
    
    #' @field n_obs Integer: the number of observations used in the trained tree.
    n_obs = function() {
      if (is.null(private$.model_info)) {
        self$update_model_info()
      }
      return(private$.model_info$n_obs)
    },
    
    #' @field n_features Integer: the number of features used in the trained tree.
    n_features = function() {
      if (is.null(private$.model_info)) {
        self$update_model_info()
      }
      return(private$.model_info$n_features)
    },
    
    #' @field depth Integer: the depth of the trained tree.
    depth = function() {
      if (is.null(private$.model_info)) {
        self$update_model_info()
      }
      return(private$.model_info$depth)
    },
    
    #' @field n_nodes Integer: the number of nodes in the trained tree.
    n_nodes = function() {
      if (is.null(private$.model_info)) {
        self$update_model_info()
      }
      return(private$.model_info$n_nodes)
    },
    
    #' @field n_leaves Integer: the number of leaves in the trained tree.
    n_leaves = function() {
      if (is.null(private$.model_info)) {
        self$update_model_info()
      }
      return(private$.model_info$n_leaves)
    },

    #' @field n_threads Integer: the number of threads used by the classifier.
    n_threads = function() {
      if (is.null(private$.model_info)) {
        self$update_model_info()
      }
      return(private$.model_info$n_threads)
    }
  )
)
