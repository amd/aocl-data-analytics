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

#' Decision forest classifier.
#' 
#' An ensemble classifier based on decision trees.
#' 
#' @param n_trees Integer, set the number of trees to train. Default = 100.
#' @param criterion Character, select scoring function to use. It can take the values
#'   'cross-entropy', 'gini', or 'misclassification'. Default = 'gini'.
#' @param seed Integer, set the random seed for the random number generator. If the value
#'   is -1, a random seed is automatically generated. In this case the resulting
#'   classification will create non-reproducible results. Default = -1.
#' @param max_depth Integer, set the maximum depth of the trees. Default = 29.
#' @param min_samples_split Integer, the minimum number of samples required to split an
#'   internal node. Default = 2.
#' @param bootstrap Logical, select whether to bootstrap the samples in the trees.
#'   Default = TRUE.
#' @param features_selection Character, select how many features to use for each split.
#'   'custom' reads the 'max_features' option, 'proportion' reads the
#'   'proportion_features' option. 'all', 'sqrt' and 'log2' select respectively all,
#'   the square root or the base-2 logarithm of the total number of features.
#'   Default = 'sqrt'.
#' @param max_features Integer, set the number of features to consider when
#'   features_selection is set to 'custom'. 0 means take all the features. Default = 0.
#' @param proportion_features Numeric, proportion of features to consider when
#'   features_selection is set to 'proportion'. Default = 0.1.
#' @param samples_factor Numeric, proportion of samples to draw from the data set to
#'   build each tree if bootstrap was set to TRUE. Default = 1.0.
#' @param min_impurity_decrease Numeric, minimum score improvement needed to consider a
#'   split from the parent node. Default = 0.0.
#' @param min_split_score Numeric, minimum score needed for a node to be considered for
#'   splitting. Default = 0.0.
#' @param feat_thresh Numeric, minimum difference in feature value required for splitting.
#'   Default = 1.0e-06.
#' @param histogram Logical, whether to use histogram-based splitting. Default = FALSE.
#' @param maximum_bins Integer, maximum number of bins to use for histogram-based splitting.
#'   Default = 256.
#' @param block_size Integer, block size for internal parallelism. Default = 256.
#' @param category_split_strategy Character, strategy to use for splitting categorical
#'   features. For a given categorical feature, 'one-vs-all' tries to split each
#'   categorical value from all the others while 'ordered' will try to split the smaller
#'   categorical from the bigger ones. Can be set to 'one-vs-all' or 'ordered'.
#'   Default = 'ordered'.
#' @param max_tree_threads Integer, maximum number of threads allocated to each tree for
#'   parallel feature evaluation. 0 means no cap. Default = 0.
#' @param check_data Logical, whether to check the data for NaNs. Default = FALSE.
#'
#' @export
#' @examples
#' # Basic decision forest
#' data <- matrix(rnorm(200), nrow = 100, ncol = 2)
#' labels <- sample(c(0, 1), 100, replace = TRUE)
#' forest_model <- decision_forest$new(n_trees = 10)
#' forest_model$fit(data, labels)
#' predictions <- forest_model$predict(data)
decision_forest <- R6::R6Class("decision_forest",
  private = list(
    .features_selection = NULL, .max_features = NULL,
    .proportion_features = NULL, .samples_factor = NULL,
    .min_impurity_decrease = NULL, .min_split_score = NULL, .feat_thresh = NULL,
    .max_tree_threads = NULL,
    .decision_forest = NULL
  ),
  public = list(
    
    #' @description Create a new decision_forest object.
    initialize = function(n_trees = 100, criterion = "gini", seed = -1, max_depth = 29,
                         min_samples_split = 2, bootstrap = TRUE,
                         features_selection = "sqrt", max_features = 0,
                         proportion_features = 0.1, samples_factor = 1.0,
                         min_impurity_decrease = 0.0, min_split_score = 0.0,
                         feat_thresh = 1.0e-06, histogram = FALSE, maximum_bins = 256,
                         block_size = 256, category_split_strategy = "ordered",
                         max_tree_threads = 0L, check_data = FALSE) {
      
      # Store wrapper state needed after initialization and for persistence.
      private$.samples_factor <- samples_factor
      private$.min_impurity_decrease <- min_impurity_decrease
      private$.min_split_score <- min_split_score
      private$.feat_thresh <- feat_thresh
      private$.max_features <- as.integer(max_features)
      private$.features_selection <- features_selection
      private$.proportion_features <- proportion_features
      private$.max_tree_threads <- as.integer(max_tree_threads)
      
      private$.decision_forest <- aoclda:::rccp_DecisionForest$new(
        as.integer(n_trees),
        criterion,
        as.integer(seed),
        as.integer(max_depth),
        as.integer(min_samples_split),
        bootstrap,
        features_selection,
        as.integer(max_features),
        histogram,
        as.integer(maximum_bins),
        as.integer(block_size),
        category_split_strategy,
        as.integer(max_tree_threads),
        check_data
      )
    },
    
    #' @description Computes the decision forest on the feature matrix X and response vector y.
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

      private$.decision_forest$fit(
        X, y,
        private$.samples_factor,
        private$.min_impurity_decrease,
        private$.min_split_score,
        private$.feat_thresh,
        private$.proportion_features,
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
      return(private$.decision_forest$score(X, y))
    },
    
    #' @description Generate labels using fitted decision forest on a new set of data X.
    #' @param X The feature matrix to evaluate the model on.
    #'   It must have n_features columns.
    #' @return Integer vector of length n_samples: the prediction vector, where n_samples
    #'   is the number of rows of X.
    predict = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.decision_forest$predict(X))
    },
    
    #' @description Generate class probabilities using fitted decision forest on a new set of data X.
    #' @param X The feature matrix to evaluate the model on.
    #'   It must have n_features columns.
    #' @return Matrix of length n_samples: the prediction vector, where n_samples
    #'   is the number of rows of X.
    predict_proba = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.decision_forest$predict_proba(X))
    },
    
    #' @description Generate class log probabilities using fitted decision forest on a new set of data X.
    #' @param X The feature matrix to evaluate the model on.
    #'   It must have n_features columns.
    #' @return Matrix of length n_samples: the prediction vector, where n_samples
    #'   is the number of rows of X.
    predict_log_proba = function(X) {
      X <- check_convert_data(X, "float", "matrix")
      return(private$.decision_forest$predict_log_proba(X))
    },

    #' @description Save the model to an RDS file. Not intended for direct use;
    #'   use \code{save_model()} from utils instead.
    #' @param path File path.
    save = function(path) {
      saveRDS(list(
        class  = "decision_forest",
        wrapper_state = list(
          max_features = private$.max_features,
          min_impurity_decrease = private$.min_impurity_decrease,
          min_split_score = private$.min_split_score,
          feat_thresh = private$.feat_thresh,
          features_selection = private$.features_selection,
          proportion_features = private$.proportion_features,
          samples_factor = private$.samples_factor,
          max_tree_threads = private$.max_tree_threads
        ),
        cpp_state = private$.decision_forest$save_model()
      ), path)
      invisible(self)
    },

    #' @description Load previously saved state into the current object. Not intended
    #'   for direct use; use \code{load_model()} from utils instead.
    #' @param cpp_state Raw list returned by save_model() at the C++ layer.
    #' @param wrapper_state Optional named list produced by \code{$save()}.
    load_state = function(cpp_state, wrapper_state = NULL) {
      private$.decision_forest$load_model(cpp_state)
      if (!is.null(wrapper_state)) {
        private$.max_features <- as.integer(wrapper_state$max_features)
        private$.min_impurity_decrease <- wrapper_state$min_impurity_decrease
        private$.min_split_score <- wrapper_state$min_split_score
        private$.feat_thresh <- wrapper_state$feat_thresh
        private$.features_selection <- wrapper_state$features_selection
        private$.proportion_features <- wrapper_state$proportion_features
        private$.samples_factor <- wrapper_state$samples_factor
        private$.max_tree_threads <- wrapper_state$max_tree_threads
      }
      invisible(self)
    },

    #' @description Print model version metadata.
    print_model_versions = function() {
      private$.decision_forest$print_model_versions()
      invisible(self)
    }
  ),
  
  active = list(
    #' @field max_features Get/set the maximum number of features
    max_features = function(value) {
      if (missing(value)) {
        return(private$.max_features)
      } else {
        value <- as.integer(value)
        private$.decision_forest$set_max_features_opt(value)
        private$.max_features <- value
      }
    },
    
    #' @field features_selection Get/set the features selection strategy
    features_selection = function(value) {
      if (missing(value)) {
        return(private$.features_selection)
      } else {
        private$.decision_forest$set_features_selection_opt(value)
        private$.features_selection <- value
      }
    },
    
    #' @field samples_factor Get/set the samples factor
    samples_factor = function(value) {
      if (missing(value)) {
        return(private$.samples_factor)
      } else {
        private$.samples_factor <- value
      }
    }
  )
)
