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

#' A Nearest Neighbors object for general nearest neighbor queries and
#' classification/regression.
#'
#' @param n_neighbors The number of nearest neighbors for
#'   \code{$kneighbors()} queries. Default = 5.
#' @param weights The weight function used for prediction.
#'   Available options are 'uniform' and 'distance'. Default = 'uniform'.
#' @param algorithm The underlying algorithm used to compute
#'   the k-nearest neighbors. Available options are 'auto', 'brute' and 'kd_tree'.
#'   k-d trees are likely to be fastest for lower dimensional datasets, but
#'   cannot not be used with the cosine distance, the squared Euclidean distance,
#'   the inner product distance, or with the Minkowski distance with power less
#'   than 1.0. Default = 'auto'.
#' @param leaf_size The leaf size passed to the k-d tree algorithm.
#'   This affects the construction of the tree and the speed of the nearest neighbor
#'   queries. Default = 30.
#' @param metric The metric used for the distance computation.
#'   Available metrics are 'euclidean', 'l2', 'sqeuclidean' (squared Euclidean distances),
#'   'manhattan', 'l1', 'cityblock', 'cosine', 'minkowski', or 'inner_product'.
#'   Default = 'euclidean'.
#' @param p The power parameter used for the Minkowski metric. For p = 1.0,
#'   this defaults to 'manhattan' metric and for p = 2.0 this defaults to 'euclidean' metric.
#'   p is only used for Minkowski distance and will be ignored otherwise. Will return an
#'   error when p is not positive. Default p = 2.0.
#' @param radius The radius of the neighborhood to consider for
#'   \code{$radius_neighbors()} queries. Default = 1.0.
#' @param outlier_handling How to handle outliers (samples with no neighbors within
#'   the search radius). Available options are NULL, 'most_frequent' or a manual
#'   value that must be an integer in case of classification or a numeric in case of
#'   regression. NULL throws an error, manual value fills output with the specified
#'   label or target, and 'most_frequent' assigns to outliers the most frequent class
#'   label for classification or the mean of the target values for regression.
#'   Default = NULL.
#' @param check_data Whether to check the data for NaNs. Default = FALSE.
#' @export
#' @examples
#' # Unsupervised nearest neighbor search
#' data <- matrix(rnorm(200), nrow = 100, ncol = 2)
#' nn_model <- nearest_neighbors$new(n_neighbors = 5)
#' nn_model$fit(data)
#' neighbors <- nn_model$kneighbors(data[1:10, ])
#'
#' # Classification
#' labels <- sample(c(0L, 1L), 100, replace = TRUE)
#' nn_model$fit(data, labels)
#' predictions <- nn_model$classifier_predict(data[1:10, ])
#'
#' # Regression
#' targets <- rnorm(100)
#' nn_model$fit(data, targets)
#' predictions <- nn_model$regressor_predict(data[1:10, ])
nearest_neighbors <- R6::R6Class("nearest_neighbors",
    private = list(
        .n_neighbors = NULL, .p = NULL, .radius = NULL,
        .outlier_handling = NULL, .y = NULL,
        .nearest_neighbors = NULL,
        .set_outlier_info = function(type = "classifier") {
            outlier_handling <- "none"
            outlier_label <- 0L
            outlier_target <- 0.0

            if (is.null(private$.outlier_handling)) {
                outlier_handling <- "none"
            } else if (identical(private$.outlier_handling, "most_frequent")) {
                outlier_handling <- "most_frequent"
            } else {
                outlier_handling <- "manual"
                if (type == "classifier") {
                    if (!is.numeric(private$.outlier_handling) ||
                        length(private$.outlier_handling) != 1 ||
                        is.na(private$.outlier_handling) ||
                        private$.outlier_handling != as.integer(private$.outlier_handling)) {
                        stop(
                            "outlier_handling must be NULL, 'most_frequent', or an integer for classification."
                        )
                    }
                    outlier_label <- as.integer(private$.outlier_handling)
                } else {
                    if (!is.numeric(private$.outlier_handling) ||
                        length(private$.outlier_handling) != 1 ||
                        is.na(private$.outlier_handling)) {
                        stop(
                            "outlier_handling must be NULL, 'most_frequent', or numeric for regression."
                        )
                    }
                    outlier_target <- as.double(private$.outlier_handling)
                }
            }

            private$.nearest_neighbors$set_outlier_info(
                outlier_handling, outlier_label, outlier_target)
        }
    ),
    public = list(

        #' @description Create a new nearest_neighbors object.
        initialize = function(n_neighbors = 5, weights = "uniform", algorithm = "auto",
                              leaf_size = 30, metric = "euclidean", p = 2.0,
                              radius = 1.0, outlier_handling = NULL, check_data = FALSE) {
            if (!is.null(n_neighbors) &&
                (!is.numeric(n_neighbors) || length(n_neighbors) != 1 ||
                 is.na(n_neighbors) || n_neighbors != as.integer(n_neighbors))) {
                stop(
                    paste0(
                        "Number of neighbors must be an integer or NULL. Got n_neighbors=",
                        n_neighbors
                    )
                )
            }
            
            private$.n_neighbors <- n_neighbors
            if (is.null(n_neighbors)) {
                n_neighbors <- 1L
            }

            if (!is.null(radius) &&
                (!is.numeric(radius) || length(radius) != 1 || !is.finite(radius))) {
                stop(paste0("Radius must be a number or NULL. Got radius=", radius))
            } else if (!is.null(radius) && radius < 0.0) {
                stop(paste0("Radius must be positive. Got radius=", radius))
            }

            private$.nearest_neighbors <- aoclda:::rccp_NearestNeighbors$new(
                n_neighbors, algorithm, as.integer(leaf_size),
                metric, weights, check_data)

            private$.p <- p
            private$.radius <- radius
            private$.outlier_handling <- outlier_handling
        },

        #' @description Fit the nearest neighbors model from the training data set provided.
        #' @param X The feature matrix on which to fit the model.
        #'   Its shape is (n_samples, n_features).
        #' @param y Labels for classification or targets for regression.
        #'   Its shape is (n_samples). If NULL, only unsupervised queries are available.
        #'   That is, only \code{$kneighbors()} and \code{$radius_neighbors()} can be used.
        #' @return Returns the instance itself.
        fit = function(X, y = NULL) {
            if (is.null(X))
                stop("fit() is not implemented for X=NULL")
            X <- check_convert_data(X, "float", "matrix")
            fit_radius <- if (is.null(private$.radius)) 0.0 else as.double(private$.radius)
            private$.nearest_neighbors$fit(X, p = private$.p, radius = fit_radius)
            private$.y <- y
            invisible(self)
        },

        #' @description Compute the indices of the nearest neighbors for each test data
        #'   point, and optionally the corresponding distances.
        #' @param X The test data for which the nearest neighbors are required.
        #'   Its shape is (n_queries, n_features).
        #' @param n_neighbors The number of neighbors. If this is NULL, then
        #'   the number of neighbors set in the constructor will be used instead.
        #'   Default=NULL.
        #' @param return_distance Denotes whether to return the distances or not.
        #'   Default=TRUE.
        #' @return A named list with \code{$distances} (matrix of shape
        #'   (n_queries, n_neighbors)): the corresponding distances to each neighbor
        #'   (only returned if return_distance=TRUE), and \code{$indices} (integer matrix
        #'   of shape (n_queries, n_neighbors)): the indices to each neighbor.
        kneighbors = function(X, n_neighbors = NULL, return_distance = TRUE) {
            if (is.null(X))
                stop("kneighbors() is not implemented for X=NULL")
            X <- check_convert_data(X, "float", "matrix")

            if (!is.null(n_neighbors) &&
                (!is.numeric(n_neighbors) || length(n_neighbors) != 1 ||
                 is.na(n_neighbors) || n_neighbors != as.integer(n_neighbors))) {
                stop(
                    paste0(
                        "Number of neighbors must be an integer or NULL. Got n_neighbors=",
                        n_neighbors
                    )
                )
            } else if (!is.null(n_neighbors) && n_neighbors != 0L && n_neighbors <= 0L) {
                stop(
                    paste0(
                        "Number of neighbors must be positive or NULL. Got n_neighbors=",
                        n_neighbors
                    )
                )
            } else if ((is.null(n_neighbors) || n_neighbors == 0L) &&
                       is.null(private$.n_neighbors)) {
                stop(
                    "n_neighbors cannot be NULL in kneighbors() if it was NULL in the constructor."
                )
            }

            if (is.null(n_neighbors)) {
                n_neighbors <- 0L
            } else {
                n_neighbors <- as.integer(n_neighbors)
            }

            if (return_distance)
                return(private$.nearest_neighbors$kneighbors(X, n_neighbors))
            return(private$.nearest_neighbors$kneighbors_indices(X, n_neighbors))
        },

        #' @description
        #' Compute the indices of the radius neighbors for each test data point,
        #' and optionally the corresponding distances.
        #'
        #' @param X The test data for which the radius neighbors are required.
        #'   Its shape is (n_queries, n_features).
        #' @param radius The radius within which to search for neighbors.
        #'   Default=NULL.
        #' @param return_distance Denotes whether to return the distances or not.
        #'   Default=TRUE.
        #' @param sort_results Whether to sort the results by distance.
        #'   Default=FALSE.
        #' @return A list of length n_queries. Each element contains
        #'   the corresponding distances to each neighbor (only when
        #'   return_distance=TRUE) and the indices to each neighbor.
        radius_neighbors = function(X, radius = NULL, return_distance = TRUE,
                                    sort_results = FALSE) {
            if (sort_results && !return_distance)
                stop("sort_results=TRUE requires return_distance=TRUE")
            if (is.null(X))
                stop("radius_neighbors() is not implemented for X=NULL")
            X <- check_convert_data(X, "float", "matrix")

            if (!is.null(radius) &&
                (!is.numeric(radius) || length(radius) != 1 || !is.finite(radius))) {
                stop(paste0("Radius must be a number or NULL. Got radius=", radius))
            }
            else if (is.null(radius) && is.null(private$.radius)) {
                stop("radius cannot be NULL in radius_neighbors() if it was NULL in the constructor.")
            }
            if (!is.null(radius)) {
                if (radius < 0.0) {
                    stop(paste0("Radius must be non-negative or NULL. Got radius=", radius))
                }
            } else {
                radius = -1.0
            }

            if (return_distance)
                return(private$.nearest_neighbors$radius_neighbors(X, radius, sort_results))
            return(private$.nearest_neighbors$radius_neighbors_indices(X, radius))
        },

        #' @description Compute the probability estimates for each of the available classes.
        #' @param X The test data for which predictions are required.
        #'   Its shape is (n_queries, n_features).
        #' @param search_mode The search mode to use for finding neighbors.
        #'   Options are 'knn' (k-nearest neighbors) or 'radius_neighbors'
        #'   (radius-based search). Default = 'knn'.
        #' @return Matrix of shape (n_queries, n_classes): The class probabilities
        #'   of the test data. Classes are sorted in ascending order.
        classifier_predict_proba = function(X, search_mode = "knn") {
            if (is.null(private$.y))
                stop("Labels y were not provided during fit(). Classification is not available.")
            if (is.null(X))
                stop("Predicting probabilities is not implemented for X=NULL")
            X <- check_convert_data(X, "float", "matrix")
            y <- check_convert_data(private$.y, "da_int", "vector")
            if ((search_mode == "radius_neighbors") && is.null(private$.radius))
                stop("radius was NULL in the constructor.")
            private$.nearest_neighbors$set_labels(y)
            if (search_mode == "radius_neighbors") {
                private$.set_outlier_info("classifier")
            }
            return(private$.nearest_neighbors$classifier_predict_proba(X, search_mode))
        },

        #' @description Compute the predicted labels for the test data (classification).
        #' @param X The test data for which predictions are required.
        #'   Its shape is (n_queries, n_features).
        #' @param search_mode The search mode to use for finding neighbors.
        #'   Options are 'knn' (k-nearest neighbors) or 'radius_neighbors'
        #'   (radius-based search). Default = 'knn'.
        #' @return Integer vector of shape (n_queries): The predicted labels of the
        #'   test data.
        classifier_predict = function(X, search_mode = "knn") {
            if (is.null(private$.y))
                stop("Labels y were not provided during fit(). Classification is not available.")
            if (is.null(X))
                stop("Predicting labels is not implemented for X=NULL")
            X <- check_convert_data(X, "float", "matrix")
            y <- check_convert_data(private$.y, "da_int", "vector")
            if ((search_mode == "radius_neighbors") && is.null(private$.radius))
                stop("radius was NULL in the constructor.")
            private$.nearest_neighbors$set_labels(y)
            if (search_mode == "radius_neighbors") {
                private$.set_outlier_info("classifier")
            }
            return(private$.nearest_neighbors$classifier_predict(X, search_mode))
        },

        #' @description Compute the predicted target values for the test data (regression).
        #' @param X The test data for which predictions are required.
        #'   Its shape is (n_queries, n_features).
        #' @param search_mode The search mode to use for finding neighbors.
        #'   Options are 'knn' (k-nearest neighbors) or 'radius_neighbors'
        #'   (radius-based search). Default = 'knn'.
        #' @return Numeric vector of shape (n_queries): The predicted target values
        #'   of the test data.
        regressor_predict = function(X, search_mode = "knn") {
            if (is.null(private$.y))
                stop("Targets y were not provided during fit(). Regression is not available.")
            if (is.null(X))
                stop("Predicting target values is not implemented for X=NULL")
            X <- check_convert_data(X, "float", "matrix")
            y <- check_convert_data(private$.y, "float", "vector")
            if ((search_mode == "radius_neighbors") && is.null(private$.radius))
                stop("radius was NULL in the constructor.")
            private$.nearest_neighbors$set_targets(y)
            if (search_mode == "radius_neighbors") {
                private$.set_outlier_info("regressor")
            }
            return(private$.nearest_neighbors$regressor_predict(X, search_mode))
        },

        #' @description Save the model to an RDS file. Not intended for direct use;
        #' use \code{save_model()} from utils instead.
        #' @param path File path.
        save = function(path) {
            saveRDS(list(
                class  = "nearest_neighbors",
                wrapper_state = list(
                    n_neighbors = private$.n_neighbors,
                    radius = private$.radius,
                    p = private$.p,
                    y = private$.y,
                    outlier_handling = private$.outlier_handling
                ),
                cpp_state = private$.nearest_neighbors$save_model()), path)
            invisible(self)
        },

        #' @description Load previously saved state into the current object. Not intended
        #' for direct use; use \code{load_model()} from utils instead.
        #' @param cpp_state Raw list returned by save_model() at the C++ layer.
        #' @param wrapper_state Optional named list produced by \code{$save()}.
        load_state = function(cpp_state, wrapper_state = NULL) {
            private$.nearest_neighbors$load_model(cpp_state)
            if (!is.null(wrapper_state)) {
                private$.n_neighbors <- if (is.null(wrapper_state$n_neighbors)) {
                    NULL
                } else {
                    as.integer(wrapper_state$n_neighbors)
                }
                private$.radius <- wrapper_state$radius
                private$.p <- wrapper_state$p
                private$.y <- wrapper_state$y
                private$.outlier_handling <- wrapper_state$outlier_handling
            }
            invisible(self)
        },

        #' @description Print model version metadata.
        print_model_versions = function() {
            private$.nearest_neighbors$print_model_versions()
            invisible(self)
        }
    )
)

#' An Approximate Nearest Neighbors object.
#'
#' @param n_neighbors The number of nearest neighbors for
#'   \code{$kneighbors()} queries. Default = 5.
#' @param algorithm The algorithm used to compute
#'   the approximate nearest neighbors. Available options are 'auto', 'ivfflat',
#'   and 'ivfsq'. 'auto' selects 'ivfflat'. Default = 'auto'.
#' @param metric The metric used for the distance computation.
#'   Available metrics are 'euclidean', 'sqeuclidean' (squared Euclidean distances),
#'   'cosine', and 'inner_product'. Default = 'sqeuclidean'.
#' @param n_list The number of inverted file lists (centroids) to construct.
#'   Default = 1.
#' @param n_probe The number of lists to probe at search time.
#'   Default = 1.
#' @param kmeans_iter Maximum number of k-means iterations to
#'   perform during training. Default = 10.
#' @param train_fraction Proportion of training data to use for
#'   k-means clustering. Default = 1.0.
#' @param seed Seed for random number generation. Set to -1 for
#'   non-deterministic results. Default = 0.
#' @param quantization Character, storage format for indexed vectors. Available
#'   options are 'int8', which applies scalar quantization, 'int4', which stores two
#'   signed codes per byte, and 'bfloat16', which stores rounded bfloat16 values.
#'   Quantization can reduce index storage and improve search performance, depending
#'   on the workload, but may change computed distances and neighbor ordering.
#'   Quantization is used only when algorithm is 'ivfsq'. Default = 'bfloat16'.
#' @param quantization_scheme Character, scheme used to map values to quantized codes.
#'   For the 'int4' and 'int8' formats, 'per_dimension' learns a separate range for
#'   each feature and 'uniform' learns one range across all features. The
#'   'direct_signed' and 'direct_unsigned' schemes use the format's fixed signed and
#'   unsigned ranges, respectively. bfloat16 storage requires 'direct_signed', which
#'   performs direct round-to-nearest-even conversion without scaling.
#'   Default = 'per_dimension'.
#' @param quantization_residual Logical, whether to encode each indexed vector as its
#'   residual from the centroid of its list instead of encoding the raw vector.
#'   Residual encoding is enabled by default. Combining residual encoding with
#'   'direct_signed' or 'direct_unsigned' is not recommended because residuals are not
#'   guaranteed to fit the fixed direct ranges and are clipped when out of range. Use
#'   'per_dimension' or 'uniform' with residual encoding. Default = TRUE.
#' @param check_data Whether to check the data for NaNs. Default = FALSE.
#' @export
#' @examples
#' set.seed(1)
#' X_train <- matrix(rnorm(1000), nrow = 200, ncol = 5)
#' X_test  <- matrix(rnorm(50),   nrow = 10,  ncol = 5)
#'
#' ann <- approximate_neighbors$new(n_neighbors = 3, n_list = 4)
#' ann$train_and_add(X_train)
#' result <- ann$kneighbors(X_test)
#' print(result$indices)
approximate_neighbors <- R6::R6Class("approximate_neighbors",
    private = list(
        .n_probe = NULL, .train_fraction = NULL, .approximate_neighbors = NULL
    ),
    active = list(
        #' @field n_probe Integer: The number of lists to probe at search time.
        n_probe = function(value) {
            if (missing(value)) {
                private$.n_probe
            } else {
                value <- as.integer(value)
                private$.approximate_neighbors$set_n_probe_opt(value)
                private$.n_probe <- value
            }
        },
        #' @field cluster_centroids Matrix: The centroid vectors computed during training.
        cluster_centroids = function() {
            return(private$.approximate_neighbors$get_cluster_centroids())
        },
        #' @field list_sizes Integer vector: The number of vectors assigned to each inverted list.
        list_sizes = function() {
            return(private$.approximate_neighbors$get_list_sizes())
        },
        #' @field n_list Integer: The number of inverted file lists.
        n_list = function() {
            return(private$.approximate_neighbors$get_n_list())
        },
        #' @field n_index Integer: The number of data points added to the index.
        n_index = function() {
            return(private$.approximate_neighbors$get_n_index())
        },
        #' @field n_features Integer: The number of features in the indexed data.
        n_features = function() {
            return(private$.approximate_neighbors$get_n_features())
        },
        #' @field kmeans_iter Integer: The number of k-means iterations performed during training.
        kmeans_iter = function() {
            return(private$.approximate_neighbors$get_kmeans_iter())
        }
    ),
    public = list(
        #' @description Create a new \code{approximate_neighbors} object.
        initialize = function(n_neighbors = 5L, algorithm = "auto",
                              metric = "sqeuclidean", n_list = 1L, n_probe = 1L,
                              kmeans_iter = 10L, train_fraction = 1.0, seed = 0L,
                              quantization = "bfloat16",
                              quantization_scheme = "per_dimension",
                              quantization_residual = TRUE, check_data = FALSE) {
            private$.approximate_neighbors <- aoclda:::rccp_ApproximateNeighbors$new(
                as.integer(n_neighbors), algorithm, metric,
                as.integer(n_list), as.integer(n_probe),
                as.integer(kmeans_iter), as.integer(seed),
                quantization, quantization_scheme, quantization_residual, check_data)
            private$.n_probe <- as.integer(n_probe)
            private$.train_fraction <- as.double(train_fraction)
        },

        #' @description
        #' Train the model by computing centroids using k-means clustering.
        #' Data is not added to the index; use \code{$add()} to populate the
        #' index after training.
        #' @param X_train The training data matrix.
        #'   Its shape is (n_samples, n_features).
        #' @return Returns the instance itself.
        train = function(X_train) {
            X_train <- check_convert_data(X_train, "float", "matrix")
            private$.approximate_neighbors$train(X_train, private$.train_fraction)
            invisible(self)
        },

        #' @description
        #' Train the model and add the training data to the index.
        #' @param X_train The training data matrix.
        #'   Its shape is (n_samples, n_features).
        #' @return Returns the instance itself.
        train_and_add = function(X_train) {
            X_train <- check_convert_data(X_train, "float", "matrix")
            private$.approximate_neighbors$train_and_add(X_train, private$.train_fraction)
            invisible(self)
        },

        #' @description
        #' Add data points to the index. The model must be trained first.
        #' @param X_add The data matrix to add to the index.
        #'   Its shape is (n_samples_add, n_features).
        #' @return Returns the instance itself.
        add = function(X_add) {
            X_add <- check_convert_data(X_add, "float", "matrix")
            private$.approximate_neighbors$add(X_add)
            invisible(self)
        },

        #' @description
        #' Compute the approximate k nearest neighbors for each query point.
        #' @param X_test The query data matrix.
        #'   Its shape is (n_queries, n_features).
        #' @param n_neighbors The number of neighbors to return.
        #'   If less than or equal to zero, uses the value set during initialization.
        #' @param return_distance Whether to return the distances.
        #'   Default = TRUE.
        #' @return A named list with \code{$distances} (matrix of shape
        #'   (n_queries, n_neighbors)): the distances to each neighbor
        #'   (only returned if return_distance=TRUE), and \code{$indices}
        #'   (integer matrix of shape (n_queries, n_neighbors)): the indices
        #'   of each neighbor.
        kneighbors = function(X_test, n_neighbors = 0L, return_distance = TRUE) {
            X_test <- check_convert_data(X_test, "float", "matrix")
            n_neighbors <- as.integer(n_neighbors)
            if (return_distance) {
                return(private$.approximate_neighbors$kneighbors(X_test, n_neighbors))
            } else {
                return(private$.approximate_neighbors$kneighbors_indices(X_test, n_neighbors))
            }
        },

        #' @description Save the model to an RDS file. Not intended for direct use;
        #' use \code{save_model()} from utils instead.
        #' @param path File path.
        save = function(path) {
            saveRDS(list(
                class  = "approximate_neighbors",
                wrapper_state = list(
                    n_probe = private$.n_probe,
                    train_fraction = private$.train_fraction
                ),
                cpp_state = private$.approximate_neighbors$save_model()), path)
            invisible(self)
        },

        #' @description Load previously saved state into the current object. Not intended
        #' for direct use; use \code{load_model()} from utils instead.
        #' @param cpp_state Raw list returned by save_model() at the C++ layer.
        #' @param wrapper_state Optional named list produced by \code{$save()}.
        load_state = function(cpp_state, wrapper_state = NULL) {
            private$.approximate_neighbors$load_model(cpp_state)
            if (!is.null(wrapper_state)) {
                private$.n_probe <- as.integer(wrapper_state$n_probe)
                private$.train_fraction <- wrapper_state$train_fraction
            }
            invisible(self)
        },

        #' @description Print model version metadata.
        print_model_versions = function() {
            private$.approximate_neighbors$print_model_versions()
            invisible(self)
        }
    )
)
