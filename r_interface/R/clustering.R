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

#' k-means clustering.
#'
#' Partition a data matrix into clusters using k-means clustering.
#'
#' @param n_clusters Integer, number of clusters to form. Default=1.
#' @param initialization_method Character, the method used to find the initial cluster centres.
#'   It can take the values 'k-means++', 'random' (initial clusters are chosen randomly from
#'   the sample data points), 'random partitions' (sample points are assigned to a random
#'   cluster and the corresponding cluster centres are computed and used as the starting
#'   point), or 'afk-mc2' (initial cluster centres are chosen using the AFK-MC^2
#'   algorithm of Bachem et al. (2016)). Default: 'k-means++'.
#' @param C Matrix, the matrix of initial cluster centres. It has
#'   shape (n_clusters, n_features). If supplied, these centres will be used as the
#'   starting point for the first iteration, otherwise the initialization method specified
#'   above will be used. Default = NULL.
#' @param n_init Integer, number of runs with different random seeds (ignored if you specify
#'   initial cluster centres). Default=10.
#' @param max_iter Integer, maximum number of iterations. Default=300.
#' @param seed Integer, seed for random number generation; set to -1 for non-deterministic
#'   results. Default=-1.
#' @param algorithm Character, the algorithm used to compute the clusters. It can take the
#'   values 'elkan', 'lloyd', 'macqueen' or 'hartigan-wong'. Default = 'lloyd'.
#' @param distance Character, the distance metric used for clustering. It can take the
#'   values 'euclidean' or 'cosine'. If 'cosine' is selected, spherical k-means
#'   is performed, which optimizes cosine similarity between data points and cluster
#'   centres. Note that cosine distance is not compatible with the Hartigan-Wong
#'   algorithm. Default = 'euclidean'.
#' @param normalize_data Logical, whether to normalize the input data before clustering.
#'   Only used if distance is set to cosine. Default = FALSE.
#' @param tol Numeric, the convergence tolerance for the iterations. Default = 1.0e-4.
#' @param empty_clusters Character, determines behaviour in the case that all sample points
#'   have been assigned to fewer than k clusters. If set to 'ignore' then empty clusters are
#'   allowed and the algorithm proceeds as normal. If set to 'error' then an error is raised
#'   when an empty cluster is encountered (if n_init > 1 then the next initialization is
#'   attempted, so an error will only be returned if all initializations led to empty
#'   clusters). If set to 'split' then the point farthest from its closest cluster centre
#'   is chosen and assigned to the empty cluster. Note that if the Hartigan-Wong algorithm
#'   is used then empty_clusters will be set to 'error' internally.
#' @param afk_mcmc_samples Integer, if the AFK-MC^2 initialization method is used, the number
#'   of MCMC samples to use in the algorithm. Default = 50.
#' @param mixed_precision Logical, whether to use mixed precision iterative refinement,
#'   in which lower precision arithmetic is used before switching to the working precision
#'   for the final iterations. Default = FALSE.
#' @param low_precision_max_iter Integer, if mixed precision iterative refinement is enabled,
#'   maximum number of iterations for the low precision phase. Default = 200.
#' @param low_precision_tol Numeric, if mixed precision iterative refinement is enabled,
#'   convergence tolerance for the low precision phase. Default = 1e-2.
#' @param check_data Logical, whether to check the data for NaNs. Default = FALSE.
#' @export
#' @examples
#' # Basic k-means clustering
#' data <- matrix(rnorm(200), nrow = 100, ncol = 2)
#' kmeans_model <- kmeans(n_clusters = 3)
#' kmeans_model$fit(data)
#' labels <- kmeans_model$labels
#' centers <- kmeans_model$cluster_centres
kmeans <- R6::R6Class("kmeans",
    private = list(
        .C = NULL, .tol = NULL, .mixed_precision = NULL,
        .low_precision_max_iter = NULL, .low_precision_tol = NULL,
        .distance = NULL, .normalize_data = NULL,
        .empty_clusters = NULL, .afk_mcmc_samples = NULL,
        .kmeans = NULL
    ),
    public = list(
        
        #' @description Create a new kmeans object.
        initialize = function(n_clusters = 1, initialization_method = 'k-means++', C = NULL, n_init = 10,
                            max_iter = 300, seed = -1, algorithm = 'lloyd',
                            distance = 'euclidean', normalize_data = FALSE,
                            tol = 1.0e-4, empty_clusters = 'ignore',
                            afk_mcmc_samples = 50L,
                            mixed_precision = FALSE, low_precision_max_iter = 200L,
                            low_precision_tol = 1e-2, check_data = FALSE) {
            
            private$.kmeans <- aoclda:::rccp_kmeans$new(
                as.integer(n_clusters), initialization_method,
                as.integer(n_init), as.integer(max_iter), as.integer(seed), algorithm,
                distance, normalize_data, empty_clusters, as.integer(afk_mcmc_samples),
                mixed_precision, as.integer(low_precision_max_iter), check_data)
            
            private$.C <- C
            private$.tol <- tol
            private$.low_precision_tol <- low_precision_tol
            private$.low_precision_max_iter <- as.integer(low_precision_max_iter)
            private$.mixed_precision <- mixed_precision
            private$.afk_mcmc_samples <- as.integer(afk_mcmc_samples)
            private$.normalize_data <- normalize_data
            private$.distance <- distance
            private$.empty_clusters <- empty_clusters
        },
        
        #' @description Computes k-means clusters for the supplied data matrix,
        #'   optionally using the supplied centres as the starting point.
        #' @param A The data matrix with which to compute the k-means clusters.
        #'   It has shape (n_samples, n_features).
        #' @return Returns the instance itself.
        fit = function(A) {
            A <- check_convert_data(A, "float", "matrix")
            
            if (!is.null(private$.C)) {
                private$.C <- check_convert_data(private$.C, "float", "matrix")
            }
            
            private$.kmeans$fit(A, private$.C, private$.tol, private$.low_precision_tol)
            invisible(self)
        },
        
        #' @description Transform a data matrix into cluster distance space.
        #'
        #'   Transforms a data matrix X from the original coordinate system into the
        #'   new coordinates in which each dimension is the distance to the cluster
        #'   centres previously computed by \code{$fit()}.
        #' @param X The data matrix to be transformed. It has shape
        #'   (m_samples, m_features). Note that m_features must match n_features,
        #'   the number of features in the data matrix originally supplied to
        #'   \code{$fit()}.
        #' @return Matrix of shape (m_samples, n_clusters): the transformed matrix.
        transform = function(X) {
            X <- check_convert_data(X, "float", "matrix")
            return(private$.kmeans$transform(X))
        },
        
        #' @description Predict the cluster each sample in a data matrix belongs to.
        #'
        #'   For each sample in the data matrix Y find the closest cluster centre
        #'   out of the clusters previously computed in \code{$fit()}.
        #' @param Y The data matrix to be transformed. It has shape
        #'   (k_samples, k_features). Note that k_features must match n_features,
        #'   the number of features in the data matrix used in \code{$fit()}.
        #' @return Integer vector of shape (k_samples): the labels.
        predict = function(Y) {
            Y <- check_convert_data(Y, "float", "matrix")
            return(private$.kmeans$predict(Y))
        },

        #' @description Save the model to an RDS file. Not intended for direct use;
        #' use \code{save_model()} from utils instead.
        #' @param path File path.
        save = function(path) {
            saveRDS(list(
                class  = "kmeans",
                wrapper_state = list(
                    C = private$.C,
                    tol = private$.tol,
                    low_precision_tol = private$.low_precision_tol,
                    low_precision_max_iter = private$.low_precision_max_iter,
                    mixed_precision = private$.mixed_precision,
                    afk_mcmc_samples = private$.afk_mcmc_samples,
                    normalize_data = private$.normalize_data,
                    distance = private$.distance,
                    empty_clusters = private$.empty_clusters
                ),
                cpp_state = private$.kmeans$save_model()
            ), path)
            invisible(self)
        },

        #' @description Load previously saved state into the current object. Not intended
        #' for direct use; use \code{load_model()} from utils instead.
        #' @param cpp_state Raw list returned by save_model() at the C++ layer.
        #' @param wrapper_state Optional named list produced by \\code{$save()}.
        load_state = function(cpp_state, wrapper_state = NULL) {
            private$.kmeans$load_model(cpp_state)
            if (!is.null(wrapper_state)) {
                private$.C <- wrapper_state$C
                private$.tol <- wrapper_state$tol
                private$.low_precision_tol <- wrapper_state$low_precision_tol
                private$.low_precision_max_iter <- as.integer(wrapper_state$low_precision_max_iter)
                private$.mixed_precision <- wrapper_state$mixed_precision
                private$.afk_mcmc_samples <- wrapper_state$afk_mcmc_samples
                private$.normalize_data <- wrapper_state$normalize_data
                private$.distance <- wrapper_state$distance
                private$.empty_clusters <- wrapper_state$empty_clusters
            }
            invisible(self)
        },

        #' @description Print model version metadata.
        print_model_versions = function() {
            private$.kmeans$print_model_versions()
            invisible(self)
        }
    ),    
    active = list(
        #' @field cluster_centres Matrix of shape (n_clusters, n_features): the coordinates
        #'   of the cluster centres.
        cluster_centres = function() {
            return(private$.kmeans$get_cluster_centers())
        },
        
        #' @field labels Integer vector of shape (n_samples): the label (which cluster) of
        #'   each sample point in the data matrix.
        labels = function() {
            return(private$.kmeans$get_labels())
        },
        
        #' @field inertia Numeric scalar: the inertia (sum of the squared distance of each
        #'   sample to its closest cluster centre).
        inertia = function() {
            return(private$.kmeans$get_inertia())
        },
        
        #' @field n_iter Integer: the number of iterations performed in the k-means computation.
        n_iter = function() {
            return(private$.kmeans$get_n_iter())
        },
        
        #' @field lp_n_iter Integer: the number of low precision iterations performed if
        #'   mixed precision was enabled.
        lp_n_iter = function() {
            return(private$.kmeans$get_lp_n_iter())
        },
        
        #' @field n_samples Integer: the number of samples in the data matrix used.
        n_samples = function() {
            return(private$.kmeans$get_n_samples())
        },
        
        #' @field n_features Integer: the number of features in the data matrix.
        n_features = function() {
            return(private$.kmeans$get_n_features())
        },
        
        #' @field n_clusters Integer: the number of clusters found.
        n_clusters = function() {
            return(private$.kmeans$get_n_clusters())
        }
    )
)

#' DBSCAN clustering.
#'
#' Partition a data matrix into clusters using DBSCAN clustering.
#'
#' @param min_samples Integer, minimum number of neighborhood samples for a sample point to be
#'   considered a core point. Default = 5.
#' @param metric Character, the distance metric used to compare sample points. Available metrics
#'   are 'euclidean', 'l2', 'sqeuclidean' (squared Euclidean distances), 'manhattan', 'l1',
#'   'cityblock', 'cosine', or 'minkowski'. Default = 'euclidean'.
#' @param algorithm Character, the algorithm used to compute the clusters. Available options are
#'   'auto', 'ball_tree', 'brute' and 'kd_tree'. k-d trees are likely to be fastest for lower
#'   dimensional datasets, and ball trees may be preferred when data is not aligned along
#'   coordinate axes, but trees cannot be used with the cosine distance, the squared
#'   Euclidean distance, or with the Minkowski distance with power less than 1.0.
#'   Default = 'auto'.
#' @param leaf_size Integer, leaf size for the k-d tree algorithm. Default = 30.
#' @param eps Numeric, maximum distance between two samples for them to be considered in
#'   each other's neighborhood. Default = 0.5.
#' @param power Numeric, power used in computing the Minkowski metric. Default = 2.0.
#' @param check_data Logical, whether to check the data for NaNs. Default = FALSE.
#' @export
#' @examples
#' # Basic DBSCAN clustering
#' data <- matrix(rnorm(200), nrow = 100, ncol = 2)
#' dbscan_model <- DBSCAN$new(eps = 0.5, min_samples = 5)
#' dbscan_model$fit(data)
#' labels <- dbscan_model$labels
DBSCAN <- R6::R6Class("DBSCAN",
    private = list(
        .DBSCAN = NULL, .eps = NULL, .power = NULL
    ),
    public = list(
        
        #' @description Create a new DBSCAN object.
        initialize = function(min_samples = 5, metric = 'euclidean', algorithm = 'auto',
                            leaf_size = 30, eps = 0.5, power = 2.0, check_data = FALSE) {
            
            private$.DBSCAN <- aoclda:::rccp_DBSCAN$new(
                as.integer(min_samples), metric, algorithm, as.integer(leaf_size), check_data)
            
            private$.eps <- eps
            private$.power <- power
        },
        
        #' @description Computes DBSCAN clusters for the supplied data matrix.
        #' @param A The data matrix with which to compute the DBSCAN clusters.
        #'   It has shape (n_samples, n_features).
        #' @return Returns the instance itself.
        fit = function(A) {
            A <- check_convert_data(A, "float", "matrix")
            
            private$.DBSCAN$fit(A, private$.eps, private$.power)
            invisible(self)
        }
    ),
    
    active = list(
        #' @field labels Integer vector of shape (n_samples): the label (which cluster) of
        #'   each sample point in the data matrix. A label of -1 indicates that the point
        #'   has been classified as noise and has not been assigned to a cluster.
        labels = function() {
            return(private$.DBSCAN$get_labels())
        },
        
        #' @field core_sample_indices Integer vector of shape (n_core_samples): the indices
        #'   of the core samples in the data matrix.
        core_sample_indices = function() {
            return(private$.DBSCAN$get_core_sample_indices())
        },
        
        #' @field n_samples Integer: the number of samples in the data matrix.
        n_samples = function() {
            return(private$.DBSCAN$get_n_samples())
        },
        
        #' @field n_core_samples Integer: the number of core samples found in the data matrix.
        n_core_samples = function() {
            return(private$.DBSCAN$get_n_core_samples())
        },
        
        #' @field n_features Integer: the number of features in the data matrix.
        n_features = function() {
            return(private$.DBSCAN$get_n_features())
        },
        
        #' @field n_clusters Integer: the number of clusters found.
        n_clusters = function() {
            return(private$.DBSCAN$get_n_clusters())
        }
    )
)
