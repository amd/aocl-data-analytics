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

#' t-SNE embedding.
#'
#' @param n_components Integer, number of embedding dimensions. Default=2.
#' @param perplexity Numeric, target perplexity. Default=30.0.
#' @param learning_rate Numeric or character, gradient descent learning rate.
#'   Use any non-positive value or "auto" for automatic selection:
#'   max(n_samples / early_exaggeration / 4, 50). Default=-1 (auto).
#' @param max_iter Integer, maximum number of gradient descent iterations. Default=1000.
#' @param n_iter_without_progress Integer, stop if no progress is made for this many
#'   iterations. Default=300.
#' @param min_grad_norm Numeric, stop if the gradient norm is below this threshold.
#'   Default=1e-7.
#' @param early_exaggeration Numeric, early exaggeration factor. Default=12.0.
#' @param theta Numeric, Barnes-Hut approximation parameter in [0, 1] (0 for exact).
#'   Default=0.5.
#' @param init Character or matrix, initialization method. Can be 'pca', 'random',
#'   or a matrix of shape (n_samples, n_components) providing the initial embedding.
#'   Default='pca'.
#' @param seed Integer, seed for randomness; set to -1 for non-deterministic. Default=0.
#' @param mixed_precision Logical, whether to use mixed precision iterative refinement,
#'   in which lower precision arithmetic is used before switching to the working precision
#'   for the final iterations. Default=FALSE.
#' @param low_precision_max_iter Integer, if mixed precision iterative refinement is
#'   enabled, maximum number of iterations for the low precision phase. Default=200.
#' @param low_precision_min_grad_norm Numeric, if mixed precision iterative refinement
#'   is enabled, gradient norm convergence threshold for the low precision phase.
#'   Default=1e-4.
#' @param check_data Logical, whether to check data for NaNs. Default=FALSE.
#' @export
#' @examples
#' data <- matrix(rnorm(500), nrow = 50, ncol = 10)
#' model <- tsne$new(n_components = 2)
#' embedding <- model$fit_transform(data)
tsne <- R6::R6Class("tsne",
    private = list(
        .tsne = NULL,
        .perplexity = NULL,
        .embedding_ = NULL,
        .learning_rate = NULL,
        .n_iter_without_progress = NULL,
        .min_grad_norm = NULL,
        .early_exaggeration = NULL,
        .theta = NULL,
        .low_precision_min_grad_norm = NULL,
        .init_embedding = NULL,

        .prepare_fit = function(X) {
            X <- check_convert_data(X, "float", "matrix")

            if (is.character(private$.learning_rate)) {
                if (private$.learning_rate == "auto") {
                    private$.learning_rate <- -1.0
                } else {
                    stop("learning_rate must be a positive float, a non-positive value (auto), or 'auto'.")
                }
            }

            init_emb <- private$.init_embedding
            if (!is.null(init_emb) && nrow(init_emb) != nrow(X))
                stop(paste0("init array has ", nrow(init_emb),
                            " rows but X has ", nrow(X), " samples."))

            list(X = X, init_emb = init_emb)
        }
    ),
    public = list(

        #' @description Create a new tsne object.
        initialize = function(n_components = 2L, perplexity = 30.0,
                              learning_rate = -1.0, max_iter = 1000L,
                              n_iter_without_progress = 300L, min_grad_norm = 1e-7,
                              early_exaggeration = 12.0, theta = 0.5, init = 'pca',
                              seed = 0L, mixed_precision = FALSE,
                              low_precision_max_iter = 200L,
                              low_precision_min_grad_norm = 1e-4,
                              check_data = FALSE) {

            private$.perplexity <- perplexity
            private$.learning_rate <- learning_rate
            private$.n_iter_without_progress <- as.integer(n_iter_without_progress)
            private$.min_grad_norm <- min_grad_norm
            private$.early_exaggeration <- early_exaggeration
            private$.theta <- theta
            private$.low_precision_min_grad_norm <- low_precision_min_grad_norm

            if (is.character(init)) {
                init_str <- init
            } else {
                init_arr <- as.matrix(init)
                if (length(dim(init_arr)) != 2)
                    stop("init array must be 2-dimensional with shape (n_samples, n_components).")
                if (ncol(init_arr) != n_components)
                    stop(paste0("init array has ", ncol(init_arr),
                                " columns but n_components=", n_components, "."))
                if (!all(is.finite(init_arr)))
                    stop("init array must contain only finite values.")
                private$.init_embedding <- init_arr
                init_str <- "supplied"
            }

            private$.tsne <- aoclda:::rccp_TSNE$new(
                as.integer(n_components), as.integer(max_iter), init_str,
                as.integer(seed), check_data, mixed_precision,
                as.integer(low_precision_max_iter))
        },

        #' @description Compute t-SNE embedding for the supplied data matrix.
        #' @param X The data matrix with shape (n_samples, n_features).
        #' @return Returns the instance itself.
        fit = function(X) {
            p <- private$.prepare_fit(X)
            private$.tsne$fit(
                p$X, private$.perplexity, private$.learning_rate,
                private$.early_exaggeration, private$.theta,
                private$.n_iter_without_progress, private$.min_grad_norm,
                p$init_emb, private$.low_precision_min_grad_norm)
            invisible(self)
        },

        #' @description Compute t-SNE embedding and return the embedding matrix.
        #' @param X The data matrix with shape (n_samples, n_features).
        #' @return Matrix of shape (n_samples, n_components) containing the embedding.
        fit_transform = function(X) {
            p <- private$.prepare_fit(X)
            private$.embedding_ <- private$.tsne$fit_transform(
                p$X, private$.perplexity, private$.learning_rate,
                private$.early_exaggeration, private$.theta,
                private$.n_iter_without_progress, private$.min_grad_norm,
                p$init_emb, private$.low_precision_min_grad_norm)
            return(private$.embedding_)
        }
    ),
    active = list(
        #' @field embedding Matrix of shape (n_samples, n_components): the embedding.
        embedding = function() {
            return(private$.tsne$get_embedding())
        },
        #' @field n_samples Integer: the number of samples in the data matrix used.
        n_samples = function() {
            return(private$.tsne$get_n_samples())
        },
        #' @field n_features Integer: the number of features in the data matrix.
        n_features = function() {
            return(private$.tsne$get_n_features())
        },
        #' @field n_components Integer: the number of embedding dimensions.
        n_components = function() {
            return(private$.tsne$get_n_components())
        },
        #' @field n_iter Integer: the number of iterations performed.
        n_iter = function() {
            return(private$.tsne$get_n_iter())
        },
        #' @field lp_n_iter Integer: the number of low precision iterations performed
        #'   if mixed precision was enabled.
        lp_n_iter = function() {
            return(private$.tsne$get_lp_n_iter())
        },
        #' @field kl_divergence Numeric: the final KL divergence.
        kl_divergence = function() {
            return(private$.tsne$get_kl_divergence())
        }
    )
)
