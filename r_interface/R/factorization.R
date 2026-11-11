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

#' Principal Component Analysis.
#'
#' Find all or some of the principal components of a data matrix.
#'
#' @param n_components Integer, number of components to keep. Default=1.
#' @param bias Character, whether to use unbiased or biased estimators for standard
#'   deviations and variances. It can take the values 'unbiased' or 'biased'.
#'   Default = 'unbiased'.
#' @param method Character, the method used to compute the PCA. Default = 'covariance'.
#'   If method = 'covariance' then the columns are mean-centered so the PCA is based on
#'   the covariance matrix.
#'   If method = 'correlation' then the columns are mean-centered and scaled by the
#'   standard deviation so the PCA is based on the correlation matrix.
#'   If method = 'svd' then no normalization occurs so the PCA reduces to the singular
#'   value decomposition.
#' @param solver Character, which LAPACK solver to use to compute the underlying singular
#'   value decomposition, allowed values: 'auto', 'gesdd', 'gesvd', 'gesvdx', 'syevd',
#'   'randomized'.
#'   If solver = 'syevd' then the SVD will be found by explicitly forming
#'   the covariance or correlation matrix and performing an eigendecomposition. This is
#'   very fast for tall, thin data matrices but for wider matrices it requires a lot of
#'   memory. The method is also more susceptible to ill-conditioning so must be used
#'   with care. It is incompatible with the 'store_U' option.
#'   If solver = 'auto' then internal heuristics will be used to choose from one of
#'   'gesdd', 'syevd' or 'randomized'.
#'   If solver = 'randomized' then a randomized SVD algorithm is used.
#'   This can be significantly faster for large matrices when only a small number of
#'   components is required. See n_oversamples, power_iterations, and
#'   power_normalization to control accuracy.
#'   Default = 'auto'.
#' @param store_U Logical, controls whether to store the matrix U from the singular value
#'   decomposition. This allows deterministic results regarding the sign of the principal
#'   components, at the expense of some extra computation. This option cannot be used if
#'   solver = 'syevd'. Default = FALSE.
#' @param whiten Logical, whether to whiten the data upon transformation. This divides
#'   each principal component by its corresponding singular value and multiplies the
#'   component by a dimensional factor so the transformed data, specifically that data
#'   used to fit the PCA, has a unit diagonal covariance matrix. Default = FALSE.
#' @param n_oversamples Integer, number of extra columns used in the randomized SVD to
#'   improve numerical accuracy. Higher values give better results at the cost of more
#'   computation. Only used when solver='randomized'. Default = 10.
#' @param power_iterations Integer, number of power iterations performed by the
#'   randomized SVD. Higher values improve accuracy for matrices whose singular values
#'   decay slowly. -1 selects an automatic heuristic. Only used when
#'   solver='randomized'. Default = -1.
#' @param power_normalization Character, normalization method used during power
#'   iterations in the randomized SVD. 'QR' uses QR decomposition (numerically stable);
#'   'LU' uses LU decomposition (faster, slightly less stable); 'none' skips
#'   normalization entirely (fastest, least stable). Only used when
#'   solver='randomized'. Default = 'QR'.
#' @param seed Integer, seed for random number generation. Set to -1 for
#'   non-deterministic results. Only used when solver='randomized'. Default = -1.
#' @param check_data Logical, whether to check the data for NaNs. Default = FALSE.
#' @export
#' @examples
#' # Basic PCA
#' data <- matrix(rnorm(200), nrow = 100, ncol = 2)
#' pca_model <- PCA$new(n_components = 2)
#' pca_model$fit(data)
#' 
#' # Transform data
#' transformed <- pca_model$transform(data)
#' 
#' # Get explained variance
#' variance <- pca_model$variance
PCA <- R6::R6Class("PCA",
    private = list(
        .pca = NULL
    ),
    public = list(
        
        #' @description Create a new PCA object.
        initialize = function(n_components = 1, bias = 'unbiased', method = 'covariance',
                            solver = 'auto', store_U = FALSE, whiten = FALSE,
                            n_oversamples = 10L, power_iterations = -1L,
                            power_normalization = 'QR', seed = -1L,
                            check_data = FALSE) {
            
            private$.pca <- aoclda:::rccp_PCADecomposition$new(
            as.integer(n_components), bias, method, solver, store_U, whiten,
            as.integer(n_oversamples), as.integer(power_iterations),
            tolower(power_normalization), as.integer(seed), check_data)
        },
        
        #' @description Computes the principal component analysis on the supplied data matrix.
        #' @param A The data matrix with which to compute the PCA. It has shape
        #'   (n_samples, n_features).
        #' @return Returns the instance itself.
        fit = function(A) {
            A <- check_convert_data(A, "float", "matrix")
            private$.pca$fit(A)
            invisible(self)
        },
        
        #' @description Transform a data matrix into new feature space.
        #'
        #'   Transforms a data matrix X from the original coordinate system into the new
        #'   coordinates previously computed by \code{$fit()}. The transformation is computed
        #'   by applying any standardization used on the original data matrix to X, then
        #'   projecting X into the previously computed principal components.
        #' @param X The data matrix to be transformed. It has shape
        #'   (m_samples, m_features). Note that m_features must match n_features,
        #'   the number of features in the data matrix originally supplied to
        #'   \code{$fit()}.
        #' @return Matrix of shape (m_samples, n_components): the transformed matrix.
        transform = function(X) {
            X <- check_convert_data(X, "float", "matrix")
            return(private$.pca$transform(X))
        },
        
        #' @description Transform a data matrix into the original coordinate space.
        #'
        #'   Transforms a data matrix Y in the new feature space back into the original
        #'   coordinate space used by the matrix which was supplied to \code{$fit()}. The
        #'   transformation is computed by projecting Y into the original coordinate space,
        #'   then inverting any standardization used on the original data matrix.
        #' @param Y The data matrix to be transformed. It has shape
        #'   (k_samples, k_features). Note that k_features must match n_components,
        #'   the number of principal components computed by \code{$fit()}.
        #' @return Matrix of shape (k_samples, n_features): the transformed matrix.
        inverse_transform = function(Y) {
            Y <- check_convert_data(Y, "float", "matrix")
            return(private$.pca$inverse_transform(Y))
        },

        #' @description Save the model to an RDS file. Not intended for direct use;
        #' use \code{save_model()} from utils instead.
        #' @param path File path.
        save = function(path) {
            saveRDS(list(
                class  = "PCA",
                cpp_state = private$.pca$save_model()
            ), path)
            invisible(self)
        },

        #' @description Load previously saved state into the current object. Not intended
        #' for direct use; use \code{load_model()} from utils instead.
        #' @param cpp_state Raw list returned by save_model() at the C++ layer.
        #' @param wrapper_state Optional named list produced by \\code{$save()}.
        load_state = function(cpp_state, wrapper_state = NULL) {
            private$.pca$load_model(cpp_state)
            invisible(self)
        },

        #' @description Print model version metadata.
        print_model_versions = function() {
            private$.pca$print_model_versions()
            invisible(self)
        }
    ),    
    active = list(
        #' @field principal_components Matrix of shape (n_components, n_features):
        #'   principal axes in feature space, representing the directions of maximum
        #'   variance in the data. Equivalently, the right singular vectors of the
        #'   normalized input data, parallel to its eigenvectors. The components are
        #'   sorted by decreasing variance.
        principal_components = function() {
            return(private$.pca$get_principal_components())
        },
        
        #' @field scores Matrix of shape (n_samples, n_components): the principal
        #'   component scores, U * Sigma.
        scores = function() {
            return(private$.pca$get_scores())
        },
        
        #' @field variance Numeric vector of shape (n_components): the amount of
        #'   variance explained by each of the selected components.
        variance = function() {
            return(private$.pca$get_variance())
        },
        
        #' @field total_variance Numeric scalar: the total amount of variance within
        #'   the dataset.
        total_variance = function() {
            return(private$.pca$get_total_variance())
        },
        
        #' @field u Matrix of shape (n_samples, n_samples): the matrix U from the SVD.
        u = function() {
            return(private$.pca$get_u())
        },
        
        #' @field sigma Numeric vector of shape (n_components): the diagonal values of
        #'   Sigma from the SVD.
        sigma = function() {
            return(private$.pca$get_sigma())
        },
        
        #' @field vt Matrix of shape (n_components, n_features): the matrix V^T from
        #'   the SVD.
        vt = function() {
            return(private$.pca$get_vt())
        },
        
        #' @field column_means Numeric vector of shape (n_features): the column means
        #'   of the data matrix. Computed if method = 'covariance' or method = 'correlation'.
        column_means = function() {
            return(private$.pca$get_column_means())
        },
        
        #' @field column_sdevs Numeric vector of shape (n_features): the column standard
        #'   deviations of the data matrix. Only computed if method = 'correlation'.
        column_sdevs = function() {
            return(private$.pca$get_column_sdevs())
        },
        
        #' @field n_samples Integer: the number of samples in the data matrix used to
        #'   compute the PCA.
        n_samples = function() {
            return(private$.pca$get_n_samples())
        },
        
        #' @field n_features Integer: the number of features in the data matrix used to
        #'   compute the PCA.
        n_features = function() {
            return(private$.pca$get_n_features())
        },
        
        #' @field n_components Integer: the number of components found in the PCA.
        n_components = function() {
            return(private$.pca$get_n_components())
        }
    )
)

#' Kernel principal component analysis (kernel PCA).
#'
#' Performs nonlinear dimensionality reduction using kernels.
#'
#' @param n_components Integer, number of components to keep. If 0, all non-zero
#'   components are kept. Default = 0.
#' @param kernel Character, kernel function to use. Possible values: 'linear',
#'   'poly', 'rbf', 'sigmoid', 'precomputed'. Default = 'linear'.
#' @param eigensolver Character, method to use for eigendecomposition of the kernel
#'   matrix. Possible values: 'auto', 'syevd', 'randomized'. If eigensolver = 'auto',
#'   then internal heuristics are used to choose between 'syevd' and 'randomized'.
#'   If eigensolver = 'randomized' then a randomized eigendecomposition algorithm is
#'   used. This can be significantly faster for large kernel matrices when only a small
#'   number of components is required. See n_oversamples, power_iterations, and
#'   power_normalization to control accuracy. Default = 'auto'.
#' @param gamma Numeric, kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels.
#'   If -1.0, it is set to 1 / n_features. Default = -1.0.
#' @param degree Integer, degree for the polynomial kernel. Default = 3.
#' @param coef0 Numeric, independent term for 'poly' and 'sigmoid' kernels.
#'   Default = 1.0.
#' @param fit_inverse_transform Logical, whether to learn the approximate inverse
#'   transform during fit, enabling inverse_transform. Cannot be used with a
#'   precomputed kernel. Default = FALSE.
#' @param alpha Numeric, ridge regularization parameter for the inverse transform.
#'   Must be positive. Default = 1.0.
#' @param remove_zero_eig Logical, whether to remove components whose eigenvalue is
#'   zero. Default = FALSE.
#' @param copy_X Logical, whether to store a copy of the user data. Default = TRUE.
#' @param n_oversamples Integer, number of extra columns used in the randomized
#'   eigendecomposition to improve numerical accuracy. Higher values give better results
#'   at the cost of more computation. Only used when eigensolver='randomized'. Default = 10.
#' @param power_iterations Integer, number of power iterations performed by the
#'   randomized eigendecomposition. Higher values improve accuracy for kernel matrices
#'   whose eigenvalues decay slowly. -1 selects an automatic heuristic. Only used
#'   when eigensolver='randomized'. Default = -1.
#' @param power_normalization Character, normalization method used during power
#'   iterations in the randomized eigendecomposition. 'QR' uses QR decomposition
#'   (numerically stable); 'LU' uses LU decomposition (faster, slightly less stable);
#'   'none' skips normalization entirely (fastest, least stable). Only used when
#'   eigensolver='randomized'. Default = 'QR'.
#' @param seed Integer, seed for random number generation. Set to -1 for
#'   non-deterministic results. Only used when eigensolver='randomized'. Default = -1.
#' @param check_data Logical, whether to check the data for NaNs prior to computation.
#'   Default = FALSE.
#' @export
KernelPCA <- R6::R6Class("KernelPCA",
    private = list(
        .kpca = NULL,
        .gamma = NULL,
        .coef0 = NULL,
        .alpha = NULL,
        .copy_X = NULL
    ),
    public = list(

        #' @description Create a new KernelPCA object.
        initialize = function(n_components = 0L, kernel = 'linear',
                              eigensolver = 'syevd', gamma = -1.0, degree = 3L,
                              coef0 = 1.0, fit_inverse_transform = FALSE, alpha = 1.0,
                              remove_zero_eig = FALSE, copy_X = TRUE,
                              n_oversamples = 10L, power_iterations = -1L,
                              power_normalization = 'QR', seed = -1L,
                              check_data = FALSE) {

            private$.gamma <- gamma
            private$.coef0 <- coef0
            private$.alpha <- alpha
            private$.copy_X <- copy_X

            private$.kpca <- aoclda:::rccp_KernelPCADecomposition$new(
                as.integer(n_components), kernel, eigensolver, as.integer(degree),
                fit_inverse_transform, remove_zero_eig, copy_X,
                as.integer(n_oversamples), as.integer(power_iterations),
                tolower(power_normalization), as.integer(seed), check_data)
        },

        #' @description Computes the kernel principal component analysis on the supplied data matrix.
        #' @param A The data matrix with which to compute the kernel PCA. It has shape
        #'   (n_samples, n_features).
        #' @return Returns the instance itself.
        fit = function(A) {
            A <- check_convert_data(A, "float", "matrix")
            private$.kpca$fit(A, private$.gamma, private$.coef0, private$.alpha)
            invisible(self)
        },

        #' @description Transform a data matrix into the Kernel PCA feature space.
        #' @param X The data matrix to be transformed. It has shape
        #'   (m_samples, m_features). Note that m_features must match n_features,
        #'   the number of features in the data matrix originally supplied to
        #'   \code{$fit()}.
        #' @return Matrix of shape (m_samples, n_components): the transformed matrix.
        transform = function(X) {
            X <- check_convert_data(X, "float", "matrix")
            return(private$.kpca$transform(X))
        },

        #' @description Transform data from the Kernel PCA feature space back to the original space.
        #'   Requires \code{fit_inverse_transform = TRUE} to have been set before calling \code{$fit()}.
        #' @param Y The data matrix to be transformed. It has shape
        #'   (k_samples, k_features). Note that k_features must match n_components.
        #' @return Matrix of shape (k_samples, n_features): the reconstructed matrix.
        inverse_transform = function(Y) {
            Y <- check_convert_data(Y, "float", "matrix")
            return(private$.kpca$inverse_transform(Y))
        },

        #' @description Print model version metadata.
        print_model_versions = function() {
            private$.kpca$print_model_versions()
            invisible(self)
        },

        #' @description Save the model to an RDS file. Not intended for direct use;
        #' use \code{save_model()} from utils instead.
        #' @param path File path.
        save = function(path) {
            saveRDS(list(
                class = "KernelPCA",
                wrapper_state = list(
                    gamma = private$.gamma,
                    coef0 = private$.coef0,
                    alpha = private$.alpha,
                    copy_X = private$.copy_X
                ),
                cpp_state = private$.kpca$save_model()
            ), path)
            invisible(self)
        },

        #' @description Load previously saved state into the current object. Not intended
        #' for direct use; use \code{load_model()} from utils instead.
        #' @param cpp_state Raw list returned by save_model() at the C++ layer.
        #' @param wrapper_state Optional named list produced by \code{$save()}.
        load_state = function(cpp_state, wrapper_state = NULL) {
            private$.kpca$load_model(cpp_state)
            if (!is.null(wrapper_state)) {
                private$.gamma <- wrapper_state$gamma
                private$.coef0 <- wrapper_state$coef0
                private$.alpha <- wrapper_state$alpha
                private$.copy_X <- wrapper_state$copy_X
            }
            invisible(self)
        }
    ),
    active = list(
        #' @field eigenvalues Numeric vector of shape (n_components): eigenvalues of the
        #'   centered kernel matrix.
        eigenvalues = function() {
            return(private$.kpca$get_eigenvalues())
        },
        #' @field eigenvectors Matrix of shape (n_samples, n_components): eigenvectors V
        #'   of the centered kernel matrix.
        eigenvectors = function() {
            return(private$.kpca$get_eigenvectors())
        },
        #' @field scores Matrix of shape (n_samples, n_components): the kernel PCA scores,
        #'   Z = V * diag(sqrt(lambda_i)).
        scores = function() {
            return(private$.kpca$get_scores())
        },
        #' @field dual_coef Matrix of shape (n_samples, n_features): dual coefficients
        #'   from the kernel ridge regression inverse transform. Only available when
        #'   fit_inverse_transform = TRUE.
        dual_coef = function() {
            return(private$.kpca$get_dual_coef())
        },
        #' @field X_fit_ Matrix of shape (n_samples, n_features): the training data used
        #'   to fit the kernel PCA, tight-packed in the fitted storage order.
        X_fit_ = function() {
            return(private$.kpca$get_X_fit())
        },
        #' @field gamma_ Numeric: the resolved gamma parameter used in the computation.
        #'   When gamma is set to a value less than 0, this returns 1 / n_features.
        gamma_ = function() {
            return(private$.kpca$get_gamma())
        },
        #' @field n_samples Integer: the number of samples in the data matrix used to
        #'   compute the kernel PCA.
        n_samples = function() {
            return(private$.kpca$get_n_samples())
        },
        #' @field n_features Integer: the number of features in the data matrix used to
        #'   compute the kernel PCA.
        n_features = function() {
            return(private$.kpca$get_n_features())
        },
        #' @field n_components Integer: the number of components found in the kernel PCA.
        n_components = function() {
            return(private$.kpca$get_n_components())
        }
    )
)

#' Nonnegative Matrix Factorization (NMF).
#'
#' Find two nonnegative matrices W (transform) and H (components) whose product
#' approximates the nonnegative input matrix A, i.e. A ~= W H with W >= 0 and H >= 0.
#'
#' @param n_components Number of components. Either a positive integer, NULL, or
#'   'auto'. If 'auto' or NULL, it is set to min(n_samples, n_features) when
#'   \code{$fit()} is called. For \code{$fit_transform()} with custom W and H, it
#'   is inferred from W. Default = 'auto'.
#' @param solver Character, numerical solver. Only 'mu' is
#'   supported. Default = 'mu'.
#' @param beta_loss Character, beta divergence to be minimized. Only 'frobenius' is
#'   supported. Default = 'frobenius'.
#' @param max_iter Integer, maximum number of iterations. Default = 200.
#' @param tol Numeric, tolerance for the stopping condition. Default = 1.0e-4.
#' @param init Character or NULL, initialization method. One of 'nndsvd', 'nndsvda',
#'   'nndsvdar', 'random', 'custom', or 'auto' (also NULL). 'auto' uses 'nndsvda'
#'   when n_components < min(n_samples, n_features), otherwise 'random'.
#'   Default = 'auto'.
#' @param random_state Integer or NULL, seed for the random number generator used by
#'   the 'random' and 'nndsvdar' init methods. Set to NULL for non-deterministic
#'   results. Default = NULL.
#' @param check_every Integer, number of MU iterations between successive convergence
#'   checks. The relative change of the Frobenius residual is evaluated once every
#'   check_every iterations and compared against tol. Larger values reduce
#'   residual-computation overhead at the cost of potentially running extra iterations
#'   after convergence. Must be at least 1. Only effective when tol > 0. Default = 10.
#' @param check_data Logical, whether to check the input matrix for NaN values before
#'   fitting. Enabling this adds a small overhead but can help diagnose ill-formed
#'   inputs. Default = FALSE.
#' @export
#' @examples
#' a <- matrix(c(1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12,
#'               4, 1, 5, 2, 5, 3, 8, 4), nrow = 5, ncol = 4, byrow = TRUE)
#' nmf_model <- NMF$new(n_components = 2, init = "nndsvda")
#' w <- nmf_model$fit_transform(a)
#' h <- nmf_model$h
NMF <- R6::R6Class("NMF",
    private = list(
        .nmf = NULL,
        .n_components = NULL
    ),
    public = list(

        #' @description Create a new NMF object.
        initialize = function(n_components = "auto", solver = "mu",
                              beta_loss = "frobenius", max_iter = 200L, tol = 1.0e-4,
                              init = "auto", random_state = NULL, check_every = 10L,
                              check_data = FALSE) {

            # Validate n_components: 'auto', NULL, or a positive integer.
            if (!is.null(n_components) && !identical(n_components, "auto")) {
                if (!is.numeric(n_components) || length(n_components) != 1 ||
                    n_components %% 1 != 0 || n_components < 1) {
                    stop("n_components must be a positive integer, NULL, or 'auto'.")
                }
                n_components <- as.integer(n_components)
            }
            private$.n_components <- n_components
            # The C API expects 0 to indicate "all components".
            n_comp_internal <- if (is.null(n_components) ||
                                   identical(n_components, "auto")) 0L
                               else as.integer(n_components)

            # NULL init is an alias for 'auto'.
            if (is.null(init)) init <- "auto"

            # Validate random_state: integer or NULL. -1 denotes non-deterministic.
            if (!is.null(random_state)) {
                if (!is.numeric(random_state) || length(random_state) != 1 ||
                    random_state %% 1 != 0) {
                    stop("random_state must be an integer or NULL.")
                }
            }
            random_internal <- if (is.null(random_state)) -1L
                               else as.integer(random_state)

            # Validate check_every: positive integer.
            if (!is.numeric(check_every) || length(check_every) != 1 ||
                check_every %% 1 != 0 || check_every < 1) {
                stop("check_every must be a positive integer.")
            }

            private$.nmf <- aoclda:::rccp_NMFDecomposition$new(
                n_comp_internal, solver, beta_loss, as.integer(max_iter),
                as.double(tol), init, random_internal, as.integer(check_every),
                check_data)
        },

        #' @description Fit the NMF model to the supplied data matrix.
        #'
        #'   Computes the factorization A ~= W H using multiplicative updates. After this
        #'   call the fitted factors W and H are available via the \code{$w} and \code{$h}
        #'   fields respectively.
        #'
        #'   When \code{init = 'custom'}, supply initial factors W and H. Both must be
        #'   provided together. Supplying only one raises an error. If n_components was set
        #'   to 'auto' or NULL, the rank is inferred from the inner dimension of the
        #'   supplied W.
        #' @param A Nonnegative data matrix of shape (n_samples, n_features). All entries
        #'   must be nonnegative.
        #' @param W Optional initial left factor of shape (n_samples, n_components).
        #'   Required when init = 'custom', ignored otherwise. Default = NULL.
        #' @param H Optional initial right factor of shape (n_components, n_features).
        #'   Required when init = 'custom', ignored otherwise. Default = NULL.
        #' @return Returns the instance itself.
        fit = function(A, W = NULL, H = NULL) {
            A <- check_convert_data(A, "float", "matrix")
            if (!is.null(W) && !is.null(H)) {
                W <- check_convert_data(W, "float", "matrix")
                H <- check_convert_data(H, "float", "matrix")
                if (is.null(private$.n_components) ||
                    identical(private$.n_components, "auto")) {
                    private$.nmf$set_n_components(ncol(W))
                }
                private$.nmf$fit_custom(A, W, H)
                return(invisible(self))
            }
            if (!is.null(W) || !is.null(H)) {
                stop("W and H must both be supplied together for init='custom'.")
            }
            private$.nmf$fit(A)
            invisible(self)
        },

        #' @description Transform a data matrix into the NMF component space.
        #'
        #'   Solves for the left factor W that minimizes ||X - W H||_F with H frozen at
        #'   the matrix computed by \code{$fit()} or \code{$fit_transform()}. Only W is
        #'   updated, H is not modified.
        #' @param X The data matrix to be transformed. It has shape (m_samples,
        #'   n_features). Note that the number of features must match n_features, the
        #'   number of features in the matrix originally supplied to \code{$fit()}.
        #' @return Matrix of shape (m_samples, n_components): the transformed matrix
        #'   representing X in the component space.
        transform = function(X) {
            X <- check_convert_data(X, "float", "matrix")
            return(private$.nmf$transform(X))
        },

        #' @description Transform data back to the original feature space.
        #'
        #'   Reconstructs approximate data in the original feature space by computing Y H,
        #'   where H is the components matrix computed by \code{$fit()} or
        #'   \code{$fit_transform()}.
        #' @param Y Transformed data matrix. It has shape (k_samples, n_components). Note
        #'   that the number of columns must match n_components.
        #' @return Matrix of shape (k_samples, n_features): the reconstructed matrix in
        #'   the original feature space.
        inverse_transform = function(Y) {
            Y <- check_convert_data(Y, "float", "matrix")
            return(private$.nmf$inverse_transform(Y))
        },

        #' @description Fit the NMF model to A and return the left factor W.
        #'
        #'   Equivalent to calling \code{$fit()} followed by reading the \code{$w} field.
        #'   Note that this is distinct from calling \code{$fit()} followed by
        #'   \code{$transform()}: \code{$transform()} re-solves for W with H frozen,
        #'   performing an additional optimization pass, whereas this method returns the W
        #'   already computed during fitting.
        #'
        #'   When \code{init = 'custom'}, supply initial factors W and H. Both must be
        #'   provided together. If n_components was set to 'auto' or NULL, the rank is
        #'   inferred from the inner dimension of the supplied W.
        #' @param A Nonnegative data matrix of shape (n_samples, n_features). All entries
        #'   must be nonnegative.
        #' @param W Optional initial left factor of shape (n_samples, n_components).
        #'   Required when init = 'custom', ignored otherwise. Default = NULL.
        #' @param H Optional initial right factor of shape (n_components, n_features).
        #'   Required when init = 'custom', ignored otherwise. Default = NULL.
        #' @return Matrix of shape (n_samples, n_components): the left factor W of the
        #'   factorization.
        fit_transform = function(A, W = NULL, H = NULL) {
            self$fit(A, W = W, H = H)
            return(private$.nmf$get_transform())
        },

        #' @description Save the model to an RDS file. Not intended for direct use;
        #'   use \code{save_model()} from utils instead.
        #' @param path File path.
        save = function(path) {
            saveRDS(list(
                class = "NMF",
                wrapper_state = list(
                    n_components = private$.n_components
                ),
                cpp_state = private$.nmf$save_model()
            ), path)
            invisible(self)
        },

        #' @description Load previously saved state into the current object. Not intended
        #'   for direct use; use \code{load_model()} from utils instead.
        #' @param cpp_state Raw list returned by save_model() at the C++ layer.
        #' @param wrapper_state Optional named list produced by \code{$save()}.
        load_state = function(cpp_state, wrapper_state = NULL) {
            private$.nmf$load_model(cpp_state)
            if (!is.null(wrapper_state)) {
                private$.n_components <- wrapper_state$n_components
            }
            invisible(self)
        },

        #' @description Print model version metadata.
        print_model_versions = function() {
            private$.nmf$print_model_versions()
            invisible(self)
        }
    ),
    active = list(
        #' @field h Matrix of shape (n_components, n_features): the right factor H of the
        #'   factorization A ~= W H. Each row is a basis vector in feature space. Available
        #'   after \code{$fit()} or \code{$fit_transform()} has been called.
        h = function() {
            return(private$.nmf$get_components())
        },
        #' @field w Matrix of shape (n_samples, n_components): the left factor W of the
        #'   factorization A ~= W H, i.e. the representation of the training data in the
        #'   component space. Available after \code{$fit()} or \code{$fit_transform()} has
        #'   been called.
        w = function() {
            return(private$.nmf$get_transform())
        },
        #' @field reconstruction_err Numeric: Frobenius reconstruction error ||A - W H||_F
        #'   evaluated at the final iterate. Available after \code{$fit()} or
        #'   \code{$fit_transform()} has been called.
        reconstruction_err = function() {
            return(private$.nmf$get_reconstruction_error())
        },
        #' @field n_iter Integer: number of multiplicative-update iterations actually
        #'   performed. May be less than max_iter if convergence was detected earlier.
        #'   Available after \code{$fit()} or \code{$fit_transform()} has been called.
        n_iter = function() {
            return(private$.nmf$get_n_iter())
        },
        #' @field n_components Integer: rank of the factorization, i.e. the number of
        #'   columns of W and rows of H. Available after \code{$fit()} or
        #'   \code{$fit_transform()} has been called.
        n_components = function() {
            return(private$.nmf$get_n_components())
        },
        #' @field n_samples Integer: number of samples in the training matrix supplied to
        #'   \code{$fit()} or \code{$fit_transform()}.
        n_samples = function() {
            return(private$.nmf$get_n_samples())
        },
        #' @field n_features Integer: number of features in the training matrix supplied to
        #'   \code{$fit()} or \code{$fit_transform()}.
        n_features = function() {
            return(private$.nmf$get_n_features())
        }
    )
)
