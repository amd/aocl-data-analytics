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

#' RBF (Radial Basis Function) Kernel
#'
#' Compute the RBF (Radial Basis Function) kernel matrix.
#'
#' K(x, y) = exp(-gamma ||x - y||^2)
#'
#' If Y is not provided, the RBF kernel is computed between the rows of X,
#' resulting in a matrix of shape (n_samples_X, n_samples_X). Otherwise, it computes
#' the RBF kernel between X and Y, with shape (n_samples_X, n_samples_Y).
#'
#' @param X The feature matrix of shape (n_samples_X, n_features).
#' @param Y The optional matrix of shape (n_samples_Y, n_features).
#' @param gamma The kernel coefficient. Has to be greater or equal to 0.
#' @return The RBF kernel matrix of shape (n_samples_X, n_samples_X) if Y is NULL,
#'   or (n_samples_X, n_samples_Y) otherwise.
#' @export
#' @examples
#' # Compute RBF kernel
#' X <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' Y <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' K <- rbf_kernel(X, Y, gamma = 1.0)
rbf_kernel <- function(X, Y = NULL, gamma = 1.0) {
    X <- check_convert_data(X, "float", "matrix")
    if (!is.null(Y)) {
        Y <- check_convert_data(Y, "float", "matrix")
    }
    
    return(rcpp_rbf_kernel(X, Y, gamma))
}

#' Linear Kernel
#'
#' Compute the linear kernel matrix.
#'
#' K(x, y) = x * y
#'
#' If Y is not provided, the linear kernel is computed between the rows of X,
#' resulting in a matrix of shape (n_samples_X, n_samples_X). Otherwise, it computes
#' the linear kernel between X and Y, with shape (n_samples_X, n_samples_Y).
#'
#' @param X The feature matrix of shape (n_samples_X, n_features).
#' @param Y The optional matrix of shape (n_samples_Y, n_features).
#' @return The linear kernel matrix of shape (n_samples_X, n_samples_X) if Y is NULL,
#'   or (n_samples_X, n_samples_Y) otherwise.
#' @export
#' @examples
#' # Compute linear kernel
#' X <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' Y <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' K <- linear_kernel(X, Y)
linear_kernel <- function(X, Y = NULL) {
    X <- check_convert_data(X, "float", "matrix")
    if (!is.null(Y)) {
        Y <- check_convert_data(Y, "float", "matrix")
    }
    
    return(rcpp_linear_kernel(X, Y))
}

#' Polynomial Kernel
#'
#' Compute the polynomial kernel matrix.
#'
#' K(x, y) = (gamma x * y + c)^d
#'
#' If Y is not provided, the polynomial kernel is computed between the rows of X,
#' resulting in a matrix of shape (n_samples_X, n_samples_X). Otherwise, it computes
#' the polynomial kernel between X and Y, with shape (n_samples_X, n_samples_Y).
#'
#' @param X The feature matrix of shape (n_samples_X, n_features).
#' @param Y The optional matrix of shape (n_samples_Y, n_features).
#' @param degree The degree of the polynomial.
#' @param gamma The kernel coefficient. Has to be greater or equal to 0.
#' @param coef0 The independent term in the polynomial kernel.
#' @return The polynomial kernel matrix of shape (n_samples_X, n_samples_X)
#'   if Y is NULL, or (n_samples_X, n_samples_Y) otherwise.
#' @export
#' @examples
#' # Compute polynomial kernel
#' X <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' Y <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' K <- polynomial_kernel(X, Y, degree = 3, gamma = 1.0, coef0 = 1.0)
polynomial_kernel <- function(X, Y = NULL, degree = 3, gamma = 1.0, coef0 = 1.0) {
    X <- check_convert_data(X, "float", "matrix")
    if (!is.null(Y)) {
        Y <- check_convert_data(Y, "float", "matrix")
    }
    
    return(rcpp_polynomial_kernel(X, Y, as.integer(degree), gamma, coef0))
}

#' Sigmoid Kernel
#'
#' Compute the sigmoid kernel matrix.
#'
#' K(x, y) = tanh(gamma x * y + c)
#'
#' If Y is not provided, the sigmoid kernel is computed between the rows of X,
#' resulting in a matrix of shape (n_samples_X, n_samples_X). Otherwise, it computes
#' the sigmoid kernel between X and Y, with shape (n_samples_X, n_samples_Y).
#'
#' @param X The feature matrix of shape (n_samples_X, n_features).
#' @param Y The optional matrix of shape (n_samples_Y, n_features).
#' @param gamma The kernel coefficient. Has to be greater or equal to 0.
#' @param coef0 The constant factor for the sigmoid kernel.
#' @return The sigmoid kernel matrix of shape (n_samples_X, n_samples_X) if Y is NULL,
#'   or (n_samples_X, n_samples_Y) otherwise.
#' @export
#' @examples
#' # Compute sigmoid kernel
#' X <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' Y <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' K <- sigmoid_kernel(X, Y, gamma = 1.0, coef0 = 1.0)
sigmoid_kernel <- function(X, Y = NULL, gamma = 1.0, coef0 = 1.0) {
    X <- check_convert_data(X, "float", "matrix")
    if (!is.null(Y)) {
        Y <- check_convert_data(Y, "float", "matrix")
    }
    
    return(rcpp_sigmoid_kernel(X, Y, gamma, coef0))
}