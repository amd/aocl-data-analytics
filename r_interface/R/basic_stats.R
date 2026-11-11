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

#' Calculate arithmetic mean
#'
#' Arithmetic mean of a data matrix along the specified axis.
#'
#' For a dataset {x_1, ..., x_n}, the arithmetic mean, x̄, is defined as
#' x̄ = (1/n) * sum(x_i) for i=1 to n.
#'
#' @param x A numeric matrix of shape (n_samples, n_features)
#' @param axis Axis along which to compute the mean ("col" for columns, "row" for rows, "all" for overall)
#' @return A numeric vector with the mean along the specified axis. Depending on 'axis' can have 
#'   shape (n_samples, ), (n_features, ) or (1, ).
#' @export
#' @examples
#' # Calculate column means of a matrix
#' data <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' da_mean(data)
#' 
#' # Calculate row means
#' da_mean(data, axis = "row")
da_mean <- function(x, axis = "col") {
    x <- check_convert_data(x, "float", "matrix")
    return(rcpp_mean(x, axis))
}

#' Calculate harmonic mean
#'
#' Harmonic mean of a data matrix along the specified axis.
#'
#' For a dataset {x_1, ..., x_n}, the harmonic mean, x̄_harm, is defined as
#' x̄_harm = n / sum(1/x_i) for i=1 to n.
#'
#' @param x A numeric matrix of shape (n_samples, n_features)
#' @param axis Axis along which to compute the harmonic mean ("col", "row", or "all")
#' @return A numeric vector with the harmonic mean along the specified axis. Depending on 'axis' can have 
#'   shape (n_samples, ), (n_features, ) or (1, ).
#' @export
#' @examples
#' # Calculate harmonic mean
#' data <- matrix(abs(rnorm(100)) + 0.1, nrow = 10, ncol = 10)
#' da_harmonic_mean(data)
da_harmonic_mean <- function(x, axis = "col") {
    x <- check_convert_data(x, "float", "matrix")
    return(rcpp_harmonic_mean(x, axis))
}

#' Calculate geometric mean
#'
#' Geometric mean of a data matrix along the specified axis.
#'
#' For a dataset {x_1, ..., x_n}, the geometric mean, x̄_geom, is defined as
#' x̄_geom = (prod(x_i))^(1/n) = exp((1/n) * sum(ln(x_i))) for i=1 to n.
#'
#' @param x A numeric matrix of shape (n_samples, n_features)
#' @param axis Axis along which to compute the geometric mean ("col", "row", or "all")
#' @return A numeric vector with the geometric mean along the specified axis. Depending on 'axis' can have 
#'   shape (n_samples, ), (n_features, ) or (1, ).
#' @export
#' @examples
#' # Calculate geometric mean
#' data <- matrix(abs(rnorm(100)) + 0.1, nrow = 10, ncol = 10)
#' da_geometric_mean(data)
da_geometric_mean <- function(x, axis = "col") {
    x <- check_convert_data(x, "float", "matrix")
    return(rcpp_geometric_mean(x, axis))
}

#' Calculate variance
#'
#' Variance of a data matrix along the specified axis.
#'
#' For a dataset {x_1, ..., x_n}, the variance, s², is defined as
#' s² = (1/dof) * sum((x_i - x̄)²) for i=1 to n,
#' where dof is the number of degrees of freedom. Setting dof = n gives the sample 
#' variance, whereas setting dof = n-1 gives an unbiased estimate of the population variance.
#'
#' @param x A numeric matrix of shape (n_samples, n_features)
#' @param dof Integer, number of degrees of freedom used to compute the variance:
#'   - If dof < 0: degrees of freedom will be set to the number of observations
#'   - If dof = 0: degrees of freedom will be set to the number of observations - 1
#'   - If dof > 0: degrees of freedom will be set to the specified value
#' @param axis Axis along which to compute the variance ("col", "row", or "all")
#' @return A numeric vector with the variance along the specified axis. Depending on 'axis' can have 
#'   shape (n_samples, ), (n_features, ) or (1, ).
#' @export
#' @examples
#' # Calculate sample variance (default)
#' data <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' da_variance(data)
#' 
#' # Calculate population variance
#' da_variance(data, dof = 0)
da_variance <- function(x, dof = 0, axis = "col") {
    x <- check_convert_data(x, "float", "matrix")
    return(rcpp_variance(x, as.integer(dof), axis))
}

#' Calculate skewness
#'
#' Skewness of a data matrix along the specified axis.
#'
#' The skewness is computed as the Fischer-Pearson coefficient of skewness
#' (that is, with the central moments scaled by the number of observations).
#' For a dataset {x_1, ..., x_n}, the skewness, g₁, is defined as
#' g₁ = [(1/n) * sum((x_i - x̄)³)] / [(1/n) * sum((x_i - x̄)²)]^(3/2)
#'
#' @param x A numeric matrix of shape (n_samples, n_features)
#' @param axis Axis along which to compute the skewness ("col", "row", or "all")
#' @return A numeric vector with the skewness along the specified axis. Depending on 'axis' can have 
#'   shape (n_samples, ), (n_features, ) or (1, ).
#' @export
#' @examples
#' # Calculate skewness
#' data <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' da_skewness(data)
da_skewness <- function(x, axis = "col") {
    x <- check_convert_data(x, "float", "matrix")
    return(rcpp_skewness(x, axis))
}

#' Calculate kurtosis
#'
#' Kurtosis of a data matrix along the specified axis.
#'
#' The kurtosis is computed using Fischer's coefficient of excess kurtosis
#' (that is, with the central moments scaled by the number of observations
#' and 3 subtracted to ensure normally distributed data gives a value of 0).
#' For a dataset {x_1, ..., x_n}, the kurtosis, g₂, is defined as
#' g₂ = [(1/n) * sum((x_i - x̄)⁴)] / [(1/n) * sum((x_i - x̄)²)]² - 3
#'
#' @param x A numeric matrix of shape (n_samples, n_features)
#' @param axis Axis along which to compute the kurtosis ("col", "row", or "all")
#' @return A numeric vector with the kurtosis along the specified axis. Depending on 'axis' can have 
#'   shape (n_samples, ), (n_features, ) or (1, ).
#' @export
#' @examples
#' # Calculate kurtosis
#' data <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' da_kurtosis(data)
da_kurtosis <- function(x, axis = "col") {
    x <- check_convert_data(x, "float", "matrix")
    return(rcpp_kurtosis(x, axis))
}

#' Calculate statistical moment
#'
#' Central moment of a data matrix along the specified axis.
#'
#' For a dataset {x_1, ..., x_n}, the k-th central moment, m_k, is defined as
#' m_k = (1/n) * sum((x_i - x̄)^k) for i=1 to n.
#' Here, the moments are scaled by the number of observations along the specified axis.
#' The function gives you the option of supplying precomputed means (via the argument
#' 'mean') about which the moments are computed. Otherwise it will compute the means itself.
#'
#' @param x A numeric matrix of shape (n_samples, n_features)
#' @param k Integer, the order of the moment to be computed
#' @param mean Optional 1D array with precomputed means. If NULL, will be computed automatically
#' @param axis Axis along which to compute the moment ("col", "row", or "all")
#' @return A numeric vector with the moment along the specified axis. Depending on 'axis' can have 
#'   shape (n_samples, ), (n_features, ) or (1, ).
#' @export
#' @examples
#' # Calculate second moment
#' data <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' da_moment(data, k = 2)
da_moment <- function(x, k, mean = NULL, axis = "col") {
    x <- check_convert_data(x, "float", "matrix")
    
    # Validate mean parameter if provided
    if (!is.null(mean)) {
        mean <- check_convert_data(mean, "float", "vector")
    }

    return(rcpp_moment(x, as.integer(k), mean, axis))
}

#' Calculate quantile
#'
#' Selected quantile of a data matrix along the specified axis.
#'
#' Computes the q-th quantiles of a data matrix along the specified axis.
#' Note that there are multiple ways to define quantiles. The available quantile types
#' correspond to the 9 different quantile types commonly used. These can be specified using the 'method' parameter.
#' In each case a number h is computed, corresponding to the approximate location in the
#' data array of the required quantile 'q'.
#'
#' Note:
#' - Methods 'inverted_cdf', 'averaged_inverted_cdf' and 'closest_observation' give discontinuous results.
#' - Method 'median_unbiased' is recommended if the sample distribution function is unknown.
#' - Method 'normal_unbiased' is recommended if the sample distribution function is known to be normal.
#' - Method 'closest_observation' rounds to nearest order statistic and NOT nearest even order statistic.
#'
#' @param x A numeric matrix of shape (n_samples, n_features)
#' @param q Numeric scalar or vector of quantiles, each must lie in the interval [0,1]
#' @param method Character, specifies the method used to compute the quantiles:
#'   - 'inverted_cdf': h = n*q, return x[ceiling(h)]
#'   - 'averaged_inverted_cdf': h = n*q + 0.5, return (x[i] + x[j])/2
#'   - 'closest_observation': h = n*q - 0.5, return x[round(h)]
#'   - 'interpolated_inverted_cdf': h = n*q, linear interpolation
#'   - 'hazen': h = n*q + 0.5, linear interpolation
#'   - 'weibull': h = (n+1)*q, linear interpolation
#'   - 'linear': h = (n-1)*q + 1, linear interpolation (default)
#'   - 'median_unbiased': h = (n+1/3)*q + 1/3, linear interpolation
#'   - 'normal_unbiased': h = (n+1/4)*q + 3/8, linear interpolation
#' @param axis Axis along which to compute the quantile ("col", "row", or "all")
#' @return If q is a scalar, a numeric vector with the quantile along the specified axis.
#'   If q is a vector and axis is not "all", a matrix of shape (length(q), n_features) or
#'   (length(q), n_samples). If axis is "all", a vector of length length(q).
#' @export
#' @examples
#' # Calculate median (0.5 quantile)
#' data <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' da_quantile(data, q = 0.5)
#' # Calculate multiple quantiles
#' da_quantile(data, q = c(0.25, 0.5, 0.75))
da_quantile <- function(x, q, method = "linear", axis = "col") {
    x <- check_convert_data(x, "float", "matrix")
    q <- as.double(q)
    return(rcpp_quantile(x, q, method, axis))
}

#' Calculate five-point summary
#'
#' Summary statistics of a data matrix along the specified axis.
#'
#' Computes the maximum, minimum, median and upper/lower hinges of a data array along
#' the specified axis.
#'
#' Note:
#' - On large datasets, this function is more efficient than calling da_quantile() five
#'   times because it uses partly sorted arrays after each stage.
#' - The 'weibull' definition of quantiles is used to calculate the statistics.
#'
#' @param x A numeric matrix of shape (n_samples, n_features)
#' @param axis Axis along which to compute the summary ("col", "row", or "all")
#' @return A list with calculated minimum, lower hinge, median, upper hinge and maximum.
#'   Depending on 'axis' each element can have shape (n_samples, ), (n_features, ) or (1, ).
#' @export
#' @examples
#' # Calculate five-point summary
#' data <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' summary_stats <- da_five_point_summary(data)
#' print(summary_stats)
da_five_point_summary <- function(x, axis = "col") {
    x <- check_convert_data(x, "float", "matrix")
    return(rcpp_five_point_summary(x, axis))
}

#' Standardize data
#'
#' Standardize a data matrix along the specified axis.
#'
#' This function can be called in various different ways:
#' - If the arrays 'shift' and 'scale' are both NULL, then the mean and standard
#'   deviations will be computed along the appropriate axis and will be
#'   used to shift and scale the data.
#' - If the arrays 'shift' and 'scale' are both supplied, then the data matrix 'x'
#'   will be shifted (by subtracting the values in 'shift') then scaled (by dividing
#'   by the values in 'scale') along the selected axis.
#' - If one of the arrays 'shift' or 'scale' is NULL then it will be ignored and only
#'   the other will be used (so that the data is only shifted or only scaled).
#' In each case, if a 0 scaling factor is encountered then it will not be used.
#'
#' An additional computational mode is available by setting reverse = TRUE. In this
#' case the standardization is reversed, so that the data matrix is multiplied by the
#' values in scale before adding the values in shift. This enables users to undo the
#' standardization after the data has been used in another computation.
#'
#'
#' @param x A numeric matrix of shape (n_samples, n_features)
#' @param shift Optional 1D array of values used for shifting the data. If NULL, will be computed automatically
#' @param scale Optional 1D array of values used for scaling the data. If NULL, will be computed automatically
#' @param dof Integer, number of degrees of freedom used to compute standard deviations:
#'   - If dof < 0: degrees of freedom will be set to the number of observations in specified axis
#'   - If dof = 0: degrees of freedom will be set to the number of observations - 1
#'   - If dof > 0: degrees of freedom will be set to the specified value
#' @param reverse Logical, determines whether or not the standardization proceeds in reverse:
#'   - If reverse = FALSE: data matrix will be shifted (by subtracting values in 'shift') then scaled (by dividing by values in 'scale')
#'   - If reverse = TRUE: data matrix will be scaled (by multiplying by values in 'scale') then shifted (by adding values in 'shift')
#' @param inplace Logical, determines whether the standardization is done without a copy
#' @param axis Axis along which to standardize ("col", "row", or "all")
#' @return A numeric matrix of shape (n_samples, n_features) with the standardized data
#' @export
#' @examples
#' # Standardize data
#' data <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' standardized <- da_standardize(data)
da_standardize <- function(x, shift = NULL, scale = NULL, dof = 0, reverse = FALSE, inplace = FALSE, axis = "col") {
    x <- check_convert_data(x, "float", "matrix")
    
    # Validate shift parameter if provided
    if (!is.null(shift)) {
        shift <- check_convert_data(shift, "float", "vector")
    }
    
    # Validate scale parameter if provided
    if (!is.null(scale)) {
        scale <- check_convert_data(scale, "float", "vector")
    }
    
    return(rcpp_standardize(x, shift, scale, as.integer(dof), reverse, inplace, axis))
}

#' Calculate covariance matrix
#'
#' Covariance matrix of a data matrix, with the rows treated as observations
#' and the columns treated as variables.
#'
#' For a dataset X = [x₁, ..., x_{n_cols}]ᵀ with column means {x̄₁, ..., x̄_{n_cols}}
#' the (i,j) element of the covariance matrix is given by covariance
#' between xᵢ and xⱼ:
#' cov(i,j) = (1/dof) * (xᵢ - x̄ᵢ) · (xⱼ - x̄ⱼ)
#' where dof is the number of degrees of freedom. Setting dof = n_cols gives the sample
#' covariances, whereas setting dof = n_cols - 1 gives unbiased estimates
#' of the population covariances.
#'
#' @param x A numeric matrix of shape (n_samples, n_features) where rows are observations and columns are variables
#' @param dof Integer, number of degrees of freedom used to compute covariances:
#'   - If dof < 0: degrees of freedom will be set to the number of observations
#'   - If dof = 0: degrees of freedom will be set to the number of observations - 1
#'   - If dof > 0: degrees of freedom will be set to the specified value
#' @param assume_centered Logical, if FALSE, centers the input matrix by subtracting the column means.
#'   If TRUE, assumes that the input data is already centered (mean = 0) and skips the centering step
#'   for computational efficiency. Default: FALSE.
#' @return A numeric matrix of shape (n_features, n_features) representing the covariance matrix
#' @export
#' @examples
#' # Calculate covariance matrix
#' data <- matrix(rnorm(200), nrow = 50, ncol = 4)
#' cov_matrix <- da_covariance_matrix(data)
#' print(cov_matrix)
da_covariance_matrix <- function(x, dof = 0, assume_centered = FALSE) {
    x <- check_convert_data(x, "float", "matrix")
    return(rcpp_covariance(x, as.integer(dof), assume_centered))
}

#' Calculate correlation matrix
#'
#' Correlation matrix of a data matrix, with the rows treated as observations and the
#' columns treated as variables.
#'
#' For a dataset X = [x₁, ..., x_{n_cols}]ᵀ with column means {x̄₁, ..., x̄_{n_cols}}
#' and column standard deviations {σ₁, ..., σ_{n_cols}} the (i,j) element of the correlation
#' matrix is given by correlation between xᵢ and xⱼ:
#' corr(i,j) = cov(i,j) / (σᵢ * σⱼ).
#' Note that the values in the correlation matrix are independent of the number of degrees
#' of freedom used to compute the standard deviations and covariances.
#'
#' @param x A numeric matrix of shape (n_samples, n_features) where rows are observations and columns are variables
#' @return A numeric matrix of shape (n_features, n_features) representing the correlation matrix
#' @export
#' @examples
#' # Calculate correlation matrix
#' data <- matrix(rnorm(200), nrow = 50, ncol = 4)
#' corr_matrix <- da_correlation_matrix(data)
#' print(corr_matrix)
da_correlation_matrix <- function(x) {
    x <- check_convert_data(x, "float", "matrix")
    return(rcpp_correlation(x))
}
