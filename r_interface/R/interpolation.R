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

#' Cubic spline interpolation.
#'
#' Construct a piecewise cubic polynomial that passes through a set of data points
#' (x_i, y_i). The interpolant is twice continuously differentiable. One or more
#' dimensions of data can be interpolated simultaneously using a single set of
#' interpolation sites.
#'
#' @param spline_type Character, the type of cubic spline to compute, which fixes the
#'   boundary conditions imposed on the spline. It can take the values 'natural' (the
#'   second derivative of the spline is set to zero at both end points), 'clamped zero'
#'   (the first derivative of the spline is set to zero at both end points), 'custom'
#'   (the boundary conditions are supplied by the user through the boundary_conditions
#'   argument of \code{$fit()}), or 'hermite' (the first derivatives of the data are
#'   supplied by the user through the y_derivative argument of \code{$fit()}).
#'   Default = 'natural'.
#' @param check_data Logical, whether to check the data for NaNs. Default = FALSE.
#' @export
#' @examples
#' # Natural cubic spline through sin(x)
#' x <- seq(0, 2 * pi, length.out = 11)
#' y <- sin(x)
#' spline <- cubic_spline$new()
#' spline$fit(x, y)
#' x_eval <- seq(0, 2 * pi, length.out = 50)
#' values <- spline$evaluate(x_eval)
cubic_spline <- R6::R6Class("cubic_spline",
    private = list(
        .spline = NULL, .dim = NULL, .x = NULL
    ),
    public = list(

        #' @description Create a new cubic_spline object.
        initialize = function(spline_type = "natural", check_data = FALSE) {
            private$.spline <- aoclda:::rccp_cubic_spline$new(spline_type, check_data)
            private$.dim <- 0
            private$.x <- NULL
        },

        #' @description Compute the cubic spline interpolant of the supplied data.
        #' @param x The interpolation sites, a vector of shape (n_sites). The sites must be
        #'   sorted in strictly increasing order with no repeated values. Constraint:
        #'   n_sites >= 2.
        #' @param y The values to interpolate at the sites. It has shape (n_sites) for
        #'   one-dimensional data, or (n_sites, n_dim) to interpolate n_dim dimensions
        #'   simultaneously.
        #' @param y_derivative The first derivatives of the data at the interpolation
        #'   sites, used only when spline_type = 'hermite'. It has the same shape as y.
        #'   Default = NULL.
        #' @param boundary_conditions The custom boundary conditions, used only when
        #'   spline_type = 'custom'. It is a pair
        #'   \code{list(list(left_order, left_values), list(right_order, right_values))}
        #'   where left_order and right_order are the derivative orders (1 for the first
        #'   derivative or 2 for the second derivative) imposed at the left and right
        #'   boundaries, and left_values and right_values are vectors of shape (n_dim)
        #'   holding the corresponding derivative values for each dimension. Default = NULL.
        #' @return Returns the instance itself.
        fit = function(x, y, y_derivative = NULL, boundary_conditions = NULL) {
            x <- check_convert_data(x, "float", "vector")
            y <- check_convert_data(y, "float", "matrix")

            y_deriv <- NULL
            if (!is.null(y_derivative)) {
                y_deriv <- check_convert_data(y_derivative, "float", "matrix")
            }

            left_order <- 2L
            left_values <- NULL
            right_order <- 2L
            right_values <- NULL
            if (!is.null(boundary_conditions)) {
                left_order <- as.integer(boundary_conditions[[1]][[1]])
                left_values <- boundary_conditions[[1]][[2]]
                right_order <- as.integer(boundary_conditions[[2]][[1]])
                right_values <- boundary_conditions[[2]][[2]]
                left_values <- check_convert_data(left_values, "float", "vector")
                right_values <- check_convert_data(right_values, "float", "vector")
            }

            private$.spline$fit(x, y, y_deriv, left_order, left_values, right_order,
                                right_values)
            private$.dim <- private$.spline$get_dim()
            private$.x <- x
            invisible(self)
        },

        #' @description Evaluate the fitted spline (or one of its derivatives) at a set of
        #'   query points.
        #' @param x_eval The query points, a vector of shape (n_eval). Points lying outside
        #'   the range of the interpolation sites are extrapolated using the nearest
        #'   boundary cell.
        #' @param orders The derivative order, or orders, to evaluate. 0 returns the spline
        #'   value, 1 the first derivative, 2 the second derivative and 3 the third
        #'   derivative. A vector of orders may be supplied to evaluate several at once.
        #'   Default = 0.
        #' @return The evaluated values, with the same shapes as the Python interface:
        #'
        #'   - When a single order is requested for one-dimensional data the result is a
        #'     vector of shape (n_eval).
        #'   - When a single order is requested for n_dim-dimensional data the result is a
        #'     matrix of shape (n_dim, n_eval).
        #'   - When multiple orders are requested the leading axis of length
        #'     length(orders) is retained, giving a matrix of shape (length(orders), n_eval)
        #'     for one-dimensional data or an array of shape (length(orders), n_dim, n_eval)
        #'     otherwise. The rows (or slices) are ordered to match the supplied orders.
        evaluate = function(x_eval, orders = 0) {
            x_eval <- check_convert_data(x_eval, "float", "vector")

            scalar_order <- length(orders) == 1
            requested <- as.integer(orders)

            # The underlying library evaluates each distinct order once, in ascending
            # order, so deduplicate the request and map the result back afterwards.
            unique_sorted <- sort(unique(requested))
            res <- private$.spline$evaluate(x_eval, unique_sorted)

            # C++ returns (n_eval, dim, n_unique); permute to the Python axis order
            # (n_orders, dim, n_eval) so shapes and indexing match across languages.
            res <- aperm(res, c(3, 2, 1))
            idx <- match(requested, unique_sorted)
            res <- res[idx, , , drop = FALSE]

            no <- dim(res)[1]
            d <- dim(res)[2]
            n_eval <- dim(res)[3]
            if (scalar_order && d == 1L) {
                res <- as.numeric(res)
            } else if (scalar_order) {
                dim(res) <- c(d, n_eval)
            } else if (d == 1L) {
                dim(res) <- c(no, n_eval)
            }

            return(res)
        },

        #' @description Find, for each query point, the index of the cell that contains it.
        #' @param x_eval The query points, a vector of shape (n_eval).
        #' @return Integer vector of shape (n_eval): for each query point x_eval[i] the
        #'   0-based index j (matching the C and Python interfaces) such that x_eval[i]
        #'   lies in the interval [x[j], x[j+1]]. Points outside the range of the
        #'   interpolation sites are clamped to the first or last cell.
        search_cells = function(x_eval) {
            x_eval <- check_convert_data(x_eval, "float", "vector")
            return(private$.spline$search_cells(x_eval))
        }
    ),
    active = list(
        #' @field x Vector of shape (n_sites): the interpolation sites used to compute the
        #'   spline.
        x = function() {
            return(private$.x)
        },

        #' @field n_sites Integer: the number of interpolation sites used to compute the
        #'   spline.
        n_sites = function() {
            return(private$.spline$get_n_sites())
        },

        #' @field n_dim Integer: the dimensionality of the values used to compute the
        #'   spline.
        n_dim = function() {
            return(private$.spline$get_dim())
        },

        #' @field coefficients The piecewise polynomial coefficients of the fitted spline.
        #'
        #'   The array has shape (n_dim, n_sites - 1, 4) (the leading dimension is dropped
        #'   when the data is one-dimensional, giving a matrix of shape (n_sites - 1, 4)).
        #'   For cell i and dimension d the spline is
        #'   S(x) = c_0 + c_1 t + c_2 t^2 + c_3 t^3, with t = x - x[i], where the four
        #'   coefficients [c_0, c_1, c_2, c_3] are stored along the trailing axis in
        #'   ascending power order, matching the Python interface.
        coefficients = function() {
            if (private$.dim == 0) {
                stop("The spline must be fitted before extracting coefficients.")
            }
            coeffs <- private$.spline$get_coefficients()
            # C++ returns (4, n_cells, dim); permute to the Python axis order
            # (dim, n_cells, 4) so shapes and indexing match across languages.
            coeffs <- aperm(coeffs, c(3, 2, 1))
            if (private$.dim == 1) {
                d <- dim(coeffs)
                dim(coeffs) <- c(d[2], d[3])
            }
            return(coeffs)
        }
    )
)
