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

#' Check and convert data to appropriate matrix type
#'
#' This function validates and converts input data to either a numeric or integer matrix
#' depending on the specified data type requirement. Optimized to avoid unnecessary copies.
#'
#' @param x Input data (matrix, data.frame, or array-like)
#' @param dtype Character string specifying the required data type: "float", "integer", or "both"
#' @param matrix_type Character string specifying expected shape: "matrix" or "vector"
#' @return A matrix or vector of the specified type and shape
#' @export
check_convert_data <- function(x, dtype = "float", matrix_type) {
    if (missing(matrix_type)) {
        stop("matrix_type must be provided and set to 'matrix' or 'vector'")
    }
    if (!matrix_type %in% c("matrix", "vector")) {
        stop("matrix_type must be 'matrix' or 'vector'")
    }
    if (!dtype %in% c("float", "integer", "da_int", "both")) {
        stop("dtype must be 'float', 'integer', 'da_int', or 'both'")
    }

    # Input validation - accept vectors, matrices, and data frames
    if (!is.vector(x) && !is.matrix(x) && !is.data.frame(x)) {
        stop("Input must be a vector, matrix, or data frame")
    }

    # Normalize shape first.
    if (matrix_type == "matrix") {
        if (is.vector(x)) {
            x <- matrix(x, ncol = 1)
        } else if (is.data.frame(x)) {
            if (!all(sapply(x, is.numeric))) {
                stop("All columns must be numeric")
            }
            x <- as.matrix(x)
        }
    } else {
        if (is.data.frame(x)) {
            if (!all(sapply(x, is.numeric))) {
                stop("All columns must be numeric")
            }
            x <- as.matrix(x)
        }
        if (is.matrix(x)) {
            x <- as.vector(x)
        }
    }

    if (matrix_type == "vector") {
        if (!is.numeric(x)) {
            stop("Vector input must be numeric")
        }

        if (dtype == "float") {
            if (!is.double(x)) {
                x <- as.double(x)
            }
        } else if (dtype == "integer" || dtype == "da_int") {
            if (!is.integer(x)) {
                x <- as.integer(x)
            }
        } else if (dtype == "both") {
            if (!is.integer(x) && !is.double(x)) {
                x <- as.double(x)
            }
        } else {
            stop("dtype must be 'float', 'integer', 'da_int', or 'both'")
        }
    } else {
        # matrix_type == "matrix"
        if (!is.numeric(x)) {
            stop("Input must be numeric")
        }

        if (dtype == "float") {
            if (!is.double(x)) {
                storage.mode(x) <- "double"
            }
        } else if (dtype == "integer" || dtype == "da_int") {
            if (!is.integer(x)) {
                storage.mode(x) <- "integer"
            }
        } else if (dtype == "both") {
            if (!(is.integer(x) || is.double(x))) {
                storage.mode(x) <- "double"
            }
        } else {
            stop("dtype must be 'float', 'integer', 'da_int', or 'both'")
        }
    }

    return(x)
}
