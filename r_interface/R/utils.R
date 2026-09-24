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

#' Train Test Split.
#'
#' Divides data into training and test sets.
#'
#' @param ... Arrays/matrices with the same number of samples.
#' @param test_size The size of the returned test array. If test_size is a
#'   float, then it should be between 0 and 1, and will represent the proportion
#'   of the dataset to include in the test data. If test_size is an integer, then
#'   it will represent the number of rows to include in the test data. If it is
#'   NULL, its value is set to the complement of the train size, or if
#'   train_size = NULL, it is set to be 0.25. Default: NULL.
#' @param train_size The size of the returned train array. If train_size is a
#'   float, then it should be between 0 and 1, and will represent the proportion
#'   of the dataset to include in the training data. If train_size is an integer,
#'   then it will represent the number of rows to include in the training data.
#'   If it is NULL, its value is set to the complement of the test size.
#'   Default: NULL.
#' @param seed Set the random seed for the random number generator. If the value
#'   is -1, a random seed is automatically generated. In this case the resulting
#'   classification will create non-reproducible results. Default: -1.
#' @param shuffle Specify whether shuffling of the data should be performed
#'   before the splitting. Default: TRUE.
#' @param stratify If not NULL, the array is used as class labels to perform a
#'   stratified shuffle. Stratified shuffling tries to preserve the proportions
#'   of the classes in the training and the test split, as they were in the
#'   original data. Default: NULL.
#' @param fp_precision An integer specifying the scaling factor applied to
#'   floating-point class labels to determine their precision. When stratify is
#'   not NULL and of type float or double, each class label is multiplied by
#'   fp_precision and then floored to obtain an integer class label for the
#'   stratified shuffling. Default: 1000.
#' @return A list of the split arrays containing the training and the test
#'   arrays: list(X1_train, X1_test, X2_train, X2_test, ...).
#' @export
train_test_split <- function(..., test_size = NULL, train_size = NULL, 
                            seed = -1L, shuffle = TRUE, stratify = NULL, 
                            fp_precision = 1000L) {
    
    # Capture input arrays
    arrays <- list(...)
    
    if (length(arrays) == 0) {
        stop("At least one array must be provided")
    }
    
    if (!is.null(stratify)) {
        if (!shuffle) {
            stop("shuffle cannot be FALSE when stratify is used")
        }
        if (length(stratify) != NROW(arrays[[1]])) {
            stop(paste0("The stratify array must have the same number of samples (m=", 
                       length(stratify), ") as the data arrays (m=", NROW(arrays[[1]]), ")."))
        }
    }

    if (!is.null(train_size) && ((train_size %% 1) != 0) && 
        (train_size <= 0 || train_size >= 1)) {
        stop("train_size cannot be a float outside the range (0.0, 1.0)")
    }
    
    if (!is.null(test_size) && ((test_size %% 1) != 0) && 
        (test_size <= 0 || test_size >= 1)) {
        stop("test_size cannot be a float outside the range (0.0, 1.0)")
    }
    
    if (!is.null(test_size) && !is.null(train_size) && ((train_size %% 1) != 0) && 
        ((test_size %% 1) != 0) && (test_size + train_size > 1.0)) {
        stop(paste0("train_size + test_size = ", train_size + test_size, 
                   ". Constraint: train_size + test_size <= 1."))
    }

    if (is.null(train_size) && is.null(test_size)) {
        train_size <- 0.75
        test_size <- 0.25
    }
    
    n_samples <- NROW(arrays[[1]])
    
    # Convert proportions to absolute sizes
    # Train size is floored, test size is ceiled to prevent exceeding sample size
    if (!is.null(train_size) && train_size < 1) {
        train_size <- as.integer(floor(n_samples * train_size))
    }
    if (!is.null(test_size) && test_size < 1) {
        test_size <- as.integer(ceiling(n_samples * test_size))
    }
    
    # Calculate missing size
    if (is.null(train_size)) {
        train_size <- n_samples - test_size
    } else if (is.null(test_size)) {
        test_size <- n_samples - train_size
    }
    
    # Convert to integers if needed
    train_size <- as.integer(train_size)
    test_size <- as.integer(test_size)
    
    # Get shuffled indices if needed
    if (shuffle) {
        if (!is.null(stratify) && !is.numeric(stratify) && !is.integer(stratify)) {
            stratify <- as.integer(factor(stratify))
        }
        shuffled_indices <- rcpp_get_shuffled_indices(
            size = n_samples, 
            seed = seed, 
            train_size = train_size, 
            test_size = test_size, 
            fp_precision = fp_precision,
            classes = stratify
        )
    } else {
        shuffled_indices <- NULL
    }
    
    # Process each array
    response <- list()
    
    for (i in seq_along(arrays)) {
        # Check array lengths match
        if (i > 1 && NROW(arrays[[i]]) != NROW(arrays[[1]])) {
            stop(paste0("Input matrices must have the same number of samples. Arrays of ",
                       "length ", NROW(arrays[[i]]), " and ", NROW(arrays[[1]]), 
                       " were encountered."))
        }
        
        # Convert and validate data - accept both float and integer for train_test_split
        current_shape <- if (is.vector(arrays[[i]])) "vector" else "matrix"
        arrays[[i]] <- check_convert_data(arrays[[i]], "both", current_shape)
        
        # Split the array
        split_result <- rcpp_train_test_split(
            X = arrays[[i]],
            test_size = test_size,
            train_size = train_size,
            shuffled_indices = shuffled_indices
        )
        
        X_train <- split_result$X_train
        X_test <- split_result$X_test
        
        # Flatten dimension when the initial input was 1D array (vector)
        if (current_shape == "vector") {
            X_train <- as.vector(X_train)
            X_test <- as.vector(X_test)
        }
        
        # Add to response list
        response <- append(response, list(X_train, X_test))
    }
    
    return(response)
}

#' Load a previously saved AOCL-DA model
#'
#' Reads an RDS file written by any model's \code{$save()} method or by
#' \code{save_model()}, recreates the model object, and restores all state.
#'
#' @param path Character. Path to the RDS file produced by \code{$save()} or
#'   \code{save_model()}.
#' @return The reconstructed model object (same class as when it was saved).
#' @export
#' @examples
#' \dontrun{
#' clf <- SVC$new(kernel = "rbf")
#' clf$fit(X_train, y_train)
#' save_model(clf, "my_svc.rds")
#' clf2 <- load_model("my_svc.rds")
#' }
load_model <- function(path) {
    state <- readRDS(path)
    valid_classes <- c("kmeans", "DBSCAN", "decision_forest", "decision_tree",
                       "PCA", "KernelPCA", "NMF", "linmod", "nearest_neighbors",
                       "approximate_neighbors", "nlls", "SVC", "SVR", "NuSVC", "NuSVR")
    if (!state$class %in% valid_classes)
        stop(paste0("Unknown model class: '", state$class, "'"))
    cls   <- get(state$class, envir = asNamespace("aoclda"))
    model <- cls$new()
    model$load_state(state$cpp_state, state$wrapper_state)
    return(model)
}

#' Save an AOCL-DA model to an RDS file
#'
#' Creates (or overwrites) an RDS file containing the full model state.
#' If a file already exists at \code{path}, it is deleted before saving.
#'
#' @param model An AOCL-DA model object (any R6 class that has a
#'   \code{$save()} method).
#' @param path Character. File path where the model will be saved.
#' @return Invisibly returns the model.
#' @export
#' @examples
#' \dontrun{
#' clf <- SVC$new(kernel = "rbf")
#' clf$fit(X_train, y_train)
#' save_model(clf, "my_svc.rds")
#' }
save_model <- function(model, path) {
    if (file.exists(path)) {
        file.remove(path)
    }
    model$save(path)
    return(invisible(model))
}

#' Print model metadata from a saved RDS file
#'
#' Prints version and build metadata embedded in a saved model file
#' without fully reconstructing the model object.
#'
#' @param path Character. Path to the RDS file produced by \code{$save()} or
#'   \code{save_model()}.
#' @export
print_model_metadata <- function(path) {
    state <- readRDS(path)
    rcpp_print_model_metadata(state$cpp_state$data)
    invisible(NULL)
}
