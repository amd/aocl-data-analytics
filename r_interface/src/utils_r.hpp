/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef UTILS_R_HPP
#define UTILS_R_HPP

#include <Rcpp.h>
#include <string>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

inline List train_test_split(SEXP X, da_int test_size, da_int train_size,
                             Nullable<IntegerVector> shuffled_indices = R_NilValue) {

    da_status status;

    if (TYPEOF(X) == REALSXP) {
        const double *X_ptr = REAL(X);
        da_int m = Rf_nrows(X);
        da_int n = Rf_ncols(X);

        NumericMatrix X_train(train_size, n);
        NumericMatrix X_test(test_size, n);

        if (shuffled_indices.isNotNull()) {
            const IntegerVector &indices = as<IntegerVector>(shuffled_indices);
            status = da_train_test_split_d(column_major, m, n, X_ptr, m, train_size,
                                           test_size, indices.begin(), X_train.begin(),
                                           train_size, X_test.begin(), test_size);
        } else {
            status = da_train_test_split_d(column_major, m, n, X_ptr, m, train_size,
                                           test_size, nullptr, X_train.begin(),
                                           train_size, X_test.begin(), test_size);
        }

        status_to_exception(status);

        return List::create(Named("X_train") = X_train, Named("X_test") = X_test);

    } else if (TYPEOF(X) == INTSXP) {
        const da_int *X_ptr = INTEGER(X);
        da_int m = Rf_nrows(X);
        da_int n = Rf_ncols(X);

        IntegerMatrix X_train(train_size, n);
        IntegerMatrix X_test(test_size, n);

        if (shuffled_indices.isNotNull()) {
            const IntegerVector &indices = as<IntegerVector>(shuffled_indices);
            status = da_train_test_split_int(column_major, m, n, X_ptr, m, train_size,
                                             test_size, indices.begin(), X_train.begin(),
                                             train_size, X_test.begin(), test_size);
        } else {
            status = da_train_test_split_int(column_major, m, n, X_ptr, m, train_size,
                                             test_size, nullptr, X_train.begin(),
                                             train_size, X_test.begin(), test_size);
        }
        status_to_exception(status);

        return List::create(Named("X_train") = X_train, Named("X_test") = X_test);

    } else {
        stop("Unsupported data type. Use numeric or integer matrix.");
    }
}

inline IntegerVector get_shuffled_indices(da_int size, da_int seed, da_int train_size,
                                          da_int test_size, da_int fp_precision,
                                          SEXP classes = R_NilValue) {
    IntegerVector shuffled_indices(size);
    da_status status;
    if (classes != R_NilValue) {
        if (TYPEOF(classes) == REALSXP) {
            const double *class_ptr = REAL(classes);
            status =
                da_get_shuffled_indices_d(size, seed, train_size, test_size, fp_precision,
                                          class_ptr, shuffled_indices.begin());
        } else if (TYPEOF(classes) == INTSXP) {
            const da_int *class_ptr = INTEGER(classes);
            status = da_get_shuffled_indices_int(size, seed, train_size, test_size,
                                                 fp_precision, class_ptr,
                                                 shuffled_indices.begin());
        } else {
            stop("Unsupported class type. Use numeric or integer vector.");
        }
    } else {
        status =
            da_get_shuffled_indices_int(size, seed, train_size, test_size, fp_precision,
                                        (da_int *)nullptr, shuffled_indices.begin());
    }
    status_to_exception(status);
    return shuffled_indices;
}

inline void print_model_metadata_from_raw(RawVector data) {
    std::vector<char> vec(data.begin(), data.end());
    std::ostringstream oss;
    std::streambuf *old = std::cout.rdbuf(oss.rdbuf());
    da_status status = da_print_model_metadata(vec);
    std::cout.rdbuf(old);
    Rcpp::Rcout << oss.str();
    status_to_exception(status);
}

#endif // UTILS_R_HPP
