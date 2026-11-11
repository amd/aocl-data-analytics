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

#ifndef METRICS_R_HPP
#define METRICS_R_HPP

#include "aoclda.h"
#include "internal_utilities_r.hpp"
#include <Rcpp.h>

using namespace Rcpp;

inline da_metric string_to_metric(const std::string &metric) {
    if (metric == "euclidean") {
        return da_euclidean;
    } else if (metric == "l2") {
        return da_l2;
    } else if (metric == "sqeuclidean") {
        return da_sqeuclidean;
    } else if (metric == "manhattan") {
        return da_manhattan;
    } else if (metric == "l1") {
        return da_l1;
    } else if (metric == "cityblock") {
        return da_cityblock;
    } else if (metric == "cosine") {
        return da_cosine;
    } else if (metric == "minkowski") {
        return da_minkowski;
    } else if (metric == "euclidean_gemm") {
        return da_euclidean_gemm;
    } else if (metric == "sqeuclidean_gemm") {
        return da_sqeuclidean_gemm;
    } else if (metric == "inner product" || metric == "inner_product") {
        return da_inner_product;
    } else {
        Rcpp::stop("Given metric does not exist. Available choices are: "
                   "'euclidean', 'l2', 'sqeuclidean', 'manhattan', "
                   "'l1', 'cityblock', 'cosine', 'minkowski', "
                   "'euclidean_gemm', 'sqeuclidean_gemm', "
                   "'inner_product'.");
    }
    return da_euclidean; // unreachable
}

inline NumericMatrix da_pairwise_distances_wrapper(NumericMatrix X,
                                                   Nullable<NumericMatrix> Y = R_NilValue,
                                                   std::string metric = "euclidean",
                                                   double p = 2.0) {
    da_status status;
    da_int m, n, k_x, k_y, ldx, ldy, ldd;

    m = X.nrow();
    k_x = X.ncol();
    ldx = m;

    da_int ncols = m;
    n = 0;

    if (Y.isNotNull()) {
        const NumericMatrix &Y_mat = Y.get();
        n = Y_mat.nrow();
        k_y = Y_mat.ncol();
        ldy = n;
        ncols = n;

        if (k_x != k_y) {
            stop("X and Y must have the same number of features");
        }
    }

    da_metric metric_enum = string_to_metric(metric);

    NumericMatrix D(m, ncols);
    ldd = m;

    if (Y.isNotNull()) {
        const NumericMatrix &Y_mat = Y.get();
        status =
            da_pairwise_distances_d(column_major, m, n, k_x, X.begin(), ldx,
                                    Y_mat.begin(), ldy, D.begin(), ldd, p, metric_enum);
    } else {
        status = da_pairwise_distances_d(column_major, m, n, k_x, X.begin(), ldx, nullptr,
                                         n, D.begin(), ldd, p, metric_enum);
    }

    status_to_exception(status);

    return D;
}

#endif // METRICS_R_HPP
