/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "pairwise_distances.hpp"
#include "aoclda.h"
#include "aoclda_metrics.h"

namespace ARCH {
namespace da_metrics {
namespace pairwise_distances {

// Create a high level template function that can be used for both single and double precision.
// Check if metric is valid and direct accordingly, otherwise throw an error.
template <typename T>
da_status pairwise_distance_kernel(da_order order, da_int m, da_int n, da_int k,
                                   const T *X, da_int ldx, const T *Y, da_int ldy, T *D,
                                   da_int ldd, T p, da_metric metric) {
    switch (metric) {
    case da_euclidean:
        return da_metrics::pairwise_distances::euclidean(order, m, n, k, X, ldx, Y, ldy,
                                                         D, ldd, false);
    case da_sqeuclidean:
        return da_metrics::pairwise_distances::euclidean(order, m, n, k, X, ldx, Y, ldy,
                                                         D, ldd, true);
    case da_manhattan:
        return da_metrics::pairwise_distances::manhattan(order, m, n, k, X, ldx, Y, ldy,
                                                         D, ldd);
    case da_cosine:
        return da_metrics::pairwise_distances::cosine(order, m, n, k, X, ldx, Y, ldy, D,
                                                      ldd);
    case da_minkowski:
        if (p == 2.0)
            return da_metrics::pairwise_distances::euclidean(order, m, n, k, X, ldx, Y,
                                                             ldy, D, ldd, false);
        else if (p == 1.0)
            return da_metrics::pairwise_distances::manhattan(order, m, n, k, X, ldx, Y,
                                                             ldy, D, ldd);
        else
            return da_metrics::pairwise_distances::minkowski(order, m, n, k, X, ldx, Y,
                                                             ldy, D, ldd, p);
    default:
        return da_status_not_implemented;
    }
}

// Create a high level template function that can be used for both single and double precision.
// Check if metric is valid and direct accordingly, otherwise throw an error.
template <typename T>
da_status pairwise_distance_error_check_kernel(da_order order, da_int m, da_int n,
                                               da_int k, const T *X, da_int ldx,
                                               const T *Y, da_int ldy, T *D, da_int ldd,
                                               T p, da_metric metric) {
    if (m < 1 || k < 1) {
        return da_status_invalid_array_dimension;
    }
    if (order == column_major && ldx < m)
        return da_status_invalid_leading_dimension;
    if (order == row_major && ldx < k)
        return da_status_invalid_leading_dimension;
    if (X == nullptr || D == nullptr)
        return da_status_invalid_pointer;
    // Check if Y is nullptr to continue with appropriate check of the leading dimension.
    if (Y != nullptr) {
        if (n < 1) {
            return da_status_invalid_array_dimension;
        }
        if (order == column_major) {
            if (ldy < n || ldd < m)
                return da_status_invalid_leading_dimension;
        } else {
            if (ldy < k || ldd < n)
                return da_status_invalid_leading_dimension;
        }
    } else {
        if (ldd < m)
            return da_status_invalid_leading_dimension;
    }

    if ((metric == da_minkowski) && (p <= 0))
        return da_status_invalid_input;

    return pairwise_distance_kernel(order, m, n, k, X, ldx, Y, ldy, D, ldd, p, metric);
}

template da_status pairwise_distance_kernel<float>(da_order order, da_int m, da_int n,
                                                   da_int k, const float *X, da_int ldx,
                                                   const float *Y, da_int ldy, float *D,
                                                   da_int ldd, float p, da_metric metric);

template da_status pairwise_distance_kernel<double>(da_order order, da_int m, da_int n,
                                                    da_int k, const double *X, da_int ldx,
                                                    const double *Y, da_int ldy,
                                                    double *D, da_int ldd, double p,
                                                    da_metric metric);

template da_status pairwise_distance_error_check_kernel<float>(
    da_order order, da_int m, da_int n, da_int k, const float *X, da_int ldx,
    const float *Y, da_int ldy, float *D, da_int ldd, float p, da_metric metric);

template da_status pairwise_distance_error_check_kernel<double>(
    da_order order, da_int m, da_int n, da_int k, const double *X, da_int ldx,
    const double *Y, da_int ldy, double *D, da_int ldd, double p, da_metric metric);
} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH
