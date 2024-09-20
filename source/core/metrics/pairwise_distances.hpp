/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "aoclda.h"
#include "aoclda_metrics.h"
#include "euclidean_distance.hpp"

// Create a high level template function that can be used for both single and double precision.
// Check if metric is valid and direct accordingly, otherwise throw an error.
// Check value of force_all_finite and direct accordingly, otherwise throw an error.
template <typename T>
da_status pairwise_distance_kernel(da_order order, da_int m, da_int n, da_int k,
                                   const T *X, da_int ldx, const T *Y, da_int ldy, T *D,
                                   da_int ldd, da_metric metric,
                                   da_data_types force_all_finite) {
    da_status status = da_status_success;
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
    // Currently no checks for NaNs/Infs are implemented.
    if (force_all_finite != da_allow_infinite)
        return da_status_not_implemented;

    if (metric == da_euclidean)
        return da_metrics::pairwise_distances::euclidean(order, m, n, k, X, ldx, Y, ldy,
                                                         D, ldd, false);
    if (metric == da_sqeuclidean)
        return da_metrics::pairwise_distances::euclidean(order, m, n, k, X, ldx, Y, ldy,
                                                         D, ldd, true);
    else
        return da_status_not_implemented;

    return status;
}