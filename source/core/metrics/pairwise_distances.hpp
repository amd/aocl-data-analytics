/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "aoclda_metrics.h"
#include "macros.h"

namespace ARCH {

template <typename T>
void euclidean_distance(da_order order, da_int m, da_int n, da_int k, const T *X,
                        da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T *X_norms,
                        da_int compute_X_norms, T *Y_norms, da_int compute_Y_norms,
                        bool square, bool X_is_Y);

// Create a high level template function that can be used for both single and double precision.
// Check if metric is valid and direct accordingly, otherwise throw an error.
// Check value of force_all_finite and direct accordingly, otherwise throw an error.
template <typename T>
da_status pairwise_distance_kernel(da_order order, da_int m, da_int n, da_int k,
                                   const T *X, da_int ldx, const T *Y, da_int ldy, T *D,
                                   da_int ldd, da_metric metric,
                                   da_data_types force_all_finite);

namespace da_metrics {
namespace pairwise_distances {

template <typename T>
da_status euclidean(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd, bool square_distances);
}
} // namespace da_metrics

} // namespace ARCH
