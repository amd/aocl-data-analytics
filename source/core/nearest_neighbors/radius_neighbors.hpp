/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include "aoclda.h"
#include "da_error.hpp"
#include "da_vector.hpp"
#include "macros.h"
#include <vector>

namespace ARCH {

namespace da_radius_neighbors {

/*
Compute the radius neighbors: for each sample point, the indices of the samples within a given
radius are returned.
*/
template <typename T>
da_status radius_neighbors_brute(da_int n_samples, da_int n_features, const T *A,
                                 da_int lda, T eps, da_metric metric, T p,
                                 std::vector<da_vector::da_vector<da_int>> &neighbors,
                                 da_errors::da_error_t *err);

template <typename T>
da_status radius_neighbors_kd_tree(da_int n_samples, da_int n_features, const T *A,
                                   da_int lda, T eps, da_metric metric, T p,
                                   da_int leaf_size,
                                   std::vector<da_vector::da_vector<da_int>> &neighbors,
                                   da_errors::da_error_t *err);

template <typename T>
da_status radius_neighbors_ball_tree(da_int n_samples, da_int n_features, const T *A,
                                     da_int lda, T eps, da_metric metric, T p,
                                     da_int leaf_size,
                                     std::vector<da_vector::da_vector<da_int>> &neighbors,
                                     da_errors::da_error_t *err);

} // namespace da_radius_neighbors

} // namespace ARCH
