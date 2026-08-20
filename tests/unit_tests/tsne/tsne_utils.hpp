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
 */

#ifndef TSNE_UTILS_HPP
#define TSNE_UTILS_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

namespace tsne_metrics {

/**
 * Compute trustworthiness metric.
 *
 * Trustworthiness measures how well the local structure is preserved in the
 * embedding.  It penalizes points that are close in the low-dimensional space
 * but far in the original (high-dimensional) space.
 *
 * Both neighbor sets and ranks are derived from pairwise distance matrices
 * (one per space) so tie-breaking is consistent within each space.
 */
template <typename T>
T compute_trustworthiness(const T *X_high, const T *X_low, da_int n_samples,
                          da_int n_features, da_int n_components, da_int k,
                          da_order order = row_major) {
    if (n_samples < 2 || k < 1)
        return T(1);

    k = std::min(k, n_samples - 1);

    da_int ld_high = (order == row_major) ? n_features : n_samples;
    da_int ld_low = (order == row_major) ? n_components : n_samples;

    // Pairwise distances for both spaces
    std::vector<T> dist_high(n_samples * n_samples);
    std::vector<T> dist_low(n_samples * n_samples);
    if (da_pairwise_distances(order, n_samples, n_samples, n_features, X_high, ld_high,
                              X_high, ld_high, dist_high.data(), n_samples, T(2),
                              da_sqeuclidean) != da_status_success)
        return T(0);
    if (da_pairwise_distances(order, n_samples, n_samples, n_components, X_low, ld_low,
                              X_low, ld_low, dist_low.data(), n_samples, T(2),
                              da_sqeuclidean) != da_status_success)
        return T(0);

    // D is stored in the layout specified by `order`
    auto didx = [&](da_int row, da_int col) -> da_int {
        return (order == row_major) ? row * n_samples + col : col * n_samples + row;
    };

    // Derive ranks + k-NN sets for high-dim, k-NN sets for low-dim
    std::vector<std::vector<da_int>> ranks_high(n_samples);
    std::vector<std::set<da_int>> neighbors_high(n_samples), neighbors_low(n_samples);
    for (da_int i = 0; i < n_samples; ++i) {
        std::vector<da_int> idx;
        idx.reserve(n_samples - 1);
        for (da_int j = 0; j < n_samples; ++j)
            if (j != i)
                idx.push_back(j);

        // High-dim: ranks and neighbor set
        std::sort(idx.begin(), idx.end(), [&](da_int a, da_int b) {
            return dist_high[didx(i, a)] < dist_high[didx(i, b)];
        });
        ranks_high[i].assign(n_samples, 0);
        for (da_int j = 0; j < (da_int)idx.size(); ++j) {
            ranks_high[i][idx[j]] = j + 1;
            if (j < k)
                neighbors_high[i].insert(idx[j]);
        }

        // Low-dim: neighbor set only
        std::sort(idx.begin(), idx.end(), [&](da_int a, da_int b) {
            return dist_low[didx(i, a)] < dist_low[didx(i, b)];
        });
        for (da_int j = 0; j < k; ++j)
            neighbors_low[i].insert(idx[j]);
    }

    T penalty = T(0);
    for (da_int i = 0; i < n_samples; ++i) {
        for (da_int nb : neighbors_low[i]) {
            if (neighbors_high[i].count(nb) == 0)
                penalty += (ranks_high[i][nb] - k);
        }
    }

    T n = static_cast<T>(n_samples);
    T kf = static_cast<T>(k);
    T denom = n * kf * (T(2) * n - T(3) * kf - T(1));

    if (denom <= T(0))
        return T(1);

    T trust = T(1) - (T(2) / denom) * penalty;
    return std::max(T(0), std::min(T(1), trust));
}

template <typename T>
bool check_embedding_finite(const T *embedding, da_int n_samples, da_int n_components) {
    for (da_int i = 0; i < n_samples * n_components; ++i) {
        if (!std::isfinite(embedding[i])) {
            return false;
        }
    }
    return true;
}

} // namespace tsne_metrics

#endif // TSNE_UTILS_HPP
