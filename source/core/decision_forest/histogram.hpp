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
#include "aoclda.h"
#include "macros.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

namespace ARCH {
namespace da_decision_forest {

template <typename T> class bins {
    /* Class computing and holding a discretized version of an (n_samples x n_features) data matrix.
     *
     * For each columns j in the input matrix X, values X[:, j] are placed into bins indexed in [0:max_bin-1]
     * Each bin is computed to contain approximately the same number of elements.
     *
     * n_samples, n_features: input matrix dimensions
     * max_bin: maximum number of bins amongst all the columns of X
     *          max_bin must be at least half of the samples.
     * thresholds: (max_bins - 1) * n_features. Holds the thresholds values for each bin and each feature.
     *             e.g., if X[i,j] = v is assigned to bin b, then thresholds[j, b] < v < thresholds[j, b+1]
     * nbins: number of bins used for each feature. nbins[j] can be < max_bins for a feature j if it contains highly clustered data.
     *        For example, if feature j is already categorical, nbins[j] will contain the number of categories of j.
     * binned_data: n_samples * n_features. Holds the bin indices for each sample and each feature.
     */
  public:
    da_int max_bin;
    da_int n_samples;
    da_int n_features;

    std::vector<T> thresholds;
    std::vector<da_int> nbins;
    std::vector<uint16_t> binned_data;

  public:
    bins(da_int max_bin, da_int n_samples, da_int n_features)
        : max_bin(max_bin), n_samples(n_samples), n_features(n_features) {
        this->max_bin = std::min(max_bin, n_samples / 2);
        if (this->max_bin < 2)
            throw std::invalid_argument(
                "Histograms: Number of samples insufficients to contain at "
                "least 2 bins with 2 samples each.");
        if (max_bin > std::numeric_limits<uint16_t>::max())
            throw std::invalid_argument("Histograms: number of bins is too big.");
        thresholds.resize((this->max_bin - 1) * n_features);
        nbins.resize(n_features);
        binned_data.resize(n_samples * n_features);
    }

    void compute_thresholds(std::vector<T> &values, da_int vals_idx, da_int feat_idx,
                            T tol = (T)1.0e-05);
    void compute_thresholds_continuous(std::vector<T> &values, da_int vals_idx,
                                       da_int feat_idx, T tol = (T)1.0e-05);
    da_status compute_histograms(const T *X, da_int n_samples, da_int n_features,
                                 da_int ldx);

    // Getters for testing purposes
    const std::vector<T> &get_thresholds() { return thresholds; }
    const std::vector<da_int> &get_nbins() { return nbins; }
    const std::vector<uint16_t> &get_bins() { return binned_data; }
    void resize_thresholds(da_int n) { thresholds.resize(n); }
    void resize_nbins(da_int n) { nbins.resize(n); }
    void resize_bins(da_int n) { binned_data.resize(n); }
};

} // namespace da_decision_forest
} // namespace ARCH