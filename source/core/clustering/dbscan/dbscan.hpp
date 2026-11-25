/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "da_vector.hpp"
#include "macros.h"
#include "nearest_neighbors_types.hpp"
#include <algorithm>
#include <random>
#include <string>

namespace ARCH {

namespace da_dbscan {

using namespace da_neighbors_types;

/* DBSCAN class */
template <typename T> class dbscan : public basic_handle<T> {
  public:
    ~dbscan();

  private:
    da_int n_samples = 0;
    da_int n_features = 0;

    // Set true when initialization is complete
    bool initdone = false;

    // User's data
    const T *A = nullptr;
    da_int lda = 0;
    da_int lda_in = 0;

    // Utility pointer to column major allocated copy of user's data
    T *A_temp = nullptr;

    // Options
    T eps = 0.5;
    da_int min_samples = 5;
    da_int leaf_size = 30;
    T p = 2.0;

    da_int algorithm = brute;
    da_int metric = da_euclidean;

    // Scalar outputs
    da_int n_core_samples = 0;
    da_int n_clusters = 0;

    // Arrays containing output data
    std::vector<da_int> labels;

    // Internal arrays
    std::vector<da_vector::da_vector<da_int>>
        neighbors; // Use da_vector since we will be dynamically expanding this array
    std::vector<bool> is_core_sample;

    // Miscellaneous variables
    da_int min_samples_m1 = 0;
    nn_algorithm alg_internal = brute;
    da_metric metric_internal = da_euclidean;

    da_status dbscan_clusters();

    da_status dbscan_clusters_parallel();

    da_status dbscan_clusters_serial();

  public:
    dbscan(da_errors::da_error_t &err);

    da_status get_result(da_result query, da_int *dim, T *result);

    da_status get_result(da_result query, da_int *dim, da_int *result);

    void refresh();

    /* Store details about user's data matrix in preparation for DBSCAN computation */
    da_status set_data(da_int n_samples, da_int n_features, const T *A_in, da_int lda_in);

    /* Compute the DBSCAN clusters */
    da_status compute();
};

} // namespace da_dbscan

} // namespace ARCH
