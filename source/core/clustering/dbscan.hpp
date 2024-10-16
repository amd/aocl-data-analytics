/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "da_vector.hpp"
#include "dbscan_types.hpp"
#include "macros.h"
#include <algorithm>
#include <random>
#include <string>

namespace ARCH {

namespace da_dbscan {

using namespace da_dbscan_types;

/* DBSCAN class */
template <typename T> class dbscan : public basic_handle<T> {
  public:
    ~dbscan();

  private:
    da_int n_samples = 0;
    da_int n_features = 0;

    // Set true when initialization is complete
    bool initdone = false;

    // Set true when dbscan clustering is computed successfully
    bool iscomputed = false;

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
    da_int metric = euclidean;

    // Scalar outputs
    da_int n_core_samples = 0;
    da_int n_clusters = 0;

    // Arrays containing output data
    da_vector::da_vector<da_int>
        core_sample_indices; // Use da_vector since we will be dynamically expanding this array
    std::vector<da_int> labels;

    // Internal arrays
    std::vector<da_vector::da_vector<da_int>>
        neighbors; // Use da_vector since we will be dynamically expanding this array

    da_status dbscan_clusters();

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
