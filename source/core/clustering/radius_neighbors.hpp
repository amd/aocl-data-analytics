/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "aoclda.h"
#include "da_error.hpp"
#include "da_vector.hpp"
#include "macros.h"
#include <vector>

namespace ARCH {

// Declare functions needed in da_utils to avoid multiple includes of unprotected file
namespace da_utils {

void blocking_scheme(da_int n_samples, da_int block_size, da_int &n_blocks,
                     da_int &block_rem);

da_int get_n_threads_loop(da_int loop_size);

} // namespace da_utils

namespace da_radius_neighbors {

/*
Compute the radius neighbors: for each sample point, the indices of the samples within a given
radius are returned. The brute-force method is used.
*/
template <typename T>
da_status radius_neighbors(da_int n_samples, da_int n_features, const T *A, da_int lda,
                           T eps, std::vector<da_vector::da_vector<da_int>> &neighbors,
                           da_errors::da_error_t *err);

} // namespace da_radius_neighbors

} // namespace ARCH
