/*
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
 */

#include "da_std.hpp"
#include "pairwise_distances.hpp"
#include "partitioning_infrastructure.hpp"

namespace ARCH {
namespace da_metrics {
namespace pairwise_distances {

// Functor for Minkowski distance calculation
template <typename T, da_int MR, da_int NR> struct MinkowskiKernelFunctor_packed {
    T p;
    MinkowskiKernelFunctor_packed(T p) : p(p) {}
    template <bool is_first_slice>
    inline void operator()(da_int m, da_int n, da_int k, const T *Atilde, const T *Btilde,
                           T *C, da_int ldc) const {

        // Call microkernel if we have a full block
        if ((m == MR) && (n == NR)) {
#if defined(__AVX512F__)
            avx512::minkowski_kernel_packed<T, MR, NR, is_first_slice>(k, Atilde, Btilde,
                                                                       C, ldc, p);
#elif defined(__AVX2__)
            avx2::minkowski_kernel_packed<T, MR, NR, is_first_slice>(k, Atilde, Btilde, C,
                                                                     ldc, p);
#endif
        } else {
            // Partial block: use masked kernel directly on C
#if defined(__AVX512F__)
            avx512::minkowski_kernel_packed_masked<T, MR, NR, is_first_slice>(
                m, n, k, Atilde, Btilde, C, ldc, p);
#elif defined(__AVX2__)
            avx2::minkowski_kernel_packed_masked<T, MR, NR, is_first_slice>(
                m, n, k, Atilde, Btilde, C, ldc, p);
#endif
        }
    }
};

// Functor for SqEuclidean distance calculation
template <typename T> struct PowPostOp {
    T invp;
    PowPostOp(T p) : invp(1.0 / p) {}
    inline void operator()(da_int m, da_int n, T *C, da_int ldc) const {
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i < m; i++) {
                C[i + j * ldc] = std::pow(C[i + j * ldc], invp);
            }
        }
    }
};

template <typename T>
da_status minkowski(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd, T p) {
    constexpr da_int MR = BlockSizes<T>::MR;
    constexpr da_int NR = BlockSizes<T>::NR;
    MinkowskiKernelFunctor_packed<T, MR, NR> kernel_op(p);
    PowPostOp<T> post_op(p);
    return dispatch_packed_distance(order, m, n, k, X, ldx, Y, ldy, D, ldd, kernel_op,
                                    post_op);
}

template da_status minkowski<float>(da_order order, da_int m, da_int n, da_int k,
                                    const float *X, da_int ldx, const float *Y,
                                    da_int ldy, float *D, da_int ldd, float p);

template da_status minkowski<double>(da_order order, da_int m, da_int n, da_int k,
                                     const double *X, da_int ldx, const double *Y,
                                     da_int ldy, double *D, da_int ldd, double p);
} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH