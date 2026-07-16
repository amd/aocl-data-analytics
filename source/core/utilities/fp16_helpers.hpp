/******************************************************************************
* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef FP16_HELPERS_HPP
#define FP16_HELPERS_HPP

// Suppress "loop not vectorized" diagnostics from `#pragma omp simd` on
// _Float16 loops when the target architecture lacks FP16 SIMD support.
// On hardware that does have FP16 vectorization the loops vectorize as
// expected; the warning is purely informational.
//
// clang reports this under -Wpass-failed (-Wpass-failed=transform-warning);
// GCC, when configured to surface vectorization remarks as warnings, uses
// the -Wopenmp-simd category. We also silence -Wunknown-pragmas/-Wpragmas
// so the suppression itself is harmless on compiler versions that don't
// recognise the inner warning name.
// The matching `pop` at the end of this file restores the including
// translation unit's diagnostic state, so these suppressions apply only to
// code within this header and do not leak into files that #include it.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wpass-failed"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wopenmp-simd"
#endif

/*
 *  Overloaded CPP wrappers for _Float16 support.
 *  Provides _Float16 overloads for AOCL-DLP (aocl_gemm), BLAS, LAPACK,
 *  and standard math functions.
 */

#include "aoclda_types.h"
#include "da_std.hpp"
#include "macros.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace da_fp16 {

// "Wider" intermediate type used to perform arithmetic that would otherwise
// overflow or lose precision in T. For _Float16 this is float; for other
// floating point types it is T itself (so the code paths are zero-overhead at runtime).
template <typename T>
using wider_t = std::conditional_t<std::is_same_v<T, _Float16>, float, T>;

// Guard against generating _Float16 code when AVX512FP16 is not available
// For non-_Float16 U this is always true, so float/double paths are unaffected.
template <typename U>
inline constexpr bool fp16_codegen_ok =
#if defined(__AVX512FP16__)
    true;
#else
    !std::is_same_v<U, _Float16>;
#endif

// Helpers for arithmetic involving integer sample counts. Quantities such as
// `T(n_samples)` can overflow when T is _Float16 (max ~65504) and even when
// representable can lose precision (the _Float16 significand is 11 bits, so
// integers above 2048 are not all representable exactly). The following
// helpers perform the arithmetic in `wider_t<T>` and convert back to T at the
// end. For T in {float, double} they are zero-overhead.
template <typename T> __attribute__((always_inline)) inline T div_int(T x, da_int n) {
    if constexpr (std::is_same_v<T, _Float16>) {
        return static_cast<T>(static_cast<float>(x) / static_cast<float>(n));
    } else {
        return x / static_cast<T>(n);
    }
}
template <typename T> __attribute__((always_inline)) inline T mul_int(T x, da_int n) {
    if constexpr (std::is_same_v<T, _Float16>) {
        return static_cast<T>(static_cast<float>(x) * static_cast<float>(n));
    } else {
        return x * static_cast<T>(n);
    }
}
template <typename T> __attribute__((always_inline)) inline T inv_int(da_int n) {
    if constexpr (std::is_same_v<T, _Float16>) {
        return static_cast<T>(1.0f / static_cast<float>(n));
    } else {
        return T(1) / static_cast<T>(n);
    }
}
template <typename T> __attribute__((always_inline)) inline T sqrt_int(da_int n) {
    if constexpr (std::is_same_v<T, _Float16>) {
        return static_cast<T>(std::sqrt(static_cast<float>(n)));
    } else {
        return std::sqrt(static_cast<T>(n));
    }
}

// float (binary32) -> half (binary16) bit pattern, round-to-nearest-even
inline uint16_t f32_to_f16_bits(float a) {
    uint32_t fbits;
    std::memcpy(&fbits, &a, sizeof(fbits));
    uint16_t sign = static_cast<uint16_t>((fbits >> 16) & 0x8000u);
    int32_t exponent = static_cast<int32_t>((fbits >> 23) & 0xFFu) - 127;
    uint32_t mantissa = fbits & 0x7FFFFFu;
    if (exponent == 128)
        return sign | 0x7C00u | static_cast<uint16_t>(mantissa ? 0x0200u : 0u);
    if (exponent > 15)
        return sign | 0x7C00u;
    if (exponent > -15) {
        uint16_t hmant = static_cast<uint16_t>(mantissa >> 13);
        uint32_t rem = mantissa & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (hmant & 1u))) {
            hmant++;
            if (hmant == 0x0400u) {
                hmant = 0;
                exponent++;
            }
            if (exponent > 15)
                return sign | 0x7C00u;
        }
        return sign | static_cast<uint16_t>((exponent + 15) << 10) | hmant;
    }
    if (exponent >= -24) {
        mantissa |= 0x800000u;
        int32_t shift = 13 - (exponent + 14);
        uint16_t hmant = static_cast<uint16_t>(mantissa >> shift);
        uint32_t rem = mantissa & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (hmant & 1u)))
            hmant++;
        return sign | hmant;
    }
    return sign;
}

// half (binary16) bit pattern -> float (binary32)
inline float f16_bits_to_f32(uint16_t a) {
    uint32_t sign = static_cast<uint32_t>(a & 0x8000u) << 16;
    int32_t exponent = (a >> 10) & 0x1F;
    uint32_t mantissa = a & 0x3FFu;
    uint32_t fbits;
    if (exponent == 0) {
        if (mantissa == 0) {
            fbits = sign;
        } else {
            exponent = 1;
            while (!(mantissa & 0x400u)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FFu;
            fbits = sign | (static_cast<uint32_t>(exponent + 127 - 15) << 23) |
                    (mantissa << 13);
        }
    } else if (exponent == 31) {
        fbits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        fbits =
            sign | (static_cast<uint32_t>(exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &fbits, sizeof(result));
    return result;
}

inline _Float16 bits_to_f16(uint16_t bits) {
#if defined(__FLT16_MAX__)
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return h;
#else
    // _Float16 is the non-native da_float16 here; so just return a placeholder.
    (void)bits;
    return _Float16();
#endif
}

inline uint16_t f16_to_bits(_Float16 h) {
#if defined(__FLT16_MAX__)
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
#else
    (void)h;
    return 0;
#endif
}

} // namespace da_fp16

extern "C" {
#define BLIS_ENABLE_CBLAS
#ifndef _WIN32
#include "aocl_dlp.h"
#endif
#include "cblas.h"
}

// Some toolchains' default runtime libraries (clang-cl on Windows, older
// libgcc on Linux) do not ship the compiler-rt builtins for half<->single/
// double conversion (__truncsfhf2, __truncdfhf2, __extendhfsf2). The link therefore
// breaks whenever clang selects the older toolchain.
// To be robust regardless of the detected libgcc/compiler-rt, exactly one TU
// (fp16_helpers.cpp) defines DA_FP16_DEFINE_BUILTINS and emits these as
// non-inline definitions into the AOCL-DA library, so the archive member is
// pulled in to satisfy references. The definitions are given hidden ELF
// visibility so they bind only to AOCL-DA's own internal references and do
// NOT interpose on the host program's libgcc/compiler-rt copies (which may
// be SIMD-accelerated and are also used by any non-AOCL-DA code in the
// same process for _Float16 casts). All other TUs that include this header
// see only prototypes.

// Every TU that includes this header sees only prototypes; the compiler-generated
// half<->single/double conversions become ordinary external references that the
// linker resolves against the single hidden-visibility definition emitted by the
// TU that defines DA_FP16_DEFINE_BUILTINS (fp16_helpers.cpp).
//
// ABI NOTE: these builtins MUST use the _Float16 (HFmode) calling convention,
// i.e. the half is passed/returned in an SSE register, NOT a uint16_t in a GPR.
// This is the convention both GCC and Clang hard-wire into the libcalls they
// emit for `(_Float16)`/`(float)` casts. On targets with F16C the casts inline
// to vcvtps2ph/vcvtph2ps and never call these symbols, which masks an incorrect
// ABI; but on a target WITHOUT F16C the compiler emits real calls and a
// uint16_t-based signature would read/return the half from the wrong register,
// silently corrupting every conversion.
extern "C" {
_Float16 __truncsfhf2(float a);
_Float16 __truncdfhf2(double a);
float __extendhfsf2(_Float16 a);
}

#ifdef DA_FP16_DEFINE_BUILTINS

// Hidden visibility prevents these definitions from interposing on the host
// program's libgcc/compiler-rt builtins of the same name. AOCL-DA's own
// references still resolve to them locally; everyone else in the process
// keeps using whatever (potentially SIMD-accelerated) copy their toolchain
// provided.
#if defined(__GNUC__) || defined(__clang__)
#define DA_FP16_BUILTIN_ATTR __attribute__((visibility("hidden")))
#else
#define DA_FP16_BUILTIN_ATTR
#endif

// float (binary32) -> half (binary16) with round-to-nearest-even
extern "C" DA_FP16_BUILTIN_ATTR _Float16 __truncsfhf2(float a) {
    return da_fp16::bits_to_f16(da_fp16::f32_to_f16_bits(a));
}

// double (binary64) -> half (binary16), via float (acceptable double-rounding)
extern "C" DA_FP16_BUILTIN_ATTR _Float16 __truncdfhf2(double a) {
    return __truncsfhf2(static_cast<float>(a));
}

// half (binary16) -> float (binary32)
extern "C" DA_FP16_BUILTIN_ATTR float __extendhfsf2(_Float16 a) {
    return da_fp16::f16_bits_to_f32(da_fp16::f16_to_bits(a));
}

#endif // DA_FP16_DEFINE_BUILTINS

namespace da_blas {

// -----------------------------------------------------------------------------
// cblas_gemm for _Float16
// -----------------------------------------------------------------------------
inline void cblas_gemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                       da_int m, da_int n, da_int k, _Float16 alpha, _Float16 const *A,
                       da_int lda, _Float16 const *B, da_int ldb, _Float16 beta,
                       _Float16 *C, da_int ldc) {
#ifndef _WIN32
    // Use aocl_gemm_f16f16f16of16 (DLP) on Linux
    const char order = (layout == CblasRowMajor) ? 'r' : 'c';
    const char transa = (transA == CblasNoTrans) ? 'n' : 't';
    const char transb = (transB == CblasNoTrans) ? 'n' : 't';
    // alpha and beta are passed as float16 (= uint16_t bit patterns); the JIT
    // kernel consumes them directly via vpbroadcastw + vmulph. They must be the
    // raw fp16 bit pattern, NOT a value cast to float (which would be truncated
    // to an integer by the implicit float -> uint16_t conversion, producing
    // garbage scalars - e.g. the gemm_scalar of -2.0 becomes nonsense).
    aocl_gemm_f16f16f16of16(order, transa, transb, m, n, k, da_fp16::f16_to_bits(alpha),
                            reinterpret_cast<const float16 *>(A), lda, 'N',
                            reinterpret_cast<const float16 *>(B), ldb, 'N',
                            da_fp16::f16_to_bits(beta), reinterpret_cast<float16 *>(C),
                            ldc, nullptr);
#else
    // No DLP on Windows: cast to float, call sgemm, cast back, though this codepath should not be reached
    // LCOV_EXCL_START
    std::vector<float> Af(m * k), Bf(k * n), Cf(m * n);
    for (da_int i = 0; i < m * k; i++)
        Af[i] = static_cast<float>(A[i]);
    for (da_int i = 0; i < k * n; i++)
        Bf[i] = static_cast<float>(B[i]);
    for (da_int i = 0; i < m * n; i++)
        Cf[i] = static_cast<float>(C[i]);
    float salpha = static_cast<float>(alpha);
    float sbeta = static_cast<float>(beta);
    cblas_sgemm(layout, transA, transB, m, n, k, salpha, Af.data(), lda, Bf.data(), ldb,
                sbeta, Cf.data(), ldc);
    for (da_int i = 0; i < m * n; i++)
        C[i] = static_cast<_Float16>(Cf[i]);
        // LCOV_EXCL_STOP
#endif
}
// -----------------------------------------------------------------------------
// cblas_scal for _Float16 (simple loop)
// -----------------------------------------------------------------------------
inline void cblas_scal(da_int n, _Float16 alpha, _Float16 *x, da_int incx) {
    for (da_int i = 0; i < n; i++) {
        x[i * incx] *= alpha;
    }
}

// -----------------------------------------------------------------------------
// cblas_nrm2 for _Float16 (float accumulation)
//
// Elements are loaded in half precision but each square is computed and
// accumulated in `float` to avoid overflowing _Float16's ~65504 range (and
// the precision loss of a half-precision running sum) when n is large. The
// result is cast back to _Float16 to keep the BLAS signature unchanged.
// -----------------------------------------------------------------------------
inline _Float16 cblas_nrm2(da_int n, _Float16 const *x, da_int incx) {
    float sum = 0.0f;
    for (da_int i = 0; i < n; i++) {
        float xi = static_cast<float>(x[i * incx]);
        sum += xi * xi;
    }
    return static_cast<_Float16>(std::sqrt(sum));
}

// -----------------------------------------------------------------------------
// cblas_dot for _Float16
//
// Block-then-promote accumulation: elements are loaded and multiplied in
// half precision and reduced into a per-block half-precision partial sum,
// which is then promoted to a `float` running accumulator before starting
// the next block. This preserves the half-precision memory bandwidth and
// per-FMA throughput inside the block (so AVX-512 FP16 / matrix engines
// can still be used) while preventing the running sum from overflowing
// `_Float16`'s ~65504 representable range when `n` is large.
//
// The final result is cast back to `_Float16` to keep the BLAS signature
// unchanged; callers that need to scale the result by `1/n` (or similar)
// to keep the value in half-precision range should do so on the final
// scalar.
// -----------------------------------------------------------------------------
inline _Float16 cblas_dot(da_int n, _Float16 const *x, da_int incx, _Float16 const *y,
                          da_int incy) {
    constexpr da_int BLOCK = 1024;
    float acc = 0.0f;
    da_int i = 0;
    if (incx == 1 && incy == 1) {
        for (; i + BLOCK <= n; i += BLOCK) {
            _Float16 block = (_Float16)0;
            // The reduction clause is only valid when _Float16 is a native
            // scalar type.
#if defined(__FLT16_MAX__)
#pragma omp simd reduction(+ : block)
#endif
            for (da_int j = 0; j < BLOCK; ++j)
                block += x[i + j] * y[i + j];
            acc += static_cast<float>(block);
        }
        _Float16 tail = (_Float16)0;
        for (; i < n; ++i)
            tail += x[i] * y[i];
        acc += static_cast<float>(tail);
    } else {
        for (; i + BLOCK <= n; i += BLOCK) {
            _Float16 block = (_Float16)0;
            for (da_int j = 0; j < BLOCK; ++j)
                block += x[(i + j) * incx] * y[(i + j) * incy];
            acc += static_cast<float>(block);
        }
        _Float16 tail = (_Float16)0;
        for (; i < n; ++i)
            tail += x[i * incx] * y[i * incy];
        acc += static_cast<float>(tail);
    }
    return static_cast<_Float16>(acc);
}

// -----------------------------------------------------------------------------
// cblas_dot_wide: dot product whose return type is the *widened* accumulator
// type (`da_fp16::wider_t<T>`).
//
// For T in {float, double, complex} this is just a thin wrapper around
// `cblas_dot` (since `wider_t<T> == T` there) and adds zero overhead.
//
// For T = _Float16, elements are loaded as half precision (preserving memory
// bandwidth) but immediately promoted to `float` for the multiplication and
// accumulation, and the result is returned as `float` *without* the final
// cast back to `_Float16`. This is the only safe routine to use when the
// caller will subsequently divide / scale the result before storing it,
// because individual products `x[i]*y[i]` are no longer subject to the
// _Float16 dynamic range (~6e-5 to 65504) for either underflow or overflow.
//
// Use this for quantities such as column variances `<X[:,j],X[:,j]>/N`
// that must be storable as `float` to remain non-zero / finite when the
// per-element magnitude is far from unity.
// -----------------------------------------------------------------------------
template <typename T>
inline da_fp16::wider_t<T> cblas_dot_wide(da_int n, T const *x, da_int incx, T const *y,
                                          da_int incy) {
    if constexpr (std::is_same_v<T, _Float16>) {
        float acc = 0.0f;
        if (incx == 1 && incy == 1) {
#pragma omp simd reduction(+ : acc)
            for (da_int i = 0; i < n; ++i) {
                acc += static_cast<float>(x[i]) * static_cast<float>(y[i]);
            }
        } else {
            for (da_int i = 0; i < n; ++i) {
                acc += static_cast<float>(x[i * incx]) * static_cast<float>(y[i * incy]);
            }
        }
        return acc;

    } else {
        return cblas_dot(n, x, incx, y, incy);
    }
}

// -----------------------------------------------------------------------------
// cblas_gemv for _Float16
// When incx == incy == 1 we treat x and y as M-by-1 / K-by-1 matrices and
// dispatch to cblas_gemm above (which uses aocl_gemm_f16f16f16of16 on Linux,
// or sgemm-via-float on Windows). For strided x/y we fall back to a direct
// loop performing native _Float16 arithmetic.
// -----------------------------------------------------------------------------
inline void cblas_gemv(CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, da_int m, da_int n,
                       _Float16 alpha, _Float16 const *A, da_int lda, _Float16 const *x,
                       da_int incx, _Float16 beta, _Float16 *y, da_int incy) {
    if (incx == 1 && incy == 1) {
        // Map GEMV to GEMM with N=1.
        //   y = alpha * op(A) * x + beta * y
        //   op(A) is (M_out by K) where M_out = (trans==NoTrans ? m : n),
        //                              K     = (trans==NoTrans ? n : m).
        const da_int M_out = (trans == CblasNoTrans) ? m : n;
        const da_int K = (trans == CblasNoTrans) ? n : m;
        // Leading dimensions for the K-by-1 (x) and M_out-by-1 (y) operands.
        // - Column major: ldx >= K, ldy >= M_out.
        // - Row major   : ldx >= 1, ldy >= 1 (single column).
        const da_int ldx = (layout == CblasColMajor) ? K : da_int(1);
        const da_int ldy = (layout == CblasColMajor) ? M_out : da_int(1);
        cblas_gemm(layout, trans, CblasNoTrans, M_out, da_int(1), K, alpha, A, lda, x,
                   ldx, beta, y, ldy);
        return;
    }

    // Strided fallback: native _Float16 accumulation.
    if (trans == CblasNoTrans) {
        // y(m) = alpha * A(m,n) * x(n) + beta * y(m)
        for (da_int i = 0; i < m; ++i) {
            _Float16 acc = (_Float16)0;
            for (da_int j = 0; j < n; ++j) {
                _Float16 aij =
                    (layout == CblasColMajor) ? A[i + j * lda] : A[j + i * lda];
                acc += aij * x[j * incx];
            }
            y[i * incy] = beta * y[i * incy] + alpha * acc;
        }
    } else {
        // y(n) = alpha * A^T(n,m) * x(m) + beta * y(n)
        for (da_int j = 0; j < n; ++j) {
            _Float16 acc = (_Float16)0;
            for (da_int i = 0; i < m; ++i) {
                _Float16 aij =
                    (layout == CblasColMajor) ? A[i + j * lda] : A[j + i * lda];
                acc += aij * x[i * incx];
            }
            y[j * incy] = beta * y[j * incy] + alpha * acc;
        }
    }
}

} // namespace da_blas

namespace da_fp16 {

// Leaf-case helper for cblas_syrk: pack-cast-ssyrk-cast on a small n-by-n
// triangle. Copies the actual n-by-k (or k-by-n) A entries into a tightly
// packed float buffer and the n-by-n C entries into another, calls
// cblas_ssyrk, and casts C back. The packed buffers avoid touching any of
// the (potentially huge) strided padding implied by lda / ldc.
constexpr da_int SYRK_FP16_LEAF = 32;

inline void cblas_syrk_fp16_leaf(CBLAS_ORDER layout, CBLAS_UPLO uplo,
                                 CBLAS_TRANSPOSE trans, da_int n, da_int k, float alpha,
                                 _Float16 const *A, da_int lda, float beta, _Float16 *C,
                                 da_int ldc) {
    const da_int a_rows = (trans == CblasNoTrans) ? n : k;
    const da_int a_cols = (trans == CblasNoTrans) ? k : n;
    const da_int lda_f = (layout == CblasColMajor) ? a_rows : a_cols;
    const da_int ldc_f = n;
    std::vector<float> Af(
        static_cast<size_t>(lda_f) *
        static_cast<size_t>((layout == CblasColMajor) ? a_cols : a_rows));
    std::vector<float> Cf(static_cast<size_t>(ldc_f) * static_cast<size_t>(n));
    if (layout == CblasColMajor) {
        for (da_int j = 0; j < a_cols; ++j)
            for (da_int i = 0; i < a_rows; ++i)
                Af[i + j * lda_f] = static_cast<float>(A[i + j * lda]);
        for (da_int j = 0; j < n; ++j)
            for (da_int i = 0; i < n; ++i)
                Cf[i + j * ldc_f] = static_cast<float>(C[i + j * ldc]);
    } else {
        for (da_int i = 0; i < a_rows; ++i)
            for (da_int j = 0; j < a_cols; ++j)
                Af[j + i * lda_f] = static_cast<float>(A[j + i * lda]);
        for (da_int i = 0; i < n; ++i)
            for (da_int j = 0; j < n; ++j)
                Cf[j + i * ldc_f] = static_cast<float>(C[j + i * ldc]);
    }
    cblas_ssyrk(layout, uplo, trans, n, k, alpha, Af.data(), lda_f, beta, Cf.data(),
                ldc_f);
    if (layout == CblasColMajor) {
        for (da_int j = 0; j < n; ++j)
            for (da_int i = 0; i < n; ++i)
                C[i + j * ldc] = static_cast<_Float16>(Cf[i + j * ldc_f]);
    } else {
        for (da_int i = 0; i < n; ++i)
            for (da_int j = 0; j < n; ++j)
                C[j + i * ldc] = static_cast<_Float16>(Cf[j + i * ldc_f]);
    }
}
} // namespace da_fp16

namespace da_blas {

// -----------------------------------------------------------------------------
// cblas_syrk for _Float16
//
// SYRK has no native _Float16 BLAS entry point in AOCL DLP, so we synthesize
// it as a recursive block decomposition:
//
//   For n > SYRK_FP16_LEAF, split n = n1 + n2 (n1 = n/2, n2 = n - n1) and
//   compute the two diagonal triangles with recursive cblas_syrk<_Float16>
//   calls, and the off-diagonal rectangular block with a single cblas_gemm
//   call that uses the native FP16 aocl_gemm_f16f16f16of16 path. The two
//   recursive triangles and the GEMM each consume `beta` exactly once on
//   disjoint regions of C, matching SYRK semantics.
//
//   For n <= SYRK_FP16_LEAF we fall back to da_fp16::cblas_syrk_fp16_leaf
//   (pack-cast-ssyrk-cast).
//
// Splitting moves the dominant ~3/4 of the work into native FP16 GEMM after
// the first level of recursion, so only the small leaf triangles ever
// promote to float.
// -----------------------------------------------------------------------------
inline void cblas_syrk(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       da_int n, da_int k, _Float16 alpha, _Float16 const *A, da_int lda,
                       _Float16 beta, _Float16 *C, da_int ldc) {
    if (n <= da_fp16::SYRK_FP16_LEAF) {
        da_fp16::cblas_syrk_fp16_leaf(layout, uplo, trans, n, k,
                                      static_cast<float>(alpha), A, lda,
                                      static_cast<float>(beta), C, ldc);
        return;
    }

    // Split n = n1 + n2; n1 = floor(n/2), n2 = ceil(n/2). When n is odd
    // n2 == n1 + 1 and the lower/right blocks are slightly larger.
    const da_int n1 = n / 2;
    const da_int n2 = n - n1;

    // Pointer offsets into A and C for the second half.
    //   trans == NoTrans : A is n-by-k, split row-wise   -> A2 offset by n1 rows
    //   trans == Trans   : A is k-by-n, split column-wise -> A2 offset by n1 cols
    //   C is always n-by-n, split into a 2x2 block grid.
    // Column-major: row offset is +n1, column offset is +n1*ld.
    // Row-major   : row offset is +n1*ld, column offset is +n1.
    _Float16 const *A2;
    _Float16 *C11 = C;
    _Float16 *C22;
    _Float16 *C_off; // C21 (Lower) or C12 (Upper)
    if (layout == CblasColMajor) {
        A2 = (trans == CblasNoTrans) ? (A + n1) : (A + static_cast<size_t>(n1) * lda);
        C22 = C + n1 + static_cast<size_t>(n1) * ldc;
        C_off = (uplo == CblasLower) ? (C + n1) : (C + static_cast<size_t>(n1) * ldc);
    } else {
        A2 = (trans == CblasNoTrans) ? (A + static_cast<size_t>(n1) * lda) : (A + n1);
        C22 = C + static_cast<size_t>(n1) * ldc + n1;
        C_off = (uplo == CblasLower) ? (C + static_cast<size_t>(n1) * ldc) : (C + n1);
    }

    // Diagonal triangles (recurse).
    cblas_syrk(layout, uplo, trans, n1, k, alpha, A, lda, beta, C11, ldc);
    cblas_syrk(layout, uplo, trans, n2, k, alpha, A2, lda, beta, C22, ldc);

    // Off-diagonal rectangular block via native FP16 GEMM.
    //   Lower: C21(n2 x n1) = alpha * op(A2) * op(A1)^t + beta * C21
    //   Upper: C12(n1 x n2) = alpha * op(A1) * op(A2)^t + beta * C12
    // where op is determined by `trans`. cblas_gemm's transA/transB form
    // the same op(.) and op(.)^t pair regardless of `trans`.
    const CBLAS_TRANSPOSE transA_gemm = trans;
    const CBLAS_TRANSPOSE transB_gemm =
        (trans == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    if (uplo == CblasLower) {
        cblas_gemm(layout, transA_gemm, transB_gemm, n2, n1, k, alpha, A2, lda, A, lda,
                   beta, C_off, ldc);
    } else {
        cblas_gemm(layout, transA_gemm, transB_gemm, n1, n2, k, alpha, A, lda, A2, lda,
                   beta, C_off, ldc);
    }
}

// -----------------------------------------------------------------------------
// cblas_axpy for _Float16 (native half-precision arithmetic)
// -----------------------------------------------------------------------------
inline void cblas_axpy(da_int n, _Float16 alpha, _Float16 const *x, da_int incx,
                       _Float16 *y, da_int incy) {
    for (da_int i = 0; i < n; ++i) {
        y[i * incy] += alpha * x[i * incx];
    }
}

// -----------------------------------------------------------------------------
// cblas_copy for _Float16
// -----------------------------------------------------------------------------
inline void cblas_copy(da_int n, _Float16 const *x, da_int incx, _Float16 *y,
                       da_int incy) {
    for (da_int i = 0; i < n; ++i)
        y[i * incy] = x[i * incx];
}

// -----------------------------------------------------------------------------
// omatcopy for _Float16: out-of-place matrix copy (column-major, MKL/AOCL
// convention). trans is 'N' for no-transpose, 'T' for transpose. Source is
// m-by-n with leading dimension lda_in; destination is m-by-n (no trans) or
// n-by-m (trans) with leading dimension ldb_out. Native _Float16 arithmetic.
// -----------------------------------------------------------------------------
inline void omatcopy(char trans, da_int m, da_int n, _Float16 alpha, _Float16 const *A,
                     da_int lda_in, _Float16 *B, da_int ldb_out) {
    const bool transpose = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    if (!transpose) {
        for (da_int j = 0; j < n; ++j) {
            for (da_int i = 0; i < m; ++i) {
                B[i + j * ldb_out] = alpha * A[i + j * lda_in];
            }
        }
    } else {
        for (da_int j = 0; j < n; ++j) {
            for (da_int i = 0; i < m; ++i) {
                B[j + i * ldb_out] = alpha * A[i + j * lda_in];
            }
        }
    }
}

// cblas_asum for _Float16: native half-precision accumulation of |x_i|.
inline _Float16 cblas_asum(da_int n, _Float16 const *x, da_int incx) {
    _Float16 s = (_Float16)0;
    for (da_int i = 0; i < n; ++i) {
        s += ARCH::da_std::abs(x[i * incx]);
    }
    return s;
}

// cblas_iamax for _Float16: index of element with maximum |x_i|, half-precision
// comparison.
inline da_int cblas_iamax(da_int n, _Float16 const *x, da_int incx) {
    if (n <= 0)
        return 0;
    da_int imax = 0;
    _Float16 vmax = ARCH::da_std::abs(x[0]);
    for (da_int i = 1; i < n; ++i) {
        _Float16 v = ARCH::da_std::abs(x[i * incx]);
        if (v > vmax) {
            vmax = v;
            imax = i;
        }
    }
    return imax;
}

} // namespace da_blas

namespace da {

// -----------------------------------------------------------------------------
// lange for _Float16
// -----------------------------------------------------------------------------
inline _Float16 lange(char const *norm, da_int const *m, da_int const *n,
                      _Float16 const *A, da_int const *lda, _Float16 *work) {
    const _Float16 zero = (_Float16)0.0;
    const _Float16 one = (_Float16)1.0;
    _Float16 value = zero;

    if (*m == 0 || *n == 0) {
        return zero;
    }

    char c = *norm;
    if (c >= 'a' && c <= 'z')
        c -= 32; // toupper

    if (c == 'M') {
        // max(abs(A(i,j)))
        for (da_int j = 0; j < *n; j++) {
            for (da_int i = 0; i < *m; i++) {
                _Float16 temp = A[i + j * (*lda)];
                if (temp < zero)
                    temp = -temp;
                if (value < temp || temp != temp)
                    value = temp;
            }
        }
    } else if (c == 'O' || *norm == '1') {
        // norm1(A)
        for (da_int j = 0; j < *n; j++) {
            _Float16 sum = zero;
            for (da_int i = 0; i < *m; i++) {
                _Float16 temp = A[i + j * (*lda)];
                if (temp < zero)
                    temp = -temp;
                sum += temp;
            }
            if (value < sum || sum != sum)
                value = sum;
        }
    } else if (c == 'I') {
        // normI(A)
        for (da_int i = 0; i < *m; i++)
            work[i] = zero;
        for (da_int j = 0; j < *n; j++) {
            for (da_int i = 0; i < *m; i++) {
                _Float16 temp = A[i + j * (*lda)];
                if (temp < zero)
                    temp = -temp;
                work[i] += temp;
            }
        }
        for (da_int i = 0; i < *m; i++) {
            _Float16 temp = work[i];
            if (value < temp || temp != temp)
                value = temp;
        }
    } else if (c == 'F' || c == 'E') {
        // normF(A) using scaled sum of squares
        _Float16 scale = zero;
        _Float16 sum = one;
        for (da_int j = 0; j < *n; j++) {
            for (da_int i = 0; i < *m; i++) {
                _Float16 temp = A[i + j * (*lda)];
                if (temp < zero)
                    temp = -temp;
                if (scale < temp) {
                    _Float16 ratio = scale / temp;
                    sum = one + sum * ratio * ratio;
                    scale = temp;
                } else if (temp > zero) {
                    _Float16 ratio = temp / scale;
                    sum += ratio * ratio;
                }
            }
        }
        value = scale * ARCH::da_std::sqrt(sum);
    }

    return value;
}

} // namespace da

// Restore the including translation unit's diagnostic state; the suppressions
// at the top of this header apply only to code within it.
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif // FP16_HELPERS_HPP
