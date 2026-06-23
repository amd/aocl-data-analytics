/*
 * Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef __KERNEL_UTILS__
#define __KERNEL_UTILS__

#include "context.hpp"
#include "da_utils.hpp"
#include <array>
#include <immintrin.h>
#include <stdint.h>
#include <string>
#include <type_traits>

// ISA type definitions
// Oracle aux tools expect to use these has index entries: scalar<=idx-1<=count
enum vectorization_type : da_int {
    undefined = -1,
    scalar = 1,
    avx = 2,
    avx2 = 3,
    avx512 = 4,
    count = 4
};

// Kernel Generic Dispatcher Framework
// The following objects work together to select and dispatch SIMD kernels at
// runtime based on architecture, data type, and a tuning parameter:
//
//  1. vectorization_type      Enum of supported ISAs (scalar, avx, avx2, avx512).
//                             Used as index and return value throughout.
//
//  2. KernelSelection         A {threshold, kernel} pair. With the default auxiliary
//                             oracle: "if param <= threshold,
//                             use this kernel (ISA)". Rows are sorted ascending;
//                             the last entry MUST use DA_INT_MAX as a catch-all.
//
//  3. tblRow / TBL            A tblRow<ROW> binds {arch, dtype} to a small array
//                             of KernelSelection breakpoints. TBL<ROW>::type is a
//                             flat array of 14 tblRows (2 dtypes × 7 archs) that
//                             callers define as a constexpr table and ALL must be
//                             defined.
//                             Note that the array of KernelSelection breakpoints
//                             is of size 4 but only one can be defined, no need
//                             to fill the unused entries.
//
//  4. Oracle                  The lookup engine. Given a table, dtype, and a tuning
//                             parameter, it walks the matching row and returns the
//                             best vectorization_type. Optionally accepts a custom
//                             predicate or a hidden-settings override key for testing.
//
//  5. kernel_implementations  Parallel structure: maps each vectorization_type to an
//                             actual function pointer (float and double variants).
//                             Call .get<T>(isa) with the vectorization_type returned
//                             by Oracle to retrieve the callable kernel.
//
//  Typical call sequence:
//    vectorization_type optimal_isa = Oracle(my_table, tid<T>(), my_param, "my.isa");
//    auto kernel            = my_implementations.get<T>(optimal_isa);
//
// For usage example, see kmeans elkan kernel assignment.

// Ringfence macro for AVX512 entries in the Kernel Implementations Table (KIT)
#ifdef __AVX512F__
#define ORL_AVX512F(...) __VA_ARGS__
#else
#define ORL_AVX512F(...) nullptr
#endif

// Kernel Implementations Table (KIT), one for each vectorization type (isa).
// Maps ISA -> kernel function pointer
template <typename KS, typename KD> struct kernel_implementations {
    std::array<KS, vectorization_type::count> kernels_s;
    std::array<KD, vectorization_type::count> kernels_d;
    template <typename T> constexpr auto get(vectorization_type v) const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
        if constexpr (std::is_same_v<T, float>) {
            if (v >= vectorization_type::scalar && v <= vectorization_type::count)
                return kernels_s[static_cast<size_t>(v) -
                                 (size_t)1]; // -1 to adjust for scalar=1
            else
                return KS{}; // empty std::function
        } else {
            if (v >= vectorization_type::scalar && v <= vectorization_type::count)
                return kernels_d[static_cast<size_t>(v) - 1]; // -1 to adjust for scalar=1
            else
                return KD{}; // empty std::function
        }
    }
};
// Example
// using T = float;
// using K = std::function<void(da_int, T *, da_int, T *, T *, da_int *, da_int)>;
// // Implementation tables must have all vectorization_type::count entries
// constexpr kernel_implementations<K> kernel_implementations_api = {{
//             /* scalar    */  k_iteration_scalar<double>,
//             /* avx (sse) */  k_iteration_kt<bsz::b128, double>,
//             /* avx2      */  k_iteration_kt<bsz::b256, double>,
// ORL_AVX512F(/* avx512    */  k_iteration_kt<bsz::b512, double>)
// }}; // repeat for type float
// Note that AVX512 implementations MUST be ring-fenced using ORL macro that
// takes care of the cases where these are not built or available.
//
// To query the table use the .get<typename>() operator with the vectorization
// type, for example:
// kernel_implementations_api<double>[vectorization_type::avx2] will return the
// kernel function pointer `k_iteration_kt<bsz::b256, double>`

template <typename ROW> struct tblRow {
    dispatch_architecture arch;
    da_type type;
    std::array<ROW, 4> table;
};

// Default kernel tuning table of (thresholds, kernel) (ISA) "breakpoints"
// The tuning table is made of a sequence (array) of at least one breakpoint
// of the form {thr1, isa1}, {thr2, isa2}, ... with 0<thr1 < thr2 < ...
// This is used along with an auxiliary kernel where it selects the "optimal"
// kernel based on where a "tuning parameter" falls in which breakpoint.
struct KernelSelection {
    da_int threshold{-1};
    vectorization_type kernel{vectorization_type::undefined};
    KernelSelection() = default; // allow default initialization
    constexpr KernelSelection(vectorization_type k) : threshold(DA_INT_MAX), kernel(k){};
    constexpr KernelSelection(da_int t, vectorization_type k) : threshold(t), kernel(k){};
};

// Default auxiliary oracle function returns
// true for the first match where param <= threshold with threshold > 0.
template <typename T> bool oracle_default(T param, T threshold) {
    return (threshold > T(0)) && (param <= threshold);
};
// Nondefault auxiliary oracle (strict inequality)
template <typename T> bool oracle_lt(T param, T threshold) {
    return (threshold > T(0)) && (param < threshold);
};

// Helper for building tables. Size of array must match with
// double the context::dispatch_architecture size!
template <typename ROW> struct TBL {
    static constexpr size_t nrows{/*dtypes*/ 2 * 7 /*archs*/};
    using type = std::array<tblRow<ROW>, nrows>;
};

// Oracle - kernel ISA selector
//
// Consults a pre-built lookup table (tbl) to return the optimal vectorization_type
// (scalar / avx / avx2 / avx512) for the current hardware and data type.
//
// Template parameters:
//   ROW  - row entry type stored in the table (e.g. KernelSelection {threshold, kernel})
//   P    - type of the tuning parameter (e.g. da_int)
//   O    - auxiliary oracle (predicate): bool oracle(P param, P threshold)
//            Returns true when 'param' falls into the bucket described by 'threshold'.
//            Default predicate (oracle_default): param <= threshold.
//
// Arguments:
//   tbl      - flat array of tblRow<ROW>; size = 2 dtypes × 7 architectures = 14 rows.
//              Each row carries {arch, dtype, table[]}, where table[] is a sorted list of
//              {threshold, kernel} breakpoints (ascending, last entry has DA_INT_MAX).
//   dtype    - da_type of the operand (float / double), used to select the right row.
//   param    - the parametrized tuning parameter used in the thresholding auxiliary oracle.
//   oracle   - predicate that decides membership in a bucket (default is oracle_default).
//   override - optional hidden-settings key (e.g. "kmeans.isa"); if present and set,
//              bypasses the table and forces the named ISA.
//   The Oracle downgrades avx512 ISA if hardware support is absent or library was built without it.
//
// Returns the "optimal" vectorization_type (ISA); falls back to scalar if no row/bucket matches.
#define FORCE_INLINE __attribute__((always_inline)) inline
template <typename ROW, typename P, typename O>
FORCE_INLINE vectorization_type
Oracle(const std::array<tblRow<ROW>, TBL<ROW>::nrows> &tbl, da_type dtype, P param,
       O oracle, const char *override = nullptr) {
    using namespace std::string_literals;
    using v = vectorization_type;
    auto *ctx = context::get_context();

    // Check to see if there is an override
    if (override && ctx->hidden_settings.find(override) != ctx->hidden_settings.end()) {
        std::string kernel = ctx->hidden_settings[override];
        if (kernel == "avx"s) {
            return v::avx;
        } else if (kernel == "avx2"s) {
            return v::avx2;
        } else if (kernel == "avx512"s) {
#ifdef __AVX512F__
            return (ctx->has_avx512) ? v::avx512 : v::avx2;
#else
            // This build does not have AVX512 kernels
            return v::avx2;
#endif
        }
        return v::scalar;
    }

    v isa{undefined};

    // Get optimal vector length
    const dispatch_architecture arch{ctx->arch};
    for (const auto &sel : tbl) {
        if (sel.arch != arch || sel.type != dtype)
            continue;
        for (const auto &t : sel.table) {
            if (oracle(param, t.threshold)) {
                isa = t.kernel;
                // Downgrade if ISA not supported
                if (isa == v::avx512) {
#ifdef __AVX512F__
                    if (!ctx->has_avx512)
                        isa = v::avx2;
#else
                    // This build does not have AVX512 kernels
                    isa = v::avx2;
#endif
                }
                // Assume minimum ISA is AVX2
                return isa;
            }
        }
    }
    return v::scalar; // Default if no match
}

// Convenience overload with default aux oracle
template <typename ROW, typename P>
FORCE_INLINE vectorization_type
Oracle(const std::array<tblRow<ROW>, TBL<ROW>::nrows> &tbl, da_type dtype, P param,
       const char *override = nullptr) {
    return Oracle(tbl, dtype, param, oracle_default<P>, override);
}

// Convenience overload for simple ISA selection
// Returns the highest vectorization ISA
FORCE_INLINE vectorization_type Oracle(const char *override = nullptr) {
    using namespace std::string_literals;
    using v = vectorization_type;
    auto *ctx = context::get_context();

    // Check to see if there is an override
    if (override && ctx->hidden_settings.find(override) != ctx->hidden_settings.end()) {
        std::string kernel = ctx->hidden_settings[override];
        if (kernel == "avx"s) {
            return v::avx;
        } else if (kernel == "avx2"s) {
            return v::avx2;
        } else if (kernel == "avx512"s) {
#ifdef __AVX512F__
            return (ctx->has_avx512) ? v::avx512 : v::avx2;
#else
            // This build does not have AVX512 kernels
            return v::avx2;
#endif
        }
        return v::scalar;
    }

#ifdef __AVX512F__
    return (ctx->has_avx512) ? v::avx512 : v::avx2;
#else
    // This build does not have AVX512 kernels
    return v::avx2;
#endif
}

// --------------------------- Generic Dispatchers -----------------------------
namespace da_dispatch::tuning {
/// @brief A template skeleton to define a "parameter tuning table" where each
/// row has "toggle" functionality that is checked against a given condition.
///
///
/// This structure combines a toggle flag with an array of rows, providing a flexible
/// way to manage table data with associated metadata.
///
/// @tparam TGGL The type of the toggle flag (e.g., bool, enum, or other indicator type)
///         can be a scalar element or an array, but must support operator!= for comparison.
///         The toggle elements must *all* match to toggle the row ON, otherwise it stays OFF.
///         Only rows that are ON are considered in the "oracle predicate" search for optimal
///         parameter values.
/// @tparam ROW The type (generic) of individual row elements stored in the table array
/// @tparam M The compile-time size of the table array (number of rows).
/// @param optv_t Type alias for the optional value type, derived from ROW::optv_t
template <typename TGGL, typename ROW, size_t M> struct tblRow {
    TGGL toggle;
    std::array<ROW, M> table;
    using optv_t = typename ROW::optv_t;
};
// convenience alias
template <typename T, size_t N> struct TBL {
    using type = std::array<T, N>;
};

/// @brief Oracle function that searches a table of parameters to find the best match based on a custom predicate.
///
/// This function iterates through a table of rows, filtering by a toggle value, and applies a user-defined
/// "oracle predicate" to find a matching entry. When a match is found, the "parameter bucket-list" is searched
/// for the optimal value.
/// If no match is found, a default value is used (passed as template parameter D).
///
/// @tparam D The default value to use if no matching entry is found in the table, needs to be explicitly specified
/// @tparam T The type of the table container (should have value_type with optv_t member). Deduced.
/// @tparam TGGL The type of the toggle flag used to filter table rows. Deduced.
/// @tparam P The type of the parameter to be evaluated against thresholds. Deduced.
/// @tparam O The type of the oracle predicate function. Deduced.
///
/// @param tbl Reference to the table container to search through.
/// @param tggl The toggle value to filter rows (only rows matching this value are considered).
/// @param param The parameter value to compare against each entry's threshold.
/// @param optv Reference to the output variable that will store the optimal value.
/// @param oracle A callable "oracle predicate" that takes (param, threshold) and returns true if
///        param satisfies the threshold condition.
///
/// @note The template parameter D must have the same type as T::value_type::optv_t
/// @note TGGL type must support operator!= comparison
/// @note The oracle predicate should return bool and accept two parameters: the param and the threshold
template <auto D, typename T, typename TGGL, typename P, typename O>
FORCE_INLINE void Oracle(const T &tbl, TGGL tggl, P param,
                         typename T::value_type::optv_t &optv, O oracle) {
    static_assert(std::is_same_v<typename T::value_type::optv_t, decltype(D)>,
                  "Default value must match with oracle's optv type");

    for (const auto &tr : tbl) {
        if (tr.toggle != tggl) // != must exist for typename TGGL (an all constituents)
            continue;
        for (const auto &t : tr.table) {
            if (oracle(param, t.threshold)) {
                optv = t.optv;
                return;
            }
        }
    }
    optv = D; // Use Default
}
} // namespace da_dispatch::tuning

template <class T> FORCE_INLINE da_int get_padding(vectorization_type isa) {
    da_int value;

    switch (isa) {
    case vectorization_type::avx:
        value = std::is_same<T, float>::value ? 4 : 2;
        break;
    case vectorization_type::avx2:
        value = std::is_same<T, float>::value ? 8 : 4;
        break;
    case vectorization_type::avx512:
        value = std::is_same<T, float>::value ? 16 : 8;
        break;
    default:
        value = 0;
        break;
    }

    return value;
}

#undef FORCE_INLINE
// -----------------------------------------------------------------------------

/*****************************
  * Internal types
  *****************************/
typedef union {
    float f;
    int32_t i;
    uint32_t u;
} flt32_t;

typedef union {
    double d;
    int64_t i;
    uint64_t u;
} flt64_t;

/*****************************
  * Internal vector types
  *****************************/

#ifdef __AVX__
/*
  * (u)int32 - 4 elements - 128 bits
  */
typedef union {
    int32_t i[4] __attribute__((aligned(16)));
    uint32_t u[4] __attribute__((aligned(16)));
    __m128i v;
} v4i32_t;

/*
  * (u)int64 - 2 elements - 128 bits
  */
typedef union {
    int64_t i[2] __attribute__((aligned(16)));
    uint64_t u[2] __attribute__((aligned(16)));
    __m128i v;
} v2i64_t;

/*
  * float32 - 4 elements - 128 bits
  */
typedef union {
    uint32_t u[4] __attribute__((aligned(16)));
    int32_t i[4] __attribute__((aligned(16)));
    float f[4] __attribute__((aligned(16)));
    __m128 v;
} v4sf_t;

/*
  * float64 - 2 element - 128 bits
  */
typedef union {
    uint64_t u[2] __attribute__((aligned(16)));
    int64_t i[2] __attribute__((aligned(16)));
    double d[2] __attribute__((aligned(16)));
    __m128d v;
} v2df_t;

#endif

#ifdef __AVX2__
/*
  * (u)int32 - 8 elements - 256 bits
  */
typedef union {
    int32_t i[8] __attribute__((aligned(32)));
    uint32_t u[8] __attribute__((aligned(32)));
    __m256i v;
} v8i32_t;

/*
  * (u)int64 - 4 elements - 256 bits
  */
typedef union {
    int64_t i[4] __attribute__((aligned(32)));
    uint64_t u[4] __attribute__((aligned(32)));
    __m256i v;
} v4i64_t;

/*
  * float32 - 8 elements - 256 bits
  */
typedef union {
    uint32_t u[8] __attribute__((aligned(32)));
    int32_t i[8] __attribute__((aligned(32)));
    float f[8] __attribute__((aligned(32)));
    __m256 v;
} v8sf_t;

/*
  * float64 - 4 elements - 256 bits
  */
typedef union {
    uint64_t u[4] __attribute__((aligned(32)));
    int64_t i[4] __attribute__((aligned(32)));
    double d[4] __attribute__((aligned(32)));
    __m256d v;
} v4df_t;

#endif

#ifdef __AVX512F__

/*
  * (u)int32 - 16 elements - 512 bits
  */
typedef union {
    int32_t i[16] __attribute__((aligned(64)));
    uint32_t u[16] __attribute__((aligned(64)));
    __m512i v;
} v16i32_t;

/*
  * (u)int64 - 8 elements - 512 bits
  */
typedef union {
    int64_t i[8] __attribute__((aligned(64)));
    uint64_t u[8] __attribute__((aligned(64)));
    __m512i v;
} v8i64_t;

/*
  * float64 - 16 elements - 512 bits
  */
typedef union {
    uint32_t u[16] __attribute__((aligned(64)));
    int32_t i[16] __attribute__((aligned(64)));
    float f[16] __attribute__((aligned(64)));
    __m512 v;
} v16sf_t;

/*
  * float64 - 8 elements - 512 bits
  */
typedef union {
    uint64_t u[8] __attribute__((aligned(64)));
    int64_t i[8] __attribute__((aligned(64)));
    double d[8] __attribute__((aligned(64)));
    __m512d v;
} v8df_t;

#endif

#endif
