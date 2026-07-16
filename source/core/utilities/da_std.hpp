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

#ifndef DA_STD_HPP
#define DA_STD_HPP

#include "aoclda_types.h"
#include "boost/random/uniform_int_distribution.hpp"
#include "macros.h"
#include <cmath>
#include <immintrin.h>
#include <limits>
#include <ostream>
#include <random>
#include <type_traits>

// Provide an ostream insertion for _Float16 because libstdc++ does not supply
// one and an unqualified `std::cout << val` for _Float16 would otherwise be
// ambiguous between the integer/floating overloads. Promote to float for
// formatting; this is the canonical representation for human-readable output.
// Use a separate (non-undef'd) guard so it is defined exactly once per TU even
// though DA_STD_HPP itself is undef'd between ARCH includes.
#ifndef DA_STD_FLOAT16_OSTREAM_GUARD
#define DA_STD_FLOAT16_OSTREAM_GUARD
inline std::ostream &operator<<(std::ostream &os, _Float16 v) {
    return os << static_cast<float>(v);
}
#endif

namespace ARCH {
namespace da_std {

/* These functions are AOCL-DA specific implementations of common STL functions. They exist because
   in certain STL implementations they might be compiled with AVX512 instructions, which can cause
   illegal instruction exceptions on AVX2 or older CPUs. Implementing our own versions within
   namespaces we control enables us to use the functionality safely.
*/

// std::fill equivalent
template <class ForwardIt, class T>
void fill(ForwardIt first, ForwardIt last, const T &value) {
    if (first >= last)
        return;
    // Cast through the iterator's value type so that a narrowing conversion
    // (e.g. fill(_Float16*, _Float16*, 0.0)) does not trip GCC's diagnostic
    // about implicit conversions to _Float16 from a higher-rank type.
    using V = typename std::iterator_traits<ForwardIt>::value_type;
    const V v = static_cast<V>(value);
    for (; first != last; ++first) {
        *first = v;
    }
}

// std::copy equivalent
template <class InputIt, class OutputIt>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
    for (; first != last; ++first, ++d_first) {
        *d_first = *first;
    }
    return d_first;
}

// std::iota equivalent
template <class ForwardIt, class T> void iota(ForwardIt first, ForwardIt last, T value) {
    for (; first != last; ++first, ++value) {
        *first = value;
    }
}

template <class RandomAccessIterator, class URNG>
void shuffle(RandomAccessIterator first, RandomAccessIterator last, URNG &&g) {
    for (auto i = (last - first) - 1; i > 0; --i) {
        boost::random::uniform_int_distribution<decltype(i)> d(0, i);
        std::swap(first[i], first[d(g)]);
    }
}

// std::sample equivalent using selection sampling (Algorithm S)
template <class PopulationIterator, class SampleIterator, class Distance, class URNG>
SampleIterator sample(PopulationIterator first, PopulationIterator last,
                      SampleIterator out, Distance n, URNG &&g) {
    Distance pop_size = std::distance(first, last);
    n = std::min(n, pop_size);

    for (; n > 0; ++first, --pop_size) {
        std::uniform_int_distribution<Distance> d(0, pop_size - 1);
        if (d(g) < n) {
            *out++ = *first;
            --n;
        }
    }

    return out;
}

/* The following functions provide additional instantiations of varios C++ STL functions for _Float16,
   which is not fully supported in all STL implementations (e.g. std::isfinite and std::numeric_limits
   specializations are missing in libstdc++). These are used in various places in the codebase where
   we want to support half precision.
*/

// Type trait that behaves like std::is_floating_point but also reports true
// for _Float16, which the standard trait does not recognise as a floating
// point type. Use this anywhere the codebase needs to accept the half
// precision type alongside float/double.
template <typename T>
struct is_floating_point
    : std::integral_constant<bool, std::is_floating_point<T>::value ||
                                       std::is_same<T, _Float16>::value> {};

template <typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

// Safe numeric_limits: inherits from std::numeric_limits for all types,
// with an explicit specialization for _Float16 (not specialized in libstdc++).
template <typename T> struct numeric_limits : std::numeric_limits<T> {};

template <> struct numeric_limits<_Float16> {
    static constexpr bool is_specialized = true;
    static constexpr bool has_quiet_NaN = true;
    // Construct from the float builtins (not __builtin_inff16/__builtin_nanf16)
    // so this compiles on toolchains lacking native _Float16 support
    static constexpr _Float16 infinity() noexcept { return (_Float16)__builtin_inff(); }
    static constexpr _Float16 max() noexcept { return (_Float16)65504.0f; }
    static constexpr _Float16 lowest() noexcept { return (_Float16)-65504.0f; }
    static constexpr _Float16 min() noexcept { return (_Float16)6.103515625e-05f; }
    static constexpr _Float16 epsilon() noexcept { return (_Float16)9.765625e-04f; }
    static constexpr _Float16 quiet_NaN() noexcept {
        return (_Float16)__builtin_nanf("");
    }
};

// std::isfinite equivalent that handles _Float16
// std::isfinite is not guaranteed to work with _Float16 in libstdc++
template <typename T> bool isfinite(T val) { return std::isfinite(val); }

inline bool isfinite(_Float16 val) {
    // Cast to float where std::isfinite is well-defined
    return std::isfinite(static_cast<float>(val));
}

// std::isinf equivalent that handles _Float16
template <typename T> bool isinf(T val) { return std::isinf(val); }

inline bool isinf(_Float16 val) { return std::isinf(static_cast<float>(val)); }

// std::isnan does not have a _Float16 overload in libstdc++. Provide one that
// promotes to float so we can call da_std::isnan generically from templates.
__attribute__((always_inline)) inline bool isnan(_Float16 x) {
    return std::isnan(static_cast<float>(x));
}

__attribute__((always_inline)) inline bool isnan(float x) { return std::isnan(x); }

__attribute__((always_inline)) inline bool isnan(double x) { return std::isnan(x); }

// =============================================================================
// Standard functions, always-inlined wrappers
// For _Float16: promote to float, compute, cast back.
// For float/double: pass-through to std::sqrt / std::acos etc
// =============================================================================
__attribute__((always_inline)) inline _Float16 sqrt(_Float16 x) {
    return static_cast<_Float16>(std::sqrt(static_cast<float>(x)));
}

__attribute__((always_inline)) inline float sqrt(float x) { return std::sqrt(x); }

__attribute__((always_inline)) inline double sqrt(double x) { return std::sqrt(x); }

__attribute__((always_inline)) inline _Float16 acos(_Float16 x) {
    return static_cast<_Float16>(std::acos(static_cast<float>(x)));
}

__attribute__((always_inline)) inline float acos(float x) { return std::acos(x); }

__attribute__((always_inline)) inline double acos(double x) { return std::acos(x); }

__attribute__((always_inline)) inline _Float16 exp(_Float16 x) {
    return static_cast<_Float16>(std::exp(static_cast<float>(x)));
}

__attribute__((always_inline)) inline float exp(float x) { return std::exp(x); }

__attribute__((always_inline)) inline double exp(double x) { return std::exp(x); }

__attribute__((always_inline)) inline _Float16 log(_Float16 x) {
    return static_cast<_Float16>(std::log(static_cast<float>(x)));
}

__attribute__((always_inline)) inline float log(float x) { return std::log(x); }

__attribute__((always_inline)) inline double log(double x) { return std::log(x); }

__attribute__((always_inline)) inline _Float16 abs(_Float16 x) {
    return static_cast<_Float16>(std::abs(static_cast<float>(x)));
}

__attribute__((always_inline)) inline float abs(float x) { return std::abs(x); }

__attribute__((always_inline)) inline double abs(double x) { return std::abs(x); }

// Generic max / min wrappers. These behave like std::max / std::min but live
// in the da_std namespace so that ADL is unambiguous for types like _Float16
// for which the standard does not (always) provide overloads that play nicely
// with templated callers. _Float16 supports operator< so std::max/min already
// work for it; these wrappers simply give a single consistent spelling.
template <typename T>
__attribute__((always_inline)) inline const T &max(const T &a, const T &b) {
    return (a < b) ? b : a;
}
template <typename T>
__attribute__((always_inline)) inline const T &min(const T &a, const T &b) {
    return (b < a) ? b : a;
}

} // namespace da_std
} // namespace ARCH

#endif // DA_STD_HPP
