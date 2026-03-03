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
#include <random>

/* These functions are AOCL-DA specific implementations of common STL functions. They exist because
   in certain STL implementations they might be compiled with AVX512 intstructions, which can cause
   illegal instruction exceptions on AVX2 or older CPUs. Implementing our own versions within
   namespaces we control enables us to use the functionality safely.
*/

namespace ARCH {
namespace da_std {

// std::fill equivalent
template <class ForwardIt, class T>
void fill(ForwardIt first, ForwardIt last, const T &value) {
    for (; first != last; ++first) {
        *first = value;
    }
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

} // namespace da_std
} // namespace ARCH

#endif // DA_STD_HPP
