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

#include "aoclda_types.h"
#include "context.hpp"
#include "da_kernel_utils.hpp"
#include <array>
#include <type_traits>

/*
  This file contains a series of overloads for the select_simd_size function, which is used to determine
  the SIMD size for different architectures and data types. The function is specialized for
  various architectures, including Zen2, Zen3, Zen4, and Zen5. Each specialization takes the working set size,
  the architecture and the data type, and uses those to choose the optimal kernel type (AVX, AVX2, AVX512)
  and the appropriate padding required for memory allocation to ensure the simd length fits neatly in the kernels.
  */

// Define a struct to hold threshold and corresponding kernel type
struct KernelSelection {
    da_int threshold;
    vectorization_type kernel;
};

// Default lookup tables
namespace da_kernel_functions {

// Defaults for AVX2 systems

constexpr std::array<KernelSelection, 3> float_ = {{
    {4, scalar},       // Up to 4 -> scalar
    {8, avx},          // Up to 8 -> avx
    {DA_INT_MAX, avx2} // >= 8 -> avx2
}};

constexpr std::array<KernelSelection, 3> double_ = {{
    {2, scalar},       // Up to 2 -> scalar
    {4, avx},          // Up to 4 -> avx
    {DA_INT_MAX, avx2} // >= 4 -> avx2
}};

// Defaults for AVX512 systems

constexpr std::array<KernelSelection, 4> float_avx512_ = {{
    {4, scalar},         // Up to 4 -> scalar
    {8, avx},            // Up to 8 -> avx
    {16, avx2},          // Up to 16 -> avx2
    {DA_INT_MAX, avx512} // >= 16 -> avx512
}};

constexpr std::array<KernelSelection, 4> double_avx512_ = {{
    {2, scalar},         // Up to 2 -> scalar
    {4, avx},            // Up to 4 -> avx
    {8, avx2},           // Up to 8 -> avx2
    {DA_INT_MAX, avx512} // >= 8 -> avx512
}};

} // namespace da_kernel_functions

// Further specific lookup tables can be added here for other architectures

//Select kernel type based on a lookup table and a parameter
template <std::size_t N>
vectorization_type lookup_kernel_kf(const std::array<KernelSelection, N> &selections,
                                    da_int param) {
    using namespace std::string_literals;
    // Check to see if there is an override
    const char isa[]{"kf.isa"};
    if (context::get_context()->hidden_settings.find(isa) !=
        context::get_context()->hidden_settings.end()) {
        std::string kernel = context::get_context()->hidden_settings[isa];
        if (kernel == "avx"s) {
            return avx;
        } else if (kernel == "avx2"s) {
            return avx2;
        } else if (kernel == "avx512"s) {
            return (context::get_context()->has_avx512) ? avx512 : avx2;
        }
        return scalar;
    }

    // Get best kernel
    for (const auto &selection : selections) {
        if (param < selection.threshold) {
            return selection.kernel;
        }
    }
    // Default to scalar if no match found - this should not happen
    return scalar;
}

template <class T>
void select_simd_size_default(da_int size, vectorization_type &kernel_type) {
    kernel_type = std::is_same<T, float>::value
                      ? lookup_kernel_kf(da_kernel_functions::float_, size)
                      : lookup_kernel_kf(da_kernel_functions::double_, size);
}

template <class T>
void select_simd_size_avx512(da_int size, vectorization_type &kernel_type) {
    kernel_type = std::is_same<T, float>::value
                      ? lookup_kernel_kf(da_kernel_functions::float_avx512_, size)
                      : lookup_kernel_kf(da_kernel_functions::double_avx512_, size);
}

// Specializations for different architectures

namespace da_dynamic_dispatch_generic {
namespace da_kernel_functions {
template <class T> void select_simd_size(da_int size, vectorization_type &kernel_type) {

    select_simd_size_default<T>(size, kernel_type);
}

// Explicit instantiations
template void select_simd_size<float>(da_int size, vectorization_type &kernel_type);
template void select_simd_size<double>(da_int size, vectorization_type &kernel_type);

} // namespace da_kernel_functions
} // namespace da_dynamic_dispatch_generic

namespace da_dynamic_dispatch_generic_avx512 {
namespace da_kernel_functions {
template <class T> void select_simd_size(da_int size, vectorization_type &kernel_type) {

    select_simd_size_avx512<T>(size, kernel_type);
}

// Explicit instantiations
template void select_simd_size<float>(da_int size, vectorization_type &kernel_type);
template void select_simd_size<double>(da_int size, vectorization_type &kernel_type);

} // namespace da_kernel_functions
} // namespace da_dynamic_dispatch_generic_avx512

namespace da_dynamic_dispatch_zen2 {
namespace da_kernel_functions {
template <class T> void select_simd_size(da_int size, vectorization_type &kernel_type) {

    select_simd_size_default<T>(size, kernel_type);
}

// Explicit instantiations
template void select_simd_size<float>(da_int size, vectorization_type &kernel_type);
template void select_simd_size<double>(da_int size, vectorization_type &kernel_type);
} // namespace da_kernel_functions
} // namespace da_dynamic_dispatch_zen2

namespace da_dynamic_dispatch_zen3 {
namespace da_kernel_functions {
template <class T> void select_simd_size(da_int size, vectorization_type &kernel_type) {

    select_simd_size_default<T>(size, kernel_type);
}

// Explicit instantiations
template void select_simd_size<float>(da_int size, vectorization_type &kernel_type);
template void select_simd_size<double>(da_int size, vectorization_type &kernel_type);
} // namespace da_kernel_functions
} // namespace da_dynamic_dispatch_zen3

namespace da_dynamic_dispatch_zen4 {
namespace da_kernel_functions {
template <class T> void select_simd_size(da_int size, vectorization_type &kernel_type) {

    select_simd_size_avx512<T>(size, kernel_type);
}

// Explicit instantiations
template void select_simd_size<float>(da_int size, vectorization_type &kernel_type);
template void select_simd_size<double>(da_int size, vectorization_type &kernel_type);
} // namespace da_kernel_functions
} // namespace da_dynamic_dispatch_zen4

namespace da_dynamic_dispatch_zen5 {
namespace da_kernel_functions {
template <class T> void select_simd_size(da_int size, vectorization_type &kernel_type) {

    select_simd_size_avx512<T>(size, kernel_type);
}

// Explicit instantiations
template void select_simd_size<float>(da_int size, vectorization_type &kernel_type);
template void select_simd_size<double>(da_int size, vectorization_type &kernel_type);
} // namespace da_kernel_functions
} // namespace da_dynamic_dispatch_zen5