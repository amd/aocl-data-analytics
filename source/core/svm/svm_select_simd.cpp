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
 the SIMD size and padding for different architectures and data types. The function is specialized for
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
namespace da_svm {

// Default lookup tables for AVX2 systems

constexpr std::array<KernelSelection, 3> wssi_float = {{
    {4, scalar},       // Up to 4 -> scalar
    {1024, avx},       // Up to 1024 -> avx
    {DA_INT_MAX, avx2} // >= 1024 -> avx2
}};

constexpr std::array<KernelSelection, 3> wssi_double = {{
    {4, scalar},       // Up to 4 -> scalar
    {512, avx},        // Up to 512 -> avx
    {DA_INT_MAX, avx2} // >= 512 -> avx2
}};

constexpr std::array<KernelSelection, 2> wssj_float = {{
    {128, avx},        // Up to 128 -> avx
    {DA_INT_MAX, avx2} // >= 128 -> avx2
}};

constexpr std::array<KernelSelection, 2> wssj_double = {{
    {64, avx},         // Up to 64 -> avx
    {DA_INT_MAX, avx2} // >= 64 -> avx2
}};

// Default lookup tables for AVX512 systems

constexpr std::array<KernelSelection, 3> wssi_float_avx512 = {{
    {8, scalar},       // Up to 8 -> scalar
    {1024, avx},       // Up to 1024 -> avx
    {DA_INT_MAX, avx2} // >= 1024 -> avx2
}};

constexpr std::array<KernelSelection, 3> wssi_double_avx512 = {{
    {8, avx},            // Up to 8 -> avx
    {1024, avx2},        // Up to 1024 -> avx2
    {DA_INT_MAX, avx512} // >= 1024 -> avx512
}};

constexpr std::array<KernelSelection, 2> wssj_float_avx512 = {{
    {128, avx2},         // Up to 128 -> avx2
    {DA_INT_MAX, avx512} // >= 128 -> avx512
}};

constexpr std::array<KernelSelection, 2> wssj_double_avx512 = {{
    {64, avx2},          // Up to 64 -> avx2
    {DA_INT_MAX, avx512} // >= 64 -> avx512
}};

} // namespace da_svm

// Specific Zen 4 lookup tables for certain cases, which override the defaults
namespace da_dynamic_dispatch_zen4 {
namespace da_svm {

constexpr std::array<KernelSelection, 3> wssi_float = {{
    {4, scalar},       // Up to 4 -> scalar
    {1024, avx},       // Up to 1024 -> avx
    {DA_INT_MAX, avx2} // >= 1024 -> avx2
}};

constexpr std::array<KernelSelection, 3> wssi_double = {{
    {4, scalar},       // Up to 4 -> scalar
    {512, avx},        // Up to 512 -> avx
    {DA_INT_MAX, avx2} // >= 512 -> avx2
}};

constexpr std::array<KernelSelection, 3> wssj_float = {{
    {64, avx},           // Up to 64 -> avx
    {128, avx2},         // Up to 128 -> avx2
    {DA_INT_MAX, avx512} // >= 128 -> avx512
}};

constexpr std::array<KernelSelection, 2> wssj_double = {{
    {64, avx2},          // Up to 64 -> avx2
    {DA_INT_MAX, avx512} // >= 64 -> avx512
}};

} // namespace da_svm
} // namespace da_dynamic_dispatch_zen4

// Further specific lookup tables can be added here for other architectures

//Select kernel type based on a lookup table and a parameter
template <std::size_t N>
vectorization_type lookup_kernel_svm(const std::array<KernelSelection, N> &selections,
                                     da_int param) {
    using namespace std::string_literals;
    // Check to see if there is an override
    const char isa[]{"svm.isa"};
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

//Select the amount of padding based on the kernel type and data type
template <class T> da_int get_padding(vectorization_type kernel_type) {

    da_int value;

    switch (kernel_type) {
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

template <class T>
void select_simd_size_default_wss(da_int ws_size, da_int &padding,
                                  vectorization_type &kernel_type_wssi,
                                  vectorization_type &kernel_type_wssj) {
    // Choose kernel type and padding for WSSI algorithm (default case)

    kernel_type_wssi = std::is_same<T, float>::value
                           ? lookup_kernel_svm(da_svm::wssi_float, ws_size)
                           : lookup_kernel_svm(da_svm::wssi_double, ws_size);
    kernel_type_wssj = std::is_same<T, float>::value
                           ? lookup_kernel_svm(da_svm::wssj_float, ws_size)
                           : lookup_kernel_svm(da_svm::wssj_double, ws_size);

    // Get padding based on the larger kernel type
    vectorization_type kernel_type_larger =
        (kernel_type_wssi > kernel_type_wssj) ? kernel_type_wssi : kernel_type_wssj;

    padding = get_padding<T>(kernel_type_larger);
}

template <class T>
void select_simd_size_wss_avx512(da_int ws_size, da_int &padding,
                                 vectorization_type &kernel_type_wssi,
                                 vectorization_type &kernel_type_wssj) {
    // Choose kernel type and padding for WSSI algorithm (default case)

    kernel_type_wssi = std::is_same<T, float>::value
                           ? lookup_kernel_svm(da_svm::wssi_float_avx512, ws_size)
                           : lookup_kernel_svm(da_svm::wssi_double_avx512, ws_size);
    kernel_type_wssj = std::is_same<T, float>::value
                           ? lookup_kernel_svm(da_svm::wssj_float_avx512, ws_size)
                           : lookup_kernel_svm(da_svm::wssj_double_avx512, ws_size);

    // Get padding based on the larger kernel type
    vectorization_type kernel_type_larger =
        (kernel_type_wssi > kernel_type_wssj) ? kernel_type_wssi : kernel_type_wssj;

    padding = get_padding<T>(kernel_type_larger);
}

// Specializations for different architectures

namespace da_dynamic_dispatch_generic {
namespace da_svm {
template <class T>
void select_simd_size_wss(da_int ws_size, da_int &padding,
                          vectorization_type &kernel_type_wssi,
                          vectorization_type &kernel_type_wssj) {

    select_simd_size_default_wss<T>(ws_size, padding, kernel_type_wssi, kernel_type_wssj);
}

// Explicit instantiations
template void select_simd_size_wss<float>(da_int ws_size, da_int &padding,
                                          vectorization_type &kernel_type_wssi,
                                          vectorization_type &kernel_type_wssj);
template void select_simd_size_wss<double>(da_int ws_size, da_int &padding,
                                           vectorization_type &kernel_type_wssi,
                                           vectorization_type &kernel_type_wssj);

} // namespace da_svm
} // namespace da_dynamic_dispatch_generic

namespace da_dynamic_dispatch_generic_avx512 {
namespace da_svm {
template <class T>
void select_simd_size_wss(da_int ws_size, da_int &padding,
                          vectorization_type &kernel_type_wssi,
                          vectorization_type &kernel_type_wssj) {

    select_simd_size_wss_avx512<T>(ws_size, padding, kernel_type_wssi, kernel_type_wssj);
}

// Explicit instantiations
template void select_simd_size_wss<float>(da_int ws_size, da_int &padding,
                                          vectorization_type &kernel_type_wssi,
                                          vectorization_type &kernel_type_wssj);
template void select_simd_size_wss<double>(da_int ws_size, da_int &padding,
                                           vectorization_type &kernel_type_wssi,
                                           vectorization_type &kernel_type_wssj);

} // namespace da_svm
} // namespace da_dynamic_dispatch_generic_avx512

namespace da_dynamic_dispatch_zen2 {
namespace da_svm {
template <class T>
void select_simd_size_wss(da_int ws_size, da_int &padding,
                          vectorization_type &kernel_type_wssi,
                          vectorization_type &kernel_type_wssj) {

    select_simd_size_default_wss<T>(ws_size, padding, kernel_type_wssi, kernel_type_wssj);
}

// Explicit instantiations
template void select_simd_size_wss<float>(da_int ws_size, da_int &padding,
                                          vectorization_type &kernel_type_wssi,
                                          vectorization_type &kernel_type_wssj);
template void select_simd_size_wss<double>(da_int ws_size, da_int &padding,
                                           vectorization_type &kernel_type_wssi,
                                           vectorization_type &kernel_type_wssj);
} // namespace da_svm
} // namespace da_dynamic_dispatch_zen2

namespace da_dynamic_dispatch_zen3 {
namespace da_svm {
template <class T>
void select_simd_size_wss(da_int ws_size, da_int &padding,
                          vectorization_type &kernel_type_wssi,
                          vectorization_type &kernel_type_wssj) {

    select_simd_size_default_wss<T>(ws_size, padding, kernel_type_wssi, kernel_type_wssj);
}

// Explicit instantiations
template void select_simd_size_wss<float>(da_int ws_size, da_int &padding,
                                          vectorization_type &kernel_type_wssi,
                                          vectorization_type &kernel_type_wssj);
template void select_simd_size_wss<double>(da_int ws_size, da_int &padding,
                                           vectorization_type &kernel_type_wssi,
                                           vectorization_type &kernel_type_wssj);
} // namespace da_svm
} // namespace da_dynamic_dispatch_zen3

namespace da_dynamic_dispatch_zen4 {
namespace da_svm {
template <class T>
void select_simd_size_wss(da_int ws_size, da_int &padding,
                          vectorization_type &kernel_type_wssi,
                          vectorization_type &kernel_type_wssj) {

    kernel_type_wssi =
        std::is_same<T, float>::value
            ? lookup_kernel_svm(da_dynamic_dispatch_zen4::da_svm::wssi_float, ws_size)
            : lookup_kernel_svm(da_dynamic_dispatch_zen4::da_svm::wssi_double, ws_size);
    kernel_type_wssj =
        std::is_same<T, float>::value
            ? lookup_kernel_svm(da_dynamic_dispatch_zen4::da_svm::wssj_float, ws_size)
            : lookup_kernel_svm(da_dynamic_dispatch_zen4::da_svm::wssj_double, ws_size);

    // Get padding based on the larger kernel type
    vectorization_type kernel_type_larger =
        (kernel_type_wssi > kernel_type_wssj) ? kernel_type_wssi : kernel_type_wssj;

    padding = get_padding<T>(kernel_type_larger);
}

// Explicit instantiations
template void select_simd_size_wss<float>(da_int ws_size, da_int &padding,
                                          vectorization_type &kernel_type_wssi,
                                          vectorization_type &kernel_type_wssj);
template void select_simd_size_wss<double>(da_int ws_size, da_int &padding,
                                           vectorization_type &kernel_type_wssi,
                                           vectorization_type &kernel_type_wssj);
} // namespace da_svm
} // namespace da_dynamic_dispatch_zen4

namespace da_dynamic_dispatch_zen5 {
namespace da_svm {
template <class T>
void select_simd_size_wss(da_int ws_size, da_int &padding,
                          vectorization_type &kernel_type_wssi,
                          vectorization_type &kernel_type_wssj) {

    select_simd_size_wss_avx512<T>(ws_size, padding, kernel_type_wssi, kernel_type_wssj);
}

// Explicit instantiations
template void select_simd_size_wss<float>(da_int ws_size, da_int &padding,
                                          vectorization_type &kernel_type_wssi,
                                          vectorization_type &kernel_type_wssj);
template void select_simd_size_wss<double>(da_int ws_size, da_int &padding,
                                           vectorization_type &kernel_type_wssi,
                                           vectorization_type &kernel_type_wssj);
} // namespace da_svm
} // namespace da_dynamic_dispatch_zen5