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
#include "kmeans_types.hpp"
#include <array>
#include <type_traits>

/*
This file contains a series of overloads for the select_simd_size function, which is used to determine
the SIMD size and padding for different architectures and data types. The function is specialized for
various architectures, including Zen2, Zen3, Zen4, and Zen5. Each specialization takes the number of
clusters, the k-means method, the architecture and the data type, and uses those to choose the
optimal kernel type (AVX, AVX2, AVX512) and the appropriate padding required for memory allocation to
ensure the simd length fits neatly in the kernels.
*/

using namespace da_kmeans_types;

// Define a struct to hold threshold and corresponding kernel type
struct KernelSelection {
    da_int threshold;
    vectorization_type kernel;
};

// Default lookup tables
namespace da_kmeans {

// Lookup tables for AVX2 machines

constexpr std::array<KernelSelection, 3> lloyd_float = {{
    {2, scalar},       // Up to 2 -> scalar
    {16, avx},         // Up to 16 -> avx
    {DA_INT_MAX, avx2} // > 16 -> avx2
}};

constexpr std::array<KernelSelection, 2> lloyd_double = {{
    {6, avx},          // Up to 6 -> avx
    {DA_INT_MAX, avx2} // > 6 -> avx2
}};

constexpr std::array<KernelSelection, 1> elkan_reduce_float = {{
    {DA_INT_MAX, scalar} // always avx2
}};

constexpr std::array<KernelSelection, 3> elkan_reduce_double = {{
    {4, scalar},       // Up to 4 -> scalar
    {16, avx},         // > 16 -> avx
    {DA_INT_MAX, avx2} // > 16 -> avx2
}};

constexpr std::array<KernelSelection, 2> elkan_update_float = {{
    {4, avx},           // Up to 4 -> avx
    {DA_INT_MAX, avx2}, // >4 -> avx2
}};

constexpr std::array<KernelSelection, 1> elkan_update_double = {{
    {DA_INT_MAX, avx2} // always avx2
}};

//Lookup tables for AVX512 machines

constexpr std::array<KernelSelection, 4> lloyd_double_avx512 = {{
    {4, scalar},         // Up to 4 -> scalar
    {6, avx},            // Up to 6 -> avx
    {19, avx2},          // Up to 19 -> avx2
    {DA_INT_MAX, avx512} // > 19 -> avx512
}};

constexpr std::array<KernelSelection, 3> lloyd_float_avx512 = {{
    {4, scalar},       // Up to 4 -> scalar
    {16, avx},         // < 16 -> avx
    {DA_INT_MAX, avx2} // > 16 -> avx2
}};

constexpr std::array<KernelSelection, 2> elkan_reduce_float_avx512 = {{
    {8, avx},         // Up to 8 scalar
    {DA_INT_MAX, avx} // > 8 -> avx
}};

constexpr std::array<KernelSelection, 4> elkan_reduce_double_avx512 = {{
    {4, scalar},         // Up to 4 -> scalar
    {8, avx},            // Up to 8 -> avx
    {15, avx2},          // Up to 15 -> avx2
    {DA_INT_MAX, avx512} // > 15 -> avx512
}};

constexpr std::array<KernelSelection, 2> elkan_update_float_avx512 = {{
    {4, avx},           // Up to 4 -> avx
    {DA_INT_MAX, avx2}, // >4 -> avx2
}};

constexpr std::array<KernelSelection, 3> elkan_update_double_avx512 = {{
    {6, avx},            // Up to 6 -> avx
    {15, avx2},          // Up to 15 -> avx2
    {DA_INT_MAX, avx512} // > 15 -> avx512
}};

} // namespace da_kmeans

// Further specific lookup tables can be added here for other architectures

//Select kernel type based on a lookup table and a parameter
template <std::size_t N>
vectorization_type lookup_kernel_kmeans(const std::array<KernelSelection, N> &selections,
                                        da_int param) {
    using namespace std::string_literals;
    // Check to see if there is an override
    const char isa[]{"kmeans.isa"};
    if (context::get_context()->hidden_settings.find(isa) !=
        context::get_context()->hidden_settings.end()) {
        std::string kernel = context::get_context()->hidden_settings[isa];
        if (kernel == "avx"s) {
            return vectorization_type::avx;
        } else if (kernel == "avx2"s) {
            return vectorization_type::avx2;
        } else if (kernel == "avx512"s) {
            return (context::get_context()->has_avx512) ? vectorization_type::avx512
                                                        : vectorization_type::avx2;
        }
        return vectorization_type::scalar;
    }

    // Get best kernel
    for (const auto &selection : selections) {
        if (param <= selection.threshold) {
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

// Default selection functions - AVX2 only

template <class T>
void select_simd_size_default_lloyd(da_int n_clusters, da_int &padding,
                                    vectorization_type &kernel_type) {
    // Choose kernel type and padding for Lloyd algorithm (default case)

    kernel_type = std::is_same<T, float>::value
                      ? lookup_kernel_kmeans(da_kmeans::lloyd_float, n_clusters)
                      : lookup_kernel_kmeans(da_kmeans::lloyd_double, n_clusters);

    padding = get_padding<T>(kernel_type);
}

template <class T>
void select_simd_size_default_elkan([[maybe_unused]] da_int n_clusters,
                                    [[maybe_unused]] da_int n_features, da_int &padding,
                                    vectorization_type &update_kernel_type,
                                    vectorization_type &reduce_kernel_type) {
    // Choose kernel type and padding for Elkan algorithm (default case)

    if (std::is_same<T, float>::value) {
        update_kernel_type =
            lookup_kernel_kmeans(da_kmeans::elkan_update_float, n_clusters);
        reduce_kernel_type =
            lookup_kernel_kmeans(da_kmeans::elkan_reduce_float, n_features);
    } else {
        update_kernel_type =
            lookup_kernel_kmeans(da_kmeans::elkan_update_double, n_clusters);
        reduce_kernel_type =
            lookup_kernel_kmeans(da_kmeans::elkan_reduce_double, n_features);
    }

    padding = get_padding<T>(update_kernel_type);
}

// Selection functions when AVX512 is deemed beneficial

template <class T>
void select_simd_size_lloyd_avx512(da_int n_clusters, da_int &padding,
                                   vectorization_type &kernel_type) {
    // Choose kernel type and padding for Lloyd algorithm (default case)

    kernel_type = std::is_same<T, float>::value
                      ? lookup_kernel_kmeans(da_kmeans::lloyd_float_avx512, n_clusters)
                      : lookup_kernel_kmeans(da_kmeans::lloyd_double_avx512, n_clusters);

    padding = get_padding<T>(kernel_type);
}

template <class T>
void select_simd_size_elkan_avx512([[maybe_unused]] da_int n_clusters,
                                   [[maybe_unused]] da_int n_features, da_int &padding,
                                   vectorization_type &update_kernel_type,
                                   vectorization_type &reduce_kernel_type) {
    // Choose kernel type and padding for Elkan algorithm (default case)

    if (std::is_same<T, float>::value) {
        update_kernel_type =
            lookup_kernel_kmeans(da_kmeans::elkan_update_float_avx512, n_clusters);
        reduce_kernel_type =
            lookup_kernel_kmeans(da_kmeans::elkan_reduce_float_avx512, n_features);
    } else {
        update_kernel_type =
            lookup_kernel_kmeans(da_kmeans::elkan_update_double_avx512, n_clusters);
        reduce_kernel_type =
            lookup_kernel_kmeans(da_kmeans::elkan_reduce_double_avx512, n_features);
    }

    padding = get_padding<T>(update_kernel_type);
}

// Specializations for different architectures

namespace da_dynamic_dispatch_generic {
namespace da_kmeans {
template <class T>
void select_simd_size_lloyd(da_int n_clusters, da_int &padding,
                            vectorization_type &kernel_type) {

    select_simd_size_default_lloyd<T>(n_clusters, padding, kernel_type);
}

template <class T>
void select_simd_size_elkan(da_int n_clusters, da_int n_features, da_int &padding,
                            vectorization_type &update_kernel_type,
                            vectorization_type &reduce_kernel_type) {

    select_simd_size_default_elkan<T>(n_clusters, n_features, padding, update_kernel_type,
                                      reduce_kernel_type);
}

// Explicit instantiations
template void select_simd_size_lloyd<float>(da_int n_clusters, da_int &padding,
                                            vectorization_type &kernel_type);
template void select_simd_size_lloyd<double>(da_int n_clusters, da_int &padding,
                                             vectorization_type &kernel_type);
template void select_simd_size_elkan<float>(da_int n_clusters, da_int n_features,
                                            da_int &padding,
                                            vectorization_type &update_kernel_type,
                                            vectorization_type &reduce_kernel_type);
template void select_simd_size_elkan<double>(da_int n_clusters, da_int n_features,
                                             da_int &padding,
                                             vectorization_type &update_kernel_type,
                                             vectorization_type &reduce_kernel_type);

} // namespace da_kmeans
} // namespace da_dynamic_dispatch_generic

namespace da_dynamic_dispatch_generic_avx512 {
namespace da_kmeans {
template <class T>
void select_simd_size_lloyd(da_int n_clusters, da_int &padding,
                            vectorization_type &kernel_type) {

    select_simd_size_lloyd_avx512<T>(n_clusters, padding, kernel_type);
}

template <class T>
void select_simd_size_elkan(da_int n_clusters, da_int n_features, da_int &padding,
                            vectorization_type &update_kernel_type,
                            vectorization_type &reduce_kernel_type) {

    select_simd_size_elkan_avx512<T>(n_clusters, n_features, padding, update_kernel_type,
                                     reduce_kernel_type);
}

// Explicit instantiations
template void select_simd_size_lloyd<float>(da_int n_clusters, da_int &padding,
                                            vectorization_type &kernel_type);
template void select_simd_size_lloyd<double>(da_int n_clusters, da_int &padding,
                                             vectorization_type &kernel_type);
template void select_simd_size_elkan<float>(da_int n_clusters, da_int n_features,
                                            da_int &padding,
                                            vectorization_type &update_kernel_type,
                                            vectorization_type &reduce_kernel_type);
template void select_simd_size_elkan<double>(da_int n_clusters, da_int n_features,
                                             da_int &padding,
                                             vectorization_type &update_kernel_type,
                                             vectorization_type &reduce_kernel_type);

} // namespace da_kmeans
} // namespace da_dynamic_dispatch_generic_avx512

namespace da_dynamic_dispatch_zen2 {
namespace da_kmeans {
template <class T>
void select_simd_size_lloyd(da_int n_clusters, da_int &padding,
                            vectorization_type &kernel_type) {

    select_simd_size_default_lloyd<T>(n_clusters, padding, kernel_type);
}

template <class T>
void select_simd_size_elkan(da_int n_clusters, da_int n_features, da_int &padding,
                            vectorization_type &update_kernel_type,
                            vectorization_type &reduce_kernel_type) {

    select_simd_size_default_elkan<T>(n_clusters, n_features, padding, update_kernel_type,
                                      reduce_kernel_type);
}

// Explicit instantiations
template void select_simd_size_lloyd<float>(da_int n_clusters, da_int &padding,
                                            vectorization_type &kernel_type);
template void select_simd_size_lloyd<double>(da_int n_clusters, da_int &padding,
                                             vectorization_type &kernel_type);
template void select_simd_size_elkan<float>(da_int n_clusters, da_int n_features,
                                            da_int &padding,
                                            vectorization_type &update_kernel_type,
                                            vectorization_type &reduce_kernel_type);
template void select_simd_size_elkan<double>(da_int n_clusters, da_int n_features,
                                             da_int &padding,
                                             vectorization_type &update_kernel_type,
                                             vectorization_type &reduce_kernel_type);
} // namespace da_kmeans
} // namespace da_dynamic_dispatch_zen2

namespace da_dynamic_dispatch_zen3 {
namespace da_kmeans {
template <class T>
void select_simd_size_lloyd(da_int n_clusters, da_int &padding,
                            vectorization_type &kernel_type) {

    select_simd_size_default_lloyd<T>(n_clusters, padding, kernel_type);
}

template <class T>
void select_simd_size_elkan(da_int n_clusters, da_int n_features, da_int &padding,
                            vectorization_type &update_kernel_type,
                            vectorization_type &reduce_kernel_type) {

    select_simd_size_default_elkan<T>(n_clusters, n_features, padding, update_kernel_type,
                                      reduce_kernel_type);
}

// Explicit instantiations
template void select_simd_size_lloyd<float>(da_int n_clusters, da_int &padding,
                                            vectorization_type &kernel_type);
template void select_simd_size_lloyd<double>(da_int n_clusters, da_int &padding,
                                             vectorization_type &kernel_type);
template void select_simd_size_elkan<float>(da_int n_clusters, da_int n_features,
                                            da_int &padding,
                                            vectorization_type &update_kernel_type,
                                            vectorization_type &reduce_kernel_type);
template void select_simd_size_elkan<double>(da_int n_clusters, da_int n_features,
                                             da_int &padding,
                                             vectorization_type &update_kernel_type,
                                             vectorization_type &reduce_kernel_type);
} // namespace da_kmeans
} // namespace da_dynamic_dispatch_zen3

namespace da_dynamic_dispatch_zen4 {
namespace da_kmeans {
template <class T>
void select_simd_size_lloyd(da_int n_clusters, da_int &padding,
                            vectorization_type &kernel_type) {

    select_simd_size_default_lloyd<T>(n_clusters, padding, kernel_type);
}

template <class T>
void select_simd_size_elkan(da_int n_clusters, da_int n_features, da_int &padding,
                            vectorization_type &update_kernel_type,
                            vectorization_type &reduce_kernel_type) {

    select_simd_size_default_elkan<T>(n_clusters, n_features, padding, update_kernel_type,
                                      reduce_kernel_type);
}

// Explicit instantiations
template void select_simd_size_lloyd<float>(da_int n_clusters, da_int &padding,
                                            vectorization_type &kernel_type);
template void select_simd_size_lloyd<double>(da_int n_clusters, da_int &padding,
                                             vectorization_type &kernel_type);
template void select_simd_size_elkan<float>(da_int n_clusters, da_int n_features,
                                            da_int &padding,
                                            vectorization_type &update_kernel_type,
                                            vectorization_type &reduce_kernel_type);
template void select_simd_size_elkan<double>(da_int n_clusters, da_int n_features,
                                             da_int &padding,
                                             vectorization_type &update_kernel_type,
                                             vectorization_type &reduce_kernel_type);
} // namespace da_kmeans
} // namespace da_dynamic_dispatch_zen4

namespace da_dynamic_dispatch_zen5 {
namespace da_kmeans {
template <class T>
void select_simd_size_lloyd(da_int n_clusters, da_int &padding,
                            vectorization_type &kernel_type) {

    select_simd_size_lloyd_avx512<T>(n_clusters, padding, kernel_type);
}

template <class T>
void select_simd_size_elkan(da_int n_clusters, da_int n_features, da_int &padding,
                            vectorization_type &update_kernel_type,
                            vectorization_type &reduce_kernel_type) {

    select_simd_size_elkan_avx512<T>(n_clusters, n_features, padding, update_kernel_type,
                                     reduce_kernel_type);
}

// Explicit instantiations
template void select_simd_size_lloyd<float>(da_int n_clusters, da_int &padding,
                                            vectorization_type &kernel_type);
template void select_simd_size_lloyd<double>(da_int n_clusters, da_int &padding,
                                             vectorization_type &kernel_type);
template void select_simd_size_elkan<float>(da_int n_clusters, da_int n_features,
                                            da_int &padding,
                                            vectorization_type &update_kernel_type,
                                            vectorization_type &reduce_kernel_type);
template void select_simd_size_elkan<double>(da_int n_clusters, da_int n_features,
                                             da_int &padding,
                                             vectorization_type &update_kernel_type,
                                             vectorization_type &reduce_kernel_type);
} // namespace da_kmeans
} // namespace da_dynamic_dispatch_zen5
