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
#include "svm_types.hpp"
#include <array>
#include <type_traits>

/*
 This file contains a series of overloads for the select_ws_size function, which is used to determine
 the working set size for different architectures and data types. The function is specialized for
 various architectures, including Zen2, Zen3, Zen4, and Zen5. Each specialization takes the number of rows and
 the architecture, and uses those to choose the optimal ws size.
 */

// Define a struct to hold threshold and corresponding optimal working set size
struct WsSizeSelection {
    da_int threshold;
    da_int ws_size;
};

namespace da_svm {

constexpr std::array<WsSizeSelection, 3> ws_size_rbf = {{
    {5000, 512},       // Up to 5000 -> 512
    {50000, 1024},     // Up to 50000 -> 1024
    {DA_INT_MAX, 2048} // >= 50000 -> 2048
}};
constexpr std::array<WsSizeSelection, 3> ws_size_linear = {{
    {20000, 256},      // Up to 20000 -> 256
    {50000, 512},      // Up to 50000 -> 512
    {DA_INT_MAX, 1024} // >= 50000 -> 1024
}};
constexpr std::array<WsSizeSelection, 3> ws_size_poly = {{
    {5000, 512},       // Up to 5000 -> 512
    {50000, 1024},     // Up to 50000 -> 1024
    {DA_INT_MAX, 2048} // >= 50000 -> 2048
}};
constexpr std::array<WsSizeSelection, 3> ws_size_sigmoid = {{
    {5000, 512},       // Up to 5000 -> 512
    {50000, 1024},     // Up to 50000 -> 1024
    {DA_INT_MAX, 2048} // >= 50000 -> 2048
}};
} // namespace da_svm

// Specific Zen 5 lookup tables for certain cases, which override the defaults
namespace da_dynamic_dispatch_zen5 {
namespace da_svm {

constexpr std::array<WsSizeSelection, 3> ws_size_rbf = {{
    {5000, 512},       // Up to 5000 -> 512
    {50000, 1024},     // Up to 50000 -> 1024
    {DA_INT_MAX, 2048} // >= 50000 -> 2048
}};
constexpr std::array<WsSizeSelection, 3> ws_size_linear = {{
    {20000, 256},      // Up to 20000 -> 256
    {50000, 512},      // Up to 50000 -> 512
    {DA_INT_MAX, 1024} // >= 50000 -> 1024
}};
constexpr std::array<WsSizeSelection, 3> ws_size_poly = {{
    {5000, 512},       // Up to 5000 -> 512
    {50000, 1024},     // Up to 50000 -> 1024
    {DA_INT_MAX, 2048} // >= 50000 -> 2048
}};
constexpr std::array<WsSizeSelection, 3> ws_size_sigmoid = {{
    {5000, 512},       // Up to 5000 -> 512
    {50000, 1024},     // Up to 50000 -> 1024
    {DA_INT_MAX, 2048} // >= 50000 -> 2048
}};
} // namespace da_svm
} // namespace da_dynamic_dispatch_zen5

// Further specific lookup tables can be added here for other architectures

// Select ws_size based on a lookup table and a parameter
template <std::size_t N>
da_int lookup_ws_size(const std::array<WsSizeSelection, N> &selections, da_int n) {
    // Get best ws_size
    for (const auto &selection : selections) {
        if (n <= selection.threshold) {
            return selection.ws_size;
        }
    }
    // Default to 1024 if no match found - this should not happen
    return 1024; // LCOV_EXCL_LINE
}

void select_ws_size_default(da_int n, da_svm_types::svm_kernel kernel, da_int &ws_size) {
    // Choose working set size (default case)
    if (kernel == da_svm_types::svm_kernel::rbf) {
        ws_size = lookup_ws_size(da_svm::ws_size_rbf, n);
    } else if (kernel == da_svm_types::svm_kernel::linear) {
        ws_size = lookup_ws_size(da_svm::ws_size_linear, n);
    } else if (kernel == da_svm_types::svm_kernel::polynomial) {
        ws_size = lookup_ws_size(da_svm::ws_size_poly, n);
    } else if (kernel == da_svm_types::svm_kernel::sigmoid) {
        ws_size = lookup_ws_size(da_svm::ws_size_sigmoid, n);
    } else {
        ws_size = 1024; // LCOV_EXCL_LINE
    }
}

// Specializations for different architectures

namespace da_dynamic_dispatch_generic {
namespace da_svm {
void select_ws_size(da_int n, da_svm_types::svm_kernel kernel, da_int &ws_size) {

    select_ws_size_default(n, kernel, ws_size);
}

} // namespace da_svm
} // namespace da_dynamic_dispatch_generic

namespace da_dynamic_dispatch_generic_avx512 {
namespace da_svm {
void select_ws_size(da_int n, da_svm_types::svm_kernel kernel, da_int &ws_size) {

    select_ws_size_default(n, kernel, ws_size);
}

} // namespace da_svm
} // namespace da_dynamic_dispatch_generic_avx512

namespace da_dynamic_dispatch_zen2 {
namespace da_svm {
void select_ws_size(da_int n, da_svm_types::svm_kernel kernel, da_int &ws_size) {

    select_ws_size_default(n, kernel, ws_size);
}

} // namespace da_svm
} // namespace da_dynamic_dispatch_zen2

namespace da_dynamic_dispatch_zen3 {
namespace da_svm {
void select_ws_size(da_int n, da_svm_types::svm_kernel kernel, da_int &ws_size) {

    select_ws_size_default(n, kernel, ws_size);
}

} // namespace da_svm
} // namespace da_dynamic_dispatch_zen3

namespace da_dynamic_dispatch_zen4 {
namespace da_svm {
void select_ws_size(da_int n, da_svm_types::svm_kernel kernel, da_int &ws_size) {

    select_ws_size_default(n, kernel, ws_size);
}

} // namespace da_svm
} // namespace da_dynamic_dispatch_zen4

namespace da_dynamic_dispatch_zen5 {
namespace da_svm {
void select_ws_size(da_int n, da_svm_types::svm_kernel kernel, da_int &ws_size) {

    if (kernel == da_svm_types::svm_kernel::rbf) {
        ws_size = lookup_ws_size(da_dynamic_dispatch_zen5::da_svm::ws_size_rbf, n);
    } else if (kernel == da_svm_types::svm_kernel::linear) {
        ws_size = lookup_ws_size(da_dynamic_dispatch_zen5::da_svm::ws_size_linear, n);
    } else if (kernel == da_svm_types::svm_kernel::polynomial) {
        ws_size = lookup_ws_size(da_dynamic_dispatch_zen5::da_svm::ws_size_poly, n);
    } else if (kernel == da_svm_types::svm_kernel::sigmoid) {
        ws_size = lookup_ws_size(da_dynamic_dispatch_zen5::da_svm::ws_size_sigmoid, n);
    } else {
        ws_size = 1024; // LCOV_EXCL_LINE
    }
}

} // namespace da_svm
} // namespace da_dynamic_dispatch_zen5