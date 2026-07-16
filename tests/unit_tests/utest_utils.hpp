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

#ifndef UTEST_UTILS_HPP
#define UTEST_UTILS_HPP
#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#ifdef _WIN32
#include <stdlib.h>
#include <windows.h>
#else
#include <cstdlib>
#include <unistd.h>
#endif
#include <iostream>
#include <string>
#include <vector>

// Auxiliary macro for token concatenation
#define CONCAT_HELPER(a, b) a##b
#define CONCAT(a, b) CONCAT_HELPER(a, b)

// Define a macro that will define the dynamic dispatch namespace used for internal tests
#define TEST_ARCH CONCAT(da_dynamic_dispatch_, ZNVER_MAX)

#define EXPECT_ARR_NEAR(n, x, y, abs_error)                                              \
    for (da_int j = 0; j < (n); j++)                                                     \
    EXPECT_NEAR((x[j]), (y[j]), abs_error)                                               \
        << "*** Vectors " #x "[" << j << "] and " #y "[" << j << "] are different!     "

#define EXPECT_ARR_EQ(n, x, y, incx, incy, startx, starty)                               \
    for (da_int j = 0; j < (n); j++)                                                     \
    EXPECT_EQ((x[startx + j * incx]), (y[starty + j * incy]))                            \
        << "*** Vectors " #x "[" << j << "] and " #y "[" << j << "] are different!     "

#define EXPECT_ARR_ABS_NEAR(n, x, y, abs_error)                                          \
    for (da_int j = 0; j < (n); j++)                                                     \
    EXPECT_NEAR((std::abs(x[j])), (std::abs(y[j])), abs_error)                           \
        << "*** Vectors " #x "[" << j << "] and " #y "[" << j << "] are different!     "

#define ASSERT_ARR_NEAR(n, x, y, abs_error)                                              \
    for (da_int j = 0; j < (n); j++)                                                     \
    ASSERT_NEAR((x[j]), (y[j]), abs_error)                                               \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

#define ASSERT_ARR_EQ(n, x, y, incx, incy, startx, starty)                               \
    for (da_int j = 0; j < (n); j++)                                                     \
    ASSERT_EQ((x[startx + j * incx]), (y[starty + j * incy]))                            \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

#define ASSERT_ARR_ABS_NEAR(n, x, y, abs_error)                                          \
    for (da_int j = 0; j < (n); j++)                                                     \
    ASSERT_NEAR((std::abs(x[j])), (std::abs(y[j])), abs_error)                           \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

#ifdef __COVERITY__
void __coverity_panic__(void); // modeled as no-return
#define ASSERT_EQ(ret, sta)                                                              \
    do {                                                                                 \
        if (!((ret) == (sta))) {                                                         \
            __coverity_panic__();                                                        \
        }                                                                                \
    } while (0)

#define FAIL()                                                                           \
    do {                                                                                 \
        __coverity_panic__();                                                            \
    } while (0) coverity_null_stream()
#endif

// return precision as a string literal to set CSV options
template <typename T> constexpr const char *prec_name();
template <> constexpr const char *prec_name<float>() { return "single"; }
template <> constexpr const char *prec_name<double>() { return "double"; }

template <typename T> constexpr const char *type_opt_name();
template <> constexpr const char *type_opt_name<float>() { return "float"; }
template <> constexpr const char *type_opt_name<double>() { return "double"; }

namespace da_numeric {
// Safe numerical tolerances to be used with single and double precision float types
template <class T> struct tolerance {
    static constexpr T eps{std::numeric_limits<T>::epsilon()};
    static const T safe_tol() { return std::sqrt(T(2) * eps); };
    static const T tol(T numerator = T(1), T denominator = T(1)) {
        return numerator * safe_tol() / denominator;
    }
};
} // namespace da_numeric

/* Convert std::vector from one type to another, to avoid warnings in templated tests*/
template <class T_in, class T_out>
std::vector<T_out> convert_vector(const std::vector<T_in> &input) {
    std::vector<T_out> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(),
                   [](T_in x) { return static_cast<T_out>(x); });
    return output;
}

inline da_status da_linmod_get_coef(da_handle handle, da_int *nc, double *x) {
    return da_handle_get_result_d(handle, da_result::da_linmod_coef, nc, x);
}
inline da_status da_linmod_get_coef(da_handle handle, da_int *nc, float *x) {
    return da_handle_get_result_s(handle, da_result::da_linmod_coef, nc, x);
}

namespace da_test {

template <typename T>
bool read_csv_data(const std::string &filename, std::vector<T> &data, da_int &n_rows,
                   da_int &n_cols, da_order order = column_major) {
    da_datastore csv_store = nullptr;
    bool ok =
        da_datastore_init(&csv_store) == da_status_success &&
        da_datastore_options_set_string(csv_store, "datastore precision",
                                        prec_name<T>()) == da_status_success &&
        da_datastore_options_set_string(csv_store, "datatype", type_opt_name<T>()) ==
            da_status_success &&
        da_data_load_from_csv(csv_store, filename.c_str()) == da_status_success &&
        da_data_get_n_cols(csv_store, &n_cols) == da_status_success &&
        da_data_get_n_rows(csv_store, &n_rows) == da_status_success &&
        da_data_select_columns(csv_store, "data", 0, n_cols - 1) == da_status_success;
    if (ok) {
        data.resize(n_rows * n_cols);
        da_int ld = (order == row_major) ? n_cols : n_rows;
        ok = da_data_extract_selection(csv_store, "data", order, data.data(), ld) ==
             da_status_success;
    }
    if (!ok)
        std::cerr << "read_csv_data: failed to load '" << filename << "'" << std::endl;
    da_datastore_destroy(&csv_store);
    return ok;
}

template <typename T> inline void free_data(T **arr, [[maybe_unused]] da_int n) {
    if (*arr)
        free(*arr);
}

inline void free_data(char ***arr, da_int n) { da_delete_string_array(arr, n); }

// Helper function to set an environment variable
inline int da_setenv(const char *name, const char *value, int overwrite) {
#ifdef _WIN32
    return _putenv_s(name, value);
#else
    return setenv(name, value, overwrite);
#endif
}

void sleep(int milliseconds) {
#ifdef _WIN32
    Sleep(milliseconds);
#else
    usleep(1000 * milliseconds);
#endif
}

} // namespace da_test

#endif