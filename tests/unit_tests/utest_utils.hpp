/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <vector>

#define EXPECT_ARR_NEAR(n, x, y, abs_error)                                              \
    for (da_int j = 0; j < (n); j++)                                                     \
    EXPECT_NEAR((x[j]), (y[j]), abs_error)                                               \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

#define EXPECT_ARR_EQ(n, x, y, incx, incy, startx, starty)                               \
    for (da_int j = 0; j < (n); j++)                                                     \
    EXPECT_EQ((x[startx + j * incx]), (y[starty + j * incy]))                            \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

#define EXPECT_ARR_ABS_NEAR(n, x, y, abs_error)                                          \
    for (da_int j = 0; j < (n); j++)                                                     \
    EXPECT_NEAR((std::abs(x[j])), (std::abs(y[j])), abs_error)                           \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

/* Convert std::vector from one type to another, to avoid warnings in templated tests*/
template <class T_in, class T_out>
std::vector<T_out> convert_vector(const std::vector<T_in> &input) {
    std::vector<T_out> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(),
                   [](T_out x) { return static_cast<T_in>(x); });
    return output;
}

inline da_status da_linmod_get_coef(da_handle handle, da_int *nc, double *x) {
    return da_handle_get_result_d(handle, da_result::da_linmod_coeff, nc, x);
}
inline da_status da_linmod_get_coef(da_handle handle, da_int *nc, float *x) {
    return da_handle_get_result_s(handle, da_result::da_linmod_coeff, nc, x);
}

#endif
