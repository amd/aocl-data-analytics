/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <numeric>

template <typename T> struct test_math_func_vec_type {
    std::vector<T> input;
    da_int power, first_dim, second_dim, size;
    T coef0, multiplier;
};

// It is important that datasets have at least 16 as a first_dim. Otherwise we interpret the
// data that cannot fit into register as a remainder and calculate with scalar version.
template <typename T> void set_zero_data(test_math_func_vec_type<T> &data) {
    data.first_dim = 16;
    data.second_dim = 2;
    data.size = data.first_dim * data.second_dim;
    data.power = 0;
    data.coef0 = 1.0;
    data.multiplier = 10.0;
    data.input = std::vector<T>(data.size, 0);
}
template <typename T> void set_iota_data(test_math_func_vec_type<T> &data) {
    data.first_dim = 16;
    data.second_dim = 2;
    data.size = data.first_dim * data.second_dim;
    data.power = 4;
    data.coef0 = -4.0;
    data.multiplier = 2.5;
    data.input.resize(data.size);
    std::iota(data.input.begin(), data.input.end(), 1);
}
template <typename T> void set_large_numbers_data(test_math_func_vec_type<T> &data) {
    data.first_dim = 16;
    data.second_dim = 2;
    data.size = data.first_dim * data.second_dim;
    data.power = 4;
    data.coef0 = 50.0;
    data.multiplier = 10;
    data.input = {23.45, -76.32, 0.56,  89.10,  -67.89, 45.67,  -12.34, 99.99,
                  55.55, -33.33, 10.01, -90.90, 42.42,  -58.58, 7.77,   -21.21,
                  64.64, -3.03,  81.81, -46.46, 2.22,   -37.37, 48.48,  -59.59,
                  15.15, -80.80, 29.29, -11.11, 66.66,  -44.44, 3.14,   -5.67};
}
template <typename T> void set_very_large_numbers_data(test_math_func_vec_type<T> &data) {
    data.first_dim = 16;
    data.second_dim = 2;
    data.size = data.first_dim * data.second_dim;
    data.power = 10;
    data.coef0 = 1e8;
    data.multiplier = 1e6;
    data.input = {1e15,   1.1e15, 1.2e15, 1.3e15, 1.4e15, 1.5e15, 1.6e15, 1.7e15,
                  1.8e15, 1.9e15, 2e15,   2.1e15, 2.2e15, 2.3e15, 2.4e15, 2.5e15,
                  2.6e15, 2.7e15, 2.8e15, 2.9e15, 3e15,   3.1e15, 3.2e15, 3.3e15,
                  3.4e15, 3.5e15, 3.6e15, 3.7e15, 3.8e15, 3.9e15, 4e15,   4.1e15};
}