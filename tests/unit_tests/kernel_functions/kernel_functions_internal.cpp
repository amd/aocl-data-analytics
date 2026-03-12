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

#include "../utest_utils.hpp"
#include "kernel_functions.hpp"
#include "kernel_functions_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstring>
#include <list>

using namespace TEST_ARCH;
using namespace da_kernel_functions;
using namespace da_kernel_functions_types;

template <typename T> class kernel_internal_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(kernel_internal_test, FloatTypes);

// Helper function to test a kernel across different ISA implementations
template <typename T, typename KernelSelector, typename KernelExecutor>
void test_kernel_across_isa(
    const std::vector<T> &input, const std::vector<T> &expected,
    const std::unordered_map<std::string, vectorization_type> &isa_list, da_int count,
    KernelSelector kernel_selector, KernelExecutor kernel_executor) {
    for (const auto &isa : isa_list) {
        std::cout << "Dataset: " << std::to_string(count)
                  << ", vectorisation: " << isa.first << std::endl;
        auto kernel_func = kernel_selector(isa.second);
        std::vector<T> outcome = input;
        kernel_executor(kernel_func, outcome);

        // Check for 4ULP accuracy
        for (size_t i = 0; i < expected.size(); i++) {
            if constexpr (std::is_same<T, float>::value)
                EXPECT_FLOAT_EQ(expected[i], outcome[i]);
            else
                EXPECT_DOUBLE_EQ(expected[i], outcome[i]);
        }
    }
}

TYPED_TEST(kernel_internal_test, math_func) {
    std::function<void(test_math_func_vec_type<TypeParam> & data)> set_test_data[] = {
        set_zero_data<TypeParam>, set_iota_data<TypeParam>,
        set_large_numbers_data<TypeParam>, set_very_large_numbers_data<TypeParam>,
        set_very_large_negative_numbers_data<TypeParam>};
    test_math_func_vec_type<TypeParam> data;
    da_int count = 0;
    std::vector<test_math_func_vec_type<TypeParam>> params;

    std::unordered_map<std::string, vectorization_type> isa_list = {
        {"avx", avx},
        {"avx2", avx2},
        // will trickle down to AVX2 where is AVX512 not available
        {"avx512", avx512}};

    for (auto &data_fun : set_test_data) {
        data_fun(data);

        // EXP KERNEL TEST
        std::cout << "EXP TEST" << std::endl;
        std::vector<TypeParam> expected = data.input;
        exp_kernel<TypeParam, scalar>(data.first_dim, data.second_dim, expected.data(),
                                      data.first_dim, data.multiplier, data.X_norm.data(),
                                      data.Y_norm.data());
        test_kernel_across_isa<TypeParam>(
            data.input, expected, isa_list, count,
            [](vectorization_type vt) {
                return select_exp_kernel_function<TypeParam>(vt);
            },
            [&data](auto kernel_func, std::vector<TypeParam> &outcome) {
                kernel_func(data.first_dim, data.second_dim, outcome.data(),
                            data.first_dim, data.multiplier, data.X_norm.data(),
                            data.Y_norm.data());
            });

        // POW KERNEL TEST
        std::cout << "POW TEST" << std::endl;
        expected = data.input;
        pow_kernel<TypeParam, scalar>(data.first_dim, data.second_dim, expected.data(),
                                      data.first_dim, data.coef0, data.power);
        test_kernel_across_isa<TypeParam>(
            data.input, expected, isa_list, count,
            [](vectorization_type vt) {
                return select_pow_kernel_function<TypeParam>(vt);
            },
            [&data](auto kernel_func, std::vector<TypeParam> &outcome) {
                kernel_func(data.first_dim, data.second_dim, outcome.data(),
                            data.first_dim, data.coef0, data.power);
            });

        // TANH KERNEL TEST
        std::cout << "TANH TEST" << std::endl;
        expected = data.input;
        tanh_kernel<TypeParam, scalar>(data.first_dim, data.second_dim, expected.data(),
                                       data.first_dim, data.coef0);
        test_kernel_across_isa<TypeParam>(
            data.input, expected, isa_list, count,
            [](vectorization_type vt) {
                return select_tanh_kernel_function<TypeParam>(vt);
            },
            [&data](auto kernel_func, std::vector<TypeParam> &outcome) {
                kernel_func(data.first_dim, data.second_dim, outcome.data(),
                            data.first_dim, data.coef0);
            });

        count++;
    }
}