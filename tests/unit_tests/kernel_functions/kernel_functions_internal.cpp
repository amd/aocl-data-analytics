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

TYPED_TEST(kernel_internal_test, math_func) {
    std::function<void(test_math_func_vec_type<TypeParam> & data)> set_test_data[] = {
        set_zero_data<TypeParam>, set_iota_data<TypeParam>,
        set_large_numbers_data<TypeParam>, set_very_large_numbers_data<TypeParam>};
    test_math_func_vec_type<TypeParam> data;
    da_int count = 0;
    std::vector<test_math_func_vec_type<TypeParam>> params;
    for (auto &data_fun : set_test_data) {
        data_fun(data);
        std::vector<TypeParam> expected = data.input;
        exp_kernel<TypeParam, scalar>(data.first_dim, data.second_dim, expected.data(),
                                      data.first_dim, data.multiplier);
        std::unordered_map<std::string, vectorization_type> isa_list;
        isa_list = {{"avx", avx},
                    {"avx2", avx2},
                    // will trickle down to AVX2 where is AVX512 not available
                    {"avx512", avx512}};
        for (const auto &isa : isa_list) {
            std::cout << "Dataset: " << std::to_string(count)
                      << ", vectorisation: " << isa.first << std::endl;
            auto exp_kernel_func = select_exp_kernel_function<TypeParam>(isa.second);
            std::vector<TypeParam> outcome = data.input;
            exp_kernel_func(data.first_dim, data.second_dim, outcome.data(),
                            data.first_dim, data.multiplier);
            // Check for 4ULP accuracy (this is stated accuracy in LIBM docs), EXPECT_*_EQ does exactly so
            for (size_t i = 0; i < expected.size(); i++) {
                if constexpr (std::is_same<TypeParam, float>::value)
                    EXPECT_FLOAT_EQ(expected[i], outcome[i]);
                else
                    EXPECT_DOUBLE_EQ(expected[i], outcome[i]);
            }
        }
        count++;
    }
}