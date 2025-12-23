/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../datests_cblas.hh"
#include "../utest_utils.hpp"
#include "aoclda_utils.h"
#include "da_syrk.hpp"
#include "da_syrk_test_data.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstring>

template <typename T> class SYRKTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> std::vector<SYRKParamType<T>> getParams() {
    std::vector<SYRKParamType<T>> params;
    GetSYRKData(params);
    return params;
}

template <typename T>
void check_syrk_answer(da_order order, da_uplo uplo, da_int n, da_int ld_result,
                       T epsilon, const std::vector<T> &result1,
                       const std::vector<T> &result2) {
    if (order == column_major) {
        if (uplo == da_upper) {
            for (da_int j = 0; j < n; j++) {
                for (da_int i = 0; i <= j; i++) {
                    EXPECT_NEAR(result1[i + ld_result * j], result2[i + ld_result * j],
                                epsilon);
                }
            }
        } else {
            for (da_int j = 0; j < n; j++) {
                for (da_int i = j; i < n; i++) {
                    EXPECT_NEAR(result1[i + ld_result * j], result2[i + ld_result * j],
                                epsilon);
                }
            }
        }
    } else if (order == row_major) {
        if (uplo == da_upper) {
            for (da_int i = 0; i < n; i++) {
                for (da_int j = i; j < n; j++) {
                    EXPECT_NEAR(result1[j + ld_result * i], result2[j + ld_result * i],
                                epsilon);
                }
            }
        } else {
            for (da_int i = 0; i < n; i++) {
                for (da_int j = 0; j <= i; j++) {
                    EXPECT_NEAR(result1[j + ld_result * i], result2[j + ld_result * i],
                                epsilon);
                }
            }
        }
    }
}

template <typename T> void test_functionality(const SYRKParamType<T> &param) {
    std::cout << "Functionality test: " << param.test_name << std::endl;

    if (!param.block_size_override.empty()) {
        EXPECT_EQ(
            da_debug_set("syrk.block_size_override", param.block_size_override.c_str()),
            da_status_success);
    }

    std::vector<T> C_da_syrk(param.n * param.ldC);
    fill_with_uniform_random(C_da_syrk, 2506);

    // compute blocked syrk
    EXPECT_EQ(TEST_ARCH::da_syrk(param.order, param.uplo, param.transpose, param.n,
                                 param.k, param.alpha, param.A.data(), param.ldA,
                                 param.beta, C_da_syrk.data(), param.ldC),
              da_status_success);

    char answer[100];
    if (!param.block_size_override.empty()) {
        EXPECT_EQ(da_debug_get("syrk.block_size", 100, answer), da_status_success);
        EXPECT_EQ(std::string(answer), param.block_size_override);
    }

    std::vector<T> C_blas_syrk(param.n * param.ldC);
    fill_with_uniform_random(C_blas_syrk, 2506);

    datest_blas::cblas_syrk(
        TEST_ARCH::da_utils::da_order_to_cblas_order(param.order),
        TEST_ARCH::da_utils::da_uplo_to_cblas_uplo(param.uplo),
        TEST_ARCH::da_utils::da_transpose_to_cblas_transpose(param.transpose), param.n,
        param.k, param.alpha, param.A.data(), param.ldA, param.beta, C_blas_syrk.data(),
        param.ldC);

    check_syrk_answer(param.order, param.uplo, param.n, param.ldC, param.epsilon,
                      C_da_syrk, C_blas_syrk);
}

class DoubleFunctionalityTest : public testing::TestWithParam<SYRKParamType<double>> {};
class FloatFunctionalityTest : public testing::TestWithParam<SYRKParamType<float>> {};

template <typename T> void PrintTo(const SYRKParamType<T> &param, ::std::ostream *os) {
    *os << param.test_name;
}

TEST_P(DoubleFunctionalityTest, ParameterizedTest) {
    const SYRKParamType<double> &p = GetParam();
    test_functionality(p);
}

TEST_P(FloatFunctionalityTest, ParameterizedTest) {
    const SYRKParamType<float> &p = GetParam();
    test_functionality(p);
}

INSTANTIATE_TEST_SUITE_P(SYRK_Functionality_Tests_Double, DoubleFunctionalityTest,
                         ::testing::ValuesIn(getParams<double>()));
INSTANTIATE_TEST_SUITE_P(SYRK_Functionality_Tests_Float, FloatFunctionalityTest,
                         ::testing::ValuesIn(getParams<float>()));

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SYRKTest, FloatTypes);

TYPED_TEST(SYRKTest, NoBlockingPath) {
    // Check we perform regular syrk for inputs where 1d blocking is not appropriate
    da_int m = 10;
    da_int n = 5;
    TypeParam alpha = 1.0;
    TypeParam beta = 0.0;

    std::vector<TypeParam> A(m * n);
    std::mt19937 gen(1234);
    std::uniform_real_distribution<TypeParam> dist(-1.0, 1.0);
    std::generate(A.begin(), A.end(), [&]() { return dist(gen); });

    std::vector<TypeParam> C(m * m, 0.0);
    std::vector<TypeParam> C_blas(m * m, 0.0);

    TypeParam epsilon = 100 * std::numeric_limits<TypeParam>::epsilon();
    // Set a block size override and check telemetry to see if it was actually used
    EXPECT_EQ(da_debug_set("syrk.block_size_override", "2"), da_status_success);

    // Check no transpose
    EXPECT_EQ(TEST_ARCH::da_syrk(column_major, da_upper, da_no_trans, m, n, alpha,
                                 A.data(), m, beta, C.data(), m),
              da_status_success);

    char answer[100];
    EXPECT_EQ(da_debug_get("syrk.block_size", 100, answer), da_status_success);
    // block size should be m meaning n_blocks == 1 and no blocking was performed
    EXPECT_EQ(std::string(answer), std::to_string(m));

    datest_blas::cblas_syrk(
        TEST_ARCH::da_utils::da_order_to_cblas_order(column_major),
        TEST_ARCH::da_utils::da_uplo_to_cblas_uplo(da_upper),
        TEST_ARCH::da_utils::da_transpose_to_cblas_transpose(da_no_trans), m, n, alpha,
        A.data(), m, beta, C_blas.data(), m);

    check_syrk_answer(column_major, da_upper, m, m, epsilon, C, C_blas);

    // Check transpose
    EXPECT_EQ(TEST_ARCH::da_syrk(column_major, da_upper, da_trans, m, n, alpha, A.data(),
                                 n, beta, C.data(), m),
              da_status_success);

    EXPECT_EQ(da_debug_get("syrk.block_size", 100, answer), da_status_success);
    // block size should be m meaning n_blocks == 1 and no blocking was performed
    EXPECT_EQ(std::string(answer), std::to_string(m));

    datest_blas::cblas_syrk(
        TEST_ARCH::da_utils::da_order_to_cblas_order(column_major),
        TEST_ARCH::da_utils::da_uplo_to_cblas_uplo(da_upper),
        TEST_ARCH::da_utils::da_transpose_to_cblas_transpose(da_trans), m, n, alpha,
        A.data(), n, beta, C_blas.data(), m);

    check_syrk_answer(column_major, da_upper, m, m, epsilon, C, C_blas);
}