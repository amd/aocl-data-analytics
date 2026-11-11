/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "bfloat16_helpers.hpp"
#include "quantized_backend.hpp"
#include "scalar_quantizer.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// Fixture: 3 samples x 3 features, chosen so scale/bias and every raw
// pre-round ratio are exact in floating point (no 2/255-style fraction),
// since x == vmax always maps to raw ratio 126.5 and x == vmin to -128.5 --
// exact round()-halfway cases that fractional scale/bias would land on
// either side of due to accumulated FP error.
//   feature 0: {0, 255, 50}     -> scale=1, bias=128.5
//   feature 1: {5, 5, 5}        -> constant, scale=1, bias=5
//   feature 2: {-510, 0, -200}  -> scale=2, bias=-253
// Expected codes round(clamp((x-bias)/scale,-128,127)):
//   sample0 {0,5,-510}   -> {-128, 0, -128}
//   sample1 {255,5,0}    -> { 127, 0,  127}
//   sample2 {50,5,-200}  -> { -79, 0,   27}

template <typename T> class ScalarQuantizerTest : public testing::Test {};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ScalarQuantizerTest, FloatTypes);

namespace {

float float_from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int8_t unpack_s4(const uint8_t *packed, da_int feature) {
    return TEST_ARCH::scalar_quantizer::int4_packing::load(packed, feature);
}

TEST(bfloat16HelpersTest, ConversionAndRoundToNearestEven) {
    EXPECT_EQ(da_bf16::to_bits(da_bf16::from_float(1.0f)), 0x3f80u);
    EXPECT_EQ(da_bf16::to_bits(da_bf16::from_float(-0.0f)), 0x8000u);

    // Exact halfway cases round toward an even retained least-significant bit.
    EXPECT_EQ(da_bf16::to_bits(da_bf16::from_float(float_from_bits(0x3f808000u))),
              0x3f80u);
    EXPECT_EQ(da_bf16::to_bits(da_bf16::from_float(float_from_bits(0x3f818000u))),
              0x3f82u);

    EXPECT_FLOAT_EQ(da_bf16::bf16_bits_to_f32(0x3f81u), 1.0078125f);
    EXPECT_EQ(da_bf16::to_bits(da_bf16::from_float(float_from_bits(0x7f800001u))),
              0x7fc0u);
}

template <typename T> std::vector<T> expected_scale() {
    std::vector<double> exp{1.0, 1.0, 2.0};
    return convert_vector<double, T>(exp);
}

template <typename T> std::vector<T> expected_bias() {
    std::vector<double> exp{128.5, 5.0, -253.0};
    return convert_vector<double, T>(exp);
}

} // namespace

TEST(QuantizedGemmTest, bfloat16BackendConvertsGathersAndComputes) {
    using backend = TEST_ARCH::da_quantized_backend::backend<da_bf16::storage_type>;
    constexpr da_int query_block_size = 3, n_rows = 2, n_cols = 2, n_features = 3;
    backend::workspace_type workspace;
    backend::resize_workspace(workspace, query_block_size, n_cols, n_features);
    const std::vector<float> queries{1.0f, 2.0f,  -1.0f, 0.0f, 0.0f,
                                     0.0f, -2.0f, 1.0f,  3.0f};
    for (size_t i = 0; i < queries.size(); i++)
        backend::set_query_value(workspace, i, queries[i]);
    const std::vector<float> code_values{1.0f, -2.0f, 3.0f, 2.0f, 1.0f, -1.0f};
    std::vector<da_bf16::storage_type> codes(code_values.size());
    for (size_t i = 0; i < code_values.size(); i++)
        codes[i] = da_bf16::from_float(code_values[i]);
    const std::vector<da_int> query_indices{12, 10};

    backend::gather_list_queries(workspace, query_indices.data(), 10, n_rows, n_features);
    backend::compute_products(workspace, codes.data(), n_rows, n_cols, n_features);

    EXPECT_FLOAT_EQ(workspace.dot_products[0], 5.0f);
    EXPECT_FLOAT_EQ(workspace.dot_products[1], -6.0f);
    EXPECT_FLOAT_EQ(workspace.dot_products[2], -6.0f);
    EXPECT_FLOAT_EQ(workspace.dot_products[3], 5.0f);
}

TEST(QuantizedHardwareSupport, OptimizedBackendSelection) {
    using namespace TEST_ARCH::da_quantized_backend;
    using TEST_ARCH::scalar_quantizer::code_kind;

#ifndef _WIN32
    EXPECT_FALSE(use_optimized_backend(code_kind::none, true));
    EXPECT_FALSE(use_optimized_backend(code_kind::int8, true));
    EXPECT_FALSE(use_optimized_backend(code_kind::int4, false));
    EXPECT_TRUE(use_optimized_backend(code_kind::int4, true));
    EXPECT_TRUE(use_optimized_backend(code_kind::bfloat16, false));
#else
    EXPECT_FALSE(use_optimized_backend(code_kind::none, true));
    EXPECT_FALSE(use_optimized_backend(code_kind::int8, true));
    EXPECT_FALSE(use_optimized_backend(code_kind::int4, true));
    EXPECT_FALSE(use_optimized_backend(code_kind::bfloat16, true));
#endif
}

TYPED_TEST(ScalarQuantizerTest, bfloat16IdentityEncoding) {
    using T = TypeParam;
    const std::vector<T> input{T(1.0), T(1.00390625), T(-2.5)};
    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> quantizer(
        TEST_ARCH::scalar_quantizer::scheme::direct_signed,
        TEST_ARCH::scalar_quantizer::code_kind::bfloat16);

    ASSERT_EQ(quantizer.train(row_major, input.data(), 1, 3, 3), da_status_success);
    ASSERT_EQ(quantizer.scale(), std::vector<T>({T(1), T(1), T(1)}));
    ASSERT_EQ(quantizer.bias(), std::vector<T>({T(0), T(0), T(0)}));

    std::vector<da_bf16::storage_type> codes(3);
    quantizer.quantize(row_major, input.data(), 1, 3, 3, codes.data(), 3);
    EXPECT_EQ(da_bf16::to_bits(codes[0]), 0x3f80u);
    EXPECT_EQ(da_bf16::to_bits(codes[1]), 0x3f80u);
    EXPECT_EQ(da_bf16::to_bits(codes[2]), 0xc020u);
}

TYPED_TEST(ScalarQuantizerTest, int4PacksOddFeatureRows) {
    using T = TypeParam;
    const std::vector<T> input{T(-8), T(7), T(-1), T(0), T(3)};
    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> quantizer(
        TEST_ARCH::scalar_quantizer::scheme::direct_signed,
        TEST_ARCH::scalar_quantizer::code_kind::int4);
    ASSERT_EQ(quantizer.train(row_major, nullptr, 0, 5, 5), da_status_success);

    std::vector<uint8_t> codes(3, 0xff);
    quantizer.quantize(row_major, input.data(), 1, 5, 5, codes.data(), 3);

    EXPECT_EQ(codes, std::vector<uint8_t>({0x78, 0x0f, 0x03}));
    for (da_int feature = 0; feature < 5; feature++)
        EXPECT_EQ(unpack_s4(codes.data(), feature), input[feature])
            << "feature " << feature;
    EXPECT_EQ(codes.back() & 0xf0, 0);
}

TYPED_TEST(ScalarQuantizerTest, int4PacksColumnMajorSamples) {
    using T = TypeParam;
    const std::vector<T> input{T(-8), T(7), T(0), T(1), T(-1), T(3), T(4), T(5), T(-4)};
    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> quantizer(
        TEST_ARCH::scalar_quantizer::scheme::direct_signed,
        TEST_ARCH::scalar_quantizer::code_kind::int4);
    ASSERT_EQ(quantizer.train(column_major, nullptr, 0, 3, 3), da_status_success);

    std::vector<uint8_t> codes(6, 0xff);
    quantizer.quantize(column_major, input.data(), 3, 3, 3, codes.data(), 2);

    EXPECT_EQ(codes, std::vector<uint8_t>({0x78, 0x00, 0xf1, 0x03, 0x54, 0x0c}));
}

TYPED_TEST(ScalarQuantizerTest, TrainRangesColumnMajor) {
    using T = TypeParam;

    std::vector<double> Xd{0.0, 255.0, 50.0, 5.0, 5.0, 5.0, -510.0, 0.0, -200.0};
    std::vector<T> X = convert_vector<double, T>(Xd);

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q;
    ASSERT_EQ(q.train(column_major, X.data(), 3, 3, 3), da_status_success);

    std::vector<T> exp_scale = expected_scale<T>();
    std::vector<T> exp_bias = expected_bias<T>();
    T tol = da_numeric::tolerance<T>::tol(T(100));

    ASSERT_EQ(q.scale().size(), static_cast<size_t>(3));
    EXPECT_ARR_NEAR(3, q.scale(), exp_scale, tol);
    EXPECT_ARR_NEAR(3, q.bias(), exp_bias, tol);
}

TYPED_TEST(ScalarQuantizerTest, TrainRangesRowMajor) {
    using T = TypeParam;

    std::vector<double> Xd{0.0, 5.0, -510.0, 255.0, 5.0, 0.0, 50.0, 5.0, -200.0};
    std::vector<T> X = convert_vector<double, T>(Xd);

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q;
    ASSERT_EQ(q.train(row_major, X.data(), 3, 3, 3), da_status_success);

    std::vector<T> exp_scale = expected_scale<T>();
    std::vector<T> exp_bias = expected_bias<T>();
    T tol = da_numeric::tolerance<T>::tol(T(100));

    EXPECT_ARR_NEAR(3, q.scale(), exp_scale, tol);
    EXPECT_ARR_NEAR(3, q.bias(), exp_bias, tol);
}

TYPED_TEST(ScalarQuantizerTest, quantizeColumnMajor) {
    using T = TypeParam;

    std::vector<double> Xd{0.0, 255.0, 50.0, 5.0, 5.0, 5.0, -510.0, 0.0, -200.0};
    std::vector<T> X = convert_vector<double, T>(Xd);

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q;
    ASSERT_EQ(q.train(column_major, X.data(), 3, 3, 3), da_status_success);

    std::vector<std::vector<int>> expected_codes{
        {-128, 0, -128}, {127, 0, 127}, {-79, 0, 27}};

    // quantize the whole block; codes column-major with ldc = n_samples = 3.
    std::vector<int8_t> codes(9);
    q.quantize(column_major, X.data(), 3, 3, 3, codes.data(), 3);
    for (da_int i = 0; i < 3; i++) {
        for (da_int d = 0; d < 3; d++) {
            EXPECT_EQ(static_cast<int>(codes[i + d * 3]), expected_codes[i][d])
                << "sample " << i << ", feature " << d;
        }
    }
}

TYPED_TEST(ScalarQuantizerTest, quantizeRowMajor) {
    using T = TypeParam;

    std::vector<double> Xd{0.0, 5.0, -510.0, 255.0, 5.0, 0.0, 50.0, 5.0, -200.0};
    std::vector<T> X = convert_vector<double, T>(Xd);

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q;
    ASSERT_EQ(q.train(row_major, X.data(), 3, 3, 3), da_status_success);

    std::vector<std::vector<int>> expected_codes{
        {-128, 0, -128}, {127, 0, 127}, {-79, 0, 27}};

    // quantize the whole block; codes row-major with ldc = n_features = 3.
    std::vector<int8_t> codes(9);
    q.quantize(row_major, X.data(), 3, 3, 3, codes.data(), 3);
    for (da_int i = 0; i < 3; i++) {
        for (da_int d = 0; d < 3; d++) {
            EXPECT_EQ(static_cast<int>(codes[i * 3 + d]), expected_codes[i][d])
                << "sample " << i << ", feature " << d;
        }
    }
}

TYPED_TEST(ScalarQuantizerTest, ConstantDimensionNoNanInf) {
    using T = TypeParam;

    std::vector<double> Xd{0.0, 255.0, 50.0, 5.0, 5.0, 5.0, -510.0, 0.0, -200.0};
    std::vector<T> X = convert_vector<double, T>(Xd);

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q;
    ASSERT_EQ(q.train(column_major, X.data(), 3, 3, 3), da_status_success);

    // Feature 1 is constant: scale must be a nonzero placeholder, never 0,
    // and no NaN/Inf should appear anywhere in scale/bias.
    EXPECT_NE(q.scale()[1], T(0));
    for (da_int d = 0; d < 3; d++) {
        EXPECT_TRUE(std::isfinite(static_cast<double>(q.scale()[d])));
        EXPECT_TRUE(std::isfinite(static_cast<double>(q.bias()[d])));
    }

    // Every value in the constant dimension quantizes to shifted-code 0.
    std::vector<int8_t> codes(9);
    q.quantize(column_major, X.data(), 3, 3, 3, codes.data(), 3);
    for (da_int i = 0; i < 3; i++)
        EXPECT_EQ(static_cast<int>(codes[i + 1 * 3]), 0) << "sample " << i;
}

TYPED_TEST(ScalarQuantizerTest, BoundaryClamping) {
    using T = TypeParam;

    std::vector<double> Xd{1.0, 5.0, -2.0, 3.0, 5.0, 4.0, 2.0, 5.0, 0.0};
    std::vector<T> X = convert_vector<double, T>(Xd);

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q;
    ASSERT_EQ(q.train(row_major, X.data(), 3, 3, 3), da_status_success);

    // Far outside the trained [1,3]/[5,5]/[-2,4] ranges: must clamp, not
    // overflow int8_t or produce garbage.
    std::vector<double> far_d{1000.0, 5.0, -1000.0};
    std::vector<T> far_values = convert_vector<double, T>(far_d);

    std::vector<int8_t> code(3);
    q.quantize(row_major, far_values.data(), 1, 3, 3, code.data(), 3);

    EXPECT_EQ(static_cast<int>(code[0]), 127);
    EXPECT_EQ(static_cast<int>(code[1]), 0);
    EXPECT_EQ(static_cast<int>(code[2]), -128);
}

// Fixture: 3 samples x 3 features, global min/max across every sample and
// feature combined is {0,255} -> scale=1, bias=128.5, broadcast to all 3
// features (unlike the per-dimension fixture above, where each feature has
// its own range).
//   feature 0: {0, 100, 50}
//   feature 1: {200, 5, 150}
//   feature 2: {255, 0, 10}

TYPED_TEST(ScalarQuantizerTest, TrainUniform) {
    using T = TypeParam;

    std::vector<double> Xd{0.0, 100.0, 50.0, 200.0, 5.0, 150.0, 255.0, 0.0, 10.0};
    std::vector<T> X = convert_vector<double, T>(Xd);

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q(
        TEST_ARCH::scalar_quantizer::scheme::uniform);
    ASSERT_EQ(q.train(column_major, X.data(), 3, 3, 3), da_status_success);

    T tol = da_numeric::tolerance<T>::tol(T(100));

    EXPECT_NEAR(q.scale()[0], T(1.0), tol);
    EXPECT_NEAR(q.bias()[0], T(128.5), tol);
}

TYPED_TEST(ScalarQuantizerTest, quantizeUniform) {
    using T = TypeParam;

    std::vector<double> Xd{0.0, 200.0, 255.0, 100.0, 5.0, 0.0, 50.0, 150.0, 10.0};
    std::vector<T> X = convert_vector<double, T>(Xd);

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q(
        TEST_ARCH::scalar_quantizer::scheme::uniform);
    ASSERT_EQ(q.train(row_major, X.data(), 3, 3, 3), da_status_success);

    std::vector<std::vector<int>> expected_codes{{-128, 72, 127}, {-29, -124, -128}};

    // quantize the first 2 rows as a block; codes row-major with ldc = 3.
    std::vector<int8_t> codes(6);
    q.quantize(row_major, X.data(), 2, 3, 3, codes.data(), 3);
    for (da_int i = 0; i < 2; i++) {
        for (da_int d = 0; d < 3; d++) {
            EXPECT_EQ(static_cast<int>(codes[i * 3 + d]), expected_codes[i][d])
                << "sample " << i << ", feature " << d;
        }
    }
}

TYPED_TEST(ScalarQuantizerTest, TrainDirectIgnoresData) {
    using T = TypeParam;

    // Data is nowhere near [0,255]: direct must not scan it.
    std::vector<double> Xd{-9999.0, 42.5, 123456.0, 7.0, -3.0, 1.0};
    std::vector<T> X = convert_vector<double, T>(Xd);

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q(
        TEST_ARCH::scalar_quantizer::scheme::direct_unsigned);
    ASSERT_EQ(q.train(row_major, X.data(), 2, 3, 3), da_status_success);

    T tol = da_numeric::tolerance<T>::tol(T(100));

    EXPECT_NEAR(q.scale()[0], T(1.0), tol);
    EXPECT_NEAR(q.bias()[0], T(128.0), tol);
}

TYPED_TEST(ScalarQuantizerTest, quantizeDirect) {
    using T = TypeParam;

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q(
        TEST_ARCH::scalar_quantizer::scheme::direct_unsigned);
    ASSERT_EQ(q.train(row_major, nullptr, 0, 3, 3), da_status_success);

    std::vector<double> xd{50.0, 300.0, -20.0};
    std::vector<T> x = convert_vector<double, T>(xd);

    std::vector<int8_t> code(3);
    q.quantize(row_major, x.data(), 1, 3, 3, code.data(), 3);

    EXPECT_EQ(static_cast<int>(code[0]), -78);
    EXPECT_EQ(static_cast<int>(code[1]), 127);
    EXPECT_EQ(static_cast<int>(code[2]), -128);
}

TYPED_TEST(ScalarQuantizerTest, TrainDirectSignedIgnoresData) {
    using T = TypeParam;

    std::vector<double> Xd{-9999.0, 42.5, 123456.0};
    std::vector<T> X = convert_vector<double, T>(Xd);

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q(
        TEST_ARCH::scalar_quantizer::scheme::direct_signed);
    ASSERT_EQ(q.train(row_major, X.data(), 1, 3, 3), da_status_success);

    T tol = da_numeric::tolerance<T>::tol(T(100));

    EXPECT_NEAR(q.scale()[0], T(1.0), tol);
    EXPECT_NEAR(q.bias()[0], T(0.0), tol);
}

TYPED_TEST(ScalarQuantizerTest, quantizeDirectSigned) {
    using T = TypeParam;

    TEST_ARCH::scalar_quantizer::scalar_quantizer<T> q(
        TEST_ARCH::scalar_quantizer::scheme::direct_signed);
    ASSERT_EQ(q.train(row_major, nullptr, 0, 3, 3), da_status_success);

    std::vector<double> xd{50.0, 200.0, -200.0};
    std::vector<T> x = convert_vector<double, T>(xd);

    std::vector<int8_t> code(3);
    q.quantize(row_major, x.data(), 1, 3, 3, code.data(), 3);

    EXPECT_EQ(static_cast<int>(code[0]), 50);
    EXPECT_EQ(static_cast<int>(code[1]), 127);
    EXPECT_EQ(static_cast<int>(code[2]), -128);
}
