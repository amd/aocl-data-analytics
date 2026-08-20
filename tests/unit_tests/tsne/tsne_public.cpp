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
 */

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "tsne_positive.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

// =============================================================================
// Typed test suite for float/double precision tests
// =============================================================================

template <typename T> class tsne_public_test : public testing::Test {};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(tsne_public_test, FloatTypes);

// =============================================================================
// Basic Public API Tests
// =============================================================================

TYPED_TEST(tsne_public_test, BasicEmbedding) {
    // Three well-separated clusters of 3 points each, row-major layout.
    // Samples 0-2 ≈ 0, samples 3-5 ≈ 10, samples 6-8 ≈ 20.
    const da_int n_samples = 9;
    const da_int n_features = 3;
    const da_int n_components = 2;
    const da_int k_neighbors = 2;
    // clang-format off
    TypeParam X[n_samples * n_features] = {
         0.1,  0.0, -0.1,
         0.0,  0.1,  0.1,
        -0.1, -0.1,  0.0,
        10.0, 10.1,  9.9,
        10.1,  9.9, 10.0,
         9.9, 10.0, 10.1,
        20.0, 20.1, 19.9,
        20.1, 19.9, 20.0,
        19.9, 20.0, 20.1
    };
    // clang-format on

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(2)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "theta", TypeParam(0)), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
    EXPECT_EQ(da_options_set(handle, "learning rate", TypeParam(10)), da_status_success);
    // Ensure we run exactly max_iter iterations
    EXPECT_EQ(da_options_set(handle, "min_grad_norm", TypeParam(0)), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_iter_without_progress", 0),
              da_status_success);
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X, n_features),
              da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    da_int dim = 6;
    TypeParam rinfo[6];
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo), da_status_success);
    EXPECT_EQ(rinfo[0], n_samples);
    EXPECT_EQ(rinfo[1], n_features);
    EXPECT_EQ(rinfo[2], n_components);
    EXPECT_EQ(rinfo[3], 1000);

    da_int emb_dim = n_samples * n_components;
    std::vector<TypeParam> embedding(emb_dim);
    EXPECT_EQ(da_handle_get_result(handle, da_tsne_embedding, &emb_dim, embedding.data()),
              da_status_success);
    for (TypeParam v : embedding) {
        EXPECT_TRUE(std::isfinite(v));
    }

    TypeParam kl_div = rinfo[4];
    TypeParam trust = tsne_metrics::compute_trustworthiness(
        X, embedding.data(), n_samples, n_features, n_components, k_neighbors);

    EXPECT_NEAR(kl_div, TypeParam(0), TypeParam(0.001));
    EXPECT_EQ(trust, TypeParam(1));

    da_handle_destroy(&handle);
}

TYPED_TEST(tsne_public_test, ComputeWithoutData) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_no_data);
    da_handle_destroy(&handle);
}

TYPED_TEST(tsne_public_test, InvalidInputs) {
    const da_int n_samples = 6;
    const da_int n_features = 3;
    const da_int ldx = n_samples; // column-major default
    TypeParam X[n_samples * n_features];
    for (da_int i = 0; i < n_samples * n_features; ++i)
        X[i] = static_cast<TypeParam>(i);

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);

    // Null data pointer
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, (TypeParam *)nullptr, ldx),
              da_status_invalid_pointer);

    // Invalid array dimensions (zero sizes or impossible shape).
    EXPECT_EQ(da_tsne_set_data(handle, 0, n_features, X, ldx),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, 0, X, ldx),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_tsne_set_data(handle, 1, n_features, X, 1),
              da_status_invalid_array_dimension);

    // Leading dimension must satisfy layout constraints.
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X, n_samples - 1),
              da_status_invalid_leading_dimension);

    TypeParam Y_init[n_samples * 2] = {};
    // Init embedding cannot be set before data is available.
    EXPECT_EQ(da_tsne_set_init_embedding(handle, Y_init, 2), da_status_no_data);

    EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(2)), da_status_success);
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X, ldx), da_status_success);

    da_int dim = 6;
    TypeParam rinfo[6];
    // Results are unavailable before a successful compute.
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo), da_status_no_data);

    // Supplied init path must fail compute when embedding is missing.
    EXPECT_EQ(da_options_set_string(handle, "init", "supplied"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "theta", TypeParam(0)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(2)), da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_no_data);

    // Supplied embedding dimension mismatch: set embedding with n_components=2,
    // then change n_components to 1 before compute.
    TypeParam Y_mismatch[n_samples * 2] = {};
    EXPECT_EQ(da_tsne_set_init_embedding(handle, Y_mismatch, n_samples),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", 1), da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_invalid_input);
    EXPECT_EQ(da_options_set_int(handle, "n_components", 2), da_status_success);

    // Null pointer for supplied init embedding.
    EXPECT_EQ(da_tsne_set_init_embedding(handle, (TypeParam *)nullptr, n_samples),
              da_status_invalid_pointer);

    // Valid random init path should compute successfully.
    EXPECT_EQ(da_options_set_string(handle, "init", "random"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 10), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    da_int small_dim = 2;
    // rinfo query with undersized output buffer.
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &small_dim, rinfo),
              da_status_invalid_array_dimension);

    small_dim = 1;
    TypeParam emb_buf[1];
    // Embedding query with undersized output buffer.
    EXPECT_EQ(da_handle_get_result(handle, da_tsne_embedding, &small_dim, emb_buf),
              da_status_invalid_array_dimension);

    dim = 100;
    TypeParam big_buf[100];
    // Unknown result identifier should return unknown query.
    EXPECT_EQ(da_handle_get_result(handle, static_cast<da_result>(9999), &dim, big_buf),
              da_status_unknown_query);

    // get_result rejects null size/output pointers.
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, (da_int *)nullptr, rinfo),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, (TypeParam *)nullptr),
              da_status_invalid_input);

    da_int int_result[6];
    dim = 6;
    // Type-mismatched output buffer should be rejected.
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, int_result),
              da_status_unknown_query);

    da_handle_destroy(&handle);
}

TYPED_TEST(tsne_public_test, NonFiniteInput) {
    const da_int n_samples = 6;
    const da_int n_features = 3;
    const da_int ldx = n_features; // row-major

    auto run_case = [&](TypeParam special) {
        TypeParam X[n_samples * n_features];
        for (da_int i = 0; i < n_samples * n_features; ++i)
            X[i] = static_cast<TypeParam>(i + 1) * TypeParam(0.1);
        X[4] = special;

        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "check data", "yes"), da_status_success);
        EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X, ldx),
                  da_status_invalid_input);
        da_handle_destroy(&handle);
    };
    run_case(std::numeric_limits<TypeParam>::quiet_NaN());
    run_case(std::numeric_limits<TypeParam>::infinity());
}

TYPED_TEST(tsne_public_test, SuppliedInitializationPositivePath) {
    const da_int n_samples = 4;
    const da_int n_features = 2;
    const da_int n_components = 2;
    const da_int ldx = n_features; // row-major
    const da_int ldy = n_components;

    TypeParam X[n_samples * n_features];
    for (da_int i = 0; i < n_samples * n_features; ++i)
        X[i] = static_cast<TypeParam>(i + 1);

    // Trivial supplied embedding: all points identical. Zero gradient
    TypeParam Y_init[n_samples * n_components] = {2, -3, 2, -3, 2, -3, 2, -3};
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(1)), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "init", "supplied"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "learning rate", TypeParam(200)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "storage order", "row-major"), da_status_success);
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X, ldx), da_status_success);
    EXPECT_EQ(da_tsne_set_init_embedding(handle, Y_init, ldy), da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    da_int emb_dim = n_samples * n_components;
    std::vector<TypeParam> embedding(emb_dim);
    EXPECT_EQ(da_handle_get_result(handle, da_tsne_embedding, &emb_dim, embedding.data()),
              da_status_success);
    for (da_int i = 0; i < emb_dim; ++i)
        EXPECT_EQ(embedding[i], Y_init[i]);

    da_handle_destroy(&handle);

    // Repeat with column-major data and column-major supplied embedding
    TypeParam X_col[n_samples * n_features];
    for (da_int i = 0; i < n_samples; ++i)
        for (da_int j = 0; j < n_features; ++j)
            X_col[i + j * n_samples] = X[i * n_features + j];

    TypeParam Y_init_col[n_samples * n_components];
    for (da_int i = 0; i < n_samples; ++i)
        for (da_int j = 0; j < n_components; ++j)
            Y_init_col[i + j * n_samples] = Y_init[i * n_components + j];

    da_handle handle_col = nullptr;
    ASSERT_EQ(da_handle_init<TypeParam>(&handle_col, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_int(handle_col, "n_components", n_components),
              da_status_success);
    EXPECT_EQ(da_options_set(handle_col, "perplexity", TypeParam(1)), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_col, "init", "supplied"), da_status_success);
    EXPECT_EQ(da_options_set(handle_col, "learning rate", TypeParam(200)),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle_col, "storage order", "column-major"),
              da_status_success);
    EXPECT_EQ(da_tsne_set_data(handle_col, n_samples, n_features, X_col, n_samples),
              da_status_success);
    EXPECT_EQ(da_tsne_set_init_embedding(handle_col, Y_init_col, n_samples),
              da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle_col), da_status_success);

    std::vector<TypeParam> emb_col(emb_dim);
    EXPECT_EQ(
        da_handle_get_result(handle_col, da_tsne_embedding, &emb_dim, emb_col.data()),
        da_status_success);
    for (da_int i = 0; i < emb_dim; ++i)
        EXPECT_EQ(emb_col[i], Y_init_col[i]);

    da_handle_destroy(&handle_col);
}

TYPED_TEST(tsne_public_test, PerplexityClampingWarning) {
    const da_int n_samples = 6;
    const da_int n_features = 3;
    TypeParam X[n_samples * n_features];
    for (da_int i = 0; i < n_samples * n_features; ++i)
        X[i] = static_cast<TypeParam>(i) * TypeParam(0.1);

    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(10)), da_status_success);

    da_status status = da_tsne_set_data(handle, n_samples, n_features, X, n_samples);
    EXPECT_EQ(status, da_status_incompatible_options)
        << "Perplexity > n_samples-1 should trigger clamping warning";

    EXPECT_EQ(da_options_set(handle, "theta", TypeParam(0)), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 10), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    da_handle_destroy(&handle);
}

TYPED_TEST(tsne_public_test, NComponentsClampingWarning) {
    const da_int n_samples = 10;
    const da_int n_features = 1;
    TypeParam X[n_samples];
    for (da_int i = 0; i < n_samples; ++i)
        X[i] = static_cast<TypeParam>(i);

    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", 2), da_status_success);

    da_status status = da_tsne_set_data(handle, n_samples, n_features, X, n_samples);
    EXPECT_EQ(status, da_status_incompatible_options)
        << "n_components > n_features should trigger clamping warning";

    EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(3)), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 10), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    da_int dim = 6;
    TypeParam rinfo[6];
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo), da_status_success);
    EXPECT_EQ(rinfo[2], 1) << "n_components should have been clamped to 1";

    da_handle_destroy(&handle);
}

TYPED_TEST(tsne_public_test, BadHandleNullAndWrongType) {
    da_handle handle = nullptr;
    TypeParam X[6] = {1, 2, 3, 4, 5, 6};
    TypeParam Y_init[4] = {0};

    EXPECT_EQ(da_tsne_set_data(handle, 2, 3, X, 2), da_status_handle_not_initialized);
    EXPECT_EQ(da_tsne_set_init_embedding(handle, Y_init, 2),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_handle_not_initialized);

    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_tsne_set_data(handle, 2, 3, X, 2), da_status_invalid_handle_type);
    EXPECT_EQ(da_tsne_set_init_embedding(handle, Y_init, 2),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_invalid_handle_type);

    da_handle_destroy(&handle);
}

TEST(TSNEPublic, IncorrectHandlePrecision) {
    double X_d[6] = {1, 2, 3, 4, 5, 6};
    float X_s[6] = {1, 2, 3, 4, 5, 6};
    double Y_d[4] = {0.1, -0.2, 0.3, -0.4};
    float Y_s[4] = {0.1, -0.2, 0.3, -0.4};

    da_handle handle_d = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_tsne_set_data_s(handle_d, 2, 3, X_s, 2), da_status_wrong_type);
    EXPECT_EQ(da_tsne_set_init_embedding_s(handle_d, Y_s, 2), da_status_wrong_type);
    EXPECT_EQ(da_tsne_compute_s(handle_d), da_status_wrong_type);
    da_handle_destroy(&handle_d);

    da_handle handle_s = nullptr;
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_tsne_set_data_d(handle_s, 2, 3, X_d, 2), da_status_wrong_type);
    EXPECT_EQ(da_tsne_set_init_embedding_d(handle_s, Y_d, 2), da_status_wrong_type);
    EXPECT_EQ(da_tsne_compute_d(handle_s), da_status_wrong_type);
    da_handle_destroy(&handle_s);
}

TYPED_TEST(tsne_public_test, MultipleSequentialComputes) {
    const da_int n_samples = 6;
    const da_int n_features = 3;
    const da_int n_components = 2;
    const da_int k_neighbors = 2;

    // Two well-separated clusters: indices 0-2 near origin, 3-5 far away
    TypeParam X1[] = {
        0.0,  0.0, 0.0, 0.1,  0.0, 0.0, 0.0,  0.1, 0.0,
        10.0, 0.0, 0.0, 10.1, 0.0, 0.0, 10.0, 0.1, 0.0,
    };

    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(2)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "max_iter", da_int(500)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "theta", TypeParam(0)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "seed", da_int(42)), da_status_success);
    // Ensure we run exactly max_iter iterations
    EXPECT_EQ(da_options_set(handle, "min_grad_norm", TypeParam(0)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "n_iter_without_progress", da_int(0)),
              da_status_success);
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X1, n_features),
              da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    da_int dim = 6;
    TypeParam rinfo1[6];
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo1), da_status_success);
    EXPECT_EQ(rinfo1[0], TypeParam(n_samples));
    EXPECT_EQ(rinfo1[1], TypeParam(n_features));
    EXPECT_EQ(rinfo1[2], TypeParam(n_components));
    EXPECT_EQ(rinfo1[3], TypeParam(500));

    da_int emb_dim = n_samples * n_components;
    std::vector<TypeParam> emb(emb_dim);
    EXPECT_EQ(da_handle_get_result(handle, da_tsne_embedding, &emb_dim, emb.data()),
              da_status_success);

    TypeParam trust1 = tsne_metrics::compute_trustworthiness(
        X1, emb.data(), n_samples, n_features, n_components, k_neighbors);
    EXPECT_EQ(trust1, TypeParam(1));

    // Recompute with more iterations: quality should not degrade
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 1000), da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    TypeParam rinfo2[6];
    dim = 6;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo2), da_status_success);
    EXPECT_EQ(rinfo2[0], TypeParam(n_samples));
    EXPECT_EQ(rinfo2[1], TypeParam(n_features));
    EXPECT_EQ(rinfo2[2], TypeParam(n_components));
    EXPECT_EQ(rinfo2[3], TypeParam(1000));

    EXPECT_EQ(da_handle_get_result(handle, da_tsne_embedding, &emb_dim, emb.data()),
              da_status_success);
    TypeParam trust2 = tsne_metrics::compute_trustworthiness(
        X1, emb.data(), n_samples, n_features, n_components, k_neighbors);
    EXPECT_EQ(trust2, TypeParam(1));

    // Replace data: different two-cluster layout (z-axis sep.)
    TypeParam X2[] = {
        0.0, 0.0, 0.0,  0.0, 0.1, 0.0,  0.0, 0.0, 0.1,
        0.0, 0.0, 10.0, 0.0, 0.1, 10.0, 0.0, 0.0, 10.1,
    };

    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X2, n_features),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 1000), da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    TypeParam rinfo3[6];
    dim = 6;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo3), da_status_success);
    EXPECT_EQ(rinfo3[0], TypeParam(n_samples));
    EXPECT_EQ(rinfo3[1], TypeParam(n_features));
    EXPECT_EQ(rinfo3[2], TypeParam(n_components));
    EXPECT_EQ(rinfo3[3], TypeParam(1000));

    EXPECT_EQ(da_handle_get_result(handle, da_tsne_embedding, &emb_dim, emb.data()),
              da_status_success);
    TypeParam trust3 = tsne_metrics::compute_trustworthiness(
        X2, emb.data(), n_samples, n_features, n_components, k_neighbors);
    EXPECT_EQ(trust3, TypeParam(1));

    da_handle_destroy(&handle);
}

// =============================================================================
// Quality behavior across iterations
// =============================================================================

template <typename T>
static void run_tsne_get_kl_and_embedding(const T *X, da_int n_samples, da_int n_features,
                                          da_int n_components, T perplexity,
                                          da_int max_iter, T theta,
                                          const char *storage_order, da_int ldx,
                                          T &kl_out, std::vector<T> &embedding_out) {
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "perplexity", perplexity), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", max_iter), da_status_success);
    EXPECT_EQ(da_options_set(handle, "theta", theta), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", storage_order),
              da_status_success);
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X, ldx), da_status_success);
    EXPECT_EQ(da_tsne_compute<T>(handle), da_status_success);

    da_int info_dim = 6;
    T rinfo[6];
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &info_dim, rinfo),
              da_status_success);
    kl_out = rinfo[4];

    da_int emb_dim = n_samples * n_components;
    embedding_out.resize(emb_dim);
    EXPECT_EQ(
        da_handle_get_result(handle, da_tsne_embedding, &emb_dim, embedding_out.data()),
        da_status_success);

    da_handle_destroy(&handle);
}

TYPED_TEST(tsne_public_test, ImproveWithMoreIterations) {
    std::string data_file = std::string(DATA_DIR) + "/tsne_data/circles_data.csv";
    std::vector<TypeParam> X;
    da_int n_samples, n_features;
    ASSERT_TRUE(da_test::read_csv_data(data_file, X, n_samples, n_features, row_major));

    TypeParam kl_100, kl_500;
    std::vector<TypeParam> emb_100, emb_500;

    run_tsne_get_kl_and_embedding<TypeParam>(X.data(), n_samples, n_features, 2,
                                             TypeParam(10), 100, TypeParam(0),
                                             "row-major", n_features, kl_100, emb_100);
    run_tsne_get_kl_and_embedding<TypeParam>(X.data(), n_samples, n_features, 2,
                                             TypeParam(10), 500, TypeParam(0),
                                             "row-major", n_features, kl_500, emb_500);

    EXPECT_LT(kl_500, kl_100) << "KL divergence should decrease with more iterations: "
                              << "KL@100=" << kl_100 << " KL@500=" << kl_500;

    TypeParam trust_100 = tsne_metrics::compute_trustworthiness(
        X.data(), emb_100.data(), n_samples, n_features, 2, 5);
    TypeParam trust_500 = tsne_metrics::compute_trustworthiness(
        X.data(), emb_500.data(), n_samples, n_features, 2, 5);

    EXPECT_GE(trust_500, trust_100)
        << "Trustworthiness should improve with more iterations: "
        << "trust@100=" << trust_100 << " trust@500=" << trust_500;
}

TYPED_TEST(tsne_public_test, ColumnMajorVsRowMajor) {
    const da_int n_samples = 10;
    const da_int n_features = 3;
    const da_int n_components = 2;

    TypeParam X_row[n_samples * n_features];
    for (da_int i = 0; i < n_samples; ++i)
        for (da_int j = 0; j < n_features; ++j)
            X_row[i * n_features + j] =
                static_cast<TypeParam>(i * 10 + j) * TypeParam(0.1);

    TypeParam X_col[n_samples * n_features];
    for (da_int i = 0; i < n_samples; ++i)
        for (da_int j = 0; j < n_features; ++j)
            X_col[i + j * n_samples] = X_row[i * n_features + j];

    TypeParam kl_row, kl_col;
    std::vector<TypeParam> emb_row, emb_col_raw;
    run_tsne_get_kl_and_embedding<TypeParam>(X_row, n_samples, n_features, n_components,
                                             TypeParam(3), 50, TypeParam(0), "row-major",
                                             n_features, kl_row, emb_row);
    run_tsne_get_kl_and_embedding<TypeParam>(
        X_col, n_samples, n_features, n_components, TypeParam(3), 50, TypeParam(0),
        "column-major", n_samples, kl_col, emb_col_raw);

    // Convert column-major output to row-major for comparison.
    da_int emb_dim = n_samples * n_components;
    std::vector<TypeParam> emb_col(emb_dim);
    for (da_int i = 0; i < n_samples; ++i)
        for (da_int j = 0; j < n_components; ++j)
            emb_col[i * n_components + j] = emb_col_raw[i + j * n_samples];

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-4) : TypeParam(1e-10);
    for (da_int i = 0; i < emb_dim; ++i)
        EXPECT_NEAR(emb_row[i], emb_col[i], tol)
            << "Row-major vs column-major mismatch at index " << i;
}

TYPED_TEST(tsne_public_test, RowMajorNonCompactLeadingDimension) {
    const da_int n_samples = 9;
    const da_int n_features = 3;
    const da_int n_components = 2;
    const da_int ldx_padded = n_features + 2;

    // clang-format off
    std::vector<TypeParam> X_compact = {
         0.1,  0.0, -0.1,
         0.0,  0.1,  0.1,
        -0.1, -0.1,  0.0,
        10.0, 10.1,  9.9,
        10.1,  9.9, 10.0,
         9.9, 10.0, 10.1,
        20.0, 20.1, 19.9,
        20.1, 19.9, 20.0,
        19.9, 20.0, 20.1
    };
    // clang-format on

    // Embed the same data in a padded row-major array with ldx > n_features
    std::vector<TypeParam> X_padded(n_samples * ldx_padded);
    for (da_int i = 0; i < n_samples; ++i)
        for (da_int j = 0; j < n_features; ++j)
            X_padded[i * ldx_padded + j] = X_compact[i * n_features + j];

    auto run = [&](const TypeParam *X, da_int ldx, std::vector<TypeParam> &emb_out) {
        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(2)), da_status_success);
        EXPECT_EQ(da_options_set(handle, "theta", TypeParam(0)), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "max_iter", 200), da_status_success);
        EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X, ldx),
                  da_status_success);
        EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

        da_int emb_dim = n_samples * n_components;
        emb_out.resize(emb_dim);
        EXPECT_EQ(
            da_handle_get_result(handle, da_tsne_embedding, &emb_dim, emb_out.data()),
            da_status_success);
        da_handle_destroy(&handle);
    };

    std::vector<TypeParam> emb_compact, emb_padded;
    run(X_compact.data(), n_features, emb_compact);
    run(X_padded.data(), ldx_padded, emb_padded);

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-6) : TypeParam(1e-10);
    for (da_int i = 0; i < n_samples * n_components; ++i)
        EXPECT_NEAR(emb_compact[i], emb_padded[i], tol)
            << "Compact vs padded row-major mismatch at index " << i;
}

TYPED_TEST(tsne_public_test, SameSeedReproducibility) {
    const da_int n_samples = 10;
    const da_int n_features = 3;
    const da_int n_components = 2;

    TypeParam X[n_samples * n_features];
    for (da_int i = 0; i < n_samples * n_features; ++i)
        X[i] = static_cast<TypeParam>(i) * TypeParam(0.1);

    std::vector<TypeParam> emb1, emb2;
    TypeParam kl1, kl2;

    auto check_same_seed = [&](TypeParam theta, const char *mode_label) {
        run_tsne_get_kl_and_embedding<TypeParam>(X, n_samples, n_features, n_components,
                                                 TypeParam(3), 100, theta, "row-major",
                                                 n_features, kl1, emb1);
        run_tsne_get_kl_and_embedding<TypeParam>(X, n_samples, n_features, n_components,
                                                 TypeParam(3), 100, theta, "row-major",
                                                 n_features, kl2, emb2);

        const TypeParam emb_tol =
            std::is_same_v<TypeParam, float> ? TypeParam(1e-6) : TypeParam(1e-12);
        const TypeParam kl_tol =
            std::is_same_v<TypeParam, float> ? TypeParam(5e-7) : TypeParam(1e-12);
        ASSERT_EQ(emb1.size(), emb2.size());
        for (size_t i = 0; i < emb1.size(); ++i)
            EXPECT_NEAR(emb1[i], emb2[i], emb_tol)
                << mode_label
                << " same-seed runs should be numerically consistent at index " << i;
        EXPECT_NEAR(kl1, kl2, kl_tol)
            << mode_label
            << " same-seed runs should produce numerically consistent KL divergence";
    };

    check_same_seed(TypeParam(0), "Exact");
    check_same_seed(TypeParam(0.5), "Barnes-Hut");
}

// When every input point is identical, PCA produces an all-zero embedding
// (zero covariance -> zero principal components). The gradient is identically
// zero (every y_i - y_j = 0), so the embedding must remain exactly zero
// throughout optimisation.
TYPED_TEST(tsne_public_test, AllZerosData) {
    const da_int n_samples = 6;
    const da_int n_features = 3;
    const da_int n_components = 2;
    TypeParam X[n_samples * n_features] = {};
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "init", "pca"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(2)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "theta", TypeParam(0.5)), da_status_success);
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X, n_features),
              da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    da_int emb_dim = n_samples * n_components;
    std::vector<TypeParam> emb(emb_dim);
    EXPECT_EQ(da_handle_get_result(handle, da_tsne_embedding, &emb_dim, emb.data()),
              da_status_success);

    for (da_int i = 0; i < emb_dim; ++i)
        EXPECT_EQ(emb[i], TypeParam(0))
            << "Embedding should be exactly zero at index " << i;

    da_int info_dim = 6;
    TypeParam rinfo[6];
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &info_dim, rinfo),
              da_status_success);
    const TypeParam kl_tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-6) : TypeParam(1e-12);
    EXPECT_NEAR(rinfo[4], TypeParam(0), kl_tol)
        << "KL divergence should be near zero for constant input";

    da_handle_destroy(&handle);
}

// n_components equals n_features, max_iter = 2, init = supplied
// The embedding should be the same as the initial embedding
TYPED_TEST(tsne_public_test, NComponentsEqualsNFeatures) {
    const da_int n_samples = 2;
    const da_int n_features = 2;
    const da_int n_components = 2;
    TypeParam X[n_samples * n_features] = {0.0, 1.0, 10.0, 11.0};
    TypeParam Y_init[n_samples * n_components] = {-1.0, 2.0, 3.0, -4.0};

    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "init", "supplied"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(1)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "theta", TypeParam(0)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "early exaggeration", TypeParam(1)),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "learning rate", TypeParam(10)), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 2), da_status_success);
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X, n_features),
              da_status_success);
    EXPECT_EQ(da_tsne_set_init_embedding(handle, Y_init, n_components),
              da_status_success);
    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    da_int dim = 6;
    TypeParam rinfo[6];
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo), da_status_success);
    const TypeParam kl_tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-6) : TypeParam(1e-12);
    EXPECT_NEAR(rinfo[4], TypeParam(0), kl_tol);

    da_int emb_dim = n_samples * n_components;
    std::vector<TypeParam> embedding(emb_dim);
    EXPECT_EQ(da_handle_get_result(handle, da_tsne_embedding, &emb_dim, embedding.data()),
              da_status_success);
    const TypeParam emb_tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-5) : TypeParam(1e-12);
    for (da_int i = 0; i < emb_dim; ++i)
        EXPECT_NEAR(embedding[i], Y_init[i], emb_tol)
            << "embedding changed at index " << i;

    da_handle_destroy(&handle);
}

// 2x1: the minimum valid array (check_2D_array requires n_samples >= 2)
TYPED_TEST(tsne_public_test, SmallestValidCases) {
    const da_int n_samples = 2;
    const da_int n_features = 1;
    const da_int n_components = 1;
    TypeParam X[2] = {0.0, 10.0};

    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "init", "random"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "perplexity", TypeParam(1)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "theta", TypeParam(0)), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 2), da_status_success);

    da_status status = da_tsne_set_data(handle, n_samples, n_features, X, n_features);
    EXPECT_EQ(status, da_status_success);

    EXPECT_EQ(da_tsne_compute<TypeParam>(handle), da_status_success);

    da_int dim = 6;
    TypeParam rinfo[6];
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo), da_status_success);
    const TypeParam kl_tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-6) : TypeParam(1e-12);
    EXPECT_NEAR(rinfo[4], TypeParam(0), kl_tol);

    da_handle_destroy(&handle);
}

// =============================================================================
// Mixed precision tests
// =============================================================================

TYPED_TEST(tsne_public_test, MixedPrecision) {
    // Run t-SNE with supplied initial embedding both with and
    // without mixed precision and check that the quality metrics are comparable.
    // Element-by-element comparison is not meaningful for t-SNE because the
    // non-convex optimization can converge to different (rotated/translated)
    // embeddings of equal quality when the trajectory is perturbed by float rounding.
    const da_int n_samples = 60;
    const da_int n_features = 10;
    const da_int n_components = 3;
    const da_int max_iter = 500;
    const da_int k_neighbors = 5;

    // Deterministic synthetic data: two clusters
    std::vector<TypeParam> X(n_samples * n_features);
    std::mt19937_64 rng(123);
    std::normal_distribution<TypeParam> normal(0.0, 1.0);
    for (da_int i = 0; i < n_samples; ++i)
        for (da_int j = 0; j < n_features; ++j)
            X[i * n_features + j] = normal(rng) + (i < n_samples / 2 ? 0.0 : 5.0);

    // Deterministic initial embedding
    std::vector<TypeParam> Y_init(n_samples * n_components);
    for (da_int i = 0; i < n_samples * n_components; ++i)
        Y_init[i] = normal(rng) * 1e-4;

    auto run = [&](bool mixed_precision, std::vector<TypeParam> &emb_out,
                   TypeParam &kl_out, da_int &n_iter_out, da_int &lp_n_iter_out,
                   da_status &status_out) {
        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_tsne), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "perplexity", static_cast<TypeParam>(5.0)),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "theta", static_cast<TypeParam>(0.0)),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "max_iter", max_iter), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "init", "supplied"), da_status_success);
        EXPECT_EQ(da_options_set(handle, "min_grad_norm", static_cast<TypeParam>(0.0)),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_iter_without_progress", 0),
                  da_status_success);
        if (mixed_precision) {
            EXPECT_EQ(da_options_set_string(handle, "mixed precision", "yes"),
                      da_status_success);
            EXPECT_EQ(da_options_set_int(handle, "low precision max_iter", 200),
                      da_status_success);
        }
        EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X.data(), n_features),
                  da_status_success);
        EXPECT_EQ(da_tsne_set_init_embedding(handle, Y_init.data(), n_components),
                  da_status_success);
        da_status status = da_tsne_compute<TypeParam>(handle);

        // Catch the case where mixed precision is not supported for this type (e.g. float, non-Zen6)
        da_int len{100};
        char arch[100], ns[100];
        ASSERT_EQ(da_get_arch_info(&len, arch, ns), da_status_success);
        const bool is_zen6 = (std::strcmp(arch, "zen6") == 0);
        if (mixed_precision && std::is_same_v<TypeParam, float> && !is_zen6) {
            EXPECT_EQ(status, da_status_invalid_option);
            da_handle_destroy(&handle);
            status_out = status;
            return;
        }

        EXPECT_EQ(status, da_status_success);

        da_int emb_dim = n_samples * n_components;
        emb_out.resize(emb_dim);
        EXPECT_EQ(
            da_handle_get_result(handle, da_tsne_embedding, &emb_dim, emb_out.data()),
            da_status_success);

        da_int dim = 6;
        TypeParam rinfo[6];
        EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, rinfo), da_status_success);
        kl_out = rinfo[4];
        n_iter_out = static_cast<da_int>(rinfo[3]);
        lp_n_iter_out = static_cast<da_int>(rinfo[5]);

        da_handle_destroy(&handle);
        status_out = da_status_success;
        return;
    };

    std::vector<TypeParam> emb_baseline, emb_mixed;
    TypeParam kl_baseline, kl_mixed;
    da_int n_iter_base, n_iter_mixed, lp_iter_base, lp_iter_mixed;
    da_status status_base, status_mixed;

    // Exercise each vectorization path. Empty string clears the override so the
    // dispatcher picks the best ISA available on the host.
    const std::vector<std::string> isa_list = {"", "scalar", "avx", "avx2", "avx512"};
    for (const auto &isa : isa_list) {
        std::cout << "MixedPrecision: tsne.isa='" << isa << "'" << std::endl;
        EXPECT_EQ(da_debug_set("tsne.isa", isa.c_str()), da_status_success);

        run(false, emb_baseline, kl_baseline, n_iter_base, lp_iter_base, status_base);
        run(true, emb_mixed, kl_mixed, n_iter_mixed, lp_iter_mixed, status_mixed);

        if (status_mixed == da_status_invalid_option) {
            // Mixed precision not supported for this type (e.g. float): test is not applicable
            SUCCEED() << "Mixed precision not supported for this type, skipping test";
            continue;
        }

        EXPECT_EQ(lp_iter_base, 0);
        EXPECT_EQ(n_iter_base, max_iter);
        EXPECT_EQ(lp_iter_mixed, 200);
        EXPECT_EQ(n_iter_mixed, max_iter);

        // Compare quality metrics: trustworthiness and KL divergence
        TypeParam trust_base = tsne_metrics::compute_trustworthiness(
            X.data(), emb_baseline.data(), n_samples, n_features, n_components,
            k_neighbors);
        TypeParam trust_mixed = tsne_metrics::compute_trustworthiness(
            X.data(), emb_mixed.data(), n_samples, n_features, n_components, k_neighbors);

        // Both should achieve good trustworthiness on this easy two-cluster dataset
        EXPECT_GE(trust_base, 0.85);
        EXPECT_GE(trust_mixed, 0.85);

        // Mixed precision should not dramatically degrade quality
        EXPECT_NEAR(trust_base, trust_mixed, 0.05);
        EXPECT_NEAR(kl_baseline, kl_mixed, 0.11);
    }

    // Clear the ISA override so subsequent tests are not affected
    EXPECT_EQ(da_debug_set("tsne.isa", ""), da_status_success);
}

// Mixed precision on a float handle requires a lower precision type.
// - On Zen5 and below there is no _Float16 LP path so the option is rejected.
// - On Zen6 the LP path runs, but the PCA initialization and the
//   Barnes-Hut (theta > 0) branches do not support _Float16, so the engine
//   must report da_status_incompatible_options.
TEST(TSNEMixedPrecision, FloatIllegalOption) {
    const da_int n_samples = 6;
    const da_int n_features = 3;
    float X[n_samples * n_features];
    for (da_int i = 0; i < n_samples * n_features; ++i)
        X[i] = static_cast<float>(i + 1);

    da_int len{100};
    char arch[100], ns[100];
    ASSERT_EQ(da_get_arch_info(&len, arch, ns), da_status_success);
    const bool is_zen6 = (std::strcmp(arch, "zen6") == 0);

    if (!is_zen6) {
        // Zen5 or below: no _Float16 LP path is compiled in, so the
        // "mixed precision" option itself is rejected by the engine.
        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init_s(&handle, da_handle_tsne), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "perplexity", 2.0f), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "mixed precision", "yes"),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
        EXPECT_EQ(da_tsne_set_data_s(handle, n_samples, n_features, X, n_features),
                  da_status_success);
        EXPECT_EQ(da_tsne_compute_s(handle), da_status_invalid_option);
        da_handle_destroy(&handle);
        return;
    } else {
        // Zen6: init="pca" combined with mixed precision is not supported.
        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init_s(&handle, da_handle_tsne), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "perplexity", 2.0f), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "mixed precision", "yes"),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "init", "pca"), da_status_success);
        EXPECT_EQ(da_options_set(handle, "theta", 0.0f), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
        EXPECT_EQ(da_tsne_set_data_s(handle, n_samples, n_features, X, n_features),
                  da_status_success);
        EXPECT_EQ(da_tsne_compute_s(handle), da_status_incompatible_options);
        da_handle_destroy(&handle);

        // Zen6: theta > 0 (Barnes-Hut / KNN path) combined with mixed precision
        // is not supported.

        ASSERT_EQ(da_handle_init_s(&handle, da_handle_tsne), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "perplexity", 2.0f), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "mixed precision", "yes"),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "init", "random"), da_status_success);
        EXPECT_EQ(da_options_set(handle, "theta", 0.5f), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
        EXPECT_EQ(da_tsne_set_data_s(handle, n_samples, n_features, X, n_features),
                  da_status_success);
        EXPECT_EQ(da_tsne_compute_s(handle), da_status_incompatible_options);
        da_handle_destroy(&handle);
    }
}

// =============================================================================
// Positive tests - Quality comparison against target solution
// =============================================================================

typedef struct tsne_param_t {
    std::string test_name; // Name of the ctest test
    std::string data_name; // Name of the dataset file
    da_int n_components;   // Embedding dimensions
    double perplexity;     // Perplexity parameter
    da_int max_iter;       // Maximum number of iterations
    double theta;          // Barnes-Hut theta (0 for exact)
    std::string init;      // Initialization method
    da_int seed;           // Random seed
    da_int k_neighbors;    // k for neighbor metrics
    double target_trust;   // target solution trustworthiness
    double target_kl_div;  // target solution KL divergence
    // Scale applied to the default absolute tolerance in tsne_positive.hpp.
    double check_tol_scale{1.0}; // multiplier applied to 0.01 absolute tolerance
} tsne_param_t;

// target solution values obtained from generate_reference.py
// clang-format off
const tsne_param_t tsne_param_pos[] = {
    // --- 1D exact (theta = 0) ---
    {"iris_exact_1d", "iris", 1, 30.0, 500, 0.0, "pca", 42, 10, 0.963747, 0.363504, 2.0},
    {"circles_exact_1d", "circles", 1, 10.0, 500, 0.0, "pca", 42, 5, 0.962217, 0.993861, 8.0},
    // --- 1D Barnes-Hut (theta > 0) ---
    {"twoclust_bh_1d", "twoclust", 1, 10.0, 500, 0.5, "pca", 42, 5, 0.919295, -1.0, 3.0 + 4.5},
    // --- 2D exact (theta = 0) ---
    {"iris_exact_2d", "iris", 2, 30.0, 1000, 0.0, "pca", 42, 10, 0.989031, 0.122057, 2.0},
    {"blobs_exact_2d", "blobs", 2, 20.0, 500, 0.0, "pca", 123, 10, 0.97184, 0.408497, 3.0},
    {"circles_exact_2d", "circles", 2, 10.0, 500, 0.0, "pca", 42, 5, 0.998196, 0.292971, 9.0},
    {"lowrank_exact_2d", "lowrank", 2, 30.0, 500, 0.0, "pca", 42, 5, 0.979143, 0.051566, 3.0},
    {"highcorr_exact_2d", "highcorr", 2, 10.0, 500, 0.0, "pca", 42, 5, 0.996795, 0.088316, 2.0},
    {"diffscales_exact_2d", "diffscales", 2, 10.0, 500, 0.0, "pca", 42, 5, 0.994952, 0.088038},
    {"moderate_exact_2d", "moderate", 2, 30.0, 500, 0.0, "pca", 42, 10, 0.830615, 0.644016, 6.0},
    // --- 2D Barnes-Hut (theta > 0) ---
    {"iris_barnes_hut", "iris", 2, 30.0, 500, 0.5, "pca", 42, 10, 0.989948, 0.127948},
    {"iris_barnes_hut_theta03", "iris", 2, 30.0, 500, 0.3, "pca", 42, 10, 0.991103, 0.120829, 4.0},
    {"blobs_barnes_hut", "blobs", 2, 30.0, 500, 0.5, "pca", 42, 10, 0.973038, 0.243888},
    {"circles_barnes_hut", "circles", 2, 10.0, 500, 0.5, "pca", 42, 5, 0.997391, 0.336932, 6.0},
    {"mnist_subset_barnes_hut_theta08", "mnist_subset", 2, 30.0, 1500, 0.8, "pca", 42, 10, 0.974921, 0.465035, 5.0},
    // --- 3D exact (theta = 0) ---
    {"iris_exact_3d", "iris", 3, 30.0, 500, 0.0, "pca", 42, 10, 0.984882, 0.418823},
    {"blobs_exact_3d", "blobs", 3, 20.0, 500, 0.0, "pca", 42, 10, 0.973041, 0.689296},
    {"twoclust_exact_3d", "twoclust", 3, 10.0, 500, 0.0, "pca", 42, 5, 0.851731, 1.269542},
    {"mnist_subset_exact_3d", "mnist_subset", 3, 30.0, 500, 0.0, "pca", 42, 10, 0.987643, 0.300049, 3.0},
    {"moderate_exact_3d", "moderate", 3, 20.0, 500, 0.0, "pca", 42, 10, 0.625929, 2.352057},
    // --- 3D Barnes-Hut (theta > 0) ---
    {"iris_bh_3d", "iris", 3, 30.0, 500, 0.5, "pca", 42, 10, 0.980734, 0.373762},
    {"blobs_barnes_hut_3d", "blobs", 3, 30.0, 500, 0.5, "pca", 42, 10, 0.982748, 0.185707},
    {"twoclust_bh_3d", "twoclust", 3, 10.0, 500, 0.5, "pca", 42, 5, 0.830513, 1.620211},
    {"mnist_subset_bh_3d", "mnist_subset", 3, 30.0, 500, 0.5, "pca", 42, 10, 0.986732, 0.340441, 7.0},
    // --- Large dataset (600x20, exercises PCA randomized SVD path: max(n,p)>500, npc<min(n,p)/5) ---
    {"large_blobs_bh_2d", "large_blobs", 2, 30.0, 500, 0.5, "pca", 123, 10, 0.972489, 0.739299, 3.0},
};
// clang-format on

class tsne_positive : public testing::TestWithParam<tsne_param_t> {};

void PrintTo(const tsne_param_t &param, ::std::ostream *os) { *os << param.test_name; }

TEST_P(tsne_positive, Double) {
    const tsne_param_t &param = GetParam();
    test_tsne_quality<double>(param.data_name, param.n_components, param.perplexity,
                              param.max_iter, param.theta, param.init, param.seed,
                              param.k_neighbors, param.target_trust, param.target_kl_div,
                              param.check_tol_scale);
}

TEST_P(tsne_positive, Single) {
    const tsne_param_t &param = GetParam();
    test_tsne_quality<float>(param.data_name, param.n_components,
                             static_cast<float>(param.perplexity), param.max_iter,
                             static_cast<float>(param.theta), param.init, param.seed,
                             param.k_neighbors, static_cast<float>(param.target_trust),
                             static_cast<float>(param.target_kl_div),
                             static_cast<float>(param.check_tol_scale));
}

INSTANTIATE_TEST_SUITE_P(tsne_pos_suite, tsne_positive,
                         testing::ValuesIn(tsne_param_pos));
