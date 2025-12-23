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

#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <stdio.h>
#include <string.h>

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "binary_tree.hpp"
#include "da_error.hpp"
#include "da_std.hpp"
#include "da_vector.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

// Given a vector ind_in of length n, and a vector dist_n, return into two arrays dist_out and ind_out
// the sorted distances and sorted indices, respectively.
template <typename T>
void sort_dist_ind(da_int n, T *dist_in, da_int *ind_in, T *dist_out, da_int *ind_out) {
    // We sort with respect to partial distances and then we use the sorted array to reorder the array of indices.
    std::vector<da_int> perm_vector(n);
    TEST_ARCH::da_std::iota(perm_vector.begin(), perm_vector.end(), 0);

    std::stable_sort(perm_vector.begin(), perm_vector.end(),
                     [&](da_int i, da_int j) { return dist_in[i] < dist_in[j]; });

    for (da_int i = 0; i < n; i++) {
        ind_out[i] = ind_in[perm_vector[i]];
        dist_out[i] = dist_out[perm_vector[i]];
    }
}

template <typename T> class KNTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(KNTest, FloatTypes);

TYPED_TEST(KNTest, k_neighbors_small) {

    da_int n_samples = 10;
    da_int n_features = 2;
    da_int lda = 10;
    da_int k = 4;
    std::vector<double> A_double = {0.0,  5.0,  6.0, 0.2,   0.1, 10.0, -0.15,
                                    -0.1, 5.5,  5.0, 0.0,   7.0, 6.01, 0.2,
                                    -0.1, 10.0, 0.1, -0.12, 5.5, 6.0};

    std::vector<TypeParam> A = convert_vector<double, TypeParam>(A_double);

    std::vector<da_int> k_ind(n_samples * k);
    std::vector<TypeParam> k_dist(n_samples * k);

    std::vector<TypeParam> sorted_dist(n_samples * k);
    std::vector<da_int> sorted_ind(n_samples * k);
    std::vector<TypeParam> sorted_dist_exp(n_samples * k);
    std::vector<da_int> sorted_ind_exp(n_samples * k);

    std::vector<da_int> k_ind_exp{3, 4, 7, 6, 5, 2, 9, 8, 1, 9, 5, 8, 4, 7,
                                  6, 0, 3, 7, 6, 0, 1, 8, 9, 2, 3, 4, 7, 0,
                                  3, 4, 6, 0, 1, 9, 5, 2, 1, 8, 5, 2};
    std::vector<double> k_dist_exp_d{
        0.08,   0.02,   0.0244, 0.0325, 34,     2,      1,      2.5,    2,      1,
        32,     0.5,    0.1,    0.1924, 0.1325, 0.08,   0.1,    0.0404, 0.1025, 0.02,
        34,     40.5,   41,     32,     0.1325, 0.1025, 0.0509, 0.0325, 0.1924, 0.0404,
        0.0509, 0.0244, 2.5,    0.5,    40.5,   0.5,    1,      0.5,    41,     1};

    std::vector<TypeParam> k_dist_exp = convert_vector<double, TypeParam>(k_dist_exp_d);

    da_errors::da_error_t *err =
        new da_errors::da_error_t(da_errors::action_t::DA_RECORD);

    da_int leaf_size = 2;
    auto tree = TEST_ARCH::da_binary_tree::kd_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)0.0);

    EXPECT_EQ(tree.k_neighbors(n_samples, n_features, nullptr, 0, k, k_ind.data(),
                               k_dist.data(), err),
              da_status_success);

    // Sort the distances and indices since the k-d tree does not guarantee that they are sorted
    for (da_int i = 0; i < n_samples; i++) {
        sort_dist_ind(k, k_dist.data() + i * k, k_ind.data() + i * k,
                      sorted_dist.data() + i * k, sorted_ind.data() + i * k);
        sort_dist_ind(k, k_dist_exp.data() + i * k, k_ind_exp.data() + i * k,
                      sorted_dist_exp.data() + i * k, sorted_ind_exp.data() + i * k);
    }

    EXPECT_ARR_EQ(n_samples * k, sorted_ind.data(), sorted_ind_exp.data(), 1, 1, 0, 0)
        << "k-d tree indices do not match expected values for small test";
    EXPECT_ARR_NEAR(n_samples * k, sorted_dist.data(), sorted_dist_exp.data(),
                    100 * std::numeric_limits<TypeParam>::epsilon())
        << "k-d tree distances do not match expected values for small test";

    // Now use the same k-d tree but treat A as a different matrix
    k = 5;
    std::vector<da_int> k_ind2(n_samples * k);
    std::vector<TypeParam> k_dist2(n_samples * k);
    sorted_dist.resize(n_samples * k);
    sorted_ind.resize(n_samples * k);
    sorted_dist_exp.resize(n_samples * k);
    sorted_ind_exp.resize(n_samples * k);

    std::vector<da_int> k_ind_exp2{3, 4, 7, 6, 0, 1, 8, 2, 9, 5, 1, 9, 2, 8, 5, 3, 4,
                                   7, 6, 0, 3, 4, 7, 6, 0, 1, 9, 2, 8, 5, 3, 4, 7, 6,
                                   0, 3, 4, 7, 6, 0, 1, 9, 2, 8, 5, 1, 9, 2, 8, 5};

    std::vector<double> k_dist_exp_d2{
        0.08,   0.02,        0.0244, 0.0325, 0,      0,           2.5,    2,
        1,      34,          2,      1,      0,      0.5,         32,     6.66134e-18,
        0.1,    0.1924,      0.1325, 0.08,   0.1,    1.66533e-18, 0.0404, 0.1025,
        0.02,   34,          41,     32,     40.5,   0,           0.1325, 0.1025,
        0.0509, 1.66533e-18, 0.0325, 0.1924, 0.0404, 1.34337e-18, 0.0509, 0.0244,
        2.5,    0.5,         0.5,    0,      40.5,   1,           0,      1,
        0.5,    41};

    std::vector<TypeParam> k_dist_exp2 = convert_vector<double, TypeParam>(k_dist_exp_d2);

    auto tree2 = TEST_ARCH::da_binary_tree::kd_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)0.0);

    EXPECT_EQ(tree2.k_neighbors(n_samples, n_features, A.data(), lda, k, k_ind2.data(),
                                k_dist2.data(), err),
              da_status_success);

    // Sort the distances and indices since the k-d tree does not guarantee that they are sorted
    for (da_int i = 0; i < n_samples; i++) {
        sort_dist_ind(k, k_dist2.data() + i * k, k_ind2.data() + i * k,
                      sorted_dist.data() + i * k, sorted_ind.data() + i * k);
        sort_dist_ind(k, k_dist_exp2.data() + i * k, k_ind_exp2.data() + i * k,
                      sorted_dist_exp.data() + i * k, sorted_ind_exp.data() + i * k);
    }

    EXPECT_ARR_EQ(n_samples * k, sorted_ind.data(), sorted_ind_exp.data(), 1, 1, 0, 0)
        << "k-d tree indices do not match expected values for small test 2";
    EXPECT_ARR_NEAR(n_samples * k, sorted_dist.data(), sorted_dist_exp.data(),
                    100 * std::numeric_limits<TypeParam>::epsilon())
        << "k-d tree distances do not match expected values for small test 2";

    // Now repeat with a ball tree
    k = 4;
    auto tree3 = TEST_ARCH::da_binary_tree::ball_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)0.0);

    EXPECT_EQ(tree3.k_neighbors(n_samples, n_features, nullptr, 0, k, k_ind.data(),
                                k_dist.data(), err),
              da_status_success);

    // Sort the distances and indices since the k-d tree does not guarantee that they are sorted
    for (da_int i = 0; i < n_samples; i++) {
        sort_dist_ind(k, k_dist.data() + i * k, k_ind.data() + i * k,
                      sorted_dist.data() + i * k, sorted_ind.data() + i * k);
        sort_dist_ind(k, k_dist_exp.data() + i * k, k_ind_exp.data() + i * k,
                      sorted_dist_exp.data() + i * k, sorted_ind_exp.data() + i * k);
    }

    EXPECT_ARR_EQ(n_samples * k, sorted_ind.data(), sorted_ind_exp.data(), 1, 1, 0, 0)
        << "ball tree indices do not match expected values for small test";
    EXPECT_ARR_NEAR(n_samples * k, sorted_dist.data(), sorted_dist_exp.data(),
                    100 * std::numeric_limits<TypeParam>::epsilon())
        << "ball tree distances do not match expected values for small test";

    // Now use the same ball tree but treat A as a different matrix
    k = 5;
    sorted_dist.resize(n_samples * k);
    sorted_ind.resize(n_samples * k);
    sorted_dist_exp.resize(n_samples * k);
    sorted_ind_exp.resize(n_samples * k);

    auto tree4 = TEST_ARCH::da_binary_tree::ball_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)0.0);

    EXPECT_EQ(tree4.k_neighbors(n_samples, n_features, A.data(), lda, k, k_ind2.data(),
                                k_dist2.data(), err),
              da_status_success);

    // Sort the distances and indices since the k-d tree does not guarantee that they are sorted
    for (da_int i = 0; i < n_samples; i++) {
        sort_dist_ind(k, k_dist2.data() + i * k, k_ind2.data() + i * k,
                      sorted_dist.data() + i * k, sorted_ind.data() + i * k);
        sort_dist_ind(k, k_dist_exp2.data() + i * k, k_ind_exp2.data() + i * k,
                      sorted_dist_exp.data() + i * k, sorted_ind_exp.data() + i * k);
    }

    EXPECT_ARR_EQ(n_samples * k, sorted_ind.data(), sorted_ind_exp.data(), 1, 1, 0, 0)
        << "ball tree indices do not match expected values for small test 2";
    EXPECT_ARR_NEAR(n_samples * k, sorted_dist.data(), sorted_dist_exp.data(),
                    100 * std::numeric_limits<TypeParam>::epsilon())
        << "ball tree distances do not match expected values for small test 2";

    delete err;
}

TYPED_TEST(KNTest, k_neighbors_large) {

    da_int n_samples = 39;
    da_int n_features = 1;
    da_int lda = n_samples;
    da_int k = 2;

    std::vector<TypeParam> A(n_samples);
    std::iota(A.begin(), A.end(), 0);

    std::vector<da_int> k_ind(n_samples * k);
    std::vector<TypeParam> k_dist(n_samples * k);

    std::vector<da_int> k_ind_exp(n_samples * k);
    std::vector<TypeParam> k_dist_exp(n_samples * k);

    for (da_int i = 1; i < n_samples - 1; i++) {
        k_ind_exp[i * k] = i - 1;
        k_ind_exp[i * k + 1] = i + 1;
        k_dist_exp[i * k] = 1;
        k_dist_exp[i * k + 1] = 1;
    }
    k_ind_exp[0] = 1;
    k_ind_exp[1] = 2;
    k_dist_exp[0] = 1;
    k_dist_exp[1] = 4;
    k_ind_exp[(n_samples - 1) * k] = n_samples - 3;
    k_ind_exp[(n_samples - 1) * k + 1] = n_samples - 2;
    k_dist_exp[(n_samples - 1) * k] = 4;
    k_dist_exp[(n_samples - 1) * k + 1] = 1;

    da_errors::da_error_t *err =
        new da_errors::da_error_t(da_errors::action_t::DA_RECORD);

    da_int leaf_size = 5;
    auto tree = TEST_ARCH::da_binary_tree::kd_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)0.0);

    EXPECT_EQ(tree.k_neighbors(n_samples, n_features, nullptr, 0, k, k_ind.data(),
                               k_dist.data(), err),
              da_status_success);

    // In-place sort to allow for different ordering of stored indices in neighbors
    for (da_int i = 0; i < n_samples; i++) {
        if (k_ind[i * k] > k_ind[i * k + 1]) {
            std::swap(k_ind[i * k], k_ind[i * k + 1]);
            std::swap(k_dist[i * k], k_dist[i * k + 1]);
        }
    }

    EXPECT_ARR_EQ(n_samples, k_ind.data(), k_ind_exp.data(), 1, 1, 0, 0)
        << "k-d tree indices do not match expected values for large test";
    EXPECT_ARR_EQ(n_samples, k_dist.data(), k_dist_exp.data(), 1, 1, 0, 0)
        << "k-d tree distances do not match expected values for large test";

    auto tree2 = TEST_ARCH::da_binary_tree::ball_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)0.0);

    EXPECT_EQ(tree2.k_neighbors(n_samples, n_features, nullptr, 0, k, k_ind.data(),
                                k_dist.data(), err),
              da_status_success);

    // In-place sort to allow for different ordering of stored indices in neighbors
    for (da_int i = 0; i < n_samples; i++) {
        if (k_ind[i * k] > k_ind[i * k + 1]) {
            std::swap(k_ind[i * k], k_ind[i * k + 1]);
            std::swap(k_dist[i * k], k_dist[i * k + 1]);
        }
    }

    EXPECT_ARR_EQ(n_samples, k_ind.data(), k_ind_exp.data(), 1, 1, 0, 0)
        << "ball tree indices do not match expected values for large test";
    EXPECT_ARR_EQ(n_samples, k_dist.data(), k_dist_exp.data(), 1, 1, 0, 0)
        << "ball tree distances do not match expected values for large test";

    delete err;
}