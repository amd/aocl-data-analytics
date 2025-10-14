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

// Test code for k-d tree construction (radius neighbors and k neighbors functionality is tested in
// radius_neighbors_internal and k_neighbors_internal)

#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "binary_tree.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

typedef struct params_t {
    std::string test_name;
    std::vector<double> A;
    std::vector<da_int> expected_indices;
    da_int n_samples;
    da_int n_features;
    da_int lda;
    da_int leaf_size;
    da_metric metric = da_euclidean;
    double p = 2.0;
} params;

const params_t ball_tree_params[] = {
    {"ball_tree_test1",
     {0.0, 1.0, 2.0, 3.0},
     {2, 3, 0, 1},
     4,
     1,
     4,
     2,
     da_euclidean,
     2.0},
    {"ball_tree_test2",
     {2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 3.0},
     {0, 1, 2, 3},
     4,
     2,
     4,
     20,
     da_euclidean,
     2.0},
    {"ball_tree_test3",
     {1.1, -1.0, 1.0, -1.0, 0.0, 0.0, -4.0, 4.1, 1.0, -1.0, -1.0, 1.0, 3.3, -3.3, -0.01,
      0.1},
     {4, 3, 6, 1, 5, 7, 0, 2},
     8,
     2,
     8,
     1,
     da_euclidean,
     2.0},
    {"ball_tree_test4",
     {1.1, -1.0, 1.0, -1.0, 0.1, -0.1, -4.0, 4.1, 1.0, -1.0, -1.0, 1.0, 3.3, -3.3, -0.01,
      0.1},
     {6, 3, 1, 5, 7, 2, 4, 0},
     8,
     2,
     8,
     1,
     da_euclidean,
     2.0},
    {"ball_tree_test5",
     {1.1, -1.0, 1.0, -1.0, 0.0, 0.0, -4.0, 4.1, 1.0, -1.0, -1.0, 1.0, 3.3, -3.3, -0.01,
      0.1},
     {4, 3, 6, 1, 5, 7, 0, 2},
     8,
     2,
     8,
     1,
     da_sqeuclidean,
     2.0},
    {"ball_tree_test6",
     {1.1, -1.0, 1.0, -1.0, 0.1, -0.1, -4.0, 4.1, 1.0, -1.0, -1.0, 1.0, 3.3, -3.3, -0.01,
      0.1},
     {6, 3, 1, 5, 7, 2, 4, 0},
     8,
     2,
     8,
     1,
     da_sqeuclidean,
     2.0},
    {"ball_tree_test7",
     {1.1, -1.0, 1.0, -1.0, 0.0, 0.0, -4.0, 4.1, 1.0, -1.0, -1.0, 1.0, 3.3, -3.3, -0.01,
      0.1},
     {6, 3, 1, 5, 7, 2, 4, 0},
     8,
     2,
     8,
     1,
     da_manhattan,
     2.0},
    {"ball_tree_test8",
     {1.1, -1.0, 1.0, -1.0, 0.1, -0.1, -4.0, 4.1, 1.0, -1.0, -1.0, 1.0, 3.3, -3.3, -0.01,
      0.1},
     {6, 3, 1, 5, 7, 2, 4, 0},
     8,
     2,
     8,
     1,
     da_minkowski,
     2.0001},
    {"ball_tree_test9",
     {1.1, -1.0, 1.0, -1.0, 0.1, -0.1, -4.0, 4.1, 1.0, -1.0, -1.0, 1.0, 3.3, -3.3, -0.01,
      0.1},
     {1, 3, 5, 6, 4, 2, 0, 7},
     8,
     2,
     8,
     4,
     da_euclidean,
     2.0},
    {"ball_tree_test10",
     {1.1, -1.0, 1.0, -1.0, 0.1, -0.1, -4.0, 4.1, 1.0, -1.0, -1.0, 1.0, 3.3, -3.3, -0.01,
      0.1},
     {1, 3, 5, 6, 4, 2, 0, 7},
     8,
     2,
     8,
     4,
     da_minkowski,
     2.0001},
    {"ball_tree_test11",
     {0.0, 1.0, 2.0, 3.0},
     {2, 3, 0, 1},
     4,
     1,
     4,
     2,
     da_manhattan,
     2.0},
};

const params_t kd_tree_params[] = {
    {"kd_tree_test1", {0.0, 1.0, 0.0, 2.0}, {0, 1}, 2, 2, 2, 2},
    {"kd_tree_test2", {0.0, 1.0, 0.0, 2.0}, {0, 1}, 2, 2, 2, 1},
    {"kd_tree_test3",
     {5.0, 2.0, 1.0, 3.0, 7.0, 4.0, 6.0, 0.0},
     {7, 2, 1, 3, 5, 0, 6, 4},
     8,
     1,
     8,
     1,
     da_euclidean,
     2.0},
    {"kd_tree_test4",
     {5.0, 8.0, 3.0, 1.0, 2.0, 4.0, 7.0, 0.0, 6.0, 5.0, 1.0, 4.0, 8.0, 3.0, 7.0, 0.0, 2.0,
      6.0},
     {7, 4, 2, 3, 5, 6, 1, 8, 0},
     9,
     2,
     9,
     1,
     da_euclidean,
     2.0},
    {"kd_tree_test5",
     {7.3, 6.0, 5.0, 4.0, 3.1, 2.0, 1.0, 0.0, 0.1, 1.0, 2.0, 3.1, 4.0, 5.0, 6.0, 6.3},
     {7, 6, 5, 4, 3, 2, 0, 1},
     8,
     2,
     8,
     1,
     da_euclidean,
     2.0},
    {"kd_tree_test6",
     {5.0, 2.0, 1.0, 3.0, 7.0, 4.0, 6.0, 0.0},
     {7, 2, 1, 3, 5, 0, 6, 4},
     8,
     1,
     8,
     1,
     da_minkowski,
     2.00001},
    {"kd_tree_test7",
     {5.0, 2.0, 1.0, 3.0, 7.0, 4.0, 6.0, 0.0},
     {7, 2, 1, 3, 5, 0, 6, 4},
     8,
     1,
     8,
     1,
     da_manhattan,
     2.0},
};

class kd_treeSmallTests : public testing::TestWithParam<params> {};

template <class T> void test_kd_tree(const params pr) {
    std::vector<T> A = convert_vector<double, T>(pr.A);
    da_int n_samples = pr.n_samples;
    da_int n_features = pr.n_features;
    da_int lda = pr.lda;
    da_int leaf_size = pr.leaf_size;

    auto tree = TEST_ARCH::da_binary_tree::kd_tree<T>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (T)2.0);

    // Check if the indices in the k-d tree match the expected indices, allowing for consecutive pairs
    // to be swapped because of the way the k-d tree is built and terminates with some leaves of size 2
    // meaning that depending on the nth_element implementation, the order of such pairs may vary
    da_int i = 0;

    while (i < n_samples) {
        if (pr.expected_indices[i] == tree.get_indices()[i]) {
            i += 1;
            continue;
        } else if (pr.expected_indices[i] == tree.get_indices()[i + 1] &&
                   pr.expected_indices[i + 1] == tree.get_indices()[i]) {
            i += 2;
            continue;
        } else {
            FAIL() << "k-d tree indices do not match expected values for small test";
        }
    }
}

void PrintTo(const params &param, ::std::ostream *os) { *os << param.test_name; }

TEST_P(kd_treeSmallTests, kd_tree_double) {
    const params &pr = GetParam();
    test_kd_tree<double>(pr);
}

TEST_P(kd_treeSmallTests, kd_tree_float) {
    const params &pr = GetParam();
    test_kd_tree<float>(pr);
}

INSTANTIATE_TEST_SUITE_P(kd_treeSuite, kd_treeSmallTests,
                         testing::ValuesIn(kd_tree_params));

class ball_treeSmallTests : public testing::TestWithParam<params> {};

template <class T> void test_ball_tree(const params pr) {
    std::vector<T> A = convert_vector<double, T>(pr.A);
    da_int n_samples = pr.n_samples;
    da_int n_features = pr.n_features;
    da_int lda = pr.lda;
    da_int leaf_size = pr.leaf_size;
    da_metric metric = pr.metric;
    T p = (T)pr.p;

    auto tree = TEST_ARCH::da_binary_tree::ball_tree<T>(n_samples, n_features, A.data(),
                                                        lda, leaf_size, metric, p);

    // Check if the indices in the ball tree match the expected indices

    for (da_int j = 0; j < n_samples; j++)
        std::cout << tree.get_indices()[j] << " ";
    std::cout << std::endl;

    EXPECT_ARR_EQ(n_samples, tree.get_indices().data(), pr.expected_indices.data(), 1, 1,
                  0, 0);
}

TEST_P(ball_treeSmallTests, ball_tree_double) {
    const params &pr = GetParam();
    test_ball_tree<double>(pr);
}

TEST_P(ball_treeSmallTests, ball_tree_float) {
    const params &pr = GetParam();
    test_ball_tree<float>(pr);
}

INSTANTIATE_TEST_SUITE_P(ball_treeSuite, ball_treeSmallTests,
                         testing::ValuesIn(ball_tree_params));

template <typename T> class typed_tree_tests : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(typed_tree_tests, FloatTypes);

TYPED_TEST(typed_tree_tests, kd_treeLargeTest) {
    da_int n_samples = 500;
    da_int n_features = 3;
    da_int lda = n_samples;
    da_int leaf_size = 1;

    std::vector<double> A_in(n_samples * n_features);
    std::vector<da_int> expected_indices(n_samples);

    for (da_int i = 0; i < n_samples; i++) {
        for (da_int j = 0; j < n_features; j++) {
            A_in[i + lda * j] = static_cast<double>(n_samples - i - 1);
        }
        expected_indices[i] = n_samples - i - 1;
    }

    std::vector<TypeParam> A = convert_vector<double, TypeParam>(A_in);

    auto tree = TEST_ARCH::da_binary_tree::kd_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)2.0);

    // Check if the indices in the k-d tree match the expected indices, allowing for consecutive pairs
    // to be swapped because of the way the k-d tree is built and terminates with some leaves of size 2
    da_int i = 0;
    while (i < n_samples) {
        if (expected_indices[i] == tree.get_indices()[i]) {
            i += 1;
            continue;
        } else if (expected_indices[i] == tree.get_indices()[i + 1] &&
                   expected_indices[i + 1] == tree.get_indices()[i]) {
            i += 2;
            continue;
        } else {
            FAIL() << "k-d tree indices do not match expected values for large test "
                      "(Euclidean metric)";
        }
    }

    auto tree2 = TEST_ARCH::da_binary_tree::kd_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_minkowski,
        (TypeParam)2.00001);

    i = 0;
    while (i < n_samples) {
        if (expected_indices[i] == tree2.get_indices()[i]) {
            i += 1;
            continue;
        } else if (expected_indices[i] == tree2.get_indices()[i + 1] &&
                   expected_indices[i + 1] == tree2.get_indices()[i]) {
            i += 2;
            continue;
        } else {
            FAIL() << "k-d tree indices do not match expected values for large test "
                      "(Minkowski metric)";
        }
    }

    auto tree3 = TEST_ARCH::da_binary_tree::kd_tree<TypeParam>(
        n_samples, (da_int)1, A.data(), lda, leaf_size, da_manhattan, (TypeParam)2.00001);

    i = 0;
    while (i < n_samples) {
        if (expected_indices[i] == tree3.get_indices()[i]) {
            i += 1;
            continue;
        } else if (expected_indices[i] == tree3.get_indices()[i + 1] &&
                   expected_indices[i + 1] == tree3.get_indices()[i]) {
            i += 2;
            continue;
        } else {
            FAIL() << "k-d tree indices do not match expected values for large test "
                      "(Manhattan metric)";
        }
    }
}

TYPED_TEST(typed_tree_tests, centroid_radius_test) {
    da_int n_samples = 1000;
    da_int n_features = 3;
    da_int lda = n_samples;

    std::vector<double> A_in(n_samples * n_features);
    std::vector<TypeParam> expected_centroid(n_features, 0.0);
    TypeParam expected_radius = 467.239466173;

    for (da_int i = 0; i < n_samples; i++) {
        for (da_int j = 0; j < n_features; j++) {
            A_in[i + lda * j] =
                static_cast<double>((i + 0.5 - n_samples / 2) * (j + 1) / 4);
        }
    }

    std::vector<TypeParam> A = convert_vector<double, TypeParam>(A_in);
    std::vector<TypeParam> centroid(n_features, 0.0);
    TypeParam radius = 0.0;

    da_status status = TEST_ARCH::da_binary_tree::parallel_centroid_radius(
        n_samples, n_features, A.data(), lda, da_euclidean, (TypeParam)2.0, centroid,
        radius);

    ASSERT_EQ(status, da_status_success);

    TypeParam tol = 10 * std::sqrt(std::numeric_limits<TypeParam>::epsilon());

    EXPECT_ARR_NEAR(n_features, centroid.data(), expected_centroid.data(), tol);

    EXPECT_NEAR(radius, expected_radius, tol);
}