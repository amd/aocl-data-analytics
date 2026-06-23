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
#include "da_omp.hpp"
#include "tsne/tsne.hpp"
#include "tsne/tsne_kernels.hpp"
#include "tsne_utils.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <limits>
#include <list>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace TEST_ARCH;

namespace {
template <typename T> T compute_entropy(const std::vector<T> &probs) {
    T H = (T)0;
    for (auto p : probs) {
        if (p > (T)0)
            H -= p * std::log(p);
    }
    return H;
}

template <typename T>
void assert_prob_vector_valid(const std::vector<T> &probs, T sum_tol = 1e-5) {
    T sum = std::accumulate(probs.begin(), probs.end(), (T)0);
    EXPECT_NEAR(sum, (T)1, sum_tol) << "Probabilities must sum to 1";
    for (size_t i = 0; i < probs.size(); ++i) {
        EXPECT_GE(probs[i], (T)0) << "Probability must be non-negative at index " << i;
    }
}

// Matches the BH kernel denominator: 1 / (dist2 + 1 + DIST2_FLOOR).
// Self-interaction (dist2=0): bh_inv(0) = 1 / (1 + DIST2_FLOOR).
template <typename T> T bh_inv(T dist2) {
    constexpr T floor = sizeof(T) <= 4 ? (T)1e-6 : (T)1e-8;
    return (T)1 / (dist2 + (T)1 + floor);
}

} // namespace

template <typename T> class tsne_internal_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(tsne_internal_test, FloatTypes);

// =============================================================================
// Quality Metrics
// =============================================================================

TYPED_TEST(tsne_internal_test, QualityMetricsBasic) {
    const da_int n_samples = 10;
    const da_int n_features = 3;
    const da_int n_components = 2;
    const da_int k = 3;

    std::vector<TypeParam> X_high(n_samples * n_features);
    std::vector<TypeParam> X_low(n_samples * n_components);

    for (da_int i = 0; i < n_samples; ++i) {
        X_high[i * n_features] = i;
        X_high[i * n_features + 1] = i * 2;
        X_high[i * n_features + 2] = i * 3;

        X_low[i * n_components] = i;
        X_low[i * n_components + 1] = i * 2;
    }

    TypeParam trust = tsne_metrics::compute_trustworthiness(
        X_high.data(), X_low.data(), n_samples, n_features, n_components, k);
    const TypeParam metric_tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-4) : TypeParam(1e-6);
    EXPECT_NEAR(trust, TypeParam(1), metric_tol);
}

// =============================================================================
// Barnes-Hut tree + repulsive force tests
// =============================================================================

// n=0 and n=1 both trigger the sum_q==0 sentinel (set to 1) once the
// diagonal self-interaction is excluded from q.
TYPED_TEST(tsne_internal_test, BarnesHutZeroAndOnePointBoundary) {
    constexpr da_int dim = 2;
    const TypeParam tol = 1e-6;

    {
        std::vector<TypeParam> pts;
        da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), 0);
        tree.build();
        EXPECT_EQ(tree.num_nodes, 0);
        EXPECT_FALSE(tree.memory_initialized);

        TypeParam sum_q = 0;
        std::vector<TypeParam> work(omp_get_max_threads());
        da_tsne::compute_repulsive_forces(tree, 0, static_cast<TypeParam *>(nullptr),
                                          sum_q, work);
        EXPECT_EQ(sum_q, TypeParam(1));
    }

    {
        std::vector<TypeParam> pts = {1.25, -2.5};
        da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), 1, TypeParam(0.5));
        tree.build();

        EXPECT_TRUE(tree.memory_initialized);
        EXPECT_EQ(tree.num_nodes, 1);
        EXPECT_EQ(tree.nodes[0].num_children, 0);
        EXPECT_EQ(tree.nodes[0].cnt, 1);
        EXPECT_EQ(tree.sorted_indices[0], 0);

        std::vector<TypeParam> rep(dim, 0);
        TypeParam sum_q = 0;
        std::vector<TypeParam> work(omp_get_max_threads());
        da_tsne::compute_repulsive_forces(tree, 1, rep.data(), sum_q, work);
        EXPECT_EQ(sum_q, TypeParam(1));
        for (da_int d = 0; d < dim; ++d)
            EXPECT_NEAR(rep[d], 0, tol);
    }
}

TYPED_TEST(tsne_internal_test, BarnesHutComputeBboxEmptyTree) {
    constexpr da_int dim = 2;
    std::vector<TypeParam> pts;
    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), 0);
    tree.compute_bbox();
    EXPECT_EQ(tree.num_nodes, 0);
}

TYPED_TEST(tsne_internal_test, BarnesHutBuildSubtreeEmptyRange) {
    constexpr da_int dim = 2;
    std::vector<TypeParam> pts = {0, 0, 1, 1};
    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), 2, TypeParam(0.5));
    tree.build();
    da_int prev_num_nodes = tree.num_nodes;

    // Call build_subtree with an empty range (start == end)
    TypeParam center[dim] = {0, 0};
    tree.build_subtree(prev_num_nodes - 1, 0, 0, 0, center, TypeParam(1));

    auto &nd = tree.nodes[prev_num_nodes - 1];
    EXPECT_EQ(nd.cnt, 0);
    EXPECT_EQ(nd.first_child, 0);
    EXPECT_EQ(nd.num_children, 0);
}

TYPED_TEST(tsne_internal_test, BarnesHutBuildSubtreeArenaFullFallback) {
    constexpr da_int dim = 2;
    // 6 well-separated points that will try to subdivide (not leaf)
    std::vector<TypeParam> pts = {0, 0, 10, 0, 0, 10, 10, 10, 5, 5, 5, 0};
    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), 6, TypeParam(0), 1);
    tree.build();

    // Artificially shrink capacity to leave no room for children, forcing the fallback
    tree.num_nodes = 1;
    tree.capacity = 1;

    TypeParam center[dim] = {tree.bbox_center[0], tree.bbox_center[1]};
    tree.build_subtree(0, 0, 6, 0, center, tree.bbox_width / TypeParam(2));

    // The root should have been turned into a leaf covering all 6 points
    EXPECT_EQ(tree.nodes[0].cnt, 6);
    EXPECT_EQ(tree.nodes[0].num_children, 0);
}

TYPED_TEST(tsne_internal_test, BarnesHutBatch4EmptyTree) {
    constexpr da_int dim = 2;
    std::vector<TypeParam> pts;
    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), 0);
    ASSERT_EQ(tree.num_nodes, 0);

    da_int pidx[4] = {0, 0, 0, 0};
    TypeParam force[4][dim], sq[4];
    int32_t node_stack[da_tsne::BarnesHutTree<TypeParam, dim>::STACK_SIZE];
    int32_t depth_stack[da_tsne::BarnesHutTree<TypeParam, dim>::STACK_SIZE];

    // Must not crash; outputs should be zeroed
    tree.compute_repulsive_batch4(pidx, force, sq, node_stack, depth_stack);
    for (da_int b = 0; b < 4; ++b) {
        EXPECT_EQ(sq[b], TypeParam(0));
        for (da_int d = 0; d < dim; ++d)
            EXPECT_EQ(force[b][d], TypeParam(0));
    }
}

// Two points in 1D at +/-a. Closed-form force and sum_q from the single
// pairwise interaction, anchoring exact arithmetic for the simplest tree.
TYPED_TEST(tsne_internal_test, BarnesHut1DSymmetricForce) {
    const da_int n = 2;
    constexpr da_int dim = 1;
    const TypeParam a = 3;
    std::vector<TypeParam> pts = {-a, +a};

    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), n, TypeParam(0), 1);
    tree.build();
    EXPECT_EQ(tree.num_nodes, 3);
    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-6) : TypeParam(1e-10);

    EXPECT_NEAR(static_cast<TypeParam>(tree.nodes[0].cnt), n, tol);
    EXPECT_NEAR(tree.nodes[0].com[0], 0, tol);
    const da_int first_c = static_cast<da_int>(tree.nodes[0].first_child);
    const da_int c_left = first_c;
    const da_int c_right = first_c + 1;
    ASSERT_GE(c_left, 0);
    ASSERT_GE(c_right, 0);
    EXPECT_EQ(tree.nodes[c_left].cnt, 1);
    EXPECT_EQ(tree.nodes[c_right].cnt, 1);
    EXPECT_NEAR(tree.nodes[c_left].com[0], -a, tol);
    EXPECT_NEAR(tree.nodes[c_right].com[0], +a, tol);

    std::vector<TypeParam> rep(n * dim, 0);
    TypeParam sum_q = 0;
    std::vector<TypeParam> work(omp_get_max_threads());
    da_tsne::compute_repulsive_forces(tree, n, rep.data(), sum_q, work);

    // dist^2 = (2a)^2, inv = 1/(dist^2 + 1 + DIST2_FLOOR), force = |2a * inv^2|
    const TypeParam dist2 = 4 * a * a;
    const TypeParam inv = bh_inv<TypeParam>(dist2);
    const TypeParam per_point_sum_q = inv;
    const TypeParam expected_force_mag = 2 * a * inv * inv;

    EXPECT_NEAR(sum_q, 2 * per_point_sum_q, tol);
    EXPECT_NEAR(rep[0], -expected_force_mag, tol);
    EXPECT_NEAR(rep[1], expected_force_mag, tol);
    EXPECT_NEAR(rep[0] + rep[1], 0, tol);
}

// Three collinear 2D points: left(-1,0), center(0,0), right(1,0).
// Center has zero net force by symmetry; side forces are analytical.
TYPED_TEST(tsne_internal_test, BarnesHut2DCenterSymmetry) {
    const da_int n = 3;
    constexpr da_int dim = 2;
    std::vector<TypeParam> pts = {
        -1, 0, // left
        0,  0, // center
        1,  0  // right
    };

    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), n, TypeParam(0), 1);
    tree.build();
    EXPECT_LE(tree.num_nodes, 9);

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(5e-7) : TypeParam(1e-8);
    EXPECT_NEAR(static_cast<TypeParam>(tree.nodes[0].cnt), n, tol);
    EXPECT_NEAR(tree.nodes[0].com[0], 0, tol);
    EXPECT_NEAR(tree.nodes[0].com[1], 0, tol);

    std::vector<TypeParam> rep(n * dim, 0);
    TypeParam sum_q = 0;
    std::vector<TypeParam> work(omp_get_max_threads());
    da_tsne::compute_repulsive_forces(tree, n, rep.data(), sum_q, work);

    const TypeParam inv1 = bh_inv<TypeParam>(1);
    const TypeParam inv4 = bh_inv<TypeParam>(4);
    const TypeParam expected_sum_q_center = 2 * inv1;
    const TypeParam expected_sum_q_side = inv1 + inv4;
    const TypeParam expected_fx_left = -(inv1 * inv1 + 2 * inv4 * inv4);

    EXPECT_NEAR(sum_q, 2 * expected_sum_q_side + expected_sum_q_center, n * tol);

    EXPECT_NEAR(rep[0 * dim + 0], expected_fx_left, tol);
    EXPECT_NEAR(rep[0 * dim + 1], 0, tol);

    EXPECT_NEAR(rep[1 * dim + 0], 0, tol);
    EXPECT_NEAR(rep[1 * dim + 1], 0, tol);

    EXPECT_NEAR(rep[2 * dim + 0], -expected_fx_left, tol);
    EXPECT_NEAR(rep[2 * dim + 1], 0, tol);
}

// 3D axis-aligned cross: center + 6 points at +/-a on each axis.
// Center has zero force by symmetry; side points have identical magnitude,
// testing D=3 dimensional indexing and leaf mass conservation.
TYPED_TEST(tsne_internal_test, BarnesHut3DCrossCenter) {
    const da_int n = 7;
    constexpr da_int dim = 3;
    const TypeParam a = 2;

    std::vector<TypeParam> pts = {
        0,  0,  0,  // center
        +a, 0,  0,  // +x
        -a, 0,  0,  // -x
        0,  +a, 0,  // +y
        0,  -a, 0,  // -y
        0,  0,  +a, // +z
        0,  0,  -a  // -z
    };

    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), n);
    tree.build();
    EXPECT_GT(tree.num_nodes, 0);
    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(5e-7) : TypeParam(1e-8);
    EXPECT_NEAR(static_cast<TypeParam>(tree.nodes[0].cnt), n, tol);
    EXPECT_NEAR(tree.nodes[0].com[0], 0, tol);
    EXPECT_NEAR(tree.nodes[0].com[1], 0, tol);
    EXPECT_NEAR(tree.nodes[0].com[2], 0, tol);
    TypeParam leaf_mass_sum = 0;
    for (da_int node = 0; node < tree.num_nodes; ++node)
        if (tree.nodes[node].num_children == 0 && tree.nodes[node].cnt > 0)
            leaf_mass_sum += static_cast<TypeParam>(tree.nodes[node].cnt);
    EXPECT_NEAR(leaf_mass_sum, n, tol);

    std::vector<TypeParam> rep(n * dim, 0);
    TypeParam sum_q = 0;
    std::vector<TypeParam> work(omp_get_max_threads());
    da_tsne::compute_repulsive_forces(tree, n, rep.data(), sum_q, work);

    // Pairwise distances: a^2 (center-side), 2a^2 (orthogonal sides), 4a^2 (opposite)
    const TypeParam inv_d1 = bh_inv<TypeParam>(a * a);
    const TypeParam inv_d2 = bh_inv<TypeParam>(2 * a * a);
    const TypeParam inv_d4 = bh_inv<TypeParam>(4 * a * a);
    const TypeParam expected_sum_q_center = 6 * inv_d1;
    EXPECT_NEAR(rep[0 * dim + 0], 0, tol);
    EXPECT_NEAR(rep[0 * dim + 1], 0, tol);
    EXPECT_NEAR(rep[0 * dim + 2], 0, tol);

    const TypeParam expected_sum_q_side = inv_d1 + inv_d4 + 4 * inv_d2;
    const TypeParam expected_force_axis =
        a * (inv_d1 * inv_d1 + 2 * inv_d4 * inv_d4 + 4 * inv_d2 * inv_d2);

    EXPECT_NEAR(sum_q, expected_sum_q_center + 6 * expected_sum_q_side, n * tol);

    const da_int side_idx[6] = {1, 2, 3, 4, 5, 6};
    const da_int axis[6] = {0, 0, 1, 1, 2, 2};
    const TypeParam sign[6] = {1, -1, 1, -1, 1, -1};

    for (da_int i = 0; i < 6; ++i) {
        for (da_int d = 0; d < dim; ++d) {
            const TypeParam expected = (d == axis[i]) ? sign[i] * expected_force_axis : 0;
            EXPECT_NEAR(rep[side_idx[i] * dim + d], expected, tol);
        }
    }
}

// Three 1D points {-1, 0, 1}: the tie point at the cell center (0) must be
// classified to the high child (p[d] >= center[d]).
TYPED_TEST(tsne_internal_test, BarnesHutCenterTieGoesToUpperChild) {
    const da_int n = 3;
    constexpr da_int dim = 1;
    std::vector<TypeParam> pts = {-1, 0, 1};

    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), n, TypeParam(0.5), 1);
    tree.build();
    ASSERT_GE(tree.num_nodes, 3);

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-6) : TypeParam(1e-12);
    const da_int first_child = static_cast<da_int>(tree.nodes[0].first_child);
    ASSERT_GE(first_child, 0);

    EXPECT_EQ(tree.nodes[first_child].cnt, 1);
    EXPECT_EQ(tree.nodes[first_child + 1].cnt, 2);
    EXPECT_NEAR(tree.nodes[first_child].com[0], TypeParam(-1), tol);
    EXPECT_NEAR(tree.nodes[first_child + 1].com[0], TypeParam(0.5), tol);
}

// Three groups of 2 identical points. Points sharing coordinates must produce
// identical forces, testing near-zero distance stability via DIST2_FLOOR.
TYPED_TEST(tsne_internal_test, BarnesHutDuplicatePoints) {
    const da_int n = 6;
    constexpr da_int dim = 2;
    std::vector<TypeParam> pts = {
        0, 0, // group 0
        0, 0, // group 0
        3, 0, // group 1
        3, 0, // group 1
        0, 4, // group 2
        0, 4  // group 2
    };

    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), n, TypeParam(0.5));
    tree.build();
    EXPECT_GT(tree.num_nodes, 0);

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-6) : TypeParam(1e-10);
    EXPECT_NEAR(static_cast<TypeParam>(tree.nodes[0].cnt), n, tol);
    EXPECT_NEAR(tree.nodes[0].com[0], TypeParam(1), tol);
    EXPECT_NEAR(tree.nodes[0].com[1], TypeParam(4) / 3, tol);

    std::vector<TypeParam> rep(n * dim, 0);
    TypeParam sum_q = 0;
    std::vector<TypeParam> work(omp_get_max_threads());
    da_tsne::compute_repulsive_forces(tree, n, rep.data(), sum_q, work);

    EXPECT_GT(sum_q, TypeParam(0));

    for (da_int g = 0; g < n; g += 2) {
        for (da_int d = 0; d < dim; ++d)
            EXPECT_NEAR(rep[g * dim + d], rep[(g + 1) * dim + d], tol);
    }
}

// Four-point square: forces must be identical after a large global translation,
// since repulsive interactions depend only on pairwise distances.
TYPED_TEST(tsne_internal_test, BarnesHutTranslationInvariance) {
    const da_int n = 4;
    constexpr da_int dim = 2;
    std::vector<TypeParam> pts = {0, 0, 1, 0, 0, 1, 1, 1};

    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), n, TypeParam(0.5));
    tree.build();
    const da_int num_nodes_first = tree.num_nodes;

    std::vector<TypeParam> rep_first(n * dim, 0);
    TypeParam sq_first = 0;
    std::vector<TypeParam> work(omp_get_max_threads());
    da_tsne::compute_repulsive_forces(tree, n, rep_first.data(), sq_first, work);

    for (da_int i = 0; i < n; ++i) {
        pts[i * dim + 0] += 1000;
        pts[i * dim + 1] += 2000;
    }

    tree.build();
    EXPECT_EQ(tree.num_nodes, num_nodes_first);

    std::vector<TypeParam> rep_second(n * dim, 0);
    TypeParam sq_second = 0;
    da_tsne::compute_repulsive_forces(tree, n, rep_second.data(), sq_second, work);

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-6) : TypeParam(1e-10);
    EXPECT_NEAR(sq_first, sq_second, tol);
    EXPECT_ARR_NEAR(n * dim, rep_first.data(), rep_second.data(), tol);
}

// Compares theta=0 (exact) baseline against theta=0.5 and 1.0 approximations.
// Forces must stay within a relative tolerance, exercising the all_far BH path.
TYPED_TEST(tsne_internal_test, BarnesHutThetaApproximationContract) {
    const da_int n = 6;
    constexpr da_int dim = 2;
    std::vector<TypeParam> pts = {
        0,   0,   // p0 query
        2.0, 0,   // p1 singleton
        3.0, 3.0, // p2 far cluster
        3.0, 4.8, // p3
        4.8, 3.0, // p4
        4.8, 4.8  // p5
    };

    da_tsne::BarnesHutTree<TypeParam, dim> tree_exact(pts.data(), n, TypeParam(0), 1);
    tree_exact.build();
    EXPECT_GT(tree_exact.num_nodes, 1);

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(5e-7) : TypeParam(1e-8);

    std::vector<TypeParam> rep_exact(n * dim, 0);
    TypeParam sq_exact = 0;
    std::vector<TypeParam> work(omp_get_max_threads());
    da_tsne::compute_repulsive_forces(tree_exact, n, rep_exact.data(), sq_exact, work);

    struct ApproxCase {
        TypeParam theta;
        TypeParam rel_tol;
    };
    const std::vector<ApproxCase> approx_cases = {{TypeParam(0.5), TypeParam(0.05)},
                                                  {TypeParam(1.0), TypeParam(0.12)}};

    for (const auto &c : approx_cases) {
        da_tsne::BarnesHutTree<TypeParam, dim> tree_bh(pts.data(), n, c.theta, 1);
        tree_bh.build();
        std::vector<TypeParam> rep_bh(n * dim, 0);
        TypeParam sq_bh = 0;
        da_tsne::compute_repulsive_forces(tree_bh, n, rep_bh.data(), sq_bh, work);

        for (da_int i = 0; i < n; ++i) {
            for (da_int d = 0; d < dim; ++d) {
                TypeParam exact_f = rep_exact[i * dim + d];
                TypeParam bh_f = rep_bh[i * dim + d];
                TypeParam mag = std::abs(exact_f);
                if (mag > tol) {
                    EXPECT_NEAR(bh_f, exact_f, mag * c.rel_tol)
                        << "theta=" << c.theta << ", point=" << i << ", dim=" << d;
                    EXPECT_EQ(std::signbit(bh_f), std::signbit(exact_f))
                        << "theta=" << c.theta << ", sign mismatch at point=" << i
                        << ", dim=" << d;
                }
            }
        }

        EXPECT_NEAR(sq_bh, sq_exact, std::abs(sq_exact) * c.rel_tol);
    }

    TypeParam sum_fx = 0, sum_fy = 0;
    for (da_int i = 0; i < n; ++i) {
        sum_fx += rep_exact[i * dim + 0];
        sum_fy += rep_exact[i * dim + 1];
    }
    EXPECT_NEAR(sum_fx, TypeParam(0), tol);
    EXPECT_NEAR(sum_fy, TypeParam(0), tol);
}

// n=17 (not a multiple of 4) at theta=0 verified against brute-force pairwise.
// Tests the batch-4 tail path and multi-point leaf iteration (leaf_threshold=3).
TYPED_TEST(tsne_internal_test, ComputeRepulsiveForcesTailBatchThetaZero) {
    constexpr da_int dim = 2;
    constexpr da_int n = 17;
    std::vector<TypeParam> pts(n * dim);
    for (da_int i = 0; i < n; ++i) {
        pts[i * dim + 0] = TypeParam(0.17) * static_cast<TypeParam>(i) +
                           TypeParam(0.03) * static_cast<TypeParam>(i % 5);
        pts[i * dim + 1] = TypeParam(-0.23) * static_cast<TypeParam>(i) +
                           TypeParam(0.41) * static_cast<TypeParam>((i * 7) % 11);
    }

    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), n, TypeParam(0), 3);
    tree.build();

    std::vector<TypeParam> rep(n * dim, 0);
    TypeParam sum_q = 0;
    std::vector<TypeParam> work(omp_get_max_threads());
    da_tsne::compute_repulsive_forces(tree, n, rep.data(), sum_q, work);

    constexpr TypeParam DIST2_FLOOR =
        sizeof(TypeParam) <= 4 ? TypeParam(1e-6) : TypeParam(1e-8);
    constexpr TypeParam EPS_INC = TypeParam(1) + DIST2_FLOOR;

    std::vector<TypeParam> ref_rep(n * dim, 0);
    TypeParam ref_sum_q = 0;
    for (da_int i = 0; i < n; ++i) {
        for (da_int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            TypeParam dx = pts[i * dim + 0] - pts[j * dim + 0];
            TypeParam dy = pts[i * dim + 1] - pts[j * dim + 1];
            TypeParam d2 = dx * dx + dy * dy;
            TypeParam inv = TypeParam(1) / (d2 + EPS_INC);
            ref_sum_q += inv;
            ref_rep[i * dim + 0] += inv * inv * dx;
            ref_rep[i * dim + 1] += inv * inv * dy;
        }
    }

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-5) : TypeParam(1e-7);

    EXPECT_NEAR(sum_q, ref_sum_q, tol);
    for (da_int i = 0; i < n * dim; ++i)
        EXPECT_NEAR(rep[i], ref_rep[i], tol) << "i=" << i;
}

// Four corners at +/-1e15: checks that extreme coordinates produce finite
// forces pointing in the correct outward direction.
TYPED_TEST(tsne_internal_test, BarnesHutExtremeCoordinates) {
    const da_int n = 4;
    constexpr da_int dim = 2;
    const TypeParam h = 1e15;
    std::vector<TypeParam> pts = {+h, +h, -h, +h, -h, -h, +h, -h};

    da_tsne::BarnesHutTree<TypeParam, dim> tree(pts.data(), n);
    tree.build();

    std::vector<TypeParam> rep(n * dim, 0);
    TypeParam sum_q = 0;
    std::vector<TypeParam> work(omp_get_max_threads());
    da_tsne::compute_repulsive_forces(tree, n, rep.data(), sum_q, work);

    EXPECT_TRUE(std::isfinite(sum_q));
    EXPECT_GT(sum_q, TypeParam(0));

    const TypeParam rel_tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-4) : TypeParam(1e-10);

    const TypeParam sign_x[4] = {+1, -1, -1, +1};
    const TypeParam sign_y[4] = {+1, +1, -1, -1};

    for (da_int i = 0; i < n; ++i) {
        EXPECT_TRUE(std::isfinite(rep[i * dim + 0]));
        EXPECT_TRUE(std::isfinite(rep[i * dim + 1]));

        if (sign_x[i] > 0)
            EXPECT_GE(rep[i * dim + 0], 0);
        else
            EXPECT_LE(rep[i * dim + 0], 0);

        if (sign_y[i] > 0)
            EXPECT_GE(rep[i * dim + 1], 0);
        else
            EXPECT_LE(rep[i * dim + 1], 0);
    }

    const TypeParam f_tol = std::abs(rep[0]) * rel_tol;
    EXPECT_NEAR(std::abs(rep[0 * dim + 0]), std::abs(rep[1 * dim + 0]), f_tol);
    EXPECT_NEAR(std::abs(rep[0 * dim + 1]), std::abs(rep[0 * dim + 0]), f_tol);
}

// =============================================================================
// compute_row_probabilities tests
// =============================================================================

// k=0 triggers the early-return path; output must remain untouched.
TYPED_TEST(tsne_internal_test, PerplexitySearchEmptyInput) {
    std::vector<TypeParam> probs;
    da_tsne::compute_row_probabilities(probs.data(), 0, std::log(TypeParam(2)),
                                       probs.data());
    EXPECT_TRUE(probs.empty());
}

// k=1 is trivially a point mass: the sole neighbor must receive probability 1.
TYPED_TEST(tsne_internal_test, PerplexitySearchSingleNeighbor) {
    std::vector<TypeParam> sq_dist = {2.5};
    const da_int k = 1;
    TypeParam log_perp = std::log(TypeParam(1));
    std::vector<TypeParam> probs(k);

    da_tsne::compute_row_probabilities(sq_dist.data(), k, log_perp, probs.data());

    TypeParam H = compute_entropy(probs);
    EXPECT_NEAR(H, log_perp, 1e-3);
    assert_prob_vector_valid(probs);
    const TypeParam expected_single[1] = {1};
    EXPECT_ARR_NEAR(1, probs.data(), expected_single, 1e-5);
}

// Standard binary-search convergence with k=3 and perplexity=2.
// Reference values computed externally; verifies the basic happy path.
TYPED_TEST(tsne_internal_test, PerplexitySearch) {
    std::vector<TypeParam> sq_dist = {1, 4, 9};
    const da_int k = 3;
    TypeParam log_perp = std::log(TypeParam(2));
    std::vector<TypeParam> probs(k);

    da_tsne::compute_row_probabilities(sq_dist.data(), k, log_perp, probs.data());

    TypeParam H = compute_entropy(probs);
    EXPECT_NEAR(H, log_perp, 1e-3);
    assert_prob_vector_valid(probs);
    std::vector<TypeParam> expected = {0.7271810917, 0.2364596816, 0.0363592267};
    EXPECT_ARR_NEAR(k, probs.data(), expected.data(), 1e-5);
}

// Equal distances must yield the uniform distribution 1/k analytically,
// regardless of the common distance value.
TYPED_TEST(tsne_internal_test, PerplexitySearchUniform) {
    const da_int k = 5;
    std::vector<TypeParam> sq_dist(k, 4);
    TypeParam log_perp = std::log(TypeParam(k));
    std::vector<TypeParam> probs(k);

    da_tsne::compute_row_probabilities(sq_dist.data(), k, log_perp, probs.data());

    TypeParam H = compute_entropy(probs);
    EXPECT_NEAR(H, log_perp, 1e-3);
    assert_prob_vector_valid(probs);

    TypeParam expected = TypeParam(1) / k;
    for (da_int j = 0; j < k; ++j)
        EXPECT_NEAR(probs[j], expected, 1e-6);
}

// Softmax probabilities are shift- and scale-invariant (the binary search
// adjusts beta to compensate). Verifies base == shifted == scaled.
TYPED_TEST(tsne_internal_test, PerplexitySearchAffineInvariance) {
    const da_int k = 4;
    std::vector<TypeParam> sq_dist = {1, 4, 9, 16};
    std::vector<TypeParam> sq_dist_shifted = {101, 104, 109, 116};
    std::vector<TypeParam> sq_dist_scaled = {5, 20, 45, 80};
    TypeParam log_perp = std::log(TypeParam(2.5));
    std::vector<TypeParam> probs_base(k), probs_shifted(k), probs_scaled(k);

    da_tsne::compute_row_probabilities(sq_dist.data(), k, log_perp, probs_base.data());
    da_tsne::compute_row_probabilities(sq_dist_shifted.data(), k, log_perp,
                                       probs_shifted.data());
    da_tsne::compute_row_probabilities(sq_dist_scaled.data(), k, log_perp,
                                       probs_scaled.data());

    for (auto *p : {&probs_base, &probs_shifted, &probs_scaled}) {
        EXPECT_NEAR(compute_entropy(*p), log_perp, 1e-3);
        assert_prob_vector_valid(*p);
    }

    EXPECT_ARR_NEAR(k, probs_base.data(), probs_shifted.data(), 1e-5);
    EXPECT_ARR_NEAR(k, probs_base.data(), probs_scaled.data(), 1e-5);
}

// Perplexity near 1 concentrates nearly all mass on the nearest neighbor,
// pushing beta toward its upper extreme.
TYPED_TEST(tsne_internal_test, PerplexitySearchNearMinimum) {
    const da_int k = 5;
    std::vector<TypeParam> sq_dist = {1, 4, 9, 16, 25};
    TypeParam log_perp = std::log(TypeParam(1.1));
    std::vector<TypeParam> probs(k);

    da_tsne::compute_row_probabilities(sq_dist.data(), k, log_perp, probs.data());

    TypeParam H = compute_entropy(probs);
    EXPECT_NEAR(H, log_perp, 1e-3);
    assert_prob_vector_valid(probs);

    std::vector<TypeParam> expected = {0.9807526980, 0.0192199209, 0.0000273782,
                                       0.0000000028, 0.0000000000};
    EXPECT_ARR_NEAR(k, probs.data(), expected.data(), 1e-5);
}

// Perplexity near k approaches the uniform distribution,
// pushing beta toward zero.
TYPED_TEST(tsne_internal_test, PerplexitySearchNearMaximum) {
    const da_int k = 5;
    std::vector<TypeParam> sq_dist = {1, 4, 9, 16, 25};
    TypeParam log_perp = std::log(TypeParam(4.9));
    std::vector<TypeParam> probs(k);

    da_tsne::compute_row_probabilities(sq_dist.data(), k, log_perp, probs.data());

    TypeParam H = compute_entropy(probs);
    EXPECT_NEAR(H, log_perp, 1e-3);
    assert_prob_vector_valid(probs);
    std::vector<TypeParam> expected = {0.2494082833, 0.2319532177, 0.2055336567,
                                       0.1735235865, 0.1395812557};
    EXPECT_ARR_NEAR(k, probs.data(), expected.data(), 1e-5);
}

// Distances spanning 8 orders of magnitude (1e-4 to 1e4) stress-test
// the binary search across extreme beta scales.
TYPED_TEST(tsne_internal_test, PerplexitySearchLargeDistanceSpread) {
    const da_int k = 4;
    std::vector<TypeParam> sq_dist = {1e-4, 1, 1e2, 1e4};
    TypeParam log_perp = std::log(TypeParam(2));
    std::vector<TypeParam> probs(k);

    da_tsne::compute_row_probabilities(sq_dist.data(), k, log_perp, probs.data());

    TypeParam H = compute_entropy(probs);
    EXPECT_NEAR(H, log_perp, 1e-3);
    assert_prob_vector_valid(probs);
    std::vector<TypeParam> expected = {0.5214186447, 0.4784847565, 0.0000965989,
                                       0.0000000000};
    EXPECT_ARR_NEAR(k, probs.data(), expected.data(), 1e-5);
}

// All distances at 1e20 cause exp(-beta*d) to underflow to zero for any
// reasonable beta, triggering the uniform fallback path in normalization.
TYPED_TEST(tsne_internal_test, PerplexitySearchExtremeUnderflowFallsBackUniform) {
    const da_int k = 4;
    std::vector<TypeParam> sq_dist(k, 1e20);
    TypeParam log_perp = std::log(TypeParam(2));
    std::vector<TypeParam> probs(k);

    da_tsne::compute_row_probabilities(sq_dist.data(), k, log_perp, probs.data());
    TypeParam H = compute_entropy(probs);
    EXPECT_NEAR(H, std::log(TypeParam(k)), 1e-3);
    assert_prob_vector_valid(probs);

    const TypeParam expected = TypeParam(1) / k;
    for (da_int j = 0; j < k; ++j)
        EXPECT_NEAR(probs[j], expected, 1e-6);
}

// =============================================================================
// symmetrize_to_csr tests
// =============================================================================

// Fully connected 3-point graph with asymmetric conditional probabilities
// (p(1|0)=0.6 vs p(0|1)=0.7). Verifies CSR structure, sorted columns,
// P[i][j]==P[j][i] symmetry, sum-to-1 normalization, and hand-computed values.
TYPED_TEST(tsne_internal_test, SymmetrizeToCsrBasic3Point) {
    const da_int n = 3, k = 2;
    std::vector<da_int> neighbor_indices = {1, 2, 0, 2, 0, 1};
    std::vector<TypeParam> neighbor_probs = {TypeParam(0.6), TypeParam(0.4),
                                             TypeParam(0.7), TypeParam(0.3),
                                             TypeParam(0.5), TypeParam(0.5)};

    std::vector<da_int> row_ptr, col_idx;
    std::vector<TypeParam> vals;
    da_tsne::symmetrize_to_csr(n, k, neighbor_indices.data(), neighbor_probs.data(),
                               row_ptr, col_idx, vals);

    // CSR structure checks
    ASSERT_EQ(row_ptr.size(), (size_t)(n + 1));
    EXPECT_EQ(row_ptr[0], 0);
    EXPECT_EQ(row_ptr[n], (da_int)vals.size());

    // Each row should have exactly 2 entries (fully connected 3-point graph)
    for (da_int i = 0; i < n; ++i)
        EXPECT_EQ(row_ptr[i + 1] - row_ptr[i], 2);

    // Column indices must be sorted within each row
    for (da_int i = 0; i < n; ++i) {
        for (da_int k = row_ptr[i]; k < row_ptr[i + 1] - 1; ++k)
            EXPECT_LT(col_idx[k], col_idx[k + 1]);
    }

    // Symmetry: P[i][j] == P[j][i]
    for (da_int i = 0; i < n; ++i) {
        for (da_int ki = row_ptr[i]; ki < row_ptr[i + 1]; ++ki) {
            da_int j = col_idx[ki];
            bool found = false;
            for (da_int kj = row_ptr[j]; kj < row_ptr[j + 1]; ++kj) {
                if (col_idx[kj] == i) {
                    EXPECT_NEAR(vals[ki], vals[kj], 1e-10);
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "Missing symmetric entry P[" << j << "][" << i << "]";
        }
    }

    // Values must sum to 1
    TypeParam total = std::accumulate(vals.begin(), vals.end(), TypeParam(0));
    EXPECT_NEAR(total, TypeParam(1), 1e-10);

    // Hand-computed expected values:
    // norm = 2*3 = 6; symm[0][1] = 0.6/6 + 0.7/6 = 13/60; etc.
    const TypeParam tol = 1e-8;
    TypeParam P01 = TypeParam(13) / 60;
    TypeParam P02 = TypeParam(9) / 60;
    TypeParam P12 = TypeParam(8) / 60;

    // Row 0: cols {1,2}
    EXPECT_EQ(col_idx[row_ptr[0]], 1);
    EXPECT_EQ(col_idx[row_ptr[0] + 1], 2);
    EXPECT_NEAR(vals[row_ptr[0]], P01, tol);
    EXPECT_NEAR(vals[row_ptr[0] + 1], P02, tol);

    // Row 1: cols {0,2}
    EXPECT_EQ(col_idx[row_ptr[1]], 0);
    EXPECT_EQ(col_idx[row_ptr[1] + 1], 2);
    EXPECT_NEAR(vals[row_ptr[1]], P01, tol);
    EXPECT_NEAR(vals[row_ptr[1] + 1], P12, tol);

    // Row 2: cols {0,1}
    EXPECT_EQ(col_idx[row_ptr[2]], 0);
    EXPECT_EQ(col_idx[row_ptr[2] + 1], 1);
    EXPECT_NEAR(vals[row_ptr[2]], P02, tol);
    EXPECT_NEAR(vals[row_ptr[2] + 1], P12, tol);
}

// Boundary: n=1 with no neighbors produces an empty CSR (row_ptr={0,0}).
TYPED_TEST(tsne_internal_test, SymmetrizeToCsrSinglePoint) {
    const da_int n = 1, k = 0;

    std::vector<da_int> row_ptr, col_idx;
    std::vector<TypeParam> vals;
    da_tsne::symmetrize_to_csr(n, k, static_cast<const da_int *>(nullptr),
                               static_cast<const TypeParam *>(nullptr), row_ptr, col_idx,
                               vals);

    ASSERT_EQ(row_ptr.size(), (size_t)2);
    EXPECT_EQ(row_ptr[0], 0);
    EXPECT_EQ(row_ptr[1], 0);
    EXPECT_TRUE(col_idx.empty());
    EXPECT_TRUE(vals.empty());
}

// =============================================================================
// compute_kl_divergence tests
// =============================================================================

// 3-point 1D toy example with pre-supplied sum_q. Verifies KL against an
// independent reference loop and checks KL > 0 (P != Q guarantees positivity).
TYPED_TEST(tsne_internal_test, KLDivergenceAnalytic) {
    // 3 points in 1D: y0=0, y1=1, y2=3
    const da_int n = 3, d = 1;
    std::vector<TypeParam> emb = {0, 1, 3};

    // Symmetric P: P[0][1]=P[1][0]=0.4, P[0][2]=P[2][0]=0.05, P[1][2]=P[2][1]=0.05
    std::vector<da_int> row_ptr = {0, 2, 4, 6};
    std::vector<da_int> col_idx = {1, 2, 0, 2, 0, 1};
    std::vector<TypeParam> p_vals = {TypeParam(0.4),  TypeParam(0.05), TypeParam(0.4),
                                     TypeParam(0.05), TypeParam(0.05), TypeParam(0.05)};

    // sum_q = sum over all directed (i,j) pairs of q_ij
    // d01^2=1, d02^2=9, d12^2=4
    TypeParam sum_q =
        TypeParam(2) * (TypeParam(1) / 2 + TypeParam(1) / 10 + TypeParam(1) / 5); // = 1.6

    std::vector<TypeParam> work(omp_get_max_threads());
    TypeParam kl =
        da_tsne::compute_kl_divergence(n, d, row_ptr, col_idx, p_vals, emb, sum_q, work);

    // Independent reference computation
    const TypeParam eps = std::numeric_limits<TypeParam>::epsilon();
    TypeParam expected_kl = 0;
    for (da_int i = 0; i < n; ++i) {
        for (da_int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            da_int j = col_idx[idx];
            TypeParam diff = emb[i] - emb[j];
            TypeParam dist2 = diff * diff;
            TypeParam q = TypeParam(1) / (TypeParam(1) + dist2);
            TypeParam Pij = std::max(p_vals[idx], eps);
            TypeParam Qij = std::max(q / sum_q, eps);
            expected_kl += Pij * std::log(Pij / Qij);
        }
    }

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-4) : TypeParam(1e-10);
    EXPECT_NEAR(kl, expected_kl, tol);
    EXPECT_GT(kl, TypeParam(0));
}

// Same dataset but passes sum_q=0, triggering the O(n^2) recomputation path.
// Verifies the recomputed result matches the pre-supplied version.
TYPED_TEST(tsne_internal_test, KLDivergenceRecomputeSumQ) {
    // Same setup, but pass sum_q_total=0 to trigger recomputation
    const da_int n = 3, d = 1;
    std::vector<TypeParam> emb = {0, 1, 3};

    std::vector<da_int> row_ptr = {0, 2, 4, 6};
    std::vector<da_int> col_idx = {1, 2, 0, 2, 0, 1};
    std::vector<TypeParam> p_vals = {TypeParam(0.4),  TypeParam(0.05), TypeParam(0.4),
                                     TypeParam(0.05), TypeParam(0.05), TypeParam(0.05)};

    TypeParam sum_q =
        TypeParam(2) * (TypeParam(1) / 2 + TypeParam(1) / 10 + TypeParam(1) / 5);

    std::vector<TypeParam> work(omp_get_max_threads());
    TypeParam kl_with_sumq =
        da_tsne::compute_kl_divergence(n, d, row_ptr, col_idx, p_vals, emb, sum_q, work);

    TypeParam kl_recomputed = da_tsne::compute_kl_divergence(
        n, d, row_ptr, col_idx, p_vals, emb, TypeParam(0), work);

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-4) : TypeParam(1e-10);
    EXPECT_NEAR(kl_with_sumq, kl_recomputed, tol);
}

// =============================================================================
// update_embedding tests
// =============================================================================

// 2x2 toy case with mixed iY/grad signs. Exercises both gain branches
// (+=0.2 and *=0.8) and analytically verifies gains, velocity, and embedding.
TYPED_TEST(tsne_internal_test, UpdateEmbeddingGainAndMomentum) {
    const da_int n = 2, d = 2;
    const TypeParam momentum = TypeParam(0.8);
    const TypeParam lr = TypeParam(1);

    std::vector<TypeParam> emb = {1, 2, 3, 4};
    std::vector<TypeParam> grad = {TypeParam(0.1), TypeParam(-0.2), TypeParam(0.3),
                                   TypeParam(0.4)};
    std::vector<TypeParam> iY = {TypeParam(0.5), TypeParam(0.5), TypeParam(-0.5),
                                 TypeParam(-0.5)};
    std::vector<TypeParam> gains = {1, 1, 1, 1};

    da_tsne::update_embedding(n, d, momentum, lr, grad, iY, gains, emb);

    const TypeParam tol = 1e-8;

    // iY[0]*grad[0] = 0.5*0.1 > 0 => gains *= 0.8 => 0.8
    EXPECT_NEAR(gains[0], TypeParam(0.8), tol);
    // iY[1]*grad[1] = 0.5*(-0.2) < 0 => gains += 0.2 => 1.2
    EXPECT_NEAR(gains[1], TypeParam(1.2), tol);
    // iY[2]*grad[2] = -0.5*0.3 < 0 => gains += 0.2 => 1.2
    EXPECT_NEAR(gains[2], TypeParam(1.2), tol);
    // iY[3]*grad[3] = -0.5*0.4 < 0 => gains += 0.2 => 1.2
    EXPECT_NEAR(gains[3], TypeParam(1.2), tol);

    // iY = momentum * iY_old - lr * gains * grad
    EXPECT_NEAR(iY[0],
                TypeParam(0.8) * TypeParam(0.5) - 1 * TypeParam(0.8) * TypeParam(0.1),
                tol);
    EXPECT_NEAR(iY[1],
                TypeParam(0.8) * TypeParam(0.5) - 1 * TypeParam(1.2) * TypeParam(-0.2),
                tol);
    EXPECT_NEAR(iY[2],
                TypeParam(0.8) * TypeParam(-0.5) - 1 * TypeParam(1.2) * TypeParam(0.3),
                tol);
    EXPECT_NEAR(iY[3],
                TypeParam(0.8) * TypeParam(-0.5) - 1 * TypeParam(1.2) * TypeParam(0.4),
                tol);

    // emb = emb_old + iY_new
    EXPECT_NEAR(emb[0], 1 + iY[0], tol);
    EXPECT_NEAR(emb[1], 2 + iY[1], tol);
    EXPECT_NEAR(emb[2], 3 + iY[2], tol);
    EXPECT_NEAR(emb[3], 4 + iY[3], tol);
}

// Boundary: zero gradient leaves embedding and velocity unchanged.
// Gains all take the else branch (*= 0.8) since iY*grad = 0.
TYPED_TEST(tsne_internal_test, UpdateEmbeddingZeroGrad) {
    const da_int n = 3, d = 2;
    const std::vector<TypeParam> emb_orig = {10, -5, 3, 7, -2, 100};
    std::vector<TypeParam> emb = emb_orig;
    std::vector<TypeParam> grad(n * d, 0);
    std::vector<TypeParam> iY(n * d, 0);
    std::vector<TypeParam> gains(n * d, 1);

    da_tsne::update_embedding(n, d, TypeParam(0.8), TypeParam(200), grad, iY, gains, emb);

    // Should not change
    for (da_int i = 0; i < n * d; ++i) {
        EXPECT_EQ(emb[i], emb_orig[i]);
        EXPECT_EQ(iY[i], TypeParam(0));
    }
    for (da_int i = 0; i < n * d; ++i)
        EXPECT_NEAR(gains[i], TypeParam(0.8), 1e-6);
}

// Two consecutive calls on a 1D scalar verify momentum accumulation across
// steps and a gain-branch switch when the gradient changes between iterations.
TYPED_TEST(tsne_internal_test, UpdateEmbeddingTwoStepMomentum) {
    const da_int n = 1, d = 1;
    const TypeParam mom = TypeParam(0.5), lr = TypeParam(0.1);
    std::vector<TypeParam> emb = {5};
    std::vector<TypeParam> iY = {0};
    std::vector<TypeParam> gains = {1};
    std::vector<TypeParam> grad = {2};
    const TypeParam tol = 1e-6;

    // Step 1: iY*grad = 0 (not < 0) → gains *= 0.8 = 0.8
    //         iY = 0.5*0 - 0.1*0.8*2 = -0.16,  emb = 5 - 0.16 = 4.84
    da_tsne::update_embedding(n, d, mom, lr, grad, iY, gains, emb);
    EXPECT_NEAR(gains[0], TypeParam(0.8), tol);
    EXPECT_NEAR(iY[0], TypeParam(-0.16), tol);
    EXPECT_NEAR(emb[0], TypeParam(4.84), tol);

    // Step 2: grad changes to 1; iY*grad = -0.16 < 0 → gains += 0.2 = 1.0
    //         iY = 0.5*(-0.16) - 0.1*1.0*1 = -0.18,  emb = 4.84 - 0.18 = 4.66
    grad[0] = 1;
    da_tsne::update_embedding(n, d, mom, lr, grad, iY, gains, emb);
    EXPECT_NEAR(gains[0], TypeParam(1.0), tol);
    EXPECT_NEAR(iY[0], TypeParam(-0.18), tol);
    EXPECT_NEAR(emb[0], TypeParam(4.66), tol);
}

// Gain of 0.012 * 0.8 = 0.0096 falls below the 0.01 floor and gets clamped.
TYPED_TEST(tsne_internal_test, UpdateEmbeddingGainFloor) {
    const da_int n = 1, d = 1;
    std::vector<TypeParam> emb = {0};
    std::vector<TypeParam> grad = {1};
    std::vector<TypeParam> iY = {1}; // same sign as grad => gains *= 0.8
    std::vector<TypeParam> gains = {TypeParam(0.012)};

    da_tsne::update_embedding(n, d, TypeParam(0), TypeParam(1), grad, iY, gains, emb);

    EXPECT_NEAR(gains[0], TypeParam(0.01), 1e-10);
}

// =============================================================================
// compute_attractive_forces tests
// =============================================================================

// 2-point analytical test: verifies the scalar d=2 kernel against hand-derived
// expected gradients. Closes the gap where other tests only cross-check kernels.
TYPED_TEST(tsne_internal_test, AttractiveForces2DAnalytic) {
    const da_int n = 2, d = 2;
    std::vector<TypeParam> emb = {0, 0, 1, 1};

    std::vector<da_int> row_ptr = {0, 1, 2};
    std::vector<da_int> col_idx = {1, 0};
    std::vector<TypeParam> p_vals = {TypeParam(0.5), TypeParam(0.5)};

    std::vector<TypeParam> repulsive(n * d, 0);
    TypeParam sum_q = 1;

    std::vector<TypeParam> grad(n * d, 0);
    da_tsne::compute_attractive_forces(
        n, d, TypeParam(1), row_ptr, col_idx, p_vals, emb, repulsive, sum_q, grad,
        &da_tsne::attractive_forces_scalar_impl<TypeParam, 2>);

    // dist2=2, q=1/3, grad_i = 4 * p_ij * q * (y_i - y_j) = +/-2/3
    const TypeParam tol = 1e-6;
    const TypeParam g = TypeParam(2) / TypeParam(3);
    EXPECT_NEAR(grad[0], -g, tol);
    EXPECT_NEAR(grad[1], -g, tol);
    EXPECT_NEAR(grad[2], g, tol);
    EXPECT_NEAR(grad[3], g, tol);
}

// d=2 with zero repulsive to isolate the attractive term. Verifies all d=2
// kernel variants (avx/avx2) against scalar, plus Newton's third law.
TYPED_TEST(tsne_internal_test, AttractiveForces2DPureAttractive) {
    using v = vectorization_type;
    // 3 points in 2D, repulsive = 0, sum_q = 1 => isolates attractive term
    const da_int n = 3, d = 2;
    std::vector<TypeParam> emb = {0, 0, 1, 0, 0, 2}; // y0=(0,0), y1=(1,0), y2=(0,2)

    // Symmetric P in CSR
    std::vector<da_int> row_ptr = {0, 2, 4, 6};
    std::vector<da_int> col_idx = {1, 2, 0, 2, 0, 1};
    std::vector<TypeParam> p_vals = {TypeParam(0.25),  TypeParam(0.125),
                                     TypeParam(0.25),  TypeParam(0.125),
                                     TypeParam(0.125), TypeParam(0.125)};

    std::vector<TypeParam> repulsive(n * d, 0);
    TypeParam sum_q = 1;

    auto implementations = da_tsne::testing::get_attractive_forces_implementations();

    std::vector<std::tuple<std::string, da_tsne::attractive_forces_kernel_fn<TypeParam>>>
        kernel_list = {{"avx", implementations[d - 2]->get<TypeParam>(v::avx)},
                       {"avx2", implementations[d - 2]->get<TypeParam>(v::avx2)}};

    std::vector<TypeParam> expected(n * d, 0);

    da_tsne::compute_attractive_forces(
        n, d, TypeParam(1), row_ptr, col_idx, p_vals, emb, repulsive, sum_q, expected,
        &da_tsne::attractive_forces_scalar_impl<TypeParam, 2>);

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-5) : TypeParam(1e-10);

    for (const auto &[name, kernel] : kernel_list) {
        std::cout << "d=2 pure attractive, vectorisation: " << name << std::endl;
        std::vector<TypeParam> grad(n * d, 0);
        da_tsne::compute_attractive_forces(n, d, TypeParam(1), row_ptr, col_idx, p_vals,
                                           emb, repulsive, sum_q, grad, kernel);
        EXPECT_ARR_NEAR(n * d, grad.data(), expected.data(), tol);
    }

    // Newton's third law: attractive forces sum to zero for symmetric P
    TypeParam sum_x = 0, sum_y = 0;
    for (da_int i = 0; i < n; ++i) {
        sum_x += expected[i * d + 0];
        sum_y += expected[i * d + 1];
    }
    EXPECT_NEAR(sum_x, TypeParam(0), tol);
    EXPECT_NEAR(sum_y, TypeParam(0), tol);
}

// d=2 with non-zero repulsive forces and sum_q=2. Verifies repulsive
// subtraction and sum_q normalization across all d=2 kernel variants.
TYPED_TEST(tsne_internal_test, AttractiveForces2DWithRepulsive) {
    using v = vectorization_type;
    const da_int n = 3, d = 2;
    std::vector<TypeParam> emb = {0, 0, 1, 0, 0, 2};

    std::vector<da_int> row_ptr = {0, 2, 4, 6};
    std::vector<da_int> col_idx = {1, 2, 0, 2, 0, 1};
    std::vector<TypeParam> p_vals = {TypeParam(0.25),  TypeParam(0.125),
                                     TypeParam(0.25),  TypeParam(0.125),
                                     TypeParam(0.125), TypeParam(0.125)};

    std::vector<TypeParam> repulsive = {TypeParam(-0.1), TypeParam(-0.2), TypeParam(0.05),
                                        TypeParam(-0.1), TypeParam(0.05), TypeParam(0.3)};
    TypeParam sum_q = TypeParam(2);

    auto implementations = da_tsne::testing::get_attractive_forces_implementations();

    std::vector<std::tuple<std::string, da_tsne::attractive_forces_kernel_fn<TypeParam>>>
        kernel_list = {
            {"avx", implementations[d - 2]->get<TypeParam>(v::avx)},
            {"avx2", implementations[d - 2]->get<TypeParam>(v::avx2)},
        };

    std::vector<TypeParam> expected(n * d, 0);
    da_tsne::compute_attractive_forces(
        n, d, TypeParam(1), row_ptr, col_idx, p_vals, emb, repulsive, sum_q, expected,
        &da_tsne::attractive_forces_scalar_impl<TypeParam, 2>);

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-5) : TypeParam(1e-10);

    for (const auto &[name, kernel] : kernel_list) {
        std::cout << "d=2 with repulsive, vectorisation: " << name << std::endl;
        std::vector<TypeParam> grad(n * d, 0);
        da_tsne::compute_attractive_forces(n, d, TypeParam(1), row_ptr, col_idx, p_vals,
                                           emb, repulsive, sum_q, grad, kernel);
        EXPECT_ARR_NEAR(n * d, grad.data(), expected.data(), tol);
    }
}

// 2-point d=3 analytical test: verifies the scalar d=3 kernel against
// hand-derived expected gradients of +/-0.5 per component.
TYPED_TEST(tsne_internal_test, AttractiveForces3DAnalytic) {
    const da_int n = 2, d = 3;
    std::vector<TypeParam> emb = {0, 0, 0, 1, 1, 1};

    std::vector<da_int> row_ptr = {0, 1, 2};
    std::vector<da_int> col_idx = {1, 0};
    std::vector<TypeParam> p_vals = {TypeParam(0.5), TypeParam(0.5)};

    std::vector<TypeParam> repulsive(n * d, 0);
    TypeParam sum_q = 1;

    std::vector<TypeParam> grad(n * d, 0);
    da_tsne::compute_attractive_forces(
        n, d, TypeParam(1), row_ptr, col_idx, p_vals, emb, repulsive, sum_q, grad,
        &da_tsne::attractive_forces_scalar_impl<TypeParam, 3>);

    // dist2=3, q=1/4, grad_i = 4 * p_ij * q * (y_i - y_j) = +/-0.5
    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-5) : TypeParam(1e-10);
    const TypeParam g = TypeParam(0.5);
    EXPECT_NEAR(grad[0], -g, tol);
    EXPECT_NEAR(grad[1], -g, tol);
    EXPECT_NEAR(grad[2], -g, tol);
    EXPECT_NEAR(grad[3], g, tol);
    EXPECT_NEAR(grad[4], g, tol);
    EXPECT_NEAR(grad[5], g, tol);
}

// d=3 kernel specialization. Verifies avx/avx2 variants against scalar.
TYPED_TEST(tsne_internal_test, AttractiveForces3D) {
    using v = vectorization_type;
    const da_int n = 3, d = 3;
    // y0=(0,0,0), y1=(1,0,0), y2=(0,1,1)
    std::vector<TypeParam> emb = {0, 0, 0, 1, 0, 0, 0, 1, 1};

    std::vector<da_int> row_ptr = {0, 2, 4, 6};
    std::vector<da_int> col_idx = {1, 2, 0, 2, 0, 1};
    std::vector<TypeParam> p_vals = {TypeParam(0.20), TypeParam(0.15), TypeParam(0.20),
                                     TypeParam(0.15), TypeParam(0.15), TypeParam(0.15)};

    std::vector<TypeParam> repulsive(n * d, 0);
    TypeParam sum_q = 1;

    auto implementations = da_tsne::testing::get_attractive_forces_implementations();

    std::vector<std::tuple<std::string, da_tsne::attractive_forces_kernel_fn<TypeParam>>>
        kernel_list = {{"avx", implementations[d - 2]->get<TypeParam>(v::avx)},
                       {"avx2", implementations[d - 2]->get<TypeParam>(v::avx2)}};

    std::vector<TypeParam> expected(n * d, 0);

    da_tsne::compute_attractive_forces(
        n, d, TypeParam(1), row_ptr, col_idx, p_vals, emb, repulsive, sum_q, expected,
        &da_tsne::attractive_forces_scalar_impl<TypeParam, 3>);

    const TypeParam tol =
        std::is_same_v<TypeParam, float> ? TypeParam(1e-5) : TypeParam(1e-10);

    for (const auto &[name, kernel] : kernel_list) {
        std::cout << "d=3, vectorisation: " << name << std::endl;
        std::vector<TypeParam> grad(n * d, 0);
        da_tsne::compute_attractive_forces(n, d, TypeParam(1), row_ptr, col_idx, p_vals,
                                           emb, repulsive, sum_q, grad, kernel);
        EXPECT_ARR_NEAR(n * d, grad.data(), expected.data(), tol);
    }
}

// 2-point analytic test for d=1 scalar kernel (used when n_components == 1).
// y0=0, y1=2 => dist2=4, q=1/(1+4)=0.2, p=0.5
// attractive_i = p * q * (y_i - y_j) => +/-0.2
// grad_i = 4 * attractive_i = +/-0.8 (with zero repulsive and sum_q=1)
TYPED_TEST(tsne_internal_test, AttractiveForces1DAnalytic) {
    const da_int n = 2, d = 1;
    std::vector<TypeParam> emb = {0, 2};

    std::vector<da_int> row_ptr = {0, 1, 2};
    std::vector<da_int> col_idx = {1, 0};
    std::vector<TypeParam> p_vals = {TypeParam(0.5), TypeParam(0.5)};

    std::vector<TypeParam> repulsive(n * d, 0);
    TypeParam sum_q = 1;

    std::vector<TypeParam> grad(n * d, 0);
    da_tsne::compute_attractive_forces(
        n, d, TypeParam(1), row_ptr, col_idx, p_vals, emb, repulsive, sum_q, grad,
        &da_tsne::attractive_forces_scalar_impl<TypeParam, 1>);

    const TypeParam tol = 1e-6;
    EXPECT_NEAR(grad[0], TypeParam(-0.8), tol);
    EXPECT_NEAR(grad[1], TypeParam(0.8), tol);
}

// =============================================================================
// compute_affinities tests
// =============================================================================

// Integration test on a 4-point unit square with exact distances. Verifies
// CSR structure, non-negativity, sum-to-1, P[i][j]==P[j][i] symmetry, and sorted columns.
TYPED_TEST(tsne_internal_test, AffinitiesSymmetric4Point) {
    const da_int n_samples = 4, n_features = 2;
    std::vector<TypeParam> data = {0, 0, 1, 0, 0, 1, 1, 1};

    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    std::vector<da_int> row_ptr, col_idx;
    std::vector<TypeParam> vals;
    ASSERT_EQ(da_tsne::compute_affinities(TypeParam(2), true, n_samples, n_features,
                                          data.data(), &err, row_ptr, col_idx, vals),
              da_status_success);

    // CSR structure checks
    ASSERT_EQ(row_ptr.size(), (size_t)(n_samples + 1));
    EXPECT_EQ(row_ptr[0], 0);
    EXPECT_EQ(row_ptr[n_samples], (da_int)vals.size());

    // All values non-negative
    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_GE(vals[i], TypeParam(0)) << "Negative P value at index " << i;

    // P sums to ~1
    TypeParam total = std::accumulate(vals.begin(), vals.end(), TypeParam(0));
    EXPECT_NEAR(total, TypeParam(1), TypeParam(0.01));

    // Symmetry: P[i][j] == P[j][i]
    for (da_int i = 0; i < n_samples; ++i) {
        for (da_int ki = row_ptr[i]; ki < row_ptr[i + 1]; ++ki) {
            da_int j = col_idx[ki];
            bool found = false;
            for (da_int kj = row_ptr[j]; kj < row_ptr[j + 1]; ++kj) {
                if (col_idx[kj] == i) {
                    EXPECT_NEAR(vals[ki], vals[kj], 1e-8);
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "Missing P[" << j << "][" << i << "]";
        }
    }

    // Column indices sorted within each row
    for (da_int i = 0; i < n_samples; ++i)
        for (da_int k = row_ptr[i]; k < row_ptr[i + 1] - 1; ++k)
            EXPECT_LT(col_idx[k], col_idx[k + 1]);
}

// Boundary integration test: n=2 is the minimal non-trivial input, producing
// exactly 2 CSR entries (one neighbor per point).
TYPED_TEST(tsne_internal_test, AffinitiesTwoPoint) {
    const da_int n_samples = 2, n_features = 2;
    std::vector<TypeParam> data = {0, 0, 1, 1};

    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    std::vector<da_int> row_ptr, col_idx;
    std::vector<TypeParam> vals;
    ASSERT_EQ(da_tsne::compute_affinities(TypeParam(1), true, n_samples, n_features,
                                          data.data(), &err, row_ptr, col_idx, vals),
              da_status_success);

    ASSERT_EQ(row_ptr.size(), (size_t)(n_samples + 1));
    // Each point has exactly 1 neighbor => 2 directed entries
    EXPECT_EQ((da_int)vals.size(), 2);
}
