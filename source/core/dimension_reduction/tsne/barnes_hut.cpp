/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "da_omp.hpp"
#include "da_std.hpp"
#include "tsne.hpp"
#include <algorithm>
#include <array>
#include <numeric>

namespace ARCH {

namespace da_tsne {

// ============================================================================
// BarnesHutTree method definitions
// ============================================================================

template <typename T, int8_t D> void BarnesHutTree<T, D>::initialize_memory() {
    if (memory_initialized && n > 0 && capacity / NC >= n)
        return;

    capacity = std::max((da_int)256, NC * n + 64);
    nodes.resize(capacity);
    sorted_indices.resize(n);
    point_pos.resize(n);
    memory_initialized = true;
}

template <typename T, int8_t D> void BarnesHutTree<T, D>::compute_bbox() {
    if (n == 0)
        return;

    T bmin[D], bmax[D];
    da_std::fill(bmin, bmin + D, std::numeric_limits<T>::max());
    da_std::fill(bmax, bmax + D, std::numeric_limits<T>::lowest());

#pragma omp parallel for reduction(min : bmin[ : D]) reduction(max : bmax[ : D])
    for (da_int i = 0; i < n; ++i) {
        const T *p = points + i * D;
        for (da_int d = 0; d < D; ++d) {
            bmin[d] = std::min(bmin[d], p[d]);
            bmax[d] = std::max(bmax[d], p[d]);
        }
    }
    da_std::copy(bmin, bmin + D, bbox_min);
    da_std::copy(bmax, bmax + D, bbox_max);

    // Use the widest dimension as bbox_width so the root cell is a square/cube
    T max_range = (T)1e-10;
    for (da_int d = 0; d < D; ++d)
        max_range = std::max(max_range, bbox_max[d] - bbox_min[d]);
    bbox_width = max_range;

    const T half = bbox_width / (T)2;
    for (da_int d = 0; d < D; ++d) {
        bbox_center[d] = (bbox_min[d] + bbox_max[d]) / (T)2;
        bbox_min[d] = bbox_center[d] - half;
        bbox_max[d] = bbox_center[d] + half;
    }

    // Identity permutation; build_subtree will reorder this in-place
    da_std::iota(sorted_indices.begin(), sorted_indices.end(), da_int(0));
}

template <typename T, int8_t D>
void BarnesHutTree<T, D>::make_leaf(da_int node_idx, da_int start, da_int end) {
    const da_int count = end - start;
    T sum[D] = {};
    for (da_int i = start; i < end; ++i) {
        const T *p = points + sorted_indices[i] * D;
        for (da_int d = 0; d < D; ++d)
            sum[d] += p[d];
    }
    TreeNode<T, D> &nd = nodes[node_idx];
    nd.cnt = (int32_t)count;
    nd.point_start = (int32_t)start;
    nd.first_child = 0;
    nd.num_children = 0;
    for (da_int d = 0; d < D; ++d)
        nd.com[d] = sum[d] / count;
}

template <typename T, int8_t D>
void BarnesHutTree<T, D>::build_subtree(da_int node_idx, da_int start, da_int end,
                                        da_int level, const T *center, T hw) {
    const da_int count = end - start;

    if (count == 0) {
        TreeNode<T, D> &nd = nodes[node_idx];
        nd.cnt = 0;
        nd.point_start = (int32_t)start;
        nd.first_child = 0;
        nd.num_children = 0;
        return;
    }

    if (count <= leaf_threshold || level >= MAX_DEPTH) {
        make_leaf(node_idx, start, end);
        return;
    }

    // Map a point to its child octant: bit d is set if p[d] >= center[d],
    // giving an index in [0, NC) that identifies the child quadrant/octant.
    auto classify_point = [&](const T *p) {
        da_int idx = 0;
        for (da_int d = 0; d < D; ++d)
            if (p[d] >= center[d])
                idx |= (1 << d);
        return idx;
    };

    // Count how many points fall into each child octant
    std::array<da_int, NC + 1> boundaries;
    std::array<da_int, NC> counts{};

    for (da_int i = start; i < end; ++i)
        counts[classify_point(points + sorted_indices[i] * D)]++;

    // Prefix-sum to get the start/end boundaries for each octant
    boundaries[0] = start;
    for (da_int c = 0; c < NC; ++c)
        boundaries[c + 1] = boundaries[c] + counts[c];

    // In-place partition of sorted_indices by swapping each point into its
    // correct octant region (cycle-sort style, O(n) swaps).
    std::array<da_int, NC> write_pos;
    for (da_int c = 0; c < NC; ++c)
        write_pos[c] = boundaries[c];

    for (da_int c = 0; c < NC; ++c) {
        while (write_pos[c] < boundaries[c + 1]) {
            da_int child_idx = classify_point(points + sorted_indices[write_pos[c]] * D);
            if (child_idx == c) {
                write_pos[c]++;
            } else {
                std::swap(sorted_indices[write_pos[c]],
                          sorted_indices[write_pos[child_idx]]);
                write_pos[child_idx]++;
            }
        }
    }

    // Only allocate tree nodes for non-empty octants
    da_int nonempty = 0;
    da_int nonempty_map[NC];
    for (da_int c = 0; c < NC; ++c) {
        if (boundaries[c + 1] > boundaries[c])
            nonempty_map[nonempty++] = c;
    }

    T child_hw = hw / (T)2;

    // Reserve a contiguous block of node slots for the children (thread-safe)
    da_int first_child;
#pragma omp atomic capture
    {
        first_child = num_nodes;
        num_nodes += nonempty;
    }

    // Fall back to a leaf if the arena is full
    if (first_child + nonempty > capacity) {
#pragma omp atomic
        num_nodes -= nonempty;
        make_leaf(node_idx, start, end);
        return;
    }

    // Recurse into non-empty children; spawn OpenMP tasks for large subtrees
    for (da_int s = 0; s < nonempty; ++s) {
        const da_int c = nonempty_map[s];
        std::array<T, D> cc;
        for (da_int d = 0; d < D; ++d)
            cc[d] = center[d] + (((c >> d) & 1) ? child_hw : -child_hw);
        const da_int b_start = boundaries[c];
        const da_int b_end = boundaries[c + 1];
        const da_int child_node = first_child + s;
        const da_int next_level = level + 1;
#pragma omp task firstprivate(cc, b_start, b_end, child_node, next_level,                \
                                  child_hw) if (count > TASK_CUTOFF)
        build_subtree(child_node, b_start, b_end, next_level, cc.data(), child_hw);
    }
#pragma omp taskwait

    nodes[node_idx].first_child = (int32_t)first_child;
    nodes[node_idx].num_children = (int16_t)nonempty;

    // Compute this node's centre of mass as the weighted average of its children
    int32_t total_cnt = 0;
    T com_sum[D] = {};
    for (da_int s = 0; s < nonempty; ++s) {
        const da_int ci = first_child + s;
        const int32_t cc = nodes[ci].cnt;
        if (cc > 0) {
            total_cnt += cc;
            const T w = (T)cc;
            for (da_int d = 0; d < D; ++d)
                com_sum[d] += w * nodes[ci].com[d];
        }
    }
    TreeNode<T, D> &root_nd = nodes[node_idx];
    root_nd.cnt = total_cnt;
    root_nd.point_start = (int32_t)start;
    if (total_cnt > 0) {
        const T inv = (T)1 / (T)total_cnt;
        for (da_int d = 0; d < D; ++d)
            root_nd.com[d] = com_sum[d] * inv;
    }
}

template <typename T, int8_t D> void BarnesHutTree<T, D>::build() {
    if (n == 0)
        return;

    initialize_memory();
    num_nodes = 0;

    compute_bbox();
    num_nodes++;

#pragma omp parallel
#pragma omp single
    build_subtree(0, 0, n, 0, bbox_center, bbox_width / (T)2);

    // Materialise points in tree-sorted order for cache-friendly leaf traversal
    sorted_pos.resize(n * D);
#pragma omp parallel for schedule(static) default(none)                                  \
    shared(n, sorted_indices, points, point_pos, sorted_pos)
    for (da_int k = 0; k < n; ++k) {
        point_pos[sorted_indices[k]] = k;
        const T *src = points + sorted_indices[k] * D;
        T *dst = sorted_pos.data() + k * D;
        for (da_int dd = 0; dd < D; ++dd)
            dst[dd] = src[dd];
    }

    four_inv_theta_sq =
        (theta > (T)0) ? (T)4 / (theta * theta) : std::numeric_limits<T>::infinity();

    // Precompute per-level distance thresholds for the Barnes-Hut opening
    // criterion: cell_width^2 / theta^2.  Each level halves the cell width,
    // so the squared threshold shrinks by 4x.
    const T root_hw = bbox_width / (T)2;
    T level_dist2 = root_hw * root_hw * four_inv_theta_sq;
    for (da_int l = 0; l < MAX_DEPTH; ++l) {
        dist2_thresh[l] = level_dist2 + DIST2_FLOOR;
        level_dist2 *= (T)0.25;
    }
}

// Shared-traversal batch-4: walks the tree once for 4 query points, reducing the traversal overhead
template <typename T, int8_t D>
void BarnesHutTree<T, D>::compute_repulsive_batch4(const da_int pt_idx[4],
                                                   T force_out[4][D], T sum_q_out[4],
                                                   int32_t *node_stack,
                                                   int32_t *depth_stack) const {
    if (num_nodes == 0) {
        for (da_int b = 0; b < 4; ++b) {
            sum_q_out[b] = (T)0;
            for (da_int d = 0; d < D; ++d)
                force_out[b][d] = (T)0;
        }
        return;
    }

    // Convert points from Array of Structures (AoS) to Structure of Arrays (SoA) for vectorization
    T pt[D][4];
    for (da_int b = 0; b < 4; ++b)
        for (da_int d = 0; d < D; ++d)
            pt[d][b] = points[pt_idx[b] * D + d];

    T f[D][4] = {};
    T sq[4] = {};
    const T *sorted_pos_p = sorted_pos.data();
    const da_int *sorted_indices_p = sorted_indices.data();
    const da_int query_pos[4] = {point_pos[pt_idx[0]], point_pos[pt_idx[1]],
                                 point_pos[pt_idx[2]], point_pos[pt_idx[3]]};
    constexpr da_int include_all[4] = {1, 1, 1, 1};

    // Initialize the traversal stack with the root node (index 0) at depth 0
    da_int stack_top = 0;
    node_stack[stack_top] = 0;
    depth_stack[stack_top++] = 0;

    while (stack_top > 0) {
        --stack_top;
        const da_int node = node_stack[stack_top];
        const da_int level = depth_stack[stack_top];

        if (node < 0 || node >= num_nodes)
            continue;

        const TreeNode<T, D> &nd = nodes[node];
        const int32_t cnt = nd.cnt;
        if (cnt == 0)
            continue;

        // Compute displacement and squared distance (vectorized)
        T delta[D][4], dist2[4];
        for (da_int b = 0; b < 4; ++b)
            dist2[b] = (T)0;
        for (da_int d = 0; d < D; ++d) {
            const T com_d = nd.com[d];
#pragma omp simd
            for (da_int b = 0; b < 4; ++b) {
                delta[d][b] = pt[d][b] - com_d;
                dist2[b] += delta[d][b] * delta[d][b];
            }
        }

        const T thresh =
            (level < MAX_DEPTH) ? dist2_thresh[level] : dist2_thresh[MAX_DEPTH - 1];
        const bool is_internal = nd.num_children > 0;

        // Accumulate the t-SNE repulsive kernel: weight / (1 + ||delta||^2)^2
        // for all 4 query points. Also accumulates the normalisation sum q
        auto accum_force = [&](T(&delta_in)[D][4], T(&dist2_in)[4], T weight,
                               const da_int include[4]) {
            T scale[4];
#pragma omp simd
            for (da_int b = 0; b < 4; ++b) {
                const T dxy1 = dist2_in[b] + EPS_INC;
                const T lane_mask = (T)include[b];
                scale[b] = lane_mask * weight / (dxy1 * dxy1);
                sq[b] += scale[b] * dxy1;
            }
            for (da_int d = 0; d < D; ++d)
#pragma omp simd
                for (da_int b = 0; b < 4; ++b)
                    f[d][b] += delta_in[d][b] * scale[b];
        };

        if (cnt == 1) {
            const da_int leaf_point_idx = sorted_indices_p[nd.point_start];
            da_int include[4];
            for (da_int b = 0; b < 4; ++b)
                include[b] = (leaf_point_idx != pt_idx[b]);
            accum_force(delta, dist2, (T)1, include);
        } else if (!is_internal) {
            // Leaf with multiple points: iterate each stored point exactly
            const da_int leaf_start = (da_int)nd.point_start;
            for (da_int c = 0; c < cnt; ++c) {
                const da_int leaf_point_idx = sorted_indices_p[leaf_start + c];
                const T *leaf_point = sorted_pos_p + (leaf_start + c) * D;
                T leaf_delta[D][4], leaf_dist2[4];
                for (da_int b = 0; b < 4; ++b)
                    leaf_dist2[b] = (T)0;
                for (da_int d = 0; d < D; ++d) {
                    const T coord = leaf_point[d];
#pragma omp simd
                    for (da_int b = 0; b < 4; ++b) {
                        leaf_delta[d][b] = pt[d][b] - coord;
                        leaf_dist2[b] += leaf_delta[d][b] * leaf_delta[d][b];
                    }
                }
                da_int include[4];
                for (da_int b = 0; b < 4; ++b)
                    include[b] = (leaf_point_idx != pt_idx[b]);
                accum_force(leaf_delta, leaf_dist2, (T)1, include);
            }
        } else {
            // Most frequent case: internal node with multiple points
            bool all_far = true;
            for (da_int b = 0; b < 4; ++b) {
                if (dist2[b] < thresh) {
                    all_far = false;
                    break;
                }
            }

            const bool can_approximate =
                all_far || stack_top + nd.num_children > STACK_SIZE;
            if (can_approximate) {
                const da_int node_start = (da_int)nd.point_start;
                const da_int node_end = node_start + cnt;
                bool contains_any_query = false;
                for (da_int b = 0; b < 4; ++b) {
                    if (node_start <= query_pos[b] && query_pos[b] < node_end) {
                        contains_any_query = true;
                        break;
                    }
                }
                if (!contains_any_query) {
                    // Barnes-Hut approximation: treat the whole cell as one body
                    accum_force(delta, dist2, (T)cnt, include_all);
                    continue;
                }
            }

            // At least one query point is too close; descend into children.
            // Push in reverse order so child 0 is processed first (depth-first).
            const da_int nch = nd.num_children;
            const int32_t offset = nd.first_child;
            for (da_int c = nch - 1; c >= 0; --c) {
                node_stack[stack_top] = offset + c;
                depth_stack[stack_top++] = level + 1;
            }
        }
    }

    for (da_int b = 0; b < 4; ++b) {
        sum_q_out[b] = sq[b];
        for (da_int d = 0; d < D; ++d)
            force_out[b][d] = f[d][b];
    }
}

// Explicit template instantiations
template struct BarnesHutTree<float, 1>;
template struct BarnesHutTree<float, 2>;
template struct BarnesHutTree<float, 3>;
template struct BarnesHutTree<double, 1>;
template struct BarnesHutTree<double, 2>;
template struct BarnesHutTree<double, 3>;

} // namespace da_tsne

} // namespace ARCH
