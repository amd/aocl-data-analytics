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
#ifndef TSNE_HPP
#define TSNE_HPP

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "da_kernel_utils.hpp"
#include "macros.h"
#include <cstdint>
#include <type_traits>
#include <vector>

namespace ARCH {

namespace da_tsne {

// ============================================================================
// Barnes-Hut Tree Types
// ============================================================================

// Narrow integer types are deliberate:
// int8_t for D (always 1, 2, or 3);
// int32_t/int16_t instead of da_int keeps each node compact
// reducing cache pressure in the traversal hot loop.
// int32_t supports up to ~2 billion nodes/points;
template <typename T, int8_t D> struct TreeNode {
    T com[D];
    int32_t cnt;
    int32_t point_start;
    int32_t first_child;
    int8_t num_children;
};

template <typename T, int8_t D> struct BarnesHutTree {
    static_assert(D >= 1 && D <= 3);

    static constexpr da_int NC = 1 << D;
    static constexpr da_int MAX_DEPTH = 32;
    static constexpr da_int STACK_SIZE = NC * MAX_DEPTH + NC;
    static constexpr T DIST2_FLOOR = std::is_same_v<T, float> ? (T)1e-6 : (T)1e-8;
    static constexpr T EPS_INC = (T)1 + DIST2_FLOOR;
    static constexpr da_int DEFAULT_LEAF_THRESHOLD = 8;
    static constexpr da_int TASK_CUTOFF = 512;

    std::vector<TreeNode<T, D>> nodes;
    std::vector<T> sorted_pos;
    std::vector<da_int> point_pos;
    T dist2_thresh[MAX_DEPTH]{};

    const T *points = nullptr;
    da_int n = 0, num_nodes = 0, capacity = 0;
    T theta = (T)0;
    T four_inv_theta_sq = (T)0;
    da_int leaf_threshold = DEFAULT_LEAF_THRESHOLD;

    std::vector<da_int> sorted_indices;
    bool memory_initialized = false;

    T bbox_min[3]{};
    T bbox_max[3]{};
    T bbox_center[3]{};
    T bbox_width = (T)0;

    BarnesHutTree(const T *pts, da_int n_in, T theta_in = (T)0,
                  da_int leaf_thresh = DEFAULT_LEAF_THRESHOLD)
        : points(pts), n(n_in), theta(theta_in), leaf_threshold(leaf_thresh) {}

    void initialize_memory();
    void compute_bbox();
    void make_leaf(da_int node_idx, da_int start, da_int end);
    void build_subtree(da_int node_idx, da_int start, da_int end, da_int level,
                       const T *center, T hw);
    void build();
    void compute_repulsive_batch4(const da_int pt_idx[4], T force_out[4][D],
                                  T sum_q_out[4], int32_t *node_stack,
                                  int32_t *depth_stack) const;
};

// ============================================================================
// Kernel type aliases
// ============================================================================

template <typename T>
using attractive_forces_kernel_fn = void (*)(const T *, const da_int *, const T *,
                                             const T *, T, da_int, da_int, T *);

template <typename T> class tsne : public basic_handle<T> {
  public:
    template <class> friend class tsne;

  private:
    da_int n_samples = 0;
    da_int n_features = 0;
    da_int n_components = 2;
    da_int max_iter = 0;
    da_int n_iter_performed = 0;

    bool initdone = false;
    bool model_trained = false;

    // If we are constructing via the bypass constructor, skip option reading in compute()
    bool check_options = true;

    // Mixed precision iterative refinement
    bool use_mixed_precision = false;
    da_int lp_n_iter = 0;
    da_int start_iter = 0;

    // Lower precision type (int16_t is a proxy for bfloat16, though we don't use it yet)
    using lp_type =
        typename std::conditional<std::is_same_v<T, double>, float, _Float16>::type;

    const T *X = nullptr;  // contiguous row-major view of input data
    std::vector<T> X_copy; // owned storage (used when copy/transpose is needed)
    std::vector<T> embedding;
    std::vector<T> supplied_embedding;
    da_int supplied_n_components = 0;
    bool has_supplied_embedding = false;
    // Sparse P matrix in CSR format
    std::vector<da_int> P_row_ptr; // Size: n_samples + 1
    std::vector<da_int> P_col_idx; // Size: nnz (number of non-zeros)
    std::vector<T> P_values;       // Size: nnz
    T kl_divergence = 0;
    std::vector<T> iY;
    std::vector<T> gains;

    // Options stored as members (set by option reading or bypass constructor)
    T learning_rate = (T)-1;
    T early_exaggeration = (T)12;
    T theta = (T)0.5;
    T min_grad_norm = (T)1e-7;
    da_int n_iter_without_progress = 300;
    std::string init_method = "random";
    da_int seed = 0;

    void refresh() override;
    da_status initialize_embedding(const std::string &init_method, da_int seed);
    da_status gradient_descent(T learning_rate, T early_exaggeration, T theta,
                               da_int n_iter_without_progress, T min_grad_norm);
    template <int8_t D>
    da_status gradient_descent_impl(T learning_rate, T early_exaggeration, T theta,
                                    da_int n_iter_without_progress, T min_grad_norm);

    da_status lower_precision_init();

    attractive_forces_kernel_fn<T> attractive_force_kernel_fn = nullptr;
    void assign_attractive_force_kernel();

  public:
    tsne(da_errors::da_error_t &err);

    // Bypass constructor: sets member variables directly, skipping option reading in compute()
    tsne(da_errors::da_error_t &err, da_int n_samples, da_int n_features,
         da_int n_components, da_int max_iter, T learning_rate, T early_exaggeration,
         T theta, T min_grad_norm, da_int n_iter_without_progress,
         std::vector<da_int> &&P_row_ptr, std::vector<da_int> &&P_col_idx,
         std::vector<T> &&P_values, std::vector<T> &&embedding,
         const std::string &init_method, da_int seed);

    ~tsne() = default;

    da_status get_result(da_result query, da_int *dim, T *result) override;
    da_status get_result(da_result query, da_int *dim, da_int *result) override;

    da_status set_data(da_int n_samples, da_int n_features, const T *X_in, da_int ldx_in);
    da_status set_init_embedding(const T *Y_in, da_int ldy_in);
    da_status compute();
};

// Binary search for bandwidth (precision) to match target perplexity for one row.
template <typename T>
void compute_row_probabilities(const T *sq_distances, da_int k, T log_perplexity,
                               T *row_prob);

template <typename T>
da_status compute_affinities(T perplexity, bool use_exact, da_int n_samples,
                             da_int n_features, const T *X, da_errors::da_error_t *err,
                             std::vector<da_int> &P_row_ptr,
                             std::vector<da_int> &P_col_idx, std::vector<T> &P_values);

template <typename T, int8_t D>
void compute_repulsive_forces(BarnesHutTree<T, D> &tree, da_int n, T *repulsive,
                              T &sum_q_total, std::vector<T> &thread_sum_q);

template <typename T>
T compute_kl_divergence(da_int n, da_int d, const std::vector<da_int> &row_ptr,
                        const std::vector<da_int> &col_idx, const std::vector<T> &p_vals,
                        const std::vector<T> &emb, T sum_q_total, std::vector<T> &work);

template <typename T>
void compute_attractive_forces(da_int n, da_int d, T exaggeration,
                               const std::vector<da_int> &row_ptr,
                               const std::vector<da_int> &col_idx,
                               const std::vector<T> &p_vals, const std::vector<T> &emb,
                               const std::vector<T> &repulsive, T sum_q_total,
                               std::vector<T> &grad,
                               attractive_forces_kernel_fn<T> kernel);

template <typename T>
void update_embedding(da_int n, da_int d, T momentum, T learning_rate,
                      std::vector<T> &grad, std::vector<T> &iY, std::vector<T> &gains,
                      std::vector<T> &emb);

template <typename T>
da_status symmetrize_to_csr(da_int n, da_int k, const da_int *neighbor_indices,
                            const T *neighbor_probs, std::vector<da_int> &P_row_ptr,
                            std::vector<da_int> &P_col_idx, std::vector<T> &P_values);

// Kernel function pointer type aliases
using KFS = attractive_forces_kernel_fn<float>;
using KFD = attractive_forces_kernel_fn<double>;
using KFH = attractive_forces_kernel_fn<_Float16>;

namespace testing {
// accessors for unit tests
const std::array<const kernel_implementations<KFS, KFD, KFH> *, 2> &
get_attractive_forces_implementations();
} // namespace testing

} // namespace da_tsne

} // namespace ARCH

#endif
