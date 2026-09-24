/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TREE_INFERENCE_HPP
#define TREE_INFERENCE_HPP

#include "aoclda.h"
#include "da_kernel_utils.hpp"
#include "decision_tree.hpp"
#include "macros.h"

namespace ARCH {
namespace da_decision_forest {

constexpr std::size_t scalar_threshold = 1 << 11;   // 2 KB
constexpr std::size_t L2_xl = (std::size_t)1 << 20; // 1 MB
constexpr std::size_t L2_s = (std::size_t)1 << 19;  // 512 KB
constexpr size_t node_size = 2 * sizeof(sizeof(int32_t)) + sizeof(float);

enum traversal_mode { undefined = 0, seq, bfs };

struct traversal_mode_t {
    dispatch_architecture arch = dispatch_architecture::generic;
    size_t threshold = 0;
};

// clang-format off
constexpr traversal_mode_t
traversal_mode_table[] = {
    {generic, L2_s},
    {zen2,    L2_s},
    {zen3,    L2_s},
    {zen4,    L2_xl},
    {zen5,    L2_xl},
    {zen6,    L2_xl},
    {generic_avx512, L2_xl}
};
// clang-format on

traversal_mode traversal_dispatcher(size_t tree_footprint, da_order order,
                                    bool has_onevall_features) {
    traversal_mode tmode = traversal_mode::seq;
    auto *ctx = context::get_context();
    bool vectorization_available = order == row_major && !has_onevall_features;

    if (!vectorization_available) {
        if (tree_footprint <= scalar_threshold)
            tmode = traversal_mode::seq;
        else
            tmode = traversal_mode::bfs;

    } else {
        tmode = traversal_mode::bfs;
        for (const auto &t : traversal_mode_table) {
            if (t.arch == ctx->arch && tree_footprint <= t.threshold) {
                tmode = traversal_mode::seq;
                break;
            }
        }
    }
    return tmode;
}

template <typename T>
da_status decision_tree<T>::compact_traverse(da_int nsamp, da_int nfeat, const T *X_test,
                                             da_int ldx_test, da_int *y_pred,
                                             bool validate_input) {

    if (validate_input) {
        if (y_pred == nullptr) {
            return da_error_bypass(this->err, da_status_invalid_pointer,
                                   "y_pred is not a valid pointer.");
        }

        if (nsamp <= 0) {
            return da_error_bypass(this->err, da_status_invalid_array_dimension,
                                   "n_samples = " + std::to_string(nsamp) +
                                       " must be > 0.");
        }

        if (nfeat != n_features) {
            return da_error_bypass(this->err, da_status_invalid_input,
                                   "n_features = " + std::to_string(nfeat) +
                                       " doesn't match the expected value " +
                                       std::to_string(n_features) + ".");
        }

        if (!this->model_trained) {
            return da_error_bypass(this->err, da_status_out_of_date,
                                   "The model has not yet been trained or the data it is "
                                   "associated with is out of date.");
        }

        da_status status =
            this->check_2D_array(this->order, nsamp, nfeat, X_test, ldx_test, "n_samples",
                                 "n_features", "X_test", "ldx_test");
        if (status != da_status_success) {
            return status;
        }
    }
    vectorization_type isa = Oracle("tree.isa");
    const std::size_t tree_footprint = n_nodes * node_size;
    traversal_mode tmode =
        traversal_dispatcher(tree_footprint, this->order, has_onevall_features);

    switch (tmode) {
    case traversal_mode::seq:
        // small trees: going through input data is more likely to be a bottleneck rather
        // than traversing the tree.
        if (this->order == row_major) {
            if (!has_onevall_features)
                return compact_traverse_sequential_vectorized(nsamp, X_test, ldx_test,
                                                              y_pred, isa);
            else
                return compact_traverse_sequential<row_major, true>(nsamp, X_test,
                                                                    ldx_test, y_pred);
        } else {
            if (!has_onevall_features)
                return compact_traverse_sequential<column_major, false>(nsamp, X_test,
                                                                        ldx_test, y_pred);
            else
                return compact_traverse_sequential<column_major, true>(nsamp, X_test,
                                                                       ldx_test, y_pred);
        }
        break;
    case traversal_mode::bfs:
        // larger trees: ensure each node of the tree is visited only once
        if (this->order == row_major) {
            if (!has_onevall_features)
                return compact_traverse_BFS<row_major, false>(nsamp, X_test, ldx_test,
                                                              y_pred);
            else
                return compact_traverse_BFS<row_major, true>(nsamp, X_test, ldx_test,
                                                             y_pred);
        } else {
            if (!has_onevall_features)
                return compact_traverse_BFS<column_major, false>(nsamp, X_test, ldx_test,
                                                                 y_pred);
            else
                return compact_traverse_BFS<column_major, true>(nsamp, X_test, ldx_test,
                                                                y_pred);
        }
        break;
    default:
        return da_error_bypass(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unknown tree traversal mode.");
    }
}

template <typename T>
da_status decision_tree<T>::compact_traverse(da_int nsamp, da_int nfeat, const T *X_test,
                                             da_int ldx_test, T *y_proba, da_int n_class,
                                             da_int ldy, bool validate_input) {
    da_status status;
    if (validate_input) {
        if (y_proba == nullptr) {
            return da_error_bypass(this->err, da_status_invalid_pointer,
                                   "y_proba is not a valid pointer.");
        }

        if (nsamp <= 0) {
            return da_error_bypass(this->err, da_status_invalid_array_dimension,
                                   "n_samples = " + std::to_string(nsamp) +
                                       " must be > 0.");
        }

        if (nfeat != n_features) {
            return da_error_bypass(this->err, da_status_invalid_input,
                                   "n_features = " + std::to_string(nfeat) +
                                       " doesn't match the expected value " +
                                       std::to_string(n_features) + ".");
        }
        if (n_class != this->n_class) {
            return da_error_bypass(this->err, da_status_invalid_input,
                                   "n_class = " + std::to_string(n_class) +
                                       " doesn't match the expected value " +
                                       std::to_string(this->n_class) + ".");
        }

        if (!this->model_trained) {
            return da_error_bypass(this->err, da_status_out_of_date,
                                   "The model has not yet been trained or the data it is "
                                   "associated with is out of date.");
        }

        // call the da_utils version directly to never check the data even if the option is set
        status = da_utils::check_2D_array(false, this->order, this->err, nsamp, nfeat,
                                          X_test, ldx_test, "n_samples", "n_features",
                                          "X_test", "ldx_test", 1, 1);
        if (status != da_status_success) {
            return status;
        }
    }

    vectorization_type isa = Oracle("tree.isa");
    const std::size_t tree_footprint = n_nodes * node_size;
    traversal_mode tmode =
        traversal_dispatcher(tree_footprint, this->order, has_onevall_features);

    switch (tmode) {
    case traversal_mode::seq:
        if (this->order == row_major) {
            if (!has_onevall_features)
                return compact_traverse_sequential_vectorized(nsamp, X_test, ldx_test,
                                                              y_proba, n_class, ldy, isa);
            else
                return compact_traverse_sequential<row_major, true>(
                    nsamp, X_test, ldx_test, y_proba, n_class, ldy);
        } else {
            if (!has_onevall_features)
                return compact_traverse_sequential<column_major, false>(
                    nsamp, X_test, ldx_test, y_proba, n_class, ldy);
            else
                return compact_traverse_sequential<column_major, true>(
                    nsamp, X_test, ldx_test, y_proba, n_class, ldy);
        }
        break;
    case traversal_mode::bfs:
        if (this->order == row_major) {
            if (!has_onevall_features)
                return compact_traverse_BFS<row_major, false>(nsamp, X_test, ldx_test,
                                                              y_proba, n_class, ldy);
            else
                return compact_traverse_BFS<row_major, true>(nsamp, X_test, ldx_test,
                                                             y_proba, n_class, ldy);
        } else {
            if (!has_onevall_features)
                return compact_traverse_BFS<column_major, false>(nsamp, X_test, ldx_test,
                                                                 y_proba, n_class, ldy);
            else
                return compact_traverse_BFS<column_major, true>(nsamp, X_test, ldx_test,
                                                                y_proba, n_class, ldy);
        }

        break;
    default:
        return da_error_bypass(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unknown tree traversal mode.");
    }
}

template <typename T>
template <da_order ORDER, bool HAS_CAT_FEAT>
da_status decision_tree<T>::compact_traverse_sequential(da_int nsamp, const T *X_test,
                                                        da_int ldx_test, da_int *y_pred) {
    for (da_int i = 0; i < nsamp; i++) {
        da_int current_idx = 0;
        const T *row = nullptr;
        if constexpr (ORDER == row_major)
            row = X_test + ldx_test * i;
        while (compact_tree[current_idx].feature_idx >= 0) {
            T feat_val;
            if constexpr (ORDER == row_major)
                feat_val = row[compact_tree[current_idx].feature_idx];
            else
                feat_val = X_test[i + ldx_test * compact_tree[current_idx].feature_idx];
            if constexpr (HAS_CAT_FEAT) {
                switch (feat_is_onevall[compact_tree[current_idx].feature_idx]) {
                case false:
                    current_idx = compact_tree[current_idx].next_idx +
                                  (da_int)std::signbit(
                                      compact_tree[current_idx].threshold - feat_val);
                    break;
                case true:
                    current_idx = compact_tree[current_idx].next_idx +
                                  ((std::abs(compact_tree[current_idx].threshold -
                                             feat_val) < feat_thresh)
                                       ? 0
                                       : 1);
                    break;
                }
            } else {
                current_idx =
                    compact_tree[current_idx].next_idx +
                    (da_int)std::signbit(compact_tree[current_idx].threshold - feat_val);
            }
        }
        y_pred[i] = compact_tree[current_idx].next_idx;
    }

    return da_status_success;
}

template <typename T>
template <da_order ORDER, bool HAS_CAT_FEAT>
da_status decision_tree<T>::compact_traverse_sequential(da_int nsamp, const T *X_test,
                                                        da_int ldx_test, T *y_proba,
                                                        da_int n_class, da_int ldy) {

    for (da_int i = 0; i < nsamp; i++) {
        da_int current_idx = 0;
        const T *row = nullptr;
        if constexpr (ORDER == row_major)
            row = X_test + ldx_test * i;
        while (compact_tree[current_idx].feature_idx >= 0) {
            T feat_val;
            if constexpr (ORDER == row_major)
                feat_val = row[compact_tree[current_idx].feature_idx];
            else
                feat_val = X_test[i + ldx_test * compact_tree[current_idx].feature_idx];
            if constexpr (HAS_CAT_FEAT) {
                switch (feat_is_onevall[compact_tree[current_idx].feature_idx]) {
                case false:
                    current_idx = compact_tree[current_idx].next_idx +
                                  (da_int)std::signbit(
                                      compact_tree[current_idx].threshold - feat_val);
                    break;
                case true:
                    current_idx = compact_tree[current_idx].next_idx +
                                  ((std::abs(compact_tree[current_idx].threshold -
                                             feat_val) < feat_thresh)
                                       ? 0
                                       : 1);
                    break;
                }
            } else
                current_idx =
                    compact_tree[current_idx].next_idx +
                    (da_int)std::signbit(compact_tree[current_idx].threshold - feat_val);
        }
        for (da_int j = 0; j < n_class; j++) {
            if constexpr (ORDER == row_major)
                y_proba[i * ldy + j] = class_props[current_idx * n_class + j];
            else
                y_proba[j * ldy + i] = class_props[current_idx * n_class + j];
        }
    }

    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::compact_traverse_sequential_vectorized(
    da_int nsamp, const T *X_test, da_int ldx, da_int *y_pred, vectorization_type isa) {

    da_int block_size;
    da_status status;

    switch (isa) {
    case vectorization_type::avx512:
#ifdef __AVX512F__
        block_size = 16;
        break;
        // Fall back to AVX2 if AVX512 is not available
#endif
    default:
        block_size = 8;
        break;
    }
    da_int n_blocks = nsamp / block_size;
    da_int remainder = nsamp % block_size;

    for (da_int b = 0; b < n_blocks; b++) {
        switch (isa) {
        case vectorization_type::avx512:
#ifdef __AVX512F__
            status = compact_traverse_sequential_avx512<false>(
                X_test + b * block_size * ldx, ldx, y_pred + b * block_size);
            break;
#endif
        default:
            status = compact_traverse_sequential_avx2<false>(
                X_test + b * block_size * ldx, ldx, y_pred + b * block_size);
            break;
        }
        if (status != da_status_success)
            return status;
    }

    if (remainder > 0) {
        status = compact_traverse_sequential<row_major, false>(
            remainder, X_test + n_blocks * block_size * ldx, ldx,
            y_pred + n_blocks * block_size);
        if (status != da_status_success)
            return status;
    }

    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::compact_traverse_sequential_vectorized(
    da_int nsamp, const T *X_test, da_int ldx, T *y_proba, da_int n_class, da_int ldy,
    vectorization_type isa) {

    da_int block_size;

    switch (isa) {
    case vectorization_type::avx512:
#ifdef __AVX512F__
        block_size = 16;
        break;
#endif
    default:
        block_size = 8;
        break;
    }
    da_int n_blocks = nsamp / block_size;
    da_int remainder = nsamp % block_size;
    std::vector<da_int> y_pred_block;

    try {
        y_pred_block.resize(block_size);
    } catch (const std::bad_alloc &) {
        return da_error_bypass(this->err, da_status_memory_error, //LCOV_EXCL_LINE
                               "Memory allocation failed for y_pred_block.");
    }

    for (da_int b = 0; b < n_blocks; b++) {
        da_status status;
        switch (isa) {
        case vectorization_type::avx512:
#ifdef __AVX512F__
            status = compact_traverse_sequential_avx512<true>(
                X_test + b * block_size * ldx, ldx, y_pred_block.data());
            break;
#endif
        default:
            status = compact_traverse_sequential_avx2<true>(X_test + b * block_size * ldx,
                                                            ldx, y_pred_block.data());
            break;
        }
        if (status != da_status_success)
            return status;
        for (da_int i = 0; i < block_size; i++) {
            for (da_int j = 0; j < n_class; j++) {
                y_proba[(b * block_size + i) * ldy + j] =
                    class_props[y_pred_block[i] * n_class + j];
            }
        }
    }

    if (remainder > 0) {
        da_status status = compact_traverse_sequential<row_major, false>(
            remainder, X_test + n_blocks * block_size * ldx, ldx,
            y_proba + n_blocks * block_size * ldy, n_class, ldy);
        if (status != da_status_success)
            return status;
    }

    return da_status_success;
}

#ifdef __AVX512F__
/* compute y_pred for 16 samples at a time using AVX-512 
 * - row major
 * - no categorical features 
 * - float
 */
template <typename T>
template <bool LEAF_RET>
da_status decision_tree<T>::compact_traverse_sequential_avx512(const T *X_block,
                                                               da_int ldx,
                                                               da_int *y_pred) {

    __m512i row_start = _mm512_set_epi32(
        15 * (int32_t)ldx, 14 * (int32_t)ldx, 13 * (int32_t)ldx, 12 * (int32_t)ldx,
        11 * (int32_t)ldx, 10 * (int32_t)ldx, 9 * (int32_t)ldx, 8 * (int32_t)ldx,
        7 * (int32_t)ldx, 6 * (int32_t)ldx, 5 * (int32_t)ldx, 4 * (int32_t)ldx,
        3 * (int32_t)ldx, 2 * (int32_t)ldx, 1 * (int32_t)ldx, 0);

    // Prepare the first iteration
    __mmask16 node_is_split = f_idx_cont[0] != -1 ? 0xffff : 0x0000;
    __m512i node_idx = _mm512_set1_epi32(0);
    __m512i feat_idx = _mm512_i32gather_epi32(node_idx, f_idx_cont.data(), 4);

    while (node_is_split) {
        __m512i nexti = _mm512_mask_i32gather_epi32(node_idx, node_is_split, node_idx,
                                                    next_idx_cont.data(), 4);
        __m512 thresh = _mm512_mask_i32gather_ps(_mm512_set1_ps(0), node_is_split,
                                                 node_idx, threshold_cont.data(), 4);
        __mmask16 go_right;
        if constexpr (std::is_same_v<T, float>) {
            __m512 feat_val = _mm512_mask_i32gather_ps(
                _mm512_set1_ps(0), node_is_split, _mm512_add_epi32(row_start, feat_idx),
                X_block, 4);
            go_right =
                _mm512_mask_cmp_ps_mask(node_is_split, feat_val, thresh, _CMP_GT_OQ);
        } else if constexpr (std::is_same_v<T, double>) {
            // Handle double precision by gathering 16 doubles in two 8-wide halves
            // From Benchmarks, this works faster than using double vectors directly
            __m512i gather_idx = _mm512_add_epi32(row_start, feat_idx);
            __m256i idx_lo = _mm512_castsi512_si256(gather_idx);
            __m256i idx_hi = _mm512_extracti32x8_epi32(gather_idx, 1);
            __m512d feat_lo = _mm512_mask_i32gather_pd(
                _mm512_set1_pd(0), (__mmask8)node_is_split, idx_lo, X_block, 8);
            __m512d feat_hi = _mm512_mask_i32gather_pd(
                _mm512_set1_pd(0), (__mmask8)(node_is_split >> 8), idx_hi, X_block, 8);
            __m512 feat_val =
                _mm512_insertf32x8(_mm512_castps256_ps512(_mm512_cvtpd_ps(feat_lo)),
                                   _mm512_cvtpd_ps(feat_hi), 1);
            go_right =
                _mm512_mask_cmp_ps_mask(node_is_split, feat_val, thresh, _CMP_GT_OQ);
        } else {
            return da_status_internal_error; // LCOV_EXCL_LINE
        }

        __m512i child =
            _mm512_mask_add_epi32(nexti, go_right, nexti, _mm512_set1_epi32(1));
        node_idx = _mm512_mask_blend_epi32(node_is_split, node_idx, child);

        feat_idx = _mm512_i32gather_epi32(node_idx, f_idx_cont.data(), 4);
        node_is_split = _mm512_mask_cmp_epi32_mask(node_is_split, feat_idx,
                                                   _mm512_set1_epi32(-1), _MM_CMPINT_NE);
    }

    // node_idx contains the leaf node prediction, load in y_pred
    __m512i result;
    if constexpr (LEAF_RET) {
        result = node_idx;
    } else {
        result = _mm512_i32gather_epi32(node_idx, next_idx_cont.data(), 4);
    }
    // The compact tree stores 32-bit predictions; widen to da_int when it is 64-bit
    if constexpr (std::is_same_v<da_int, int32_t>) {
        _mm512_storeu_epi32(y_pred, result);
    } else {
        __m512i result_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(result));
        __m512i result_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(result, 1));
        _mm512_storeu_si512(reinterpret_cast<void *>(y_pred), result_lo);
        _mm512_storeu_si512(reinterpret_cast<void *>(y_pred + 8), result_hi);
    }
    return da_status_success;
}
#endif

/* compute y_pred for 8 samples at a time using AVX2
 * - row major
 * - no categorical features
 * - float and double
 */
template <typename T>
template <bool LEAF_RET>
da_status decision_tree<T>::compact_traverse_sequential_avx2(const T *X_block, da_int ldx,
                                                             da_int *y_pred) {

    __m256i row_start = _mm256_set_epi32(7 * ldx, 6 * ldx, 5 * ldx, 4 * ldx, 3 * ldx,
                                         2 * ldx, 1 * ldx, 0);

    // Prepare the first iteration
    __m256i node_is_split =
        f_idx_cont[0] != -1 ? _mm256_set1_epi32(-1) : _mm256_setzero_si256();
    __m256i node_idx = _mm256_setzero_si256();
    __m256i feat_idx = _mm256_i32gather_epi32(f_idx_cont.data(), node_idx, 4);

    while (!_mm256_testz_si256(node_is_split, node_is_split)) {
        __m256i nexti = _mm256_mask_i32gather_epi32(node_idx, next_idx_cont.data(),
                                                    node_idx, node_is_split, 4);
        __m256 thresh =
            _mm256_mask_i32gather_ps(_mm256_set1_ps(0), threshold_cont.data(), node_idx,
                                     _mm256_castsi256_ps(node_is_split), 4);
        __m256i go_right;
        if constexpr (std::is_same_v<T, float>) {
            __m256i gather_idx = _mm256_add_epi32(row_start, feat_idx);
            __m256 feat_val =
                _mm256_mask_i32gather_ps(_mm256_set1_ps(0), X_block, gather_idx,
                                         _mm256_castsi256_ps(node_is_split), 4);
            go_right = _mm256_and_si256(
                _mm256_castps_si256(_mm256_cmp_ps(feat_val, thresh, _CMP_GT_OQ)),
                node_is_split);
        } else if constexpr (std::is_same_v<T, double>) {
            // Handle double precision by gathering 8 doubles in two 4-wide halves
            __m256i gather_idx = _mm256_add_epi32(row_start, feat_idx);
            __m128i idx_lo = _mm256_castsi256_si128(gather_idx);
            __m128i idx_hi = _mm256_extracti128_si256(gather_idx, 1);
            // Expand the 32-bit lane masks to 64-bit lanes for the double gathers
            __m256i mask_lo =
                _mm256_cvtepi32_epi64(_mm256_castsi256_si128(node_is_split));
            __m256i mask_hi =
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(node_is_split, 1));
            __m256d feat_lo = _mm256_mask_i32gather_pd(_mm256_set1_pd(0), X_block, idx_lo,
                                                       _mm256_castsi256_pd(mask_lo), 8);
            __m256d feat_hi = _mm256_mask_i32gather_pd(_mm256_set1_pd(0), X_block, idx_hi,
                                                       _mm256_castsi256_pd(mask_hi), 8);
            __m256 feat_val =
                _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(feat_lo)),
                                     _mm256_cvtpd_ps(feat_hi), 1);
            go_right = _mm256_and_si256(
                _mm256_castps_si256(_mm256_cmp_ps(feat_val, thresh, _CMP_GT_OQ)),
                node_is_split);
        } else {
            return da_status_internal_error; // LCOV_EXCL_LINE
        }

        __m256i child =
            _mm256_add_epi32(nexti, _mm256_and_si256(go_right, _mm256_set1_epi32(1)));
        node_idx = _mm256_blendv_epi8(node_idx, child, node_is_split);

        feat_idx = _mm256_i32gather_epi32(f_idx_cont.data(), node_idx, 4);
        __m256i not_leaf = _mm256_xor_si256(
            _mm256_cmpeq_epi32(feat_idx, _mm256_set1_epi32(-1)), _mm256_set1_epi32(-1));
        node_is_split = _mm256_and_si256(node_is_split, not_leaf);
    }

    // node_idx contains the leaf node prediction, load in y_pred
    __m256i result;
    if constexpr (LEAF_RET) {
        result = node_idx;
    } else {
        result = _mm256_i32gather_epi32(next_idx_cont.data(), node_idx, 4);
    }
    // The compact tree stores 32-bit predictions; widen to da_int when it is 64-bit
    if constexpr (std::is_same_v<da_int, int32_t>) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(y_pred), result);
    } else {
        __m256i result_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(result));
        __m256i result_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(result, 1));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(y_pred), result_lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(y_pred + 4), result_hi);
    }
    return da_status_success;
}

template <typename T>
template <da_order ORDER, bool HAS_CAT_FEAT>
da_status decision_tree<T>::compact_traverse_BFS(da_int nsamp, const T *X_test,
                                                 da_int ldx_test, da_int *y_pred) {
    std::vector<da_int> idx_src(nsamp);

    da_std::iota(idx_src.begin(), idx_src.end(), (da_int)0);

    struct node_to_treat {
        // node_idx: index of the node in the compact tree
        // start_idx: start index in idx_src of the samples to treat for this node
        // end_idx: end index in idx_src of the samples to treat for this node
        da_int node_idx;
        da_int start_idx;
        da_int end_idx;
    };

    std::vector<node_to_treat> nodes_queue;
    nodes_queue.reserve(3 * depth + 1);
    nodes_queue.push_back({0, 0, nsamp});

    while (!nodes_queue.empty()) {
        const da_int node_idx = nodes_queue.back().node_idx;
        const da_int start_idx = nodes_queue.back().start_idx;
        const da_int end_idx = nodes_queue.back().end_idx;
        nodes_queue.pop_back();
        const auto &current_node = compact_tree[node_idx];

        if (current_node.feature_idx < 0) {
            // leaf: assign class to every sample in the group
            for (da_int i = start_idx; i < end_idx; i++) {
                y_pred[idx_src[i]] = current_node.next_idx;
            }
            continue;
        }

        const T threshold = current_node.threshold;
        const da_int feat = current_node.feature_idx;

        // partition samples in idx_src[start_idx:end_idx] according to the split
        da_int wp = end_idx - 1;
        da_int i = start_idx;
        while (i <= wp) {
            da_int idx = idx_src[i];
            T feat_val;
            if constexpr (ORDER == row_major) {
                feat_val = X_test[idx * ldx_test + feat];
            } else {
                feat_val = X_test[feat * ldx_test + idx];
            }

            if constexpr (HAS_CAT_FEAT) {
                switch (feat_is_onevall[feat]) {
                case false:
                    if (feat_val > threshold) {
                        std::swap(idx_src[i], idx_src[wp]);
                        wp--;
                    } else {
                        i++;
                    }
                    break;
                case true:
                    if (std::abs(feat_val - threshold) > feat_thresh) {
                        std::swap(idx_src[i], idx_src[wp]);
                        wp--;
                    } else {
                        i++;
                    }
                    break;
                }
            } else {
                if (feat_val > threshold) {
                    std::swap(idx_src[i], idx_src[wp]);
                    wp--;
                } else {
                    i++;
                }
            }
        }

        // push children to queue if they have samples to treat
        if (wp >= start_idx)
            nodes_queue.push_back({current_node.next_idx, start_idx, wp + 1});
        if (wp < end_idx - 1)
            nodes_queue.push_back({current_node.next_idx + 1, wp + 1, end_idx});
    }

    return da_status_success;
}

template <typename T>
template <da_order ORDER, bool HAS_CAT_FEAT>
da_status decision_tree<T>::compact_traverse_BFS(da_int nsamp, const T *X_test,
                                                 da_int ldx_test, T *y_proba,
                                                 da_int n_class, da_int ldy) {
    std::vector<da_int> idx_src(nsamp);

    da_std::iota(idx_src.begin(), idx_src.end(), (da_int)0);

    struct node_to_treat {
        // node_idx: index of the node in the compact tree
        // start_idx: start index in idx_src of the samples to treat for this node
        // end_idx: end index in idx_src of the samples to treat for this node
        da_int node_idx;
        da_int start_idx;
        da_int end_idx;
    };

    std::vector<node_to_treat> nodes_queue;
    nodes_queue.reserve(3 * depth + 1);
    nodes_queue.push_back({0, 0, nsamp});

    while (!nodes_queue.empty()) {
        const da_int node_idx = nodes_queue.back().node_idx;
        const da_int start_idx = nodes_queue.back().start_idx;
        const da_int end_idx = nodes_queue.back().end_idx;
        nodes_queue.pop_back();
        const auto &current_node = compact_tree[node_idx];

        if (current_node.feature_idx < 0) {
            // leaf: assign class probability to every sample in the group
            for (da_int i = start_idx; i < end_idx; i++) {
                da_int idx = idx_src[i];
                for (da_int j = 0; j < n_class; j++) {
                    if constexpr (ORDER == row_major)
                        y_proba[idx * ldy + j] = class_props[node_idx * n_class + j];
                    else
                        y_proba[j * ldy + idx] = class_props[node_idx * n_class + j];
                }
            }
            continue;
        }

        const T threshold = current_node.threshold;
        const da_int feat = current_node.feature_idx;

        // partition samples in idx_src[start_idx:end_idx] according to the split
        da_int wp = end_idx - 1;
        da_int i = start_idx;
        while (i <= wp) {
            da_int idx = idx_src[i];
            T feat_val;
            if constexpr (ORDER == row_major)
                feat_val = X_test[idx * ldx_test + feat];
            else
                feat_val = X_test[feat * ldx_test + idx];
            if constexpr (HAS_CAT_FEAT) {
                switch (feat_is_onevall[feat]) {
                case false:
                    if (feat_val > threshold) {
                        std::swap(idx_src[i], idx_src[wp]);
                        wp--;
                    } else {
                        i++;
                    }
                    break;
                case true:
                    if (std::abs(feat_val - threshold) > feat_thresh) {
                        std::swap(idx_src[i], idx_src[wp]);
                        wp--;
                    } else {
                        i++;
                    }
                    break;
                }
            } else {
                if (feat_val > threshold) {
                    std::swap(idx_src[i], idx_src[wp]);
                    wp--;
                } else {
                    i++;
                }
            }
        }

        // push children to queue if they have samples to treat
        if (wp >= start_idx)
            nodes_queue.push_back({current_node.next_idx, start_idx, wp + 1});
        if (wp < end_idx - 1)
            nodes_queue.push_back({current_node.next_idx + 1, wp + 1, end_idx});
    }

    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::predict(da_int nsamp, da_int nfeat, const T *X_test,
                                    da_int ldx_test, da_int *y_pred) {
    da_status status = read_storage_order();
    if (status != da_status_success)
        return status;

    const da_order user_order = this->order;
    // The vectorized traversal kernels operate on row-major data. Row-major input
    // is evaluated in place; column-major input is transposed into a temporary
    // row-major buffer first. Trees with one-vs-all features are never vectorized,
    // so their column-major input is evaluated directly without transposition.
    // Input is validated in the traverse functions.
    if (user_order == row_major || has_onevall_features)
        return compact_traverse(nsamp, nfeat, X_test, ldx_test, y_pred);

    if (nsamp <= 0)
        return da_error_bypass(this->err, da_status_invalid_array_dimension,
                               "n_samples = " + std::to_string(nsamp) + " must be > 0.");
    if (nfeat != n_features)
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");

    std::vector<T> X_row;
    try {
        X_row.resize((size_t)nsamp * nfeat);
    } catch (const std::bad_alloc &) {
        return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation failed.");
    }
    status = da_utils::switch_order_copy(user_order, nsamp, nfeat, X_test, ldx_test,
                                         X_row.data(), nfeat);
    if (status != da_status_success)
        return da_error_bypass(this->err, status,
                               "Failed to transpose the input data."); // LCOV_EXCL_LINE

    this->order = row_major;
    status = compact_traverse(nsamp, nfeat, X_row.data(), nfeat, y_pred);
    this->order = user_order;
    return status;
}

template <typename T>
da_status decision_tree<T>::predict_proba(da_int nsamp, da_int nfeat, const T *X_test,
                                          da_int ldx_test, T *y_proba_pred, da_int nclass,
                                          da_int ldy) {
    da_status status = read_storage_order();
    if (status != da_status_success)
        return status;

    const da_order user_order = this->order;
    if (user_order == row_major || has_onevall_features)
        return compact_traverse(nsamp, nfeat, X_test, ldx_test, y_proba_pred, nclass,
                                ldy);

    // Column-major input: validate here (in the user's order), transpose the data
    // into a row-major scratch buffer, evaluate the tree there and transpose the
    // resulting probabilities back into the user's column-major buffer.
    if (y_proba_pred == nullptr)
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "y_proba is not a valid pointer.");
    if (nsamp <= 0)
        return da_error_bypass(this->err, da_status_invalid_array_dimension,
                               "n_samples = " + std::to_string(nsamp) + " must be > 0.");
    if (nfeat != n_features)
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    if (nclass != this->n_class)
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_class = " + std::to_string(nclass) +
                                   " doesn't match the expected value " +
                                   std::to_string(this->n_class) + ".");
    if (!this->model_trained)
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    status = this->check_2D_array(user_order, nsamp, nfeat, X_test, ldx_test, "n_samples",
                                  "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    std::vector<T> X_row, y_row;
    try {
        X_row.resize((size_t)nsamp * nfeat);
        y_row.resize((size_t)nsamp * nclass);
    } catch (const std::bad_alloc &) {
        return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation failed.");
    }
    status = da_utils::switch_order_copy(user_order, nsamp, nfeat, X_test, ldx_test,
                                         X_row.data(), nfeat);
    if (status != da_status_success)
        return da_error_bypass(this->err, status,
                               "Failed to transpose the input data."); // LCOV_EXCL_LINE

    this->order = row_major;
    status = compact_traverse(nsamp, nfeat, X_row.data(), nfeat, y_row.data(), nclass,
                              nclass, false);
    this->order = user_order;
    if (status != da_status_success)
        return status;

    // Transpose the row-major probabilities back to the user's column-major layout
    da_utils::copy_transpose_2D_array_row_to_column_major(nsamp, nclass, y_row.data(),
                                                          nclass, y_proba_pred, ldy);
    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::predict_log_proba(da_int nsamp, da_int nfeat, const T *X_test,
                                              da_int ldx_test, T *y_log_proba,
                                              da_int nclass, da_int ldy) {
    da_status status = da_status_success;

    status = predict_proba(nsamp, nfeat, X_test, ldx_test, y_log_proba, nclass, ldy);
    if (status != da_status_success)
        return status;

    switch (this->order) {
    case column_major:
        for (da_int j = 0; j < nclass; j++) {
            for (da_int i = 0; i < nsamp; i++) {
                y_log_proba[ldy * j + i] = log(y_log_proba[ldy * j + i]);
            }
        }
        break;
    case row_major:
        for (da_int j = 0; j < nsamp; j++) {
            for (da_int i = 0; i < nclass; i++) {
                y_log_proba[j * ldy + i] = log(y_log_proba[j * ldy + i]);
            }
        }
        break;
    }
    return status;
}

template <typename T>
da_status decision_tree<T>::score(da_int nsamp, da_int nfeat, const T *X_test,
                                  da_int ldx_test, const da_int *y_test, T *accuracy) {

    da_status status;
    if (accuracy == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "mean_accuracy is not a valid pointer.");
    }

    status = read_storage_order();
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(nsamp, y_test, "n_samples", "y_test");
    if (status != da_status_success)
        return status;

    std::vector<da_int> class_samp;
    try {
        class_samp.resize(nsamp);
    } catch (const std::exception &e) {
        return da_error_bypass(this->err,
                               da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation failed");
    }

    const da_order user_order = this->order;
    if (user_order == row_major || has_onevall_features) {
        status = compact_traverse(nsamp, nfeat, X_test, ldx_test, class_samp.data());
    } else {
        // Transpose the column-major input into a row-major scratch buffer and
        // evaluate the tree on that copy.
        if (nfeat != n_features) {
            return da_error_bypass(this->err, da_status_invalid_input,
                                   "n_features = " + std::to_string(nfeat) +
                                       " doesn't match the expected value " +
                                       std::to_string(n_features) + ".");
        }
        std::vector<T> X_row;
        try {
            X_row.resize((size_t)nsamp * nfeat);
        } catch (const std::bad_alloc &) {
            return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                                   "Memory allocation failed.");
        }
        status = da_utils::switch_order_copy(user_order, nsamp, nfeat, X_test, ldx_test,
                                             X_row.data(), nfeat);
        if (status != da_status_success)
            return da_error_bypass(
                this->err, status,
                "Failed to transpose the input data."); // LCOV_EXCL_LINE
        this->order = row_major;
        status = compact_traverse(nsamp, nfeat, X_row.data(), nfeat, class_samp.data());
        this->order = user_order;
    }
    if (status != da_status_success)
        return status;

    da_int n_correct = 0;
    for (da_int i = 0; i < nsamp; i++) {
        if (class_samp[i] == y_test[i])
            n_correct += 1;
    }

    *accuracy = (T)n_correct / (T)nsamp;

    return da_status_success;
}

/* training tree based inference functions -- Kept as a baseline */
template <typename T>
da_status decision_tree<T>::predict_training_tree(da_int nsamp, da_int nfeat,
                                                  const T *X_test, da_int ldx_test,
                                                  da_int *y_pred) {
    if (y_pred == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "y_pred is not a valid pointer.");
    }

    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp;

    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }

    if (!this->model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    da_int mode = 0;
    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test", mode);
    if (status != da_status_success)
        return status;

    // Fill y_pred with the values of all the requested samples
    node<T> *current_node;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        while (!current_node->is_leaf) {
            if (current_node->prop == continuous ||
                current_node->prop == categorical_ordered) {
                T feat_val = X_test_temp[ldx_test_temp * current_node->feature + i];
                if (feat_val < current_node->x_threshold)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            } else {
                da_int cat_val =
                    std::round(X_test_temp[ldx_test_temp * current_node->feature + i]);
                if (cat_val == current_node->category)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            }
        }
        y_pred[i] = current_node->y_pred;
    }
    if (utility_ptr1)
        delete[] (utility_ptr1);
    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::predict_proba_training_tree(da_int nsamp, da_int nfeat,
                                                        const T *X_test, da_int ldx_test,
                                                        T *y_proba_pred, da_int nclass,
                                                        da_int ldy) {

    const T *X_test_temp;
    T *utility_ptr1;
    T *utility_ptr2;
    da_int ldx_test_temp;
    T *y_proba_pred_temp;
    da_int ldy_proba_pred_temp;

    if (!predict_proba_opt) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "predict_proba must be set to 1");
    }

    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }

    if (nclass != n_class) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "nclass = " + std::to_string(nclass) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_class) + ".");
    }

    if (!this->model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    da_int mode = 0;
    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test", mode);
    if (status != da_status_success)
        return status;

    status = this->store_2D_array(nsamp, nclass, y_proba_pred, ldy, &utility_ptr2,
                                  const_cast<const T **>(&y_proba_pred_temp),
                                  ldy_proba_pred_temp, "n_samples", "n_class", "y_proba",
                                  "ldy", mode);
    if (status != da_status_success)
        return status;

    // Fill y_proba_pred with the values of all the requested samples
    node<T> *current_node;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        da_int current_node_idx = 0;
        while (!current_node->is_leaf) {
            T feat_val = X_test_temp[ldx_test_temp * current_node->feature + i];
            if (feat_val < current_node->x_threshold) {
                current_node_idx = current_node->left_child_idx;
                current_node = &tree[current_node_idx];
            } else {
                current_node_idx = current_node->right_child_idx;
                current_node = &tree[current_node_idx];
            }
        }
        for (da_int j = 0; j < n_class; j++)
            y_proba_pred_temp[ldy_proba_pred_temp * j + i] =
                class_props[n_class * current_node_idx + j];
    }

    if (this->order == row_major) {

        da_utils::copy_transpose_2D_array_column_to_row_major(
            nsamp, n_class, y_proba_pred_temp, ldy_proba_pred_temp, y_proba_pred, ldy);
        if (utility_ptr1)
            delete[] (utility_ptr1);
        if (utility_ptr2)
            delete[] (utility_ptr2);
    }
    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::score_training_tree(da_int nsamp, da_int nfeat,
                                                const T *X_test, da_int ldx_test,
                                                const da_int *y_test, T *accuracy) {

    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp;

    if (accuracy == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "mean_accuracy is not a valid pointer.");
    }

    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "nfeat = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }

    if (!this->model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(nsamp, y_test, "n_samples", "y_test", 1);
    if (status != da_status_success)
        return status;

    node<T> *current_node;
    *accuracy = 0.;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        while (!current_node->is_leaf) {
            if (current_node->prop == continuous ||
                current_node->prop == categorical_ordered) {
                T feat_val = X_test_temp[ldx_test_temp * current_node->feature + i];
                if (feat_val < current_node->x_threshold)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            } else {
                da_int cat_val =
                    std::round(X_test_temp[ldx_test_temp * current_node->feature + i]);
                if (cat_val == current_node->category)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            }
        }

        if (current_node->y_pred == y_test[i])
            *accuracy += (T)1.0;
    }
    *accuracy = *accuracy / (T)nsamp;
    if (utility_ptr1)
        delete[] (utility_ptr1);

    return da_status_success;
}

} // namespace da_decision_forest
} // namespace ARCH

#endif