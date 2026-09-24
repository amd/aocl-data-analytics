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
#ifndef ANN_HPP
#define ANN_HPP

#include "aoclda.h"
#include "approximate_neighbors_options.hpp"
#include "basic_handle.hpp"
#include "bfloat16_helpers.hpp"
#include "binary_tree.hpp"
#include "da_error.hpp"
#include "da_vector.hpp"
#include "macros.h"
#include "model_persistence.hpp"
#include "quantized_backend.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>
#include <variant>

namespace ARCH {

namespace da_approx_nn {

using namespace da_approx_nn_types;

template <typename Q> struct dense_indexed_lists {
    static_assert(std::is_trivially_copyable_v<Q>,
                  "IVF dense storage requires trivially copyable elements.");

    using value_type = Q;
    std::vector<da_vector::da_vector<Q>> lists;
};

template <typename T>
using ivf_indexed_storage =
    std::variant<dense_indexed_lists<T>, dense_indexed_lists<int8_t>,
                 dense_indexed_lists<da_bf16::storage_type>,
                 // For uint8_t, each byte stores two signed int4 values. The
                 // int4_packing helpers define its logical row layout.
                 dense_indexed_lists<uint8_t>>;

template <typename T> class approximate_neighbors : public basic_handle<T> {
  private:
    bool train_data_is_set = false;
    bool data_is_added = false;

    // Number of neighbors at search time
    da_int n_neighbors = 5;

    // Public algorithm selection; both variants share the IVF implementation.
    da_int algo = approx_nn_algorithm::automatic;
    da_int internal_algo;

    // Distance metric to use
    approx_nn_metric internal_metric;
    approx_nn_metric metric;

    // Seed to use for k-means initialization (and subsampling of training data if performed)
    std::mt19937 mt_engine;
    da_int internal_seed;
    da_int seed = 0;

    // User train data
    da_int n_samples_train = 0, n_samples = 0, n_features = 0, ldx_train = 0;
    const T *X_train = nullptr;
    // Vector used to store user data after model has been loaded
    std::vector<T> X_int;
    // Fraction of training data to use for k-means clustering
    T train_fraction = 1.0;

    // nlist - number of lists to make = number of k-means clusters
    da_int n_list = 1;
    // nprobe - number of lists to probe at search time
    da_int n_probe = 1;

    // Maximum number of k-means iterations to perform
    da_int max_iter = 10;
    // Actual number of k-means iterations performed
    da_int kmeans_iter;
    // Number of k-means initializations to perform
    da_int n_init = 1;
    // Tolerance for k-means training
    T kmeans_tol = std::sqrt(std::numeric_limits<T>::epsilon());
    // Centroids returned by k-means training
    std::vector<T> centroids;
    da_int ld_centroids;

    // Number of rows of data added to the index
    da_int n_index;
    // Typed inverted lists. The active variant is selected once during training.
    ivf_indexed_storage<T> indexed_storage;
    // Squared L2 norms parallel to indexed_storage. Quantized indexes store norms of
    // quantized vectors; unquantized indexes store norms of original vectors.
    // Only populated for euclidean metric; computed once in add_ivfflat.
    std::vector<da_vector::da_vector<T>> list_norms;
    // Squared L2 norms of each centroid; only populated for euclidean metric
    // Computed once in train_ivfflat after k-means converges
    std::vector<T> centroid_norms;
    // For each list: list_assignments maps each stored row to a global index
    // which is returned at search time. Global index runs from 0 to n_index-1
    // and each sample is indexed by the order it is added.
    std::vector<da_vector::da_vector<da_int>> global_indices;
    // list_sizes stores the number of rows added to each list
    // old_list_sizes is needed for bookkeeping when add is called more than once
    std::vector<da_int> list_sizes, old_list_sizes;

    // Whether quantized codes encode residuals (x - nearest centroid) rather than raw vectors
    bool quantize_residual = true;
    // Scalar quantizer configured from "quantization_scheme"/"quantization"
    scalar_quantizer::scalar_quantizer<T> sq;

    bool quantize_db() const {
        return sq.get_code_kind() != scalar_quantizer::code_kind::none;
    }

    da_status
    store_unquantized_rows(dense_indexed_lists<T> &storage,
                           const std::vector<da_vector::da_vector<da_int>> &local_indices,
                           const T *X_add, da_int ldx_add, da_int n_threads,
                           bool is_euclidean);
    template <typename Q>
    da_status
    store_quantized_rows(dense_indexed_lists<Q> &storage,
                         const std::vector<da_vector::da_vector<da_int>> &local_indices,
                         const T *X_add, da_int ldx_add, da_int n_threads,
                         bool is_euclidean);

    template <typename Q>
    T reconstructed_squared_norm(const Q *codes, const T *centroid,
                                 da_int centroid_stride);

    // Per-thread scratch buffer pointers, set once per thread and reused across blocks.
    // query_cos is null unless cosine metric; qnorms is null unless euclidean metric.
    template <typename Backend> struct search_workspace {
        da_int *heap_indices;
        T *heap_distances;
        da_binary_tree::MaxHeap<T> *heaps;
        T *fine_distances;
        T *queries;
        T *query_cos;
        T *qnorms;
        T *coarse_dist;
        da_int *cent_idx;
        T *cent_sel_dists;
        da_int *qpc;
        da_int *qpc_count;
        da_int *topk_indices;
        // Quantized-search state, non-null only when quantize_db is set. Query
        // preparation and product scratch are owned by the selected GEMM backend;
        // bias_dot_products and distance_offsets belong to the metric correction shared by all
        // storage formats and product approaches.
        typename Backend::workspace_type *quantized;
        T *bias_dot_products;
        T *distance_offsets;
    };

    // Read-only search configuration, built once and shared across the parallel region.
    struct search_params {
        da_int n_queries, n_features, k_neigh;
        bool return_distance;
        const T *X_test_ptr;
        da_int ldx_test;
        const T *centroids_ptr;
        da_int ld_centroids_local;
        da_int query_blk_sz, list_blk_sz, n_blocks, final_query_blk_sz;
        bool is_euclidean, is_cosine;
        T *n_dist;
        da_int *n_ind;
    };

    void update_heaps_from_list_blk(da_int this_list_blk_sz, da_int q_count,
                                    const da_int *list_blk_global_idx,
                                    const T *fine_distances, da_int block_start,
                                    const da_int *queries_processed,
                                    da_binary_tree::MaxHeap<T> *heaps);

    // Process one query block: coarse centroid selection, list scan, result write-back.
    template <typename Backend>
    bool process_block(search_workspace<Backend> &ws, const search_params &p,
                       const dense_indexed_lists<typename Backend::stored_type> &storage,
                       da_int i);

    // Process one IVF list for the current block: gather probing queries, fine GEMM,
    // heap updates.
    template <typename Backend>
    bool process_list(search_workspace<Backend> &ws, const search_params &p,
                      const dense_indexed_lists<typename Backend::stored_type> &storage,
                      da_int j, da_int block_start);

    // Quantized counterpart of process_list. Backend owns the stored code type,
    // query representation, GEMM approach, accumulator, and reconstruction.
    template <typename Backend>
    bool process_list_quantized(
        search_workspace<Backend> &ws, const search_params &p,
        const dense_indexed_lists<typename Backend::stored_type> &storage, da_int j,
        da_int block_start);

    template <typename Backend>
    da_status ivfflat_search_query_parallel_impl(
        const dense_indexed_lists<typename Backend::stored_type> &storage,
        da_int n_queries, da_int n_features, da_int k_neigh, bool return_distance,
        const T *X_test_ptr, da_int ldx_test, const T *centroids_ptr,
        da_int ld_centroids_local, da_int query_blk_sz, da_int list_blk_sz,
        da_int n_blocks, da_int final_query_blk_sz, da_int n_threads, da_int *n_ind,
        T *n_dist);

    da_status ivfflat_search_query_parallel(da_int n_queries, da_int n_features,
                                            da_int k_neigh, bool return_distance,
                                            const T *X_test_ptr, da_int ldx_test,
                                            const T *centroids_ptr,
                                            da_int ld_centroids_local,
                                            da_int query_blk_sz, da_int list_blk_sz,
                                            da_int n_blocks, da_int final_query_blk_sz,
                                            da_int n_threads, da_int *n_ind, T *n_dist);

    da_status ivf_search(da_int n_queries, da_int n_features, const T *X_test,
                         da_int ldx_test, da_int *n_ind, T *n_dist, da_int k_neigh,
                         bool return_distance);

    da_status kneighbors_compute_ivfflat(da_int n_queries, da_int n_features,
                                         const T *X_test, da_int ldx_test, da_int *n_ind,
                                         T *n_dist, da_int n_neigh, bool return_distance);

  public:
    ~approximate_neighbors();

    approximate_neighbors(da_errors::da_error_t &err);

    da_status get_result(da_result query, da_int *dim, T *result) override;
    da_status get_result(da_result query, da_int *dim, da_int *result) override;
    // Set input parameters
    da_status set_params();

    // Check if the options have been updated between calls to add or kneighbors
    da_status check_options_update();

    // Set user inputted training data
    da_status set_training_data(da_int n_samples, da_int n_features, const T *X_train,
                                da_int ldx_train);

    da_status read_training_options();

    // Do we need to subsample training data?
    da_status subsample_training_data(std::vector<T> &X_train_sampled,
                                      da_int &ldx_train_sampled);

    // Run training
    da_status train();

    // ivfflat training
    da_status train_ivfflat();

    // Add some data to the index.
    da_status add(da_int n_samples_add, da_int n_features, const T *X_add,
                  da_int ldX_add);

    // ivfflat adding
    da_status add_ivfflat(da_int n_samples, da_int n_features, const T *X_add,
                          da_int ldx_add);

    // Train the index and add the training data to the index
    // This doesn't provide extra functionality, but avoids the user having
    // to call set_training_data then add on the same data
    da_status train_and_add();

    // Compute the k-nearest neighbors and optionally the corresponding distances
    da_status kneighbors(da_int n_queries, da_int n_features, const T *X_test,
                         da_int ldx_test, da_int *n_ind, T *n_dist, da_int k_neigh = 0,
                         bool return_distance = 0);

    // Model storage
    da_status serialize(da_model_persistence::serialization_buffer &buffer) override;
    da_status save_model(da_model_persistence::serialization_buffer &buffer) override;
    da_status load_model(da_model_persistence::serialization_buffer &buffer) override;
};

} // namespace da_approx_nn

} // namespace ARCH

#endif // ANN_HPP
