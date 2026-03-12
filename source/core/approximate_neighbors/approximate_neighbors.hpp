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
#include "da_error.hpp"
#include "da_vector.hpp"
#include "macros.h"

#include <cmath>
#include <random>

namespace ARCH {

namespace da_approx_nn {

using namespace da_approx_nn_types;

template <typename T> class approximate_neighbors : public basic_handle<T> {
  private:
    bool train_data_is_set = false;
    bool index_is_trained = false;
    bool data_is_added = false;

    // Number of neighbors at search time
    da_int n_neighbors = 5;

    // Algorithm to use - only ivfflat for now
    da_int algo = approx_nn_algorithm::ivfflat;
    da_int internal_algo;

    // Distance metric to use
    approx_nn_metric internal_metric;
    approx_nn_metric metric;

    // Seed to use for k-means initialisation (and subsampling of training data if performed)
    std::mt19937 mt_engine;
    da_int internal_seed;
    da_int seed = 0;

    // User train data
    da_int n_samples_train = 0, n_samples = 0, n_features = 0, ldx_train = 0;
    const T *X_train = nullptr;
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
    // Number of k-means initialisations to perform
    da_int n_init = 1;
    // Tolerance for k-means training
    T kmeans_tol = std::sqrt(std::numeric_limits<T>::epsilon());
    // Centroids returned by k-means training
    std::vector<T> centroids;
    da_int ld_centroids;

    // Number of rows of data added to the index
    da_int n_index;
    // Inner vector contains rows assigned to each centroid
    std::vector<da_vector::da_vector<T>> indexed_vectors;
    // For each list: list_assignments maps each row in indexed_vectors to a global index
    // which is returned at search time. Global index runs from 0 to n_index-1
    // and each sample is indexed by the order it is added.
    std::vector<da_vector::da_vector<da_int>> global_indices;
    // list_sizes stores the number of rows added to each list
    // old_list_sizes is needed for bookkeeping when add is called more than once
    std::vector<da_int> list_sizes, old_list_sizes;

  public:
    ~approximate_neighbors();

    approximate_neighbors(da_errors::da_error_t &err);

    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
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

    da_status euclidean_search(da_int n_queries, da_int n_features, const T *X_test,
                               da_int ldx_test, da_int *n_ind, T *n_dist, da_int k_neigh,
                               bool return_distance);
    da_status inner_product_search(da_int n_queries, da_int n_features, const T *X_test,
                                   da_int ldx_test, da_int *n_ind, T *n_dist,
                                   da_int k_neigh, bool return_distance);

    // ivfflat searching
    da_status kneighbors_compute_ivfflat(da_int n_queries, da_int n_features,
                                         const T *X_test, da_int ldx_test, da_int *n_ind,
                                         T *n_dist, da_int n_neigh, bool return_distance);
};

} // namespace da_approx_nn

} // namespace ARCH

#endif // ANN_HPP
