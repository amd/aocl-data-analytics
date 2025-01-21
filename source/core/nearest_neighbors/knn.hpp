/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "knn_options.hpp"
#include "macros.h"

namespace ARCH {

namespace da_knn {

/* k-nearest neighbors class */
template <typename T> class knn : public basic_handle<T> {
  private:
    // Set true when initialization is complete by set_params() function
    bool is_up_to_date = false;
    // Set true if training data has been provided via set_training_data()
    bool istrained = false;
    // Set true if the available classes have been computed via a call to available_classes()
    bool classes_computed = false;

    // Number of neighbors to be considered
    da_int n_neighbors = 5;
    // Algorithm to be used for the knn computation
    da_int algo = da_brute_force;
    // Metric to be used for the distance computation
    da_int metric = da_euclidean;
    // Internal metric to be used for the distance computation.
    // We want to avoid squaring the distance unless it's necessary.
    da_int internal_metric = da_sqeuclidean;
    // Weight function used to compute the k-nearest neighbors
    da_int weights = da_knn_uniform;
    // User's data
    da_int n_samples = 0, n_features = 0, ldx_train = 0;
    const T *X_train = nullptr /*n_samples-by-n_features*/;
    const da_int *y_train = nullptr /*n_samples*/;
    //Utility pointer to column major allocated copy of user's data
    T *X_train_temp = nullptr;

  public:
    std::vector<da_int> classes;
    da_int n_classes = -1;

    ~knn();

    knn(da_errors::da_error_t &err);

    /* get_result (required to be defined by basic_handle) */
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
    // Set input parameters
    da_status set_params();
    // Set the training data
    da_status set_training_data(da_int n_samples, da_int n_features, const T *X_train,
                                da_int ldx_train, const da_int *y_train);
    // Compute the k-nearest neighbors and optionally the corresponding distances
    // Includes the appropriate checks for input arguments
    da_status kneighbors(da_int n_queries, da_int n_features, const T *X_test,
                         da_int ldx_test, da_int *n_ind, T *n_dist, da_int k = 0,
                         bool return_distance = 0);
    // Compute kernel for the k-nearest neighbors and optionally the corresponding distances
    da_status kneighbors_compute(da_int n_queries, da_int n_features, const T *X_test,
                                 da_int ldx_test, da_int *n_ind, T *n_dist,
                                 da_int n_neigh, bool return_distance);
    // Computational kernel that computes kneighbors using blocking on Xtest for overall algorithm.
    // In addition, it uses blocking for Xtrain only for the distance computation.
    template <da_int XTRAIN_BLOCK, da_int XTEST_BLOCK>
    da_status kneighbors_blocked_Xtest(da_int n_queries, da_int n_features,
                                       const T *X_test, da_int ldx_test, da_int *n_ind,
                                       T *n_dist, da_int n_neigh, bool return_distance);
    // Compute the k-nearest neighbors and optionally the corresponding distances
    template <da_int XTRAIN_BLOCK>
    da_status kneighbors_kernel(da_int xtrain_block_size, da_int n_blocks_train,
                                da_int block_rem_train, da_int n_queries,
                                da_int n_features, const T *X_test, da_int ldx_test, T *D,
                                da_int *n_ind, T *n_dist, da_int k, bool return_distance);
    // Compute probability estimates for provided test data
    da_status predict_proba(da_int n_queries, da_int n_features, const T *X_test,
                            da_int ldx_test, T *proba);
    // Predict the labels for provided test data
    da_status predict(da_int n_queries, da_int n_features, const T *X_test,
                      da_int ldx_test, da_int *y_test);
    // Internal function used to compute the std::vector that holds the available classes
    da_status available_classes();

    // Implementing refresh
    void refresh();
};

} // namespace da_knn

} // namespace ARCH
