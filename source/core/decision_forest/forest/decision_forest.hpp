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

#ifndef DECISION_FOREST_HPP
#define DECISION_FOREST_HPP

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_omp.hpp"

#include "common/histogram.hpp"
#include "macros.h"
#include "options.hpp"
#include "tree/decision_tree.hpp"

#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace ARCH {

namespace da_decision_forest {

using namespace da_errors;

template <typename T> class decision_forest : public basic_handle<T> {

    bool model_trained = false;

    // User data. Never modified by the classifier
    // X[n_samples X n_features]: features -- floating point matrix, column major
    // y[n_samples]: labels -- integer array, 0,...,n_classes-1 values
    // usr_categorical_feat[n_features]: usr_categorical_feat[i] contains the number of categories for feature i if it is categorical.
    //                               <= 0 if feature i is continuous
    //                               if usr_categorical_feat == nullptr, all features are continuous
    const T *X = nullptr;
    const da_int *y = nullptr;
    da_int n_samples = 0;
    da_int ldx = 0;
    da_int n_features = 0;
    da_int n_class = 0;
    const da_int *usr_categorical_feat = nullptr;

    //Utility pointer to column major allocated copy of user's data
    T *X_temp = nullptr;

    // Options
    da_int n_tree = 0;
    da_int seed, n_obs;
    da_int block_size;

    // Histogram: data is quantized into max_bin categories
    // X_binned: class containing the binned X and auxialiary routines
    bins<T> *X_binned = nullptr;
    da_int use_hist = 0;
    da_int usr_max_bins;

    // Model data
    std::vector<std::unique_ptr<decision_tree<T>>> forest;

  public:
    decision_forest(da_errors::da_error_t &err);
    ~decision_forest();
    da_status set_training_data(da_int n_samples, da_int n_features, const T *X,
                                da_int ldx, const da_int *y, da_int n_class = 0,
                                const da_int *usr_cat_feat = nullptr);
    da_status fit();
    void parallel_count_classes(const T *X_test, da_int ldx_test, const da_int &n_blocks,
                                const da_int &block_size, const da_int &block_rem,
                                const da_int &n_threads,
                                std::vector<da_int> &count_classes,
                                std::vector<da_int> &y_pred_tree);
    da_status predict(da_int nn_samples, da_int n_features, const T *X, da_int ldx,
                      da_int *y_pred);
    da_status predict_proba(da_int nsamp, da_int nfeat, const T *X_test, da_int ldx_test,
                            T *y_proba, da_int nclass, da_int ldy);
    da_status predict_log_proba(da_int nsamp, da_int n_features, const T *X_test,
                                da_int ldx, T *y_pred, da_int n_class, da_int ldy);
    da_status score(da_int nsamp, da_int nfeat, const T *X_test, da_int ldx_test,
                    const da_int *y_test, T *score);

    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
};

} // namespace da_decision_forest

} // namespace ARCH

#endif // DECISION_FOREST_HPP