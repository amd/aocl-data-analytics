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

#include "aoclda.h"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "decision_forest.hpp"
#include "decision_tree_types.hpp"
#include "macros.h"
#include "options.hpp"
#include "random_forest_options.hpp"

#include <chrono>
#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "boost/sort/spreadsort/spreadsort.hpp"

namespace ARCH {

namespace da_decision_forest {

using namespace da_decision_tree_types;

using namespace da_errors;

template <typename T>
random_forest<T>::random_forest(da_errors::da_error_t &err) : basic_handle<T>(err) {
    // Initialize the options registry
    // Any error is stored err->status[.] and this NEEDS to be checked
    // by the caller.
    register_forest_options<T>(this->opts, *this->err);
}

template <typename T> random_forest<T>::~random_forest() {
    // Destructor needs to handle arrays that were allocated due to row major storage of input data
    if (X_temp)
        delete[] (X_temp);
}

template <typename T>
da_status random_forest<T>::get_result([[maybe_unused]] da_result query,
                                       [[maybe_unused]] da_int *dim,
                                       [[maybe_unused]] da_int *result) {
    return da_warn(this->err, da_status_unknown_query,
                   "There are no integer results available for this API.");
};

template <typename T>
da_status random_forest<T>::get_result(da_result query, da_int *dim, T *result) {

    if (!model_trained)
        return da_warn_bypass(
            this->err, da_status_unknown_query,
            "Handle does not contain data relevant to this query. Was the "
            "last call to the solver successful?");
    // Pointers were already tested in the generic get_result

    da_int rinfo_size = 5;
    switch (query) {
    case da_result::da_rinfo:
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(rinfo_size) + ".");
        }
        result[0] = (T)n_features;
        result[1] = (T)n_samples;
        result[2] = (T)n_obs;
        result[3] = (T)seed;
        result[4] = (T)n_tree;
        break;
    default:
        return da_warn_bypass(this->err, da_status_unknown_query,
                              "The requested result could not be found.");
    }
    return da_status_success;
}

template <typename T>
da_status random_forest<T>::set_training_data(da_int n_samples, da_int n_features,
                                              const T *X, da_int ldx, const da_int *y,
                                              da_int n_class) {

    // Guard against errors due to multiple calls using the same class instantiation
    if (X_temp) {
        delete[] (X_temp);
        X_temp = nullptr;
    }

    da_status status =
        this->store_2D_array(n_samples, n_features, X, ldx, &X_temp, &this->X, this->ldx,
                             "n_samples", "n_features", "X", "ldx");
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(n_samples, y, "n_samples", "y", 1);
    if (status != da_status_success)
        return status;

    this->refresh();
    this->y = y;
    this->n_samples = n_samples;
    this->n_features = n_features;
    this->n_class = n_class;
    if (n_class <= 0)
        this->n_class = *std::max_element(y, y + n_samples) + 1;

    return da_status_success;
}

template <typename T> da_status random_forest<T>::fit() {

    // Read optional parameters
    bool opt_pass = true, bootstrap;
    da_int max_depth, min_node_sample, method, build_order, nfeat_split, bootstrap_opt,
        feat_select, sort_method;
    T feat_thresh, min_split_score, min_improvement, prop;
    std::string opt_val;
    opt_pass &= this->opts.get("number of trees", n_tree) == da_status_success;
    opt_pass &= this->opts.get("maximum depth", max_depth) == da_status_success;
    opt_pass &= this->opts.get("seed", seed) == da_status_success;
    opt_pass &=
        this->opts.get("node minimum samples", min_node_sample) == da_status_success;
    opt_pass &=
        this->opts.get("node minimum samples", min_node_sample) == da_status_success;
    opt_pass &= this->opts.get("scoring function", opt_val, method) == da_status_success;
    opt_pass &=
        this->opts.get("tree building order", opt_val, build_order) == da_status_success;
    opt_pass &=
        this->opts.get("features selection", opt_val, feat_select) == da_status_success;
    opt_pass &= this->opts.get("maximum features", nfeat_split) == da_status_success;
    opt_pass &= this->opts.get("feature threshold", feat_thresh) == da_status_success;
    opt_pass &=
        this->opts.get("minimum split score", min_split_score) == da_status_success;
    opt_pass &=
        this->opts.get("minimum split improvement", min_improvement) == da_status_success;
    opt_pass &= this->opts.get("bootstrap", opt_val, bootstrap_opt) == da_status_success;
    opt_pass &= this->opts.get("bootstrap samples factor", prop) == da_status_success;
    opt_pass &= this->opts.get("block size", block_size) == da_status_success;
    opt_pass &=
        this->opts.get("sorting method", opt_val, sort_method) == da_status_success;
    if (!opt_pass)
        return da_error_trace(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                              "Unexpected error while reading the optional parameters.");

    std::vector<da_int> seed_tree;
    try {
        forest.resize(n_tree);
        seed_tree.resize(n_tree);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    // Initialize the seeds of all the trees to be able to reproduce results if required
    std::mt19937 mt_engine;
    if (seed == -1) {
        std::random_device r;
        seed = std::abs((da_int)r());
    }
    mt_engine.seed(seed);
    std::uniform_int_distribution<da_int> uniform_dist(0, 1000000);
    std::generate(seed_tree.begin(), seed_tree.end(),
                  [&uniform_dist, &mt_engine]() { return uniform_dist(mt_engine); });

    // Initialize remaining tree optional parameters
    switch (feat_select) {
    case feat_selection::all:
        nfeat_split = n_features;
        break;

    case feat_selection::sqrt:
        nfeat_split = (da_int)std::ceil(std::sqrt(n_features));
        break;

    case feat_selection::log2:
        nfeat_split = (da_int)std::ceil(std::sqrt(n_features));
        break;

    case feat_selection::custom:
        // nfeat_split was already loaded with the maximum features option value
    default:
        break;
    }
    bootstrap = bootstrap_opt == 1;

    n_obs = n_samples;
    if (bootstrap && prop < 1.0) {
        n_obs = std::max((da_int)std::round(n_samples * prop), (da_int)1);
    }
    da_int prn_times = 0;
    da_int n_failed_tree = 0;

    // Train all the trees in parallel
#pragma omp parallel for shared(                                                         \
        n_failed_tree, forest, n_tree, max_depth, min_node_sample, method, prn_times,    \
            build_order, seed_tree, min_split_score, feat_thresh, min_improvement,       \
            n_samples, n_features, X, ldx, y, n_class, n_obs, nfeat_split, bootstrap,    \
            sort_method) default(none) schedule(dynamic)
    for (da_int i = 0; i < n_tree; i++) {
        // Set tree optional parameters
        try {
            forest[i] = std::make_unique<decision_tree<T>>(
                decision_tree(max_depth, min_node_sample, method, prn_times, build_order,
                              nfeat_split, seed_tree[i], sort_method, min_split_score,
                              feat_thresh, min_improvement, bootstrap));
        } catch (std::bad_alloc &) {
#pragma omp atomic
            n_failed_tree++;
            continue;
        }
        da_status tree_status;
        tree_status = forest[i]->set_training_data(n_samples, n_features, X, ldx, y,
                                                   n_class, n_obs, nullptr);
        tree_status = forest[i]->fit();
        forest[i]->clear_working_memory();
        if (tree_status != da_status_success) {
#pragma omp atomic
            n_failed_tree++;
        }
    }

    if (n_failed_tree != 0)
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        std::to_string(n_failed_tree) +
                            " trees failed training unexpectedly.");

    model_trained = true;
    return da_status_success;
}

template <typename T>
void random_forest<T>::parallel_count_classes(
    const T *X_test, da_int ldx_test, const da_int &n_blocks, const da_int &block_size,
    const da_int &block_rem, [[maybe_unused]] const da_int &n_threads,
    std::vector<da_int> &count_classes, std::vector<da_int> &y_pred_tree) {

#pragma omp parallel for collapse(2)                                                     \
    shared(n_blocks, forest, y_pred_tree, n_features, X_test, ldx_test, count_classes,   \
               block_rem, block_size) default(none)
    for (da_int i_block = 0; i_block < n_blocks; i_block++) {
        for (auto &tree : forest) {
            da_int start_idx = i_block * block_size;
            da_int thread_id = (da_int)omp_get_thread_num();
            da_int start_pred_idx = thread_id * block_size;
            da_int n_elem = block_size;
            if (i_block == n_blocks - 1 && block_rem > 0) {
                n_elem = block_rem;
            }
            tree->predict(n_elem, n_features, &X_test[start_idx], ldx_test,
                          &(y_pred_tree.data()[start_pred_idx]), 2);
            for (da_int i = 0; i < n_elem; i++) {
                da_int c = y_pred_tree[start_pred_idx + i];
#pragma omp atomic update
                count_classes[(start_idx + i) * n_class + c] += 1;
            }
        }
    }
}

template <typename T>
da_status random_forest<T>::predict(da_int nsamp, da_int nfeat, const T *X_test,
                                    da_int ldx_test, da_int *y_pred) {

    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp;

    if (y_pred == nullptr) {
        return da_error(this->err, da_status_invalid_input,
                        "y_pred is not a valid pointer.");
    }

    if (nfeat != n_features) {
        return da_error(this->err, da_status_invalid_input,
                        "n_features = " + std::to_string(nfeat) +
                            " doesn't match the expected value " +
                            std::to_string(n_features) + ".");
    }

    if (!model_trained) {
        return da_error(this->err, da_status_out_of_date,
                        "The model has not yet been trained or the data it is "
                        "associated with is out of date.");
    }

    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    if (this->opts.get("block size", block_size) != da_status_success)
        return da_error_trace( // LCOV_EXCL_LINE
            this->err, da_status_internal_error,
            "Unexpected error while reading the optional parameter 'block size' .");

    // Set up the parallel tasks and data. X is divided into blocks of small size
    // evaluating 1 tree on one of X's blocks is considered an independent task that
    // can be evaluated in parallel.
    std::vector<da_int> count_classes, y_pred_tree;
    da_int n_blocks, block_rem;
    da_utils::blocking_scheme(nsamp, block_size, n_blocks, block_rem);
    da_int n_threads = da_utils::get_n_threads_loop(n_blocks * n_tree);
    // We need n_threads arrays of size block_size
    try {
        y_pred_tree.resize(n_threads * block_size);
        count_classes.resize(n_class * nsamp, 0);
    } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    parallel_count_classes(X_test_temp, ldx_test_temp, n_blocks, block_size, block_rem,
                           n_threads, count_classes, y_pred_tree);

#pragma omp parallel for shared(nsamp, n_class, y_pred, count_classes) default(none)
    for (da_int i = 0; i < nsamp; i++) {
        da_int max_count = -1, class_i = -1;
        for (da_int c = 0; c < n_class; c++) {
            if (count_classes[i * n_class + c] > max_count) {
                max_count = count_classes[i * n_class + c];
                class_i = c;
            }
        }
        y_pred[i] = class_i;
    }

    if (utility_ptr1)
        delete[] (utility_ptr1);

    return da_status_success;
}

template <typename T>
da_status random_forest<T>::predict_proba(da_int nsamp, da_int nfeat, const T *X_test,
                                          da_int ldx_test, T *y_proba, da_int nclass,
                                          da_int ldy) {

    const T *X_test_temp;
    T *utility_ptr1;
    T *utility_ptr2;
    da_int ldx_test_temp;
    T *y_proba_temp;
    da_int ldy_proba_temp;

    if (nfeat != n_features) {
        return da_error(this->err, da_status_invalid_input,
                        "n_features = " + std::to_string(nfeat) +
                            " doesn't match the expected value " +
                            std::to_string(n_features) + ".");
    }

    if (nclass != n_class) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_class = " + std::to_string(nclass) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_class) + ".");
    }

    if (!model_trained) {
        return da_error(this->err, da_status_out_of_date,
                        "The model has not yet been trained or the data it is "
                        "associated with is out of date.");
    }

    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    status = this->store_2D_array(nsamp, nclass, y_proba, ldy, &utility_ptr2,
                                  const_cast<const T **>(&y_proba_temp), ldy_proba_temp,
                                  "n_samples", "n_class", "y_proba", "ldy", 1);
    if (status != da_status_success)
        return status;

    std::vector<T> sum_proba, y_proba_tree;
    da_int n_blocks, block_rem;
    da_utils::blocking_scheme(nsamp, block_size, n_blocks, block_rem);
    da_int n_threads = da_utils::get_n_threads_loop(n_blocks * n_tree);
    try {
        sum_proba.resize(n_class * nsamp);
        y_proba_tree.resize(n_threads * n_class * block_size);
    } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

#pragma omp parallel for collapse(2)                                                     \
    shared(sum_proba, nsamp, nfeat, X_test_temp, ldx_test_temp, n_blocks, block_rem,     \
               y_proba_tree, n_threads) default(none)
    for (da_int i_block = 0; i_block < n_blocks; i_block++) {
        for (auto &tree : forest) {
            da_int start_idx = i_block * block_size;
            da_int thread_id = (da_int)omp_get_thread_num();
            da_int start_pred_idx = thread_id * block_size;
            da_int n_elem = block_size;
            if (i_block == n_blocks - 1 && block_rem > 0) {
                n_elem = block_rem;
            }
            tree->predict_proba(n_elem, n_features, &X_test_temp[start_idx],
                                ldx_test_temp, &y_proba_tree.data()[start_pred_idx],
                                n_class, n_threads * block_size, 2);
            for (da_int i = 0; i < n_elem; i++) {
                for (da_int j = 0; j < n_class; j++) {
#pragma omp atomic update
                    sum_proba[j * nsamp + start_idx + i] +=
                        y_proba_tree[start_pred_idx + j * n_threads * block_size + i];
                }
            }
        }
    }

#pragma omp parallel for shared(nsamp, n_class, ldy_proba_temp, y_proba_temp, sum_proba, \
                                    n_tree) default(none)
    for (da_int i = 0; i < nsamp; i++) {
        T sum_ave_prob = 0.0;
        for (da_int j = 0; j < n_class; j++) {
            T ave_proba = sum_proba[j * nsamp + i] / n_tree;
            y_proba_temp[j * ldy_proba_temp + i] = ave_proba;
            sum_ave_prob += ave_proba;
        }
        for (da_int j = 0; j < n_class; j++) {
            y_proba_temp[j * ldy_proba_temp + i] /= sum_ave_prob;
        }
    }

    if (this->order == row_major) {

        da_utils::copy_transpose_2D_array_column_to_row_major(
            nsamp, n_class, y_proba_temp, ldy_proba_temp, y_proba, ldy);

        delete[] (utility_ptr1);
        delete[] (utility_ptr2);
    }

    return da_status_success;
}

template <typename T>
da_status random_forest<T>::predict_log_proba(da_int nsamp, da_int nfeat, const T *X_test,
                                              da_int ldx_test, T *y_log_proba,
                                              da_int nclass, da_int ldy) {

    predict_proba(nsamp, nfeat, X_test, ldx_test, y_log_proba, n_class, ldy);

    if (this->order == column_major) {
        for (da_int j = 0; j < nclass; j++) {
            for (da_int i = 0; i < nsamp; i++) {
                y_log_proba[ldy * j + i] = log(y_log_proba[ldy * j + i]);
            }
        }
    } else {
        for (da_int j = 0; j < nsamp; j++) {
            for (da_int i = 0; i < nclass; i++) {
                y_log_proba[j * ldy + i] = log(y_log_proba[j * ldy + i]);
            }
        }
    }

    return da_status_success;
}

template <typename T>
da_status random_forest<T>::score(da_int nsamp, da_int nfeat, const T *X_test,
                                  da_int ldx_test, const da_int *y_test, T *score) {

    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp;

    if (score == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "mean_accuracy is not valid pointers.");
    }

    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "nfeat = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }

    if (!model_trained) {
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

    if (this->opts.get("block size", block_size) != da_status_success)
        return da_error_trace( // LCOV_EXCL_LINE
            this->err, da_status_internal_error,
            "Unexpected error while reading the optional parameter 'block size' .");

    std::vector<da_int> count_classes, y_pred_tree;
    da_int n_blocks, block_rem;
    da_utils::blocking_scheme(nsamp, block_size, n_blocks, block_rem);
    da_int n_threads = da_utils::get_n_threads_loop(n_blocks * n_tree);
    // We need n_threads arrays of size block_size
    try {
        y_pred_tree.resize(n_threads * block_size);
        count_classes.resize(n_class * nsamp, 0);
    } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    parallel_count_classes(X_test_temp, ldx_test_temp, n_blocks, block_size, block_rem,
                           n_threads, count_classes, y_pred_tree);

    *score = 0;
#pragma omp parallel for shared(nsamp, n_class, y_test, count_classes,                   \
                                    score) default(none)
    for (da_int i = 0; i < nsamp; i++) {
        da_int max_count = -1, class_i = -1;
        for (da_int c = 0; c < n_class; c++) {
            if (count_classes[i * n_class + c] > max_count) {
                max_count = count_classes[i * n_class + c];
                class_i = c;
            }
        }
        if (class_i == y_test[i]) {
#pragma omp atomic update
            (*score)++;
        }
    }
    *score /= (T)nsamp;

    if (utility_ptr1)
        delete[] (utility_ptr1);
    return da_status_success;
}

template class random_forest<double>;
template class random_forest<float>;

} // namespace da_decision_forest

} // namespace ARCH