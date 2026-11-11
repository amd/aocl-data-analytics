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

#ifndef FOREST_INFERENCE_HPP
#define FOREST_INFERENCE_HPP

#include "macros.h"

namespace ARCH {

namespace da_decision_forest {

template <typename T> da_status decision_forest<T>::read_storage_order() {
    std::string opt_order;
    da_int iorder;
    da_status status = this->opts.get("storage order", opt_order, iorder);
    if (status != da_status_success)
        return status;
    this->order = da_order(iorder);
    return da_status_success;
}

template <typename T>
void decision_forest<T>::parallel_sum_proba(const T *X_test, da_int ldx_test,
                                            da_int nsamp, const da_int &n_threads,
                                            std::vector<T> &sum_proba,
                                            std::vector<T> &thread_proba,
                                            std::vector<T> &y_proba_tree) {

#pragma omp parallel num_threads(n_threads)                                              \
    shared(X_test, ldx_test, nsamp, thread_proba, y_proba_tree) default(none)
    {
        da_int tid = (da_int)omp_get_thread_num();
        // Per-thread accumulator and per-tree scratch, both row-major [sample, class]
        T *local_proba = &thread_proba[(da_int)tid * nsamp * n_class];
        T *local_tree = &y_proba_tree[(da_int)tid * nsamp * n_class];

#pragma omp for schedule(dynamic)
        for (da_int t = 0; t < n_tree; t++) {
            forest[t]->set_storage_order(row_major);
            forest[t]->compact_traverse(nsamp, n_features, X_test, ldx_test, local_tree,
                                        n_class, n_class, false);
            for (da_int i = 0; i < nsamp; i++) {
                for (da_int j = 0; j < n_class; j++) {
                    local_proba[i * n_class + j] += local_tree[i * n_class + j];
                }
            }
        }
    }

    // Reduce the per-thread partial sums into class-major [class, sample] layout. Each
    // sample is handled by exactly one thread, so the store is non-atomic.
    da_int cl_samp = nsamp * n_class;
#pragma omp parallel for num_threads(n_threads)                                          \
    shared(nsamp, n_threads, sum_proba, thread_proba, cl_samp) default(none)
    for (da_int i = 0; i < nsamp; i++) {
        for (da_int j = 0; j < n_class; j++) {
            T sum = 0;
            da_int idx_cl = i * n_class + j;
            for (da_int t = 0; t < n_threads; t++) {
                sum += thread_proba[t * cl_samp + idx_cl];
            }
            sum_proba[j * nsamp + i] = sum;
        }
    }
}

template <typename T>
da_status
decision_forest<T>::transpose_input_if_needed(da_int nsamp, const T *X_test,
                                              da_int ldx_test, std::vector<T> &X_row,
                                              const T *&X_eval, da_int &ldx_eval) {

    X_eval = X_test;
    ldx_eval = ldx_test;

    // The vectorized tree kernels operate on row-major data.
    if (this->order != column_major || n_tree == 0)
        return da_status_success;

    try {
        X_row.resize((size_t)nsamp * n_features);
    } catch (const std::bad_alloc &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    da_status status = da_utils::switch_order_copy(this->order, nsamp, n_features, X_test,
                                                   ldx_test, X_row.data(), n_features);
    if (status != da_status_success)
        return da_error(this->err, status, // LCOV_EXCL_LINE
                        "Failed to transpose the input data.");
    X_eval = X_row.data();
    ldx_eval = n_features;
    return da_status_success;
}

template <typename T>
da_status decision_forest<T>::predict(da_int nsamp, da_int nfeat, const T *X_test,
                                      da_int ldx_test, da_int *y_pred) {

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

    if (!this->model_trained) {
        return da_error(this->err, da_status_out_of_date,
                        "The model has not yet been trained or the data it is "
                        "associated with is out of date.");
    }

    // Inference handles both storage orders natively, no input transposition
    da_status status = read_storage_order();
    if (status != da_status_success)
        return status;
    status = this->check_2D_array(this->order, nsamp, nfeat, X_test, ldx_test,
                                  "n_samples", "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    // Transpose the input once (if needed) so that every tree can reuse it.
    std::vector<T> X_row;
    const T *X_eval;
    da_int ldx_eval;
    status = transpose_input_if_needed(nsamp, X_test, ldx_test, X_row, X_eval, ldx_eval);
    if (status != da_status_success)
        return status;

    // Set up the parallel tasks: each tree is evaluated on the full set of samples and
    // accumulates its class probabilities into a private buffer, reduced afterwards (no
    // atomics). Soft voting (mean probability).
    std::vector<T> sum_proba, thread_proba, y_proba_tree;
    da_int n_threads = da_utils::get_n_threads_loop(n_tree);
    try {
        sum_proba.resize((size_t)n_class * nsamp, 0);
        thread_proba.resize((size_t)n_threads * nsamp * n_class, 0);
        y_proba_tree.resize((size_t)n_threads * nsamp * n_class);
    } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    parallel_sum_proba(X_eval, ldx_eval, nsamp, n_threads, sum_proba, thread_proba,
                       y_proba_tree);

#pragma omp parallel for shared(nsamp, n_class, y_pred, sum_proba) default(none)
    for (da_int i = 0; i < nsamp; i++) {
        T max_proba = -1;
        da_int class_i = -1;
        for (da_int c = 0; c < n_class; c++) {
            if (sum_proba[c * nsamp + i] > max_proba) {
                max_proba = sum_proba[c * nsamp + i];
                class_i = c;
            }
        }
        y_pred[i] = class_i;
    }

    return da_status_success;
}

template <typename T>
da_status decision_forest<T>::predict_proba(da_int nsamp, da_int nfeat, const T *X_test,
                                            da_int ldx_test, T *y_proba, da_int nclass,
                                            da_int ldy) {

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

    if (!this->model_trained) {
        return da_error(this->err, da_status_out_of_date,
                        "The model has not yet been trained or the data it is "
                        "associated with is out of date.");
    }

    da_status status = read_storage_order();
    if (status != da_status_success)
        return status;
    status = this->check_2D_array(this->order, nsamp, nfeat, X_test, ldx_test,
                                  "n_samples", "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    // Validate the output array in the user's storage order (no NaN check, no copy):
    // the probabilities are written straight into y_proba, so no temporary buffer is
    // needed.
    if (y_proba == nullptr)
        return da_error(this->err, da_status_invalid_pointer, "y_proba is null.");
    da_int ldy_min = (this->order == column_major) ? nsamp : nclass;
    if (ldy < ldy_min)
        return da_error(this->err, da_status_invalid_leading_dimension,
                        "The function was called with ldy = " + std::to_string(ldy) +
                            ". Constraint: ldy >= " + std::to_string(ldy_min) + ".");

    // Transpose the input once (if needed) so that every tree can reuse it.
    std::vector<T> X_row;
    const T *X_eval;
    da_int ldx_eval;
    status = transpose_input_if_needed(nsamp, X_test, ldx_test, X_row, X_eval, ldx_eval);
    if (status != da_status_success)
        return status;

    // Parallelize over trees only (this scales better than blocking over samples): each
    // thread evaluates a subset of trees on all samples and accumulates the class
    // probabilities into a private buffer, reduced afterwards (no atomics).
    std::vector<T> sum_proba, thread_proba, y_proba_tree;
    da_int n_threads = da_utils::get_n_threads_loop(n_tree);
    try {
        sum_proba.resize((size_t)n_class * nsamp);
        thread_proba.resize((size_t)n_threads * nsamp * n_class, 0);
        y_proba_tree.resize((size_t)n_threads * nsamp * n_class);
    } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    parallel_sum_proba(X_eval, ldx_eval, nsamp, n_threads, sum_proba, thread_proba,
                       y_proba_tree);

    // Write the averaged, normalized probabilities straight into the user's buffer in
    // the requested storage order
    if (this->order == column_major) {
#pragma omp parallel for shared(nsamp, n_class, ldy, y_proba, sum_proba,                 \
                                    n_tree) default(none)
        for (da_int i = 0; i < nsamp; i++) {
            T sum_ave_prob = 0.0;
            for (da_int j = 0; j < n_class; j++) {
                T ave_proba = sum_proba[j * nsamp + i] / n_tree;
                y_proba[j * ldy + i] = ave_proba;
                // cannot be 0: no guard
                sum_ave_prob += ave_proba;
            }
            for (da_int j = 0; j < n_class; j++) {
                y_proba[j * ldy + i] /= sum_ave_prob;
            }
        }
    } else { // row_major: probabilities laid out as [sample, class]
#pragma omp parallel for shared(nsamp, n_class, ldy, y_proba, sum_proba,                 \
                                    n_tree) default(none)
        for (da_int i = 0; i < nsamp; i++) {
            T sum_ave_prob = 0.0;
            for (da_int j = 0; j < n_class; j++) {
                T ave_proba = sum_proba[j * nsamp + i] / n_tree;
                y_proba[i * ldy + j] = ave_proba;
                sum_ave_prob += ave_proba;
            }
            for (da_int j = 0; j < n_class; j++) {
                y_proba[i * ldy + j] /= sum_ave_prob;
            }
        }
    }

    return da_status_success;
}

template <typename T>
da_status decision_forest<T>::predict_log_proba(da_int nsamp, da_int nfeat,
                                                const T *X_test, da_int ldx_test,
                                                T *y_log_proba, da_int nclass,
                                                da_int ldy) {

    da_status status =
        predict_proba(nsamp, nfeat, X_test, ldx_test, y_log_proba, nclass, ldy);
    if (status != da_status_success)
        return status;

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
da_status decision_forest<T>::score(da_int nsamp, da_int nfeat, const T *X_test,
                                    da_int ldx_test, const da_int *y_test, T *score) {

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

    if (!this->model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    // Inference handles both storage orders natively, no input transposition
    da_status status = read_storage_order();
    if (status != da_status_success)
        return status;
    status = this->check_2D_array(this->order, nsamp, nfeat, X_test, ldx_test,
                                  "n_samples", "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(nsamp, y_test, "n_samples", "y_test", 1);
    if (status != da_status_success)
        return status;

    // Transpose the input once (if needed) so that every tree can reuse it.
    std::vector<T> X_row;
    const T *X_eval;
    da_int ldx_eval;
    status = transpose_input_if_needed(nsamp, X_test, ldx_test, X_row, X_eval, ldx_eval);
    if (status != da_status_success)
        return status;

    std::vector<T> sum_proba, thread_proba, y_proba_tree;
    da_int n_threads = da_utils::get_n_threads_loop(n_tree);
    try {
        sum_proba.resize((size_t)n_class * nsamp, 0);
        thread_proba.resize((size_t)n_threads * nsamp * n_class, 0);
        y_proba_tree.resize((size_t)n_threads * nsamp * n_class);
    } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    parallel_sum_proba(X_eval, ldx_eval, nsamp, n_threads, sum_proba, thread_proba,
                       y_proba_tree);

    *score = 0;
#pragma omp parallel for shared(nsamp, n_class, y_test, sum_proba, score) default(none)
    for (da_int i = 0; i < nsamp; i++) {
        T max_proba = -1;
        da_int class_i = -1;
        for (da_int c = 0; c < n_class; c++) {
            if (sum_proba[c * nsamp + i] > max_proba) {
                max_proba = sum_proba[c * nsamp + i];
                class_i = c;
            }
        }
        if (class_i == y_test[i]) {
#pragma omp atomic update
            (*score)++;
        }
    }
    *score /= (T)nsamp;

    return da_status_success;
}

} // namespace da_decision_forest
} // namespace ARCH

#endif