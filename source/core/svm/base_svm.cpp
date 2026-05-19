/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "da_cache.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "da_vector.hpp"
#include "kernel_functions.hpp"
#include "macros.h"
#include "miscellaneous.hpp"
#include "options.hpp"
#include "svm.hpp"
#include "svm_options.hpp"
#include "svm_types.hpp"
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <utility>
#include <vector>
using namespace std::string_literals;

namespace ARCH {

template <typename T>
static void rbf_wrapper(da_order order, da_int m, da_int n, da_int k, const T *X,
                        T *x_norm, da_int compute_X_norms, da_int ldx, const T *Y,
                        T *y_norm, da_int compute_Y_norms, da_int ldy, T *D, da_int ldd,
                        T gamma, da_int /*degree*/, T /*coef0*/, bool X_is_Y,
                        da_int vectorisation) {
    return ARCH::da_kernel_functions::rbf_kernel_internal(
        order, m, n, k, X, x_norm, compute_X_norms, ldx, Y, y_norm, compute_Y_norms, ldy,
        D, ldd, gamma, X_is_Y, vectorisation);
}

template <typename T>
static void linear_wrapper(da_order order, da_int m, da_int n, da_int k, const T *X,
                           T * /*x_norm*/, da_int /*compute_X_norms*/, da_int ldx,
                           const T *Y, T * /*y_norm*/, da_int /*compute_Y_norms*/,
                           da_int ldy, T *D, da_int ldd, T /*gamma*/, da_int /*degree*/,
                           T /*coef0*/, bool X_is_Y, da_int /*vectorisation*/) {
    return ARCH::da_kernel_functions::linear_kernel_internal(order, m, n, k, X, ldx, Y,
                                                             ldy, D, ldd, X_is_Y);
}

template <typename T>
static void sigmoid_wrapper(da_order order, da_int m, da_int n, da_int k, const T *X,
                            T * /*x_norm*/, da_int /*compute_X_norms*/, da_int ldx,
                            const T *Y, T * /*y_norm*/, da_int /*compute_Y_norms*/,
                            da_int ldy, T *D, da_int ldd, T gamma, da_int /*degree*/,
                            T coef0, bool X_is_Y, da_int vectorisation) {
    return ARCH::da_kernel_functions::sigmoid_kernel_internal(
        order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0, X_is_Y, vectorisation);
}

template <typename T>
static void polynomial_wrapper(da_order order, da_int m, da_int n, da_int k, const T *X,
                               T * /*x_norm*/, da_int /*compute_X_norms*/, da_int ldx,
                               const T *Y, T * /*y_norm*/, da_int /*compute_Y_norms*/,
                               da_int ldy, T *D, da_int ldd, T gamma, da_int degree,
                               T coef0, bool X_is_Y, da_int vectorisation) {
    return ARCH::da_kernel_functions::polynomial_kernel_internal(
        order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, degree, coef0, X_is_Y,
        vectorisation);
}

/*
 * Base SVM handle class that contains members that
 * are common for all SVM models.
 *
 * This handle is inherited by all specialized svm handles.
 *
 * The inheritance scheme is as follows:
 *
 *                       BASE_SVM
 *                         /   \
 *                        /     \
 *                   C-SVM       Nu-SVM
 *                  /     \      /     \
 *                 /       \    /       \
 *              SVC       SVR Nu-SVC   Nu-SVR
 */

namespace da_svm {

using namespace da_svm_types;
using namespace da_model_persistence;

// This forward declaration is here to allow for "friending" it with base_svm few lines below
template <typename T>
base_svm<T>::base_svm(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train)
    : XUSR(XUSR), yusr(yusr), n(n), p(p), ldx(ldx_train){};
template <typename T> base_svm<T>::~base_svm(){};

/* Main function which contains Thunder loop */
template <typename T> da_status base_svm<T>::compute() {
    da_status status = da_status_success;
    // Define them in this scope since they are large matrices and do not need to be in the class scope
    da_cache::LRUCache<T> cache(*err);
    std::vector<T *> ptr_kernel_col;
    std::vector<T> gradient_threads;
    // Used at local_smo
    da_vector::da_vector<T> kernel_matrix;
    std::vector<T> local_kernel_matrix_row_major, kernel_diagonal;
    std::vector<da_int> real_indices;
    std::vector<T> X_temp; // Contain relevant slices of X for kernel computation

    if (mod == da_svm_model::svr || mod == da_svm_model::nusvr)
        actual_size = n * 2;
    else
        actual_size = n;

    iter = 0;
    if (max_iter == 0)
        max_iter = DA_INT_MAX;
    // Variable to keep track of number of selected indexes in working set. At iter==0 equals 0, later ws_size/2 because we copy last half to the first half.
    da_int n_selected;
    // Global convergence variables, if first diff of local SMO does not change for some number of iterations, then stop
    T first_diff = T(0), previous_first_diff = T(0);
    da_int no_diff_counter = 0;
    // For multiclass we take slices of user's data on indexes where class is i or j (those indexes are stored in idx_class and obtained in set_data())
    if (ismulticlass) {
        X = new T[n * p];
        y = new T[n];

        for (da_int i = 0; i < n; i++) {
            da_int current_idx = idx_class[i];
            for (da_int j = 0; j < p; j++) {
                X[i + j * n] = XUSR[current_idx + j * ldx];
            }
            // 0 is transformed to -1 in initialisation
            y[i] = idx_is_positive[i] ? 1.0 : 0.0;
        }
        ldx_2 = n; // At this point X array has to be in the dense layout
    } else {
        X = (T *)XUSR;
        y = (T *)yusr;
        ldx_2 = ldx;
    }
    if (max_ws_size == T(-1.0)) {
        // Heuristically chosen boundaries, they depend on
        select_ws_size(n, (svm_kernel)kernel_function, max_ws_size);
    }
    compute_ws_size(ws_size, max_ws_size);
    vectorization_type simd_type_wssi, simd_type_wssj;
    select_simd_size_wss<T>(ws_size, padding, simd_type_wssi, simd_type_wssj);
    wssi_vec_type = simd_type_wssi;
    wssj_vec_type = simd_type_wssj;
    // Add telemetry
    context_set_hidden_settings(
        "svm.setup"s, "kernel.wssi_kernel.type="s + std::to_string(wssi_vec_type) +
                          ",kernel.wssj_kernel.type="s + std::to_string(wssj_vec_type) +
                          ",kernel.padding="s + std::to_string(padding));
    /* Interpret cache_size from MB to number of columns of kernel matrix it can hold */
    // Possibility of overflow if cache_size is too big
    da_int cache_col_capacity;
    if (cache_size < 0)
        // Potentially better rules depending on n can be applied
        cache_col_capacity = std::min(n, (da_int)2048);
    else {
        uint64_t cache_size_values;
        cache_size_values = cache_size * 1024 * 1024 / sizeof(T);
        cache_col_capacity = (da_int)std::round(cache_size_values / n);
        cache_col_capacity = std::min(
            cache_col_capacity, n); // Ensure capacity is not larger than needed (n^2)
    }
    if (0 < cache_col_capacity && cache_col_capacity < ws_size)
        cache_smaller_than_ws = true;
    else
        cache_smaller_than_ws = false;
    // Initialise which kernel function will be used
    switch (kernel_function) {
    case rbf:
        kernel_f = &rbf_wrapper<T>;
        try {
            x_norm_aux.resize(n + padding);
            y_norm_aux.resize(ws_size + padding);
        } catch (std::bad_alloc &) {
            return da_error(err, da_status_memory_error, "Memory allocation error");
        }
        // Precompute x_norm since it is constant vector
        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                x_norm_aux[j] += X[i * ldx_2 + j] * X[i * ldx_2 + j];
            }
        }
        break;
    case linear:
        kernel_f = &linear_wrapper<T>;
        break;
    case polynomial:
        kernel_f = &polynomial_wrapper<T>;
        break;
    case sigmoid:
        kernel_f = &sigmoid_wrapper<T>;
        break;
    }
    status = cache.set_size(cache_col_capacity, n);
    if (status != da_status_success) {
        return da_error(err, status,
                        "Memory allocation error inside cache initialisation.");
    }
    try {
        // Outer WSS
        ws_indicator.resize(actual_size);
        index_aux.resize(actual_size); // For nu problem also used in initialisation
        // Compute kernel
        ws_indexes.resize(ws_size);
        ptr_kernel_col.resize(ws_size);
        // Large matrix, don't initialise (performance)
        kernel_matrix.resize(ws_size * n);
        X_temp.resize(ws_size * p);
        // Local SMO
        gradient.resize(actual_size);
        // This is because if compute() is called many times one after another, it causes problems in
        // nu variant because in nusvm::initialisation() gradient is not explicitly set to 0, but relies on 0 initialisation here
        // (can be modified if it's unusual design)
        da_std::fill(gradient.begin(), gradient.end(), T(0));
        response.resize(actual_size);
        alpha.resize(actual_size);
        local_alpha.resize(ws_size);
        local_gradient.resize(ws_size + padding);
        local_response.resize(ws_size);
        local_kernel_matrix_row_major.resize(ws_size * (ws_size + padding));
        kernel_diagonal.resize(ws_size + padding);
        real_indices.resize(ws_size);
        I_low_p.resize(ws_size + padding);
        I_up_p.resize(ws_size + padding);
        I_low_n.resize(ws_size + padding);
        I_up_n.resize(ws_size + padding);
        // Update gradient
        alpha_diff.resize(ws_size);
        // Result handling
        n_support_per_class.resize(2, 0);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    status = initialisation(n, gradient, response, alpha, cache);
    if (status != da_status_success)
        return status;
    for (; iter < max_iter; iter++) {
        ////////// Outer WSS
        da_std::fill(ws_indicator.begin(), ws_indicator.end(), false);
        if (iter == 0) {
            n_selected = 0;
            outer_wss(actual_size, ws_indexes, ws_indicator, n_selected);
        }
        // Before next iteration, copy last half of indexes into the first half (heuristic barely mentioned in the paper, but used in implementation)
        else {
            n_selected = ws_size / 2;
            for (da_int i = 0; i < n_selected; i++) {
                ws_indexes[i] = ws_indexes[i + n_selected];
                ws_indicator[ws_indexes[i]] = true;
            }
            outer_wss(actual_size, ws_indexes, ws_indicator, n_selected);
        }
        //////////
        // Compute kernel matrix using working set indexes
        kernel_compute(ws_indexes, ws_size, X_temp, kernel_matrix, ptr_kernel_col, cache);
        // Use kernel matrix to perform local SMO (as a result alpha, alpha_diff and first_diff are updated)
        local_smo(ws_size, ws_indexes, ptr_kernel_col, local_kernel_matrix_row_major,
                  kernel_diagonal, real_indices, alpha, local_alpha, gradient,
                  local_gradient, response, local_response, I_low_p, I_up_p, I_low_n,
                  I_up_n, first_diff, alpha_diff, std::nullopt);
        // Global gradient update based on alpha_diff
        update_gradient(gradient.data(), gradient_threads, alpha_diff, n, ws_size,
                        ptr_kernel_col);
        // Check global convergence
        // Stop when first_diff does not change for 5 iteration OR first_diff is less than tolerance
        // Additionally make sure that we perform at least 5 iterations
        no_diff_counter = std::abs(first_diff - previous_first_diff) < tol * 1e-3
                              ? no_diff_counter + 1
                              : 0;
        previous_first_diff = first_diff;
        if ((no_diff_counter > 4 || first_diff < tol) && iter > 4)
            break;
    }
    // Interpret results and save them into appropriate arrays
    status = set_bias(alpha, gradient, response, actual_size, bias);
    if (status != da_status_success)
        return status;
    status = set_sv(alpha, n_support);
    if (status != da_status_success)
        return status;

    // Build a contiguous column-major support vector matrix (n_support x p)
    // so that prediction can use direct pointer arithmetic instead of scattered gathers
    try {
        sv_matrix.resize(n_support * p);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    for (da_int i = 0; i < n_support; i++) {
        da_int src_idx = support_indexes[i];
        for (da_int j = 0; j < p; j++) {
            sv_matrix[i + j * n_support] = X[src_idx + j * ldx_2];
        }
    }

    if (ismulticlass) {
        delete[] X;
        delete[] y;
    }
    return status;
}

/* Predict SVM */
template <typename T>
da_status base_svm<T>::predict(da_int nsamples, da_int nfeat, const T *X_test,
                               da_int ldx_test, T *predictions) {
    da_status status = da_status_success;
    // Vector that will store decision values
    status = decision_function(nsamples, nfeat, X_test, ldx_test, predictions);
    if (mod == da_svm_model::svc || mod == da_svm_model::nusvc) {
        for (da_int i = 0; i < nsamples; i++) {
            predictions[i] = predictions[i] > 0 ? 1 : 0;
        }
    }
    return status;
}

/* Parallel loop for decision function, templated on norm precomputation */
template <typename T>
template <bool PrecomputedNorms>
void base_svm<T>::decision_function_loop(
    da_int nfeat, const T *X_test, da_int ldx_test, T *decision_values,
    da_int total_blocks, [[maybe_unused]] da_int active_threads, da_int inner_block_size,
    da_int outer_block_size, da_int inner_block_count, da_int outer_block_count,
    da_int inner_block_remainder, da_int outer_block_remainder,
    std::vector<T> &kernel_matrices, std::vector<T> &local_decisions,
    std::vector<T> &sv_norms, std::vector<T> &test_norms,
    vectorization_type vectorisation_full) {
    constexpr da_int compute_norms = PrecomputedNorms ? 1 : 0;
#pragma omp parallel for schedule(dynamic) num_threads(active_threads) default(none)     \
    shared(total_blocks, nfeat, X_test, ldx_test, decision_values, inner_block_size,     \
               outer_block_size, inner_block_count, outer_block_count,                   \
               inner_block_remainder, outer_block_remainder, n_support, kernel_matrices, \
               local_decisions, sv_norms, test_norms, sv_matrix, support_coefficients,   \
               gamma, degree, coef0, kernel_f, vectorisation_full)
    for (da_int block_idx = 0; block_idx < total_blocks; block_idx++) {
        da_int outer_sample_block_idx = block_idx / inner_block_count;
        da_int inner_block_idx = block_idx % inner_block_count;
        da_int tid = omp_get_thread_num();
        T *my_kernel = kernel_matrices.data() + tid * inner_block_size * outer_block_size;
        T *my_decision = local_decisions.data() + tid * outer_block_size;

        bool last_inner =
            inner_block_remainder > 0 && inner_block_idx == inner_block_count - 1;
        bool last_outer =
            outer_block_remainder > 0 && outer_sample_block_idx == outer_block_count - 1;
        da_int cur_inner = last_inner ? inner_block_remainder : inner_block_size;
        da_int cur_outer = last_outer ? outer_block_remainder : outer_block_size;
        vectorization_type vectorisation = vectorisation_full;
        if (last_inner || last_outer)
            da_kernel_functions::select_simd_size<T>(std::max(cur_inner, cur_outer),
                                                     vectorisation);

        da_int sample_start = outer_sample_block_idx * outer_block_size;
        const T *X_test_block = X_test + sample_start;
        da_int inner_offset = inner_block_idx * inner_block_size;
        const T *sv_data = sv_matrix.data() + inner_offset;
        T *x_norm_ptr = nullptr;
        T *y_norm_ptr = nullptr;
        if constexpr (PrecomputedNorms) {
            x_norm_ptr = sv_norms.data() + inner_offset;
            y_norm_ptr = test_norms.data() + sample_start;
        }
        kernel_f(column_major, cur_inner, cur_outer, nfeat, sv_data, x_norm_ptr,
                 compute_norms, n_support, X_test_block, y_norm_ptr, compute_norms,
                 ldx_test, my_kernel, cur_inner, gamma, degree, coef0, false,
                 (da_int)vectorisation);
        da_blas::cblas_gemv(
            CblasColMajor, CblasTrans, cur_inner, cur_outer, (T)1.0, my_kernel, cur_inner,
            support_coefficients.data() + inner_offset, 1, (T)0.0, my_decision, 1);
        T *decision_block = decision_values + sample_start;
        for (da_int i = 0; i < cur_outer; i++) {
#pragma omp atomic
            decision_block[i] += my_decision[i];
        }
    }
}

/* Calculate decision function */
template <typename T>
da_status base_svm<T>::decision_function(da_int nsamples, da_int nfeat, const T *X_test,
                                         da_int ldx_test, T *decision_values) {
    for (da_int i = 0; i < nsamples; i++)
        decision_values[i] = bias;
    if (n_support == 0 || nsamples == 0)
        return da_status_success;
    bool use_precomputed_norms = (kernel_function == rbf);

    // Threading and blocking parameters.
    da_int thread_count = omp_get_max_threads();
    constexpr da_int max_outer_block_size = 192;
    constexpr da_int max_inner_block_size = 256;

    // Outer (sample) blocking
    da_int outer_block_size =
        std::min((da_int)std::ceil((T)nsamples / thread_count), max_outer_block_size);
    da_int outer_block_count, outer_block_remainder;
    da_utils::blocking_scheme(nsamples, outer_block_size, outer_block_count,
                              outer_block_remainder);
    // Inner (support vector) blocking
    da_int inner_block_size = std::min(n_support, max_inner_block_size);
    da_int inner_block_count, inner_block_remainder;
    da_utils::blocking_scheme(n_support, inner_block_size, inner_block_count,
                              inner_block_remainder);

    // 2D parallelism over (outer, inner) block pairs
    da_int total_blocks = outer_block_count * inner_block_count;
    da_int active_threads = std::min(thread_count, total_blocks);

    // Per-thread buffers
    std::vector<T> kernel_matrices;
    std::vector<T> sv_norms, test_norms;
    std::vector<T> local_decisions;
    try {
        kernel_matrices.resize(active_threads * inner_block_size * outer_block_size);
        local_decisions.resize(active_threads * outer_block_size);
        if (use_precomputed_norms) {
            sv_norms.resize(n_support);
            test_norms.resize(nsamples);
        }
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    if (use_precomputed_norms) {
        da_std::fill(sv_norms.begin(), sv_norms.end(), (T)0.0);
        for (da_int j = 0; j < nfeat; j++) {
            const T *sv_col = sv_matrix.data() + j * n_support;
            for (da_int i = 0; i < n_support; i++) {
                T v = sv_col[i];
                sv_norms[i] += v * v;
            }
        }
        da_std::fill(test_norms.begin(), test_norms.end(), (T)0.0);
        for (da_int j = 0; j < nfeat; j++) {
            const T *test_col = X_test + j * ldx_test;
            for (da_int i = 0; i < nsamples; i++) {
                T v = test_col[i];
                test_norms[i] += v * v;
            }
        }
    }
    // Precompute SIMD type for the common (full-size) blocks
    vectorization_type vectorisation_full;
    da_kernel_functions::select_simd_size<T>(std::max(inner_block_size, outer_block_size),
                                             vectorisation_full);

    if (use_precomputed_norms)
        decision_function_loop<true>(
            nfeat, X_test, ldx_test, decision_values, total_blocks, active_threads,
            inner_block_size, outer_block_size, inner_block_count, outer_block_count,
            inner_block_remainder, outer_block_remainder, kernel_matrices,
            local_decisions, sv_norms, test_norms, vectorisation_full);
    else
        decision_function_loop<false>(
            nfeat, X_test, ldx_test, decision_values, total_blocks, active_threads,
            inner_block_size, outer_block_size, inner_block_count, outer_block_count,
            inner_block_remainder, outer_block_remainder, kernel_matrices,
            local_decisions, sv_norms, test_norms, vectorisation_full);
    return da_status_success;
}

/* Compute size of the outer working set */
template <typename T>
void base_svm<T>::compute_ws_size(da_int &ws_size, da_int max_ws_size) {
    // Pick minimum between maximum power of two such that it is less than n, or some constant in this case 1024
    da_int pow_two = maxpowtwo(actual_size);
    ws_size = std::min(pow_two, max_ws_size);
}

/* Given a set of indices 'idx', compute the kernel matrix between X and X_temp,
   where X is the original data matrix and X_temp is subset of X, consisting of rows specified in 'idx'.
   Only the columns, that are not stored in cache will be computed. Results are ultimately stored in ptr_kernel_col. */
template <typename T>
void base_svm<T>::kernel_compute(std::vector<da_int> &idx, da_int &idx_size,
                                 std::vector<T> &X_temp,
                                 da_vector::da_vector<T> &kernel_temp,
                                 std::vector<T *> &ptr_kernel_col,
                                 da_cache::LRUCache<T> &cache) {
    // Vector to store indexes that are not in cache
    std::vector<da_int> idx_to_compute(idx_size);
    // Vector of pointers in cache memory that we will copy computed values to
    std::vector<T *> free_pointers(idx_size);
    da_int idx_to_compute_count = 0;
    da_int rhs_kernel_matrix = idx_size; // Position of right-hand-side of kernel matrix

    // Check how many indexes are needed to compute
    for (da_int i = 0; i < idx_size; i++) {
        // Modulo n is needed for SVR/NuSVR case where idx has values from 0 to 2n
        da_int current_idx = idx[i] % n;
        T *values_ptr;
        // If index is in cache get() will return pointer to the column in the kernel matrix
        // Otherwise it will return nullptr
        values_ptr = cache.active_ ? cache.get(current_idx) : nullptr;
        if (values_ptr) {
            // If 0 < cache_size < ws_size, values_ptr will be invalidated when cache.put() is called.
            // Thus we need to copy the values to the kernel_temp
            if (cache_smaller_than_ws) {
                // kernel_temp is n by idx_size matrix, we compute new kernel values in the first idx_to_compute_count columns
                // The rest of the columns (idx_size - idx_to_compute_count) is empty, that is the space that we utilise here.
                // Effectively utilising entire array
                for (da_int j = 0; j < n; j++) {
                    kernel_temp[(rhs_kernel_matrix - 1) * n + j] = values_ptr[j];
                }
                ptr_kernel_col[i] = &kernel_temp[(rhs_kernel_matrix - 1) * n];
                rhs_kernel_matrix--;
            } else {
                // If cache is large enough, we can just use pointer to the column in the kernel matrix as it will not be overwritten
                ptr_kernel_col[i] = values_ptr;
            }
        } else {
            idx_to_compute[idx_to_compute_count] = i;
            idx_to_compute_count++;
        }
    }

    if (idx_to_compute_count > 0) {
        // Based on experiments, using more threads does not improve performance
        [[maybe_unused]] da_int n_threads_get_slices = da_utils::get_n_threads_loop(16);

#pragma omp parallel for if (idx_to_compute_count > 64)                                  \
    num_threads(n_threads_get_slices) default(none)                                      \
    shared(idx, idx_to_compute, idx_to_compute_count, X_temp, kernel_temp, X, n, p,      \
               ldx_2)
        // Get the relevant slices of original matrix (working set)
        // Note: it would be more efficient to operate on row-major order
        for (da_int i = 0; i < idx_to_compute_count; i++) {
            da_int current_idx = idx[idx_to_compute[i]] % n;
            for (da_int j = 0; j < p; j++) {
                X_temp[i + j * idx_to_compute_count] = X[current_idx + j * ldx_2];
            }
        }

        // Call to appropriate kernel function. Note that only idx_to_compute_count columns of kernel_temp will be filled.
        vectorization_type vectorisation;
        da_kernel_functions::select_simd_size<T>(n, vectorisation);
        // Variables used in euclidean_distance interface, 1 means to use precomputed norms, 2 means to compute norms
        da_int compute_X_norms = 1;
        da_int compute_y_norms = 2;
        kernel_f(column_major, n, idx_to_compute_count, p, X, x_norm_aux.data(),
                 compute_X_norms, ldx_2, X_temp.data(), y_norm_aux.data(),
                 compute_y_norms, idx_to_compute_count, kernel_temp.data(), n, gamma,
                 degree, coef0, false, (da_int)vectorisation);

        // Update cache with idx_computed_count new columns of kernel matrix, otherwise just fill result array
        if (cache.active_) {
            // Build index array and call cache.put once to store all computed columns and update the LRU list
            std::vector<da_int> idx_temp(idx_to_compute_count);
            for (da_int i = 0; i < idx_to_compute_count; i++) {
                da_int current_idx = idx_to_compute[i];
                idx_temp[i] = idx[current_idx] % n;
                // Fill result array
                ptr_kernel_col[current_idx] = &kernel_temp[i * n];
            }
            // Pass raw data pointer and stride to cache - avoids storing pointers to local data
            cache.put(idx_temp, kernel_temp.data(), n);
        } else {
            for (da_int i = 0; i < idx_to_compute_count; i++) {
                da_int current_idx = idx_to_compute[i];
                ptr_kernel_col[current_idx] = &kernel_temp[i * n];
            }
        }
    }
};

// Formula for global gradient update is:   gradient = gradient + sum_over_columns(alpha_diff[i] * i_th_column_kernel_matrix)
// Here we benefit from column-major order of kernel matrix
// alpha_diff is of length ws_size, kernel_matrix is nrow by ncol, gradient is of length nrow
// get_kernel_col: callable that takes da_int i and returns const T* to the i-th kernel column
template <typename T>
void base_svm<T>::update_gradient_impl(T *gradient, std::vector<T> &gradient_threads,
                                       std::vector<T> &alpha_diff, da_int &nrow,
                                       da_int &ncol,
                                       std::function<const T *(da_int)> get_kernel_col) {
    std::vector<T> gradient_temp(nrow, 0);
    da_int n_threads = da_utils::get_n_threads_loop(32);
    gradient_threads.resize(nrow * n_threads);

#pragma omp parallel shared(alpha_diff, nrow, ncol, gradient_threads) default(none)      \
    num_threads(n_threads) firstprivate(get_kernel_col)
    {
        da_int thread_id = omp_get_thread_num();
        T *local_grad = &gradient_threads[thread_id * nrow];
#pragma omp for schedule(static)
        for (da_int i = 0; i < ncol; i++) {
            const T *const_kernel = get_kernel_col(i);
            // Accumulate into the thread-local buffer
            da_blas::cblas_axpy(nrow, alpha_diff[i], const_kernel, 1, local_grad, 1);
        }
    }
// Merge thread-local contributions in fixed order
#pragma omp parallel for schedule(static) num_threads(n_threads) default(none)           \
    shared(gradient_temp, gradient_threads, nrow, n_threads)
    for (da_int i = 0; i < nrow; i++) {
        T sum = 0;
        for (da_int t = 0; t < n_threads; t++) {
            sum += gradient_threads[t * nrow + i];
            gradient_threads[t * nrow + i] = 0; // Zero thread-local buffer
        }
        gradient_temp[i] += sum;
    }

    // Special path for regression problems since gradient's length is 2 * nrow
    if (mod == da_svm_model::svr || mod == da_svm_model::nusvr) {
        for (da_int i = 0; i < nrow * 2; i++) {
            gradient[i] += gradient_temp[i % nrow];
        }
    } else {
        for (da_int i = 0; i < nrow; i++) {
            gradient[i] += gradient_temp[i];
        }
    }
}

template <typename T>
void base_svm<T>::update_gradient(T *gradient, std::vector<T> &gradient_threads,
                                  std::vector<T> &alpha_diff, da_int &nrow, da_int &ncol,
                                  std::vector<T *> &ptr_kernel_col) {
    update_gradient_impl(gradient, gradient_threads, alpha_diff, nrow, ncol,
                         [&ptr_kernel_col](da_int i) { return ptr_kernel_col[i]; });
}

template <typename T>
void base_svm<T>::update_gradient(T *gradient, std::vector<T> &gradient_threads,
                                  std::vector<T> &alpha_diff, da_int &nrow, da_int &ncol,
                                  const T *kernel_data, da_int stride) {
    update_gradient_impl(
        gradient, gradient_threads, alpha_diff, nrow, ncol,
        [kernel_data, stride](da_int i) { return kernel_data + i * stride; });
}

// This function calculates highest power of 2, that is smaller or equal to n (number of rows in data)
template <typename T> da_int base_svm<T>::maxpowtwo(da_int &n) {
    da_int power = 1;
    while (power * 2 <= n) {
        power *= 2;
    }
    return power;
};

// Select i-th index for the local smo working set selection
// We pick argmin of gradient such that it is in I_up set
template <typename T>
void base_svm<T>::wssi(std::vector<da_int> &I_up, std::vector<T> &gradient, da_int &i,
                       T &min_grad) {
    // Start with very large value to find minimum and its index
    T min_grad_value = std::numeric_limits<T>::max();
    da_int min_grad_idx = -1;
    switch (wssi_vec_type) {
    case vectorization_type::avx:
        wssi_kernel<T, avx>(I_up.data(), gradient.data(), min_grad_idx, min_grad_value,
                            ws_size);
        break;
    case vectorization_type::avx2:
        wssi_kernel<T, avx2>(I_up.data(), gradient.data(), min_grad_idx, min_grad_value,
                             ws_size);
        break;
    case vectorization_type::avx512:
#ifdef __AVX512F__
        wssi_kernel<T, avx512>(I_up.data(), gradient.data(), min_grad_idx, min_grad_value,
                               ws_size);
#else
        wssi_kernel<T, avx2>(I_up.data(), gradient.data(), min_grad_idx, min_grad_value,
                             ws_size);
#endif
        break;
    default:
        wssi_kernel<T, scalar>(I_up.data(), gradient.data(), min_grad_idx, min_grad_value,
                               ws_size);
        break;
    }
    i = min_grad_idx;
    min_grad = min_grad_value;
};

// Select j-th index for the local smo working set selection
// We pick argmax of (b^2)/a such that it is in I_low set, while at the same time tracking maximum gradient value in I_low set
// for the local_smo convergence test
template <typename T>
void base_svm<T>::wssj(std::vector<da_int> &I_low, std::vector<T> &gradient, da_int &i,
                       T &min_grad, da_int &j, T &max_grad,
                       std::vector<T> &kernel_matrix_row_major,
                       std::vector<T> &kernel_matrix_diagonal, T &delta, T &max_fun) {
    // Start with very large negative value to find maximum and its index
    const T lowest = std::numeric_limits<T>::lowest();
    max_grad = lowest;
    max_fun = lowest;
    j = -1;
    delta = 0;

    if (i == -1)
        return;

    T *K_ith_row = &kernel_matrix_row_major[i * ws_size];
    T K_ii = K_ith_row[i];

    switch (wssj_vec_type) {
    case vectorization_type::avx:
        wssj_kernel<T, avx>(I_low.data(), gradient.data(), K_ith_row,
                            kernel_matrix_diagonal.data(), K_ii, j, max_grad, min_grad,
                            max_fun, delta, tau, ws_size);
        break;
    case vectorization_type::avx2:
        wssj_kernel<T, avx2>(I_low.data(), gradient.data(), K_ith_row,
                             kernel_matrix_diagonal.data(), K_ii, j, max_grad, min_grad,
                             max_fun, delta, tau, ws_size);
        break;
    case vectorization_type::avx512:
#ifdef __AVX512F__
        wssj_kernel<T, avx512>(I_low.data(), gradient.data(), K_ith_row,
                               kernel_matrix_diagonal.data(), K_ii, j, max_grad, min_grad,
                               max_fun, delta, tau, ws_size);
#else
        wssj_kernel<T, avx2>(I_low.data(), gradient.data(), K_ith_row,
                             kernel_matrix_diagonal.data(), K_ii, j, max_grad, min_grad,
                             max_fun, delta, tau, ws_size);
#endif
        break;
    default:
        wssj_kernel<T, scalar>(I_low.data(), gradient.data(), K_ith_row,
                               kernel_matrix_diagonal.data(), K_ii, j, max_grad, min_grad,
                               max_fun, delta, tau, ws_size);
        break;
    }
};

template <typename T>
void base_svm<T>::prepare_kernel_matrix(std::vector<T *> &ptr_kernel_col, da_int ws_size,
                                        std::vector<T> &local_kernel_matrix_row_major,
                                        std::vector<T> &kernel_diagonal,
                                        std::vector<da_int> &real_indices) {
    // Based on experiments, using more threads does not improve performance
    [[maybe_unused]] da_int n_threads = std::min(omp_get_max_threads(), 64);
#pragma omp parallel for schedule(dynamic) default(none) num_threads(n_threads)          \
    shared(local_kernel_matrix_row_major, kernel_diagonal, ptr_kernel_col, real_indices, \
               ws_size)
    // Copy relevant kernel values once, populate diagonal, and materialise the transpose
    for (da_int j = 0; j < ws_size; j++) {
        T *const row_major_row = &local_kernel_matrix_row_major[j * ws_size];
        const T *const kernel_col_j = ptr_kernel_col[j];
        for (da_int iter = 0; iter < ws_size; iter++) {
            const da_int real_idx = real_indices[iter];
            const T value = kernel_col_j[real_idx];
            row_major_row[iter] = value;
        }
        kernel_diagonal[j] = row_major_row[j];
    }
};

template <typename T> da_status base_svm<T>::deserialize_kernel_f() {
    switch (kernel_function) {
    case rbf:
        kernel_f = &rbf_wrapper<T>;
        break;
    case linear:
        kernel_f = &linear_wrapper<T>;
        break;
    case polynomial:
        kernel_f = &polynomial_wrapper<T>;
        break;
    case sigmoid:
        kernel_f = &sigmoid_wrapper<T>;
        break;
    default:
        return da_error(err, da_status_invalid_file_data,
                        "Unknown kernel function during deserialization.");
    }
    return da_status_success;
}

template <typename T> da_status base_svm<T>::serialize(serialization_buffer &buffer) {

    da_status status = da_status_success;
    auto io_dispatch = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = buffer.dispatch_buffer_io(data);
        return;
    };

    io_dispatch(this->pos_class);
    io_dispatch(this->neg_class);
    io_dispatch(this->kernel_function);
    io_dispatch(this->gamma);
    io_dispatch(this->degree);
    io_dispatch(this->coef0);
    io_dispatch(this->mod);
    io_dispatch(this->n_support);
    io_dispatch(this->support_indexes);
    io_dispatch(this->n_support_per_class);
    io_dispatch(this->support_coefficients);
    io_dispatch(this->bias);
    io_dispatch(this->sv_matrix);

    if (status != da_status_success)
        return status;

    if (buffer.get_mode() == buffer_mode::deserialize) {
        status = this->deserialize_kernel_f();
    }

    return status;
};

template class base_svm<float>;
template class base_svm<double>;

} // namespace da_svm

template void rbf_wrapper<double>(da_order order, da_int m, da_int n, da_int k,
                                  const double *X, double *x_norm, da_int compute_X_norms,
                                  da_int ldx, const double *Y, double *y_norm,
                                  da_int compute_Y_norms, da_int ldy, double *D,
                                  da_int ldd, double gamma, da_int /*degree*/,
                                  double /*coef0*/, bool X_is_Y, da_int vectorisation);
template void rbf_wrapper<float>(da_order order, da_int m, da_int n, da_int k,
                                 const float *X, float *x_norm, da_int compute_X_norms,
                                 da_int ldx, const float *Y, float *y_norm,
                                 da_int compute_Y_norms, da_int ldy, float *D, da_int ldd,
                                 float gamma, da_int /*degree*/, float /*coef0*/,
                                 bool X_is_Y, da_int vectorisation);
template void linear_wrapper<double>(
    da_order order, da_int m, da_int n, da_int k, const double *X, double * /*x_norm*/,
    da_int /*compute_X_norms*/, da_int ldx, const double *Y, double * /*y_norm*/,
    da_int /*compute_Y_norms*/, da_int ldy, double *D, da_int ldd, double /*gamma*/,
    da_int /*degree*/, double /*coef0*/, bool X_is_Y, da_int /*vectorisation*/);
template void linear_wrapper<float>(
    da_order order, da_int m, da_int n, da_int k, const float *X, float * /*x_norm*/,
    da_int /*compute_X_norms*/, da_int ldx, const float *Y, float * /*y_norm*/,
    da_int /*compute_Y_norms*/, da_int ldy, float *D, da_int ldd, float /*gamma*/,
    da_int /*degree*/, float /*coef0*/, bool X_is_Y, da_int /*vectorisation*/);
template void sigmoid_wrapper<double>(
    da_order order, da_int m, da_int n, da_int k, const double *X, double * /*x_norm*/,
    da_int /*compute_X_norms*/, da_int ldx, const double *Y, double * /*y_norm*/,
    da_int /*compute_Y_norms*/, da_int ldy, double *D, da_int ldd, double gamma,
    da_int /*degree*/, double coef0, bool X_is_Y, da_int /*vectorisation*/);
template void sigmoid_wrapper<float>(da_order order, da_int m, da_int n, da_int k,
                                     const float *X, float * /*x_norm*/,
                                     da_int /*compute_X_norms*/, da_int ldx,
                                     const float *Y, float * /*y_norm*/,
                                     da_int /*compute_Y_norms*/, da_int ldy, float *D,
                                     da_int ldd, float gamma, da_int /*degree*/,
                                     float coef0, bool X_is_Y, da_int /*vectorisation*/);
template void polynomial_wrapper<double>(
    da_order order, da_int m, da_int n, da_int k, const double *X, double * /*x_norm*/,
    da_int /*compute_X_norms*/, da_int ldx, const double *Y, double * /*y_norm*/,
    da_int /*compute_Y_norms*/, da_int ldy, double *D, da_int ldd, double gamma,
    da_int degree, double coef0, bool X_is_Y, da_int /*vectorisation*/);
template void
polynomial_wrapper<float>(da_order order, da_int m, da_int n, da_int k, const float *X,
                          float * /*x_norm*/, da_int /*compute_X_norms*/, da_int ldx,
                          const float *Y, float * /*y_norm*/, da_int /*compute_Y_norms*/,
                          da_int ldy, float *D, da_int ldd, float gamma, da_int degree,
                          float coef0, bool X_is_Y, da_int /*vectorisation*/);
} // namespace ARCH