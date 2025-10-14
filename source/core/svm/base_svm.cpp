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
#include "da_cache.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "kernel_functions.hpp"
#include "macros.h"
#include "miscellaneous.hpp"
#include "options.hpp"
#include "svm.hpp"
#include "svm_options.hpp"
#include "svm_types.hpp"
#include <climits>
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
    std::vector<T> kernel_matrix, local_kernel_matrix, local_kernel_matrix_row_major,
        kernel_diagonal;
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
            x_norm_aux.resize(n);
            y_norm_aux.resize(ws_size);
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
        local_kernel_matrix.resize(ws_size * ws_size);
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
        local_smo(ws_size, ws_indexes, ptr_kernel_col, local_kernel_matrix,
                  local_kernel_matrix_row_major, kernel_diagonal, real_indices, alpha,
                  local_alpha, gradient, local_gradient, response, local_response,
                  I_low_p, I_up_p, I_low_n, I_up_n, first_diff, alpha_diff, std::nullopt);
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

/* Calculate decision function */
template <typename T>
da_status base_svm<T>::decision_function(da_int nsamples, da_int nfeat, const T *X_test,
                                         da_int ldx_test, T *decision_values) {
    da_status status = da_status_success;
    // Initialise decision values to constant (bias)
    for (da_int i = 0; i < nsamples; i++)
        decision_values[i] = bias;
    // Stop early if there are no support vectors
    if (n_support == 0)
        return status;
    // Perform blocked decision function evaluation (n_support can be up to n_samples and
    // then kernel_matrix (n_support by n_samples) can be too large)
    std::vector<da_int> sv_idx(n_support);
    if (ismulticlass) {
        for (da_int i = 0; i < n_support; i++) {
            sv_idx[i] = idx_class[support_indexes[i]];
        }
    } else {
        sv_idx = support_indexes;
    }
    // Block size for kernel function
    da_int block_size = std::min(n_support, SVM_MAX_BLOCK_SIZE);
    da_int n_blocks = n_support / block_size, residual = n_support % block_size;
    std::vector<T> x_aux, y_aux, kernel_matrix, block_support_vectors;
    // Variable used in euclidean_distance interface, 2 means compute norms into aux arrays
    da_int compute_norms = 2;
    try {
        x_aux.resize(block_size);
        y_aux.resize(nsamples);
        kernel_matrix.resize(block_size * nsamples);
        block_support_vectors.resize(block_size * nfeat);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
#pragma omp parallel for schedule(dynamic) if (n_blocks > 1) default(none)               \
    firstprivate(x_aux, y_aux, kernel_matrix, block_support_vectors)                     \
    shared(n_blocks, residual, block_size, X_test, sv_idx, nfeat, nsamples, ldx_test,    \
               kernel_f, gamma, degree, coef0, compute_norms)                            \
    reduction(+ : decision_values[ : nsamples])
    for (da_int i = 0; i <= n_blocks; i++) {
        da_int current_block_size = (i < n_blocks) ? block_size : residual;
        if (current_block_size == 0) {
            continue;
        }
        da_int offset;
        if (i < n_blocks) // we are in block
            offset = i * block_size;
        else // we are in residual
            offset = n_blocks * block_size;

        // Get the relevant slices of support vectors
        for (da_int j = 0; j < current_block_size; j++) {
            da_int current_idx = sv_idx[offset + j];
            for (da_int k = 0; k < nfeat; k++) {
                block_support_vectors[j + k * current_block_size] =
                    XUSR[current_idx + k * ldx];
            }
        }

        // Get vectorisation
        vectorization_type vectorisation;
        da_kernel_functions::select_simd_size<T>(current_block_size, vectorisation);

        // Compute kernel matrix K between support vectors and test data
        kernel_f(column_major, current_block_size, nsamples, nfeat,
                 block_support_vectors.data(), x_aux.data(), compute_norms,
                 current_block_size, X_test, y_aux.data(), compute_norms, ldx_test,
                 kernel_matrix.data(), current_block_size, gamma, degree, coef0, false,
                 (da_int)vectorisation);

        // Compute decision_values = K'*alpha + bias
        da_blas::cblas_gemv(CblasColMajor, CblasTrans, current_block_size, nsamples,
                            (T)1.0, kernel_matrix.data(), current_block_size,
                            support_coefficients.data() + offset, 1, (T)1.0,
                            decision_values, 1);
    }
    return status;
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
                                 std::vector<T> &X_temp, std::vector<T> &kernel_temp,
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
            // Get pointers to first values of each column, to later pass into cache.put()
            std::vector<T *> ptr_kernel_col_temp(idx_to_compute_count);
            std::vector<da_int> idx_temp(idx_to_compute_count);
            // Call put for each computed index, this will update list that tracks LRU columns in the cache
            for (da_int i = 0; i < idx_to_compute_count; i++) {
                da_int current_idx = idx_to_compute[i];
                idx_temp[i] = idx[current_idx] % n;
                ptr_kernel_col_temp[i] = &kernel_temp[i * n];
                // Fill result array
                ptr_kernel_col[current_idx] = &kernel_temp[i * n];
            }
            cache.put(idx_temp, ptr_kernel_col_temp);
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
template <typename T>
void base_svm<T>::update_gradient(T *gradient, std::vector<T> &gradient_threads,
                                  std::vector<T> &alpha_diff, da_int &nrow, da_int &ncol,
                                  std::vector<T *> &ptr_kernel_col) {
    std::vector<T> gradient_temp(nrow, 0);
    da_int n_threads = da_utils::get_n_threads_loop(32);
    gradient_threads.resize(nrow * n_threads);

#pragma omp parallel shared(alpha_diff, ptr_kernel_col, nrow, ncol,                      \
                                gradient_threads) default(none) num_threads(n_threads)
    {
        da_int thread_id = omp_get_thread_num();
        T *local_grad = &gradient_threads[thread_id * nrow];
#pragma omp for schedule(static)
        for (da_int i = 0; i < ncol; i++) {
            const T *const_kernel = ptr_kernel_col[i];
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
};

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
                                        std::vector<T> &local_kernel_matrix,
                                        std::vector<T> &local_kernel_matrix_row_major,
                                        std::vector<T> &kernel_diagonal,
                                        std::vector<da_int> &real_indices) {
    // Based on experiments, using more threads does not improve performance
    [[maybe_unused]] da_int n_threads = std::min(omp_get_max_threads(), 64);
#pragma omp parallel for default(none) num_threads(n_threads)                            \
    shared(local_kernel_matrix, ptr_kernel_col, real_indices, ws_size)
    // Second loop: Copy only relevant kernel matrix values to local_kernel_matrix
    // Most efficient storage: local_kernel_matrix - square matrix of size ws_size (column major)
    //                         kernel_matrix - original kernel matrix of size n by ws_size (column major)
    for (da_int j = 0; j < ws_size; j++) {
        const da_int local_kernel_offset = j * ws_size;
        T *kernel_col_j = ptr_kernel_col[j];
#pragma omp simd
        for (da_int iter = 0; iter < ws_size; iter++) {
            local_kernel_matrix[local_kernel_offset + iter] =
                *(kernel_col_j + real_indices[iter]);
        }
    }

// Third loop: Transpose local kernel matrix to local_kernel_matrix_row_major and get diagonal values
#pragma omp simd collapse(2)
    for (da_int i = 0; i < ws_size; i++) {
        for (da_int j = 0; j < ws_size; j++) {
            local_kernel_matrix_row_major[j * ws_size + i] =
                local_kernel_matrix[i * ws_size + j];
        }
    }
    for (da_int i = 0; i < ws_size; i++) {
        kernel_diagonal[i] = local_kernel_matrix[i * ws_size + i];
    }
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