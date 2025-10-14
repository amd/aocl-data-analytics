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
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "kernel_functions.hpp"
#include "macros.h"
#include "svm.hpp"
#include <boost/sort/spreadsort/float_sort.hpp>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace ARCH {

// This function returns whether observation is in I_up set and is a positive class
template <typename T> inline da_int is_upper_pos(const T &alpha, const T &y, const T &C) {
    return (alpha < C && y > 0) ? -1 : 0;
};

// This function returns whether observation is in I_up set and is a negative class
template <typename T> inline da_int is_upper_neg(const T &alpha, const T &y) {
    return (alpha > 0 && y < 0) ? -1 : 0;
};

// This function returns whether observation is in I_low set and is a positive class
template <typename T> inline da_int is_lower_pos(const T &alpha, const T &y) {
    return (alpha > 0 && y > 0) ? -1 : 0;
};

// This function returns whether observation is in I_low set and is a negative class
template <typename T> inline da_int is_lower_neg(const T &alpha, const T &y, const T &C) {
    return (alpha < C && y < 0) ? -1 : 0;
};

/*
 * Nu-SVM handle class that contains members that
 * are common for all SVM models that are Nu problem.
 *
 * This handle is inherited by nuSVC and nuSVR.
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

template <typename T>
nusvm<T>::nusvm(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train)
    : base_svm<T>(XUSR, yusr, n, p, ldx_train){};
template <typename T> nusvm<T>::~nusvm(){};

template <typename T>
nusvc<T>::nusvc(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train)
    : nusvm<T>(XUSR, yusr, n, p, ldx_train) {
    this->mod = da_svm_model::nusvc;
};
template <typename T> nusvc<T>::~nusvc(){};

template <typename T>
nusvr<T>::nusvr(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train)
    : nusvm<T>(XUSR, yusr, n, p, ldx_train) {
    this->mod = da_svm_model::nusvr;
};
template <typename T> nusvr<T>::~nusvr(){};

template <typename T>
void nusvm<T>::outer_wss(da_int &size, std::vector<da_int> &selected_ws_idx,
                         std::vector<bool> &selected_ws_indicator, da_int &n_selected) {
    da_int pos_left_p = 0, pos_right_p = size - 1;
    da_int pos_left_n = 0, pos_right_n = size - 1;
    da_int current_index;
    // Fill index_aux with numbers from 0, 1, ..., n
    da_std::iota(this->index_aux.begin(), this->index_aux.end(), 0);
    // Perform argsort
    auto rightshift = [this](const da_int &idx, const unsigned offset) {
        using sort_type =
            std::conditional_t<std::is_same<T, double>::value, int64_t, int32_t>;
        return boost::sort::spreadsort::float_mem_cast<T, sort_type>(
                   this->gradient[idx]) >>
               offset;
    };
    boost::sort::spreadsort::float_sort(this->index_aux.begin(), this->index_aux.end(),
                                        rightshift, [&](da_int &i, da_int &j) {
                                            // Compare the gradient values at the indices i and j
                                            return this->gradient[i] < this->gradient[j];
                                        });
    // Here index_aux is where we get indexes from, it contains argsorted gradient array
    // Select first ws_size/4 indices that are in I_up and are positive
    // Select first ws_size/4 indices that are in I_up and are negative
    // Select last ws_size/4 indices that are in I_low and are positive
    // Select last ws_size/4 indices that are in I_low and are negative
    // We start at far left and far right positions and iteratively shift our position more and more to
    // the other direction. We do this in a way that selected_ws_idx contains interleaved indexes from the left and right.
    // Second condition (potentially not) necessary because of risk of infinite loop
    while (n_selected < this->ws_size && (pos_right_p >= 0 || pos_left_p < size) &&
           (pos_right_n >= 0 || pos_left_n < size)) {
        if (pos_left_p < size) {
            current_index = this->index_aux[pos_left_p];
            // Skip to the next situation where our conditions are fulfilled. I.e, it is not in the working set already
            // and is in I_up set and is positive.
            while (selected_ws_indicator[current_index] == true ||
                   !is_upper_pos(this->alpha[current_index],
                                 this->response[current_index], this->C)) {
                pos_left_p++;
                if (pos_left_p == size)
                    break;
                current_index = this->index_aux[pos_left_p];
            }
            // When above loop stops, then `current_index` has next index that we want to include in the working set
            if (pos_left_p < size) {
                selected_ws_idx[n_selected++] = current_index;
                selected_ws_indicator[current_index] = true;
            }
        }
        if (n_selected >= this->ws_size)
            break;
        if (pos_left_n < size) {
            current_index = this->index_aux[pos_left_n];
            // Skip to the next situation where our conditions are fulfilled. I.e, it is not in the working set already
            // and is in I_up set and is negative.
            while (selected_ws_indicator[current_index] == true ||
                   !is_upper_neg(this->alpha[current_index],
                                 this->response[current_index])) {
                pos_left_n++;
                if (pos_left_n == size)
                    break;
                current_index = this->index_aux[pos_left_n];
            }
            // When above loop stops, then `current_index` has next index that we want to include in the working set
            if (pos_left_n < size) {
                selected_ws_idx[n_selected++] = current_index;
                selected_ws_indicator[current_index] = true;
            }
        }
        if (n_selected >= this->ws_size)
            break;
        if (pos_right_p >= 0) {
            current_index = this->index_aux[pos_right_p];
            // Skip to the next situation where our conditions are fulfilled. I.e, it is not in working set already
            // and is in I_low set and is positive.
            while (selected_ws_indicator[current_index] == true ||
                   !is_lower_pos(this->alpha[current_index],
                                 this->response[current_index])) {
                pos_right_p--;
                if (pos_right_p == -1)
                    break;
                current_index = this->index_aux[pos_right_p];
            }
            // When above loop stops, then `current_index` has next index that we want to include in the working set
            if (pos_right_p >= 0) {
                selected_ws_idx[n_selected++] = current_index;
                selected_ws_indicator[current_index] = true;
            }
        }
        if (n_selected >= this->ws_size)
            break;
        if (pos_right_n >= 0) {
            current_index = this->index_aux[pos_right_n];
            // Skip to the next situation where our conditions are fulfilled. I.e, it is not in working set already
            // and is in I_low set and is negative.
            while (selected_ws_indicator[current_index] == true ||
                   !is_lower_neg(this->alpha[current_index],
                                 this->response[current_index], this->C)) {
                pos_right_n--;
                if (pos_right_n == -1)
                    break;
                current_index = this->index_aux[pos_right_n];
            }
            // When above loop stops, then `current_index` has next index that we want to include in the working set
            if (pos_right_n >= 0) {
                selected_ws_idx[n_selected++] = current_index;
                selected_ws_indicator[current_index] = true;
            }
        }
    }
}

template <typename T>
void nusvm<T>::local_smo(da_int &ws_size, std::vector<da_int> &idx,
                         std::vector<T *> &ptr_kernel_col,
                         std::vector<T> &local_kernel_matrix,
                         std::vector<T> &local_kernel_matrix_row_major,
                         std::vector<T> &kernel_diagonal,
                         std::vector<da_int> &real_indices, std::vector<T> &alpha,
                         std::vector<T> &local_alpha, std::vector<T> &gradient,
                         std::vector<T> &local_gradient, std::vector<T> &response,
                         std::vector<T> &local_response, std::vector<da_int> &I_low_p,
                         std::vector<da_int> &I_up_p, std::vector<da_int> &I_low_n,
                         std::vector<da_int> &I_up_n, T &first_diff,
                         std::vector<T> &alpha_diff, std::optional<T> tol) {
    // Grab the values of alpha, gradient and response that are in the working set, so that we operate on smaller arrays
    // First loop: Copy alpha, gradient, response, and compute flags
    for (da_int iter = 0; iter < ws_size; iter++) {
        const da_int idx_iter = idx[iter];
        local_alpha[iter] = alpha[idx_iter];
        local_gradient[iter] = gradient[idx_iter];
        local_response[iter] = response[idx_iter];
        I_low_p[iter] = is_lower_pos(local_alpha[iter], local_response[iter]);
        I_up_p[iter] = is_upper_pos(local_alpha[iter], local_response[iter], this->C);
        I_low_n[iter] = is_lower_neg(local_alpha[iter], local_response[iter], this->C);
        I_up_n[iter] = is_upper_neg(local_alpha[iter], local_response[iter]);
        real_indices[iter] = idx[iter] % this->n;
    }

    this->prepare_kernel_matrix(ptr_kernel_col, ws_size, local_kernel_matrix,
                                local_kernel_matrix_row_major, kernel_diagonal,
                                real_indices);

    // i, j - indexes for update in the current iteration of SMO, domain = (0, ws_size)
    da_int i, j, i_p, i_n, j_p, j_n;
    da_int max_iter_inner = ws_size * 100;
    T min_grad_p, min_grad_n, max_grad_p, max_grad_n, max_fun_p, max_fun_n, delta,
        delta_p, delta_n, diff, epsilon = 1;
    // alpha_?_diff - Update values of alpha (we will pick minimum between alpha_i_diff and alpha_j_diff)
    T alpha_i_diff, alpha_j_diff;
    // Custom epsilon functionality is purely for the internal testing reasons
    bool is_custom_epsilon = false;
    if (tol.has_value()) {
        epsilon = tol.value();
        is_custom_epsilon = true;
    }

    for (da_int iter = 0; iter < max_iter_inner; iter++) {
        this->wssi(I_up_p, local_gradient, i_p, min_grad_p);
        this->wssi(I_up_n, local_gradient, i_n, min_grad_n);
        this->wssj(I_low_p, local_gradient, i_p, min_grad_p, j_p, max_grad_p,
                   local_kernel_matrix_row_major, kernel_diagonal, delta_p, max_fun_p);
        this->wssj(I_low_n, local_gradient, i_n, min_grad_n, j_n, max_grad_n,
                   local_kernel_matrix_row_major, kernel_diagonal, delta_n, max_fun_n);
        diff = std::max(max_grad_p - min_grad_p, max_grad_n - min_grad_n);
        if (iter == 0 && !is_custom_epsilon) {
            first_diff = diff;
            epsilon = std::max(this->tol, T(0.1) * diff);
        }
        if (diff < epsilon)
            break;
        if (max_fun_p > max_fun_n) {
            i = i_p;
            j = j_p;
            delta = delta_p;
        } else {
            i = i_n;
            j = j_n;
            delta = delta_n;
        }
        if (i == -1 || j == -1)
            break;
        alpha_i_diff = local_response[i] > 0 ? this->C - local_alpha[i] : local_alpha[i];
        alpha_j_diff = std::min(
            local_response[j] > 0 ? local_alpha[j] : this->C - local_alpha[j], delta);
        delta = std::min(alpha_i_diff, alpha_j_diff);
        // Update alpha
        local_alpha[i] += delta * local_response[i];
        local_alpha[j] -= delta * local_response[j];

        // Update I_low and I_up
        I_low_p[i] = is_lower_pos(local_alpha[i], local_response[i]);
        I_up_p[i] = is_upper_pos(local_alpha[i], local_response[i], this->C);
        I_low_p[j] = is_lower_pos(local_alpha[j], local_response[j]);
        I_up_p[j] = is_upper_pos(local_alpha[j], local_response[j], this->C);
        I_low_n[i] = is_lower_neg(local_alpha[i], local_response[i], this->C);
        I_up_n[i] = is_upper_neg(local_alpha[i], local_response[i]);
        I_low_n[j] = is_lower_neg(local_alpha[j], local_response[j], this->C);
        I_up_n[j] = is_upper_neg(local_alpha[j], local_response[j]);
        // Update gradient (local_kernel_matrix is square at this point so row/column major does not matter here)
        // Formula: gradient[k] += delta * (Q_ki - Q_kj)
        // We need to obtain two columns from kernel matrix
        T *kernel_matrix_ith = local_kernel_matrix.data() + (i * ws_size);
        T *kernel_matrix_jth = local_kernel_matrix.data() + (j * ws_size);
#pragma omp simd
        for (da_int iter = 0; iter < ws_size; iter++) {
            local_gradient[iter] +=
                delta * (kernel_matrix_ith[iter] - kernel_matrix_jth[iter]);
        }
    }
    // Compute alpha difference between start of SMO and at the end (used in further gradient update)
    // and scatter calculated alphas into global array
    for (da_int iter = 0; iter < ws_size; iter++) {
        alpha_diff[iter] = (local_alpha[iter] - alpha[idx[iter]]) * local_response[iter];
        alpha[idx[iter]] = local_alpha[iter];
    }
}

template <typename T>
da_status nusvm<T>::set_bias(std::vector<T> &alpha, std::vector<T> &gradient,
                             std::vector<T> &response, da_int &size, T &bias) {
    T gradient_sum_p = 0, gradient_sum_n = 0;
    da_int n_free_p = 0, n_free_n = 0;
    T min_value_p = std::numeric_limits<T>::max();
    T min_value_n = min_value_p;
    T max_value_p = -min_value_p, max_value_n = -min_value_n;
    T bias_p, bias_n;
    for (da_int i = 0; i < size; i++) {
        if (alpha[i] > 0 && alpha[i] < this->C && response[i] > 0) {
            gradient_sum_p += gradient[i];
            n_free_p++;
        }
        if (alpha[i] > 0 && alpha[i] < this->C && response[i] < 0) {
            gradient_sum_n -= gradient[i];
            n_free_n++;
        }
        if (is_upper_pos(alpha[i], response[i], this->C))
            min_value_p = std::min(min_value_p, gradient[i]);
        if (is_lower_pos(alpha[i], response[i]))
            max_value_p = std::max(max_value_p, gradient[i]);
        if (is_upper_neg(alpha[i], response[i]))
            min_value_n = std::min(min_value_n, gradient[i]);
        if (is_lower_neg(alpha[i], response[i], this->C))
            max_value_n = std::max(max_value_n, gradient[i]);
    }
    // If no free vectors then set bias to the middle of the two values, otherwise average of gradients of free vectors
    bias_p = n_free_p == 0 ? (min_value_p + max_value_p) / 2 : gradient_sum_p / n_free_p;
    bias_n = n_free_n == 0 ? -(min_value_n + max_value_n) / 2 : gradient_sum_n / n_free_n;

    bias = (bias_n - bias_p) / 2;
    if (this->mod == da_svm_model_::nusvc) {
        T scale = (bias_p + bias_n) / 2;
        if (scale == 0)
            return da_error(this->err, da_status_numerical_difficulties,
                            "Cannot divide by zero in bias calculation.");
        for (da_int i = 0; i < size; i++)
            alpha[i] /= scale;
        bias /= scale;
    }
    return da_status_success;
}

template <typename T>
da_status nusvm<T>::initialise_gradient(std::vector<T> &alpha_diff, da_int counter,
                                        std::vector<T> &gradient,
                                        da_cache::LRUCache<T> &cache) {
    // Early-out to avoid division by zero and unnecessary work
    if (counter <= 0) {
        return da_status_success;
    }
    da_int gradient_size = gradient.size();
    da_int block_size = std::min(counter, SVM_MAX_BLOCK_SIZE);
    da_int n_blocks = counter / block_size, residual = counter % block_size;
    std::vector<T> kernel_matrix, X_temp, gradient_threads, current_alpha_diff,
        gradient_local;
    std::vector<T *> ptr_kernel_col;
    std::vector<da_int> current_idx;
    [[maybe_unused]] da_int n_threads = da_utils::get_n_threads_loop(n_blocks);
    n_threads = std::min(n_threads, da_int(64));
    // Create local arrays for each thread
    try {
        current_idx.resize(block_size);
        current_alpha_diff.resize(block_size);
        ptr_kernel_col.resize(block_size);
        gradient_local.resize(gradient_size * n_threads);
        kernel_matrix.resize(this->n * block_size);
        X_temp.resize(block_size * this->p);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    da_int n = this->n, p = this->p, ldx_2 = this->ldx_2;
    std::vector<da_int> index_aux = this->index_aux;
    T *X = this->X, *x_norm_aux = this->x_norm_aux.data();
    kernel_f_type<T> kernel_f = this->kernel_f;
    T coef0 = this->coef0, gamma = this->gamma;
    da_int degree = this->degree, threading_error = 0;
#pragma omp parallel for schedule(dynamic) num_threads(n_threads) default(none)          \
    firstprivate(current_idx, ptr_kernel_col, current_alpha_diff, kernel_matrix, X_temp, \
                     gradient_threads)                                                   \
    shared(n_blocks, block_size, residual, index_aux, cache, n, p, ldx_2, degree, coef0, \
               gamma, kernel_f, X, x_norm_aux, alpha_diff, gradient_local,               \
               gradient_size, threading_error)
    // Loop over blocks and residual
    for (da_int i = 0; i <= n_blocks; i++) {
        da_int current_block_size = (i < n_blocks) ? block_size : residual;
        if (current_block_size == 0) {
            continue;
        }

        if (i < n_blocks) { // we are in block
            current_idx.assign(index_aux.begin() + i * block_size,
                               index_aux.begin() + (i + 1) * block_size);
            current_alpha_diff.assign(alpha_diff.begin() + i * block_size,
                                      alpha_diff.begin() + (i + 1) * block_size);
        } else { // we are in residual
            current_idx.assign(index_aux.begin() + n_blocks * block_size,
                               index_aux.end());
            current_alpha_diff.assign(alpha_diff.begin() + n_blocks * block_size,
                                      alpha_diff.end());
        }

        // Get the relevant slices of original matrix (working set)
        // Note: it would be more efficient to operate on row-major order
        for (da_int j = 0; j < current_block_size; j++) {
            da_int current_idx2 = current_idx[j] % n;
            for (da_int k = 0; k < p; k++) {
                X_temp[j + k * current_block_size] = X[current_idx2 + k * ldx_2];
            }
        }

        // Call to appropriate kernel function. Note that only idx_to_compute_count columns of kernel_temp will be filled.
        vectorization_type vectorisation;
        da_kernel_functions::select_simd_size<T>(n, vectorisation);
        // Variables used in euclidean_distance interface, 1 means to use precomputed norms, 2 means to compute norms
        da_int compute_X_norms = 1;
        da_int compute_y_norms = 2;
        // Make y-norms buffer thread-local to avoid data races
        std::vector<T> y_norm_aux_local(current_block_size);
        kernel_f(column_major, n, current_block_size, p, X, x_norm_aux, compute_X_norms,
                 ldx_2, X_temp.data(), y_norm_aux_local.data(), compute_y_norms,
                 current_block_size, kernel_matrix.data(), n, gamma, degree, coef0, false,
                 (da_int)vectorisation);

        // Update cache with idx_computed_count new columns of kernel matrix, otherwise just fill result array
        if (cache.active_) {
            // Get pointers to first values of each column, to later pass into cache.put()
            std::vector<T *> ptr_kernel_col_temp(current_block_size);
            std::vector<da_int> idx_temp(current_block_size);
            try {
                ptr_kernel_col_temp.resize(current_block_size);
                idx_temp.resize(current_block_size);
            } catch (std::bad_alloc const &) {
#pragma omp atomic write
                threading_error = 1;
            }
            // Call put for each computed index, this will update list that tracks LRU columns in the cache
            for (da_int j = 0; j < current_block_size; j++) {
                idx_temp[j] = current_idx[j] % n;
                ptr_kernel_col_temp[j] = &kernel_matrix[j * n];
                // Fill result array
                ptr_kernel_col[j] = &kernel_matrix[j * n];
            }
#pragma omp critical
            cache.put(idx_temp, ptr_kernel_col_temp);
        } else {
            for (da_int j = 0; j < current_block_size; j++) {
                ptr_kernel_col[j] = &kernel_matrix[j * n];
            }
        }
        da_int t_id = omp_get_thread_num();
        this->update_gradient(&gradient_local[t_id * gradient_size], gradient_threads,
                              current_alpha_diff, n, current_block_size, ptr_kernel_col);
        if (this->mod == da_svm_model::nusvr) {
            // alpha_diff is just of size n (when technically it should be 2n) but since
            // second half of alpha_diff is just first half negated, we can just multiply by -1
            // and call update_gradient again with new values
            std::for_each(current_alpha_diff.begin(), current_alpha_diff.end(),
                          [](T &value) { value = -value; });
            this->update_gradient(&gradient_local[t_id * gradient_size], gradient_threads,
                                  current_alpha_diff, n, current_block_size,
                                  ptr_kernel_col);
        }
    }
    if (threading_error == 1)
        return da_error(this->err, da_status_memory_error, "Memory allocation failed.");

        // Merge thread-local contributions in fixed order
#pragma omp parallel for schedule(static) num_threads(n_threads) default(none)           \
    shared(gradient_local, gradient, n_threads, gradient_size)
    for (da_int i = 0; i < gradient_size; i++) {
        T sum = 0;
        for (da_int t = 0; t < n_threads; t++) {
            sum += gradient_local[t * gradient_size + i];
        }
        gradient[i] += sum;
    }
    return da_status_success;
}

// Alphas here non-trivial, gradient needs to be calculated based on alphas
template <typename T>
da_status nusvc<T>::initialisation(da_int &size, std::vector<T> &gradient,
                                   std::vector<T> &response, std::vector<T> &alpha,
                                   da_cache::LRUCache<T> &cache) {
    std::vector<T> alpha_diff;
    this->C = 1;
    try {
        alpha_diff.resize(size);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    // Initialise response and alpha
    for (da_int i = 0; i < size; i++) {
        response[i] = this->y[i] == 0 ? -1.0 : this->y[i];
    }
    T sum_pos = this->nu * this->n / 2;
    T sum_neg = sum_pos;
    for (da_int i = 0; i < size; i++) {
        if (response[i] > 0) {
            alpha[i] = std::min((T)1.0, sum_pos);
            sum_pos -= alpha[i];
        } else {
            alpha[i] = std::min((T)1.0, sum_neg);
            sum_neg -= alpha[i];
        }
    }
    // Compute gradient based on alpha
    da_int counter = 0;
    for (da_int i = 0; i < size; i++) {
        if (alpha[i] != (T)0.0) {
            this->index_aux[counter] = i;
            alpha_diff[counter] = alpha[i] * response[i];
            counter++;
        }
    }
    da_status status = this->initialise_gradient(alpha_diff, counter, gradient, cache);
    return status;
}

template <typename T>
da_status nusvr<T>::initialisation(da_int &size, std::vector<T> &gradient,
                                   std::vector<T> &response, std::vector<T> &alpha,
                                   da_cache::LRUCache<T> &cache) {
    std::vector<T> alpha_diff;
    T sum = this->C * this->nu * size / 2;
    try {
        alpha_diff.resize(size);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    // Initialise response and alpha
    for (da_int i = 0; i < size; i++) {
        gradient[i] = -this->y[i];
        gradient[i + size] = -this->y[i];
        response[i] = 1.0;
        response[i + size] = -1.0;
        alpha[i] = std::min(this->C, sum);
        alpha[i + size] = alpha[i];
        sum -= alpha[i];
    }
    // Compute gradient based on alpha
    da_int counter = 0;
    for (da_int i = 0; i < size; i++) {
        if (alpha[i] != (T)0.0) {
            this->index_aux[counter] = i;
            alpha_diff[counter] = alpha[i];
            counter++;
        }
    }
    da_status status = this->initialise_gradient(alpha_diff, counter, gradient, cache);
    return status;
}

// The same code as in SVC problem
template <typename T>
da_status nusvc<T>::set_sv(std::vector<T> &alpha, da_int &n_support) {
    n_support = 0;
    T epsilon = std::numeric_limits<T>::epsilon();
    for (da_int i = 0; i < this->n; i++) {
        // There could be a better way to find if alpha is different than 0
        // Possibly one that would look if it is within the tolerance around 0.
        if (std::abs(alpha[i]) > epsilon) {
            n_support++;
            alpha[i] *= this->response[i];
            // n_support_per_class will hold n_support of negative class at index 0, and positive at index 1
            if (this->response[i] < 0)
                this->n_support_per_class[0]++;
            else
                this->n_support_per_class[1]++;
        }
    }
    try {
        this->support_indexes.resize(n_support);
        this->support_indexes_neg.resize(this->n_support_per_class[0]);
        this->support_indexes_pos.resize(this->n_support_per_class[1]);
        this->support_coefficients.resize(n_support);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    da_int position = 0;
    if (!this->ismulticlass) {
        for (da_int i = 0; i < this->n; i++) {
            if (std::abs(alpha[i]) > epsilon) {
                this->support_indexes[position] = i;
                this->support_coefficients[position++] = alpha[i];
            }
        }
    } else {
        da_int position_pos = 0, position_neg = 0;
        for (da_int i = 0; i < this->n; i++) {
            if (std::abs(alpha[i]) > epsilon) {
                if (this->idx_is_positive[i]) {
                    this->support_indexes_pos[position_pos++] = i;
                } else {
                    this->support_indexes_neg[position_neg++] = i;
                }
                this->support_indexes[position] = i;
                this->support_coefficients[position++] = alpha[i];
            }
        }
    }
    return da_status_success;
}

// The same code as in SVR problem
template <typename T>
da_status nusvr<T>::set_sv(std::vector<T> &alpha, da_int &n_support) {
    n_support = 0;
    T epsilon = std::numeric_limits<T>::epsilon();
    for (da_int i = 0; i < this->n; i++) {
        alpha[i] = alpha[i] - alpha[i + this->n];
        // There could be a better way to find if alpha is different than 0
        // Possibly one that would look if it is within the tolerance around 0.
        if (std::abs(alpha[i]) > epsilon)
            n_support++;
    }
    try {
        this->support_indexes.resize(n_support);
        this->support_coefficients.resize(n_support);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    da_int position = 0;
    for (da_int i = 0; i < this->n; i++) {
        if (std::abs(alpha[i]) > epsilon) {
            this->support_indexes[position] = i;
            this->support_coefficients[position] = alpha[i];
            position++;
        }
    }
    return da_status_success;
}

template class nusvm<float>;
template class nusvm<double>;
template class nusvc<float>;
template class nusvc<double>;
template class nusvr<float>;
template class nusvr<double>;

} // namespace da_svm

template da_int is_upper_pos<double>(const double &alpha, const double &y,
                                     const double &C);
template da_int is_upper_pos<float>(const float &alpha, const float &y, const float &C);
template da_int is_lower_pos<double>(const double &alpha, const double &y);
template da_int is_lower_pos<float>(const float &alpha, const float &y);
template da_int is_upper_neg<double>(const double &alpha, const double &y);
template da_int is_upper_neg<float>(const float &alpha, const float &y);
template da_int is_lower_neg<double>(const double &alpha, const double &y,
                                     const double &C);
template da_int is_lower_neg<float>(const float &alpha, const float &y, const float &C);

} // namespace ARCH