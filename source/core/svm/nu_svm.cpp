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

// Deal with some Windows compilation issues regarding max/min macros
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_std.hpp"
#include "macros.h"
#include "svm.hpp"
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
template <typename T> bool is_upper_pos(T &alpha, const T &y, T &C) {
    return (alpha < C && y > 0);
};

// This function returns whether observation is in I_up set and is a negative class
template <typename T> bool is_upper_neg(T &alpha, const T &y) {
    return (alpha > 0 && y < 0);
};

// This function returns whether observation is in I_low set and is a positive class
template <typename T> bool is_lower_pos(T &alpha, const T &y) {
    return (alpha > 0 && y > 0);
};

// This function returns whether observation is in I_low set and is a negative class
template <typename T> bool is_lower_neg(T &alpha, const T &y, T &C) {
    return (alpha < C && y < 0);
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
    std::stable_sort(
        this->index_aux.begin(), this->index_aux.end(),
        [&](size_t i, size_t j) { return this->gradient[i] < this->gradient[j]; });
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
                         std::vector<T> &kernel_matrix,
                         std::vector<T> &local_kernel_matrix, std::vector<T> &alpha,
                         std::vector<T> &local_alpha, std::vector<T> &gradient,
                         std::vector<T> &local_gradient, std::vector<T> &response,
                         std::vector<T> &local_response, std::vector<bool> &I_low_p,
                         std::vector<bool> &I_up_p, std::vector<bool> &I_low_n,
                         std::vector<bool> &I_up_n, T &first_diff,
                         std::vector<T> &alpha_diff, std::optional<T> tol) {
    for (da_int iter = 0; iter < ws_size; iter++) {
        local_alpha[iter] = alpha[idx[iter]];
        local_gradient[iter] = gradient[idx[iter]];
        local_response[iter] = response[idx[iter]];
        I_low_p[iter] = is_lower_pos(local_alpha[iter], local_response[iter]);
        I_up_p[iter] = is_upper_pos(local_alpha[iter], local_response[iter], this->C);
        I_low_n[iter] = is_lower_neg(local_alpha[iter], local_response[iter], this->C);
        I_up_n[iter] = is_upper_neg(local_alpha[iter], local_response[iter]);
        // This can benefit from kernel matrix being stored in row-major
        for (da_int j = 0; j < ws_size; j++) {
            local_kernel_matrix[j * ws_size + iter] =
                kernel_matrix[j * this->n + (idx[iter] % this->n)];
        }
    }
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
                   local_kernel_matrix, delta_p, max_fun_p);
        this->wssj(I_low_n, local_gradient, i_n, min_grad_n, j_n, max_grad_n,
                   local_kernel_matrix, delta_n, max_fun_n);
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
                                        std::vector<T> &gradient) {
    da_int block_size = std::min(counter, SVM_MAX_BLOCK_SIZE);
    da_int n_blocks = counter / block_size, residual = counter % block_size;
    std::vector<T> current_alpha_diff;
    std::vector<da_int> current_idx;
    try {
        current_idx.resize(block_size);
        current_alpha_diff.resize(block_size);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    for (da_int i = 0; i <= n_blocks; i++) {
        da_int current_block_size = (i < n_blocks) ? block_size : residual;
        if (current_block_size == 0) {
            continue;
        }

        std::vector<T> kernel_matrix, X_temp;
        try {
            kernel_matrix.resize(this->n * current_block_size);
            X_temp.resize(current_block_size * this->p);
            if (current_block_size > this->ws_size) {
                this->y_norm_aux.resize(current_block_size);
            }
        } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }
        if (i < n_blocks) { // we are in block
            current_idx.assign(this->index_aux.begin() + i * block_size,
                               this->index_aux.begin() + (i + 1) * block_size);
            current_alpha_diff.assign(alpha_diff.begin() + i * block_size,
                                      alpha_diff.begin() + (i + 1) * block_size);
        } else { // we are in residual
            current_idx.assign(this->index_aux.begin() + n_blocks * block_size,
                               this->index_aux.end());
            current_alpha_diff.assign(alpha_diff.begin() + n_blocks * block_size,
                                      alpha_diff.end());
        }

        this->kernel_compute(current_idx, current_block_size, X_temp, kernel_matrix);
        this->update_gradient(gradient, current_alpha_diff, this->n, current_block_size,
                              kernel_matrix);
        if (this->mod == da_svm_model::nusvr) {
            // alpha_diff is just of size n (when technically it should be 2n) but since
            // second half of alpha_diff is just first half negated, we can just multiply by -1
            // and call update_gradient again with new values
            std::for_each(current_alpha_diff.begin(), current_alpha_diff.end(),
                          [](T &value) { value = -value; });
            this->update_gradient(gradient, current_alpha_diff, this->n,
                                  current_block_size, kernel_matrix);
        }
    }
    return da_status_success;
}

// Alphas here non-trivial, gradient needs to be calculated based on alphas
template <typename T>
da_status nusvc<T>::initialisation(da_int &size, std::vector<T> &gradient,
                                   std::vector<T> &response, std::vector<T> &alpha) {
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
    da_status status = this->initialise_gradient(alpha_diff, counter, gradient);
    return status;
}

template <typename T>
da_status nusvr<T>::initialisation(da_int &size, std::vector<T> &gradient,
                                   std::vector<T> &response, std::vector<T> &alpha) {
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
    da_status status = this->initialise_gradient(alpha_diff, counter, gradient);
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

template bool is_upper_pos<double>(double &alpha, const double &y, double &C);
template bool is_upper_pos<float>(float &alpha, const float &y, float &C);
template bool is_lower_pos<double>(double &alpha, const double &y);
template bool is_lower_pos<float>(float &alpha, const float &y);
template bool is_upper_neg<double>(double &alpha, const double &y);
template bool is_upper_neg<float>(float &alpha, const float &y);
template bool is_lower_neg<double>(double &alpha, const double &y, double &C);
template bool is_lower_neg<float>(float &alpha, const float &y, float &C);

} // namespace ARCH