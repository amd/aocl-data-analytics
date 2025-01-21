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
#include "macros.h"
#include "svm.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

namespace ARCH {

// This function returns whether observation is in I_up set
template <typename T> bool is_upper(T &alpha, const T &y, T &C) {
    return ((alpha < C && y > 0) || (alpha > 0 && y < 0));
};
// This function returns whether observation is in I_low set
template <typename T> bool is_lower(T &alpha, const T &y, T &C) {
    return ((alpha < C && y < 0) || (alpha > 0 && y > 0));
};

namespace da_svm {

using namespace da_svm_types;

template <typename T> csvm<T>::~csvm(){};

template <typename T> svc<T>::svc() { this->mod = da_svm_model::svc; };
template <typename T> svc<T>::~svc(){};

template <typename T> svr<T>::svr() { this->mod = da_svm_model::svr; };
template <typename T> svr<T>::~svr(){};

template <typename T>
void csvm<T>::outer_wss(da_int &size, std::vector<da_int> &selected_ws_idx,
                        std::vector<bool> &selected_ws_indicator, da_int &n_selected) {
    da_int pos_left = 0, pos_right = size - 1;
    da_int current_index;
    // Fill index_aux with numbers from 0, 1, ..., n
    std::iota(this->index_aux.begin(), this->index_aux.end(), 0);
    // Perform argsort
    std::stable_sort(
        this->index_aux.begin(), this->index_aux.end(),
        [&](size_t i, size_t j) { return this->gradient[i] < this->gradient[j]; });
    // Here index_aux is where we get indexes from, it contains argsorted gradient array
    // Select first ws_size/2 indices that are in I_up
    // Select last ws_size/2 indices that are in I_low
    // We start at far left and far right positions and iteratively shift our position more and more to
    // the other direction. We do this in a way that selected_ws_idx contains interleaved indexes from the left and right.
    // Second condition necessary because of risk of infinite loop
    while (n_selected < this->ws_size && (pos_right >= 0 || pos_left < size)) {
        if (pos_left < size) {
            current_index = this->index_aux[pos_left];
            // Skip to the next situation where our conditions are fulfilled. I.e, it is not in the working set already
            // and is in I_up set.
            while (selected_ws_indicator[current_index] == true ||
                   !is_upper(this->alpha[current_index], this->response[current_index],
                             this->C)) {
                pos_left++;
                if (pos_left == size)
                    break;
                current_index = this->index_aux[pos_left];
            }
            // When above loop stops, then `current_index` has next index that we want to include in the working set
            if (pos_left < size) {
                selected_ws_idx[n_selected++] = current_index;
                selected_ws_indicator[current_index] = true;
            }
        }
        if (n_selected >= this->ws_size)
            break;
        if (pos_right >= 0) {
            current_index = this->index_aux[pos_right];
            // Skip to the next situation where our conditions are fulfilled. I.e, it is not in working set already
            // and is in I_low set.
            while (selected_ws_indicator[current_index] == true ||
                   !is_lower(this->alpha[current_index], this->response[current_index],
                             this->C)) {
                pos_right--;
                if (pos_right == -1)
                    break;
                current_index = this->index_aux[pos_right];
            }
            // When above loop stops, then `current_index` has next index that we want to include in the working set
            if (pos_right >= 0) {
                selected_ws_idx[n_selected++] = current_index;
                selected_ws_indicator[current_index] = true;
            }
        }
    }
}

template <typename T>
void csvm<T>::local_smo(da_int &ws_size, std::vector<da_int> &idx,
                        std::vector<T> &local_kernel_matrix, std::vector<T> &alpha,
                        std::vector<T> &local_alpha, std::vector<T> &gradient,
                        std::vector<T> &local_gradient, std::vector<T> &response,
                        std::vector<T> &local_response, std::vector<bool> &I_low_p,
                        std::vector<bool> &I_up_p,
                        [[maybe_unused]] std::vector<bool> &I_low_n,
                        [[maybe_unused]] std::vector<bool> &I_up_n, T &first_diff,
                        std::vector<T> &alpha_diff, std::optional<T> tol) {
    for (da_int iter = 0; iter < ws_size; iter++) {
        local_alpha[iter] = alpha[idx[iter]];
        local_gradient[iter] = gradient[idx[iter]];
        local_response[iter] = response[idx[iter]];
        I_low_p[iter] = is_lower(local_alpha[iter], local_response[iter], this->C);
        I_up_p[iter] = is_upper(local_alpha[iter], local_response[iter], this->C);
        // This can benefit from kernel matrix being stored in row-major
        for (da_int j = 0; j < ws_size; j++) {
            local_kernel_matrix[j * ws_size + iter] =
                this->kernel_matrix[j * this->n + (idx[iter] % this->n)];
        }
    }
    // i, j - indexes for update in the current iteration of SMO, domain = (0, ws_size)
    da_int i, j;
    da_int max_iter_inner = ws_size * 100;
    T min_grad, max_grad, max_fun, delta, diff, epsilon = 1;
    // alpha_x_diff - value explained in the libsvm paper, potential update value of alpha_x
    T alpha_i_diff, alpha_j_diff;
    bool is_custom_epsilon = false;
    if (tol.has_value()) {
        epsilon = tol.value();
        is_custom_epsilon = true;
    }
    da_int iter_smo = 0;
    for (; iter_smo < max_iter_inner; iter_smo++) {
        this->wssi(I_up_p, local_gradient, i, min_grad);
        this->wssj(I_low_p, local_gradient, i, min_grad, j, max_grad, local_kernel_matrix,
                   delta, max_fun);
        diff = max_grad - min_grad;
        if (iter_smo == 0 && !is_custom_epsilon) {
            first_diff = diff;
            epsilon = std::max(this->tol, T(0.1) * diff);
        }
        if (diff < epsilon)
            break;
        alpha_i_diff = local_response[i] > 0 ? this->C - local_alpha[i] : local_alpha[i];
        alpha_j_diff = std::min(
            local_response[j] > 0 ? local_alpha[j] : this->C - local_alpha[j], delta);
        delta = std::min(alpha_i_diff, alpha_j_diff);
        // Update alpha
        local_alpha[i] += delta * local_response[i];
        local_alpha[j] -= delta * local_response[j];
        // Update I_low and I_up
        I_low_p[i] = is_lower(local_alpha[i], local_response[i], this->C);
        I_up_p[i] = is_upper(local_alpha[i], local_response[i], this->C);
        I_low_p[j] = is_lower(local_alpha[j], local_response[j], this->C);
        I_up_p[j] = is_upper(local_alpha[j], local_response[j], this->C);
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
void csvm<T>::set_bias(std::vector<T> &alpha, std::vector<T> &gradient,
                       std::vector<T> &response, da_int &size, T &bias) {
    T gradient_sum = 0;
    da_int n_free = 0;
    T min_value = std::numeric_limits<T>::max();
    T max_value = -min_value;
    for (da_int i = 0; i < size; i++) {
        if (alpha[i] > 0 && alpha[i] < this->C) {
            gradient_sum += gradient[i];
            n_free++;
        }
        if (is_upper(alpha[i], response[i], this->C))
            min_value = std::min(min_value, gradient[i]);
        if (is_lower(alpha[i], response[i], this->C))
            max_value = std::max(max_value, gradient[i]);
    }
    // If no free vectors then set bias to the middle of the two values, otherwise average of gradients of free vectors
    bias = n_free == 0 ? -(min_value + max_value) / 2 : -gradient_sum / n_free;
}

template <typename T>
da_status svc<T>::initialisation(da_int &size, std::vector<T> &gradient,
                                 std::vector<T> &response, std::vector<T> &alpha) {
    for (da_int i = 0; i < size; i++) {
        gradient[i] = this->y[i] == 0 ? 1.0 : -this->y[i];
        response[i] = this->y[i] == 0 ? -1.0 : this->y[i];
        alpha[i] = 0;
    }
    return da_status_success;
}

template <typename T>
da_status svr<T>::initialisation(da_int &size, std::vector<T> &gradient,
                                 std::vector<T> &response, std::vector<T> &alpha) {
    for (da_int i = 0; i < size; i++) {
        gradient[i] = this->eps - this->y[i];
        gradient[i + size] = -this->eps - this->y[i];
        response[i] = 1.0;
        response[i + size] = -1.0;
        alpha[i] = 0;
        alpha[i + size] = 0;
    }
    return da_status_success;
}

// Find support vectors and their indexes
template <typename T> da_status svc<T>::set_sv(std::vector<T> &alpha, da_int &n_support) {
    n_support = 0;
    for (da_int i = 0; i < this->n; i++) {
        // There could be a better way to find if alpha is different than 0
        // Possibly one that would look if it is within the tolerance around 0.
        if (alpha[i] != 0) {
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
        this->support_vectors.resize(n_support * this->p);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    da_int position = 0;
    if (!this->ismulticlass) {
        for (da_int i = 0; i < this->n; i++) {
            if (alpha[i] != 0) {
                this->support_indexes[position] = i;
                this->support_coefficients[position++] = alpha[i];
            }
        }
    } else {
        da_int position_pos = 0, position_neg = 0;
        for (da_int i = 0; i < this->n; i++) {
            if (alpha[i] != 0) {
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
    // Construct a matrix consisting of only support vectors (can be optimised in row major)
    for (da_int i = 0; i < n_support; i++) {
        da_int current_idx = this->support_indexes[i];
        for (da_int j = 0; j < this->p; j++) {
            this->support_vectors[i + j * n_support] =
                this->X[current_idx + j * this->ldx_2];
        }
    }
    return da_status_success;
}

// Find support vectors and their indexes
template <typename T> da_status svr<T>::set_sv(std::vector<T> &alpha, da_int &n_support) {
    n_support = 0;
    for (da_int i = 0; i < this->n; i++) {
        alpha[i] = alpha[i] - alpha[i + this->n];
        // There could be a better way to find if alpha is different than 0
        // Possibly one that would look if it is within the tolerance around 0.
        if (alpha[i] != 0)
            n_support++;
    }
    try {
        this->support_indexes.resize(n_support);
        this->support_coefficients.resize(n_support);
        this->support_vectors.resize(n_support * this->p);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    da_int position = 0;
    for (da_int i = 0; i < this->n; i++) {
        if (alpha[i] != 0) {
            this->support_indexes[position] = i;
            this->support_coefficients[position] = alpha[i];
            position++;
        }
    }
    // Construct a matrix consisting of only support vectors (can be optimised in row major)
    for (da_int i = 0; i < n_support; i++) {
        da_int current_idx = this->support_indexes[i];
        for (da_int j = 0; j < this->p; j++) {
            this->support_vectors[i + j * n_support] =
                this->X[current_idx + j * this->ldx_2];
        }
    }
    return da_status_success;
}

template class csvm<float>;
template class csvm<double>;
template class svc<float>;
template class svc<double>;
template class svr<float>;
template class svr<double>;

} // namespace da_svm

template bool is_upper<double>(double &alpha, const double &y, double &C);
template bool is_upper<float>(float &alpha, const float &y, float &C);
template bool is_lower<double>(double &alpha, const double &y, double &C);
template bool is_lower<float>(float &alpha, const float &y, float &C);

} // namespace ARCH