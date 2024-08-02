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

#ifndef BASESVM_HPP
#define BASESVM_HPP

// Deal with some Windows compilation issues regarding max/min macros
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "kernel_function_local.hpp"
#include "moment_statistics.hpp"
#include "options.hpp"
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

/*
 * Base SVM handle class that contains members that
 * are common for all SVM models.
 *
 * This handle is inherited by all specialized svm handles.
 * 
 * The inheritance scheme is as follows:
 * 
 *                          SVM
 *                         /   \
 *                        /     \ 
 *                   C-SVM       Nu-SVM
 *                  /     \      /     \
 *                 /       \    /       \
 *              SVC       SVR Nu-SVC   Nu-SVR
 */
namespace da_svm {
// This forward declaration is here to allow for "friending" it with base_svm few lines below
template <typename T> class svm;

template <typename T> class base_svm {
  public:
    // n x p (samples x features)
    da_int n = 0;
    da_int p = 0;
    // User's data
    const T *XUSR;
    const T *yusr;
    // Data that we operate on (in SVR / binary classification it is same as user's data)
    T *X = nullptr;
    // Second ldx is here for case when we have multiclass, X is created internally dense, so we want to proceed with ldx_2 = nsamples
    // However in other cases we use user's pointer where we want ldx_2 = ldx
    da_int ldx, ldx_2;
    T *y = nullptr;
    // actual_size = 2n if (SVR or nuSVR), otherwise n
    da_int actual_size;

    // Used in multi-class classification (memory allocated in svm.hpp set_data())
    std::vector<da_int> idx_class; // indexes of observations where class is either i or j
    std::vector<da_int> support_indexes_pos, support_indexes_neg;
    std::vector<bool> idx_is_positive;
    bool ismulticlass = false;
    da_int pos_class, neg_class;

    // Kernel function to use for computation
    da_int kernel_function = rbf;
    // Kernel specific parameters
    T gamma = 1.0;
    da_int degree = 3;
    T coef0 = 0.0;
    // Regularisation parameters
    T C = 1, eps = 0.1, nu = 0.5, tau = 1e-6;
    // Convergence tolerance
    T tol = 1.0e-3;
    da_int max_iter;
    da_int iter;

    // Variable is being set at the contructors of spiecialised classes
    da_svm_model mod = svm_undefined;

    // Pointer to error trace
    da_errors::da_error_t *err = nullptr;

    // Variables for result handling
    std::vector<T> gradient, alpha, response;
    da_int n_support;                    // Number of support vectors
    std::vector<da_int> support_indexes, // Indexes of support vectors
        n_support_per_class;
    std::vector<T> support_coefficients; // Alphas of support vectors
    std::vector<T>
        support_vectors; // Observations that are support vectors (subset of X rows)
    T bias;              // Constant in decision function

    // Internal working variables
    std::vector<T> alpha_diff;
    std::vector<T> kernel_matrix;
    da_int ws_size;        // Size of working set
    std::vector<T> X_temp; // Contain relevant slices of X for kernel computation
    std::vector<T> local_alpha, local_gradient, local_response, local_kernel_matrix;
    std::vector<T> x_norm_aux, y_norm_aux; // Work array for kernel computation
    std::vector<bool> I_low_p, I_up_p, I_low_n, I_up_n;

    // ws_indexes is array with indexes of outer working set
    // index_aux is array with argsorted gradient array
    std::vector<da_int> ws_indexes, index_aux;
    std::vector<bool> ws_indicator;

  public:
    // This friend is here to allow following "svm" class to set protected members of "base_svm" class, such as kernel_function, X or y.
    friend class svm<T>;
    base_svm(){};
    virtual ~base_svm(){}; // Virtual to remove warnings

    // Main functions
    da_status compute();
    da_status predict(da_int nsamples, da_int nfeat, const T *X_test, da_int ldx_test,
                      T *decision_values);
    da_status decision_function(da_int nsamples, da_int nfeat, const T *X_test,
                                da_int ldx_test, T *decision_values);

    // General auxiliary functions
    void update_gradient(std::vector<T> &gradient, std::vector<T> &alpha_diff,
                         da_int &nrow, da_int &ncol, std::vector<T> &kernel_matrix);
    void kernel_compute(std::vector<da_int> &idx, da_int &idx_size,
                        std::vector<T> &X_temp, std::vector<T> &kernel_matrix);
    void compute_ws_size(da_int &ws_size);
    da_int maxpowtwo(da_int &n);
    void wssi(std::vector<bool> &I_up, std::vector<T> &gradient, da_int &i, T &min_grad);
    void wssj(std::vector<bool> &I_low, std::vector<T> &gradient, da_int &i, T &min_grad,
              da_int &j, T &max_grad, std::vector<T> &kernel_matrix, T &delta,
              T &max_fun);

    // Functions that need specialisation
    virtual da_status initialisation(da_int &size, std::vector<T> &gradient,
                                     std::vector<T> &response, std::vector<T> &alpha) = 0;
    virtual void outer_wss(da_int &size, std::vector<da_int> &selected_ws_idx,
                           std::vector<bool> &selected_ws_indicator,
                           da_int &n_selected) = 0;
    virtual void local_smo(da_int &ws_size, std::vector<da_int> &idx,
                           std::vector<T> &local_kernel_matrix, std::vector<T> &alpha,
                           std::vector<T> &local_alpha, std::vector<T> &gradient,
                           std::vector<T> &local_gradient, std::vector<T> &response,
                           std::vector<T> &local_response, std::vector<bool> &I_low_p,
                           std::vector<bool> &I_up_p, std::vector<bool> &I_low_n,
                           std::vector<bool> &I_up_n, T &first_diff,
                           std::vector<T> &alpha_diff, std::optional<T> tol) = 0;
    virtual void set_bias(std::vector<T> &alpha, std::vector<T> &gradient,
                          std::vector<T> &response, da_int &size, T &bias) = 0;
    virtual da_status set_sv(std::vector<T> &alpha, da_int &n_support) = 0;
};

/* Main function which contains Thunder loop */
template <typename T> da_status base_svm<T>::compute() {
    da_status status = da_status_success;
    actual_size = mod == da_svm_model::svr || mod == da_svm_model::nusvr ? n * 2 : n;
    iter = 0;
    if (max_iter == -1)
        max_iter = INT_MAX;
    else if (max_iter == 0)
        return da_error(err, da_status_invalid_option, "max_iter must be positive or -1");
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
    compute_ws_size(ws_size);
    try {
        // Outer WSS
        ws_indicator.resize(actual_size);
        index_aux.resize(actual_size); // For nu problem also used in initialisation
        // Compute kernel
        ws_indexes.resize(ws_size);
        kernel_matrix.resize(ws_size * n);
        X_temp.resize(ws_size * p);
        x_norm_aux.resize(n);
        y_norm_aux.resize(ws_size);
        // Local SMO
        gradient.resize(actual_size);
        // This is because if compute() is called many times one after another, it causes problems in
        // nu variant because in nusvm::initialisation() gradient is not explicitly set to 0, but relies on 0 initialisation here
        // (can be modified if it's unusual design)
        std::fill(gradient.begin(), gradient.end(), T(0));
        response.resize(actual_size);
        alpha.resize(actual_size);
        local_alpha.resize(ws_size);
        local_gradient.resize(ws_size);
        local_response.resize(ws_size);
        local_kernel_matrix.resize(ws_size * ws_size);
        I_low_p.resize(ws_size);
        I_up_p.resize(ws_size);
        I_low_n.resize(ws_size);
        I_up_n.resize(ws_size);
        // Update gradient
        alpha_diff.resize(ws_size);
        // Result handling
        n_support_per_class.resize(2, 0);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    status = initialisation(n, gradient, response, alpha);
    if (status != da_status_success)
        return status;
    for (; iter < max_iter; iter++) {

        ////////// Outer WSS
        std::fill(ws_indicator.begin(), ws_indicator.end(), false);
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
        kernel_compute(ws_indexes, ws_size, X_temp, kernel_matrix);
        // Use kernel matrix to perform local SMO (as a result alpha, alpha_diff and first_diff are updated)
        local_smo(ws_size, ws_indexes, local_kernel_matrix, alpha, local_alpha, gradient,
                  local_gradient, response, local_response, I_low_p, I_up_p, I_low_n,
                  I_up_n, first_diff, alpha_diff, std::nullopt);
        // Global gradient update based on alpha_diff
        update_gradient(gradient, alpha_diff, n, ws_size, kernel_matrix);
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
    set_bias(alpha, gradient, response, actual_size, bias);
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
    std::vector<T> decision_values(nsamples);
    status = decision_function(nsamples, nfeat, X_test, ldx_test, decision_values.data());
    if (mod == da_svm_model::svc || mod == da_svm_model::nusvc) {
        for (da_int i = 0; i < nsamples; i++) {
            predictions[i] = decision_values[i] > 0 ? 1 : 0;
        }
    } else {
        for (da_int i = 0; i < nsamples; i++) {
            predictions[i] = decision_values[i];
        }
    }
    return status;
}

/* Calculate decision function */
template <typename T>
da_status base_svm<T>::decision_function(da_int nsamples, da_int nfeat, const T *X_test,
                                         da_int ldx_test, T *decision_values) {
    da_status status = da_status_success;
    std::vector<T> x_aux, y_aux, kernel_matrix;
    try {
        x_aux.resize(n_support);
        y_aux.resize(nsamples);
        kernel_matrix.resize(n_support * nsamples);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    // Initialise decision values to constant (bias)
    for (da_int i = 0; i < nsamples; i++)
        decision_values[i] = bias;
    // Compute kernel matrix K between support vectors and test data
    switch (kernel_function) {
    case rbf:
        rbf_kernel_local(column_major, n_support, nsamples, nfeat, support_vectors.data(),
                         x_aux.data(), n_support, X_test, y_aux.data(), ldx_test,
                         kernel_matrix.data(), n_support, gamma, false);
        break;
    case linear:
        linear_kernel_local(column_major, n_support, nsamples, nfeat,
                            support_vectors.data(), n_support, X_test, ldx_test,
                            kernel_matrix.data(), n_support, false);
        break;
    case polynomial:
        polynomial_kernel_local(column_major, n_support, nsamples, nfeat,
                                support_vectors.data(), n_support, X_test, ldx_test,
                                kernel_matrix.data(), n_support, gamma, degree, coef0,
                                false);
        break;
    case sigmoid:
        sigmoid_kernel_local(column_major, n_support, nsamples, nfeat,
                             support_vectors.data(), n_support, X_test, ldx_test,
                             kernel_matrix.data(), n_support, gamma, coef0, false);
        break;
    }
    // Compute decision_values = K'*alpha + bias
    da_blas::cblas_gemv(CblasColMajor, CblasTrans, n_support, nsamples, (T)1.0,
                        kernel_matrix.data(), n_support, support_coefficients.data(), 1,
                        (T)1.0, decision_values, 1);

    return status;
}

/* Compute size of the outer working set */
template <typename T> void base_svm<T>::compute_ws_size(da_int &ws_size) {
    // Pick minimum between maximum power of two such that it is less than n, or some constant in this case 1024 (TODO: should be tuned to our hardware)
    da_int pow_two = maxpowtwo(actual_size);
    ws_size = std::min(pow_two, MAX_KERNEL_SIZE);
}

template <typename T>
void base_svm<T>::kernel_compute(std::vector<da_int> &idx, da_int &idx_size,
                                 std::vector<T> &X_temp, std::vector<T> &kernel_matrix) {
    // Get the relevant slices of original matrix (working set)
    // It will be more efficient to operate on row-major order
    for (da_int i = 0; i < idx_size; i++) {
        da_int current_idx = idx[i] % n;
        for (da_int j = 0; j < p; j++) {
            X_temp[i + j * idx_size] = X[current_idx + j * ldx_2];
        }
    }
    // Call to appropriate kernel function
    switch (kernel_function) {
    case rbf:
        rbf_kernel_local(column_major, n, idx_size, p, X, x_norm_aux.data(), ldx_2,
                         X_temp.data(), y_norm_aux.data(), idx_size, kernel_matrix.data(),
                         n, gamma, false);
        break;
    case linear:
        linear_kernel_local(column_major, n, idx_size, p, X, ldx_2, X_temp.data(),
                            idx_size, kernel_matrix.data(), n, false);
        break;
    case polynomial:
        polynomial_kernel_local(column_major, n, idx_size, p, X, ldx_2, X_temp.data(),
                                idx_size, kernel_matrix.data(), n, gamma, degree, coef0,
                                false);
        break;
    case sigmoid:
        sigmoid_kernel_local(column_major, n, idx_size, p, X, ldx_2, X_temp.data(),
                             idx_size, kernel_matrix.data(), n, gamma, coef0, false);
        break;
    }
};

// Formula for global gradient update is:   gradient = gradient + sum_over_columns(alpha_diff[i] * i_th_column_kernel_matrix)
// Here we benefit from column-major order of kernel matrix
// alpha_diff is of length ws_size, kernel_matrix is nrow by ncol, gradient is of length nrow
template <typename T>
void base_svm<T>::update_gradient(std::vector<T> &gradient, std::vector<T> &alpha_diff,
                                  da_int &nrow, da_int &ncol,
                                  std::vector<T> &kernel_matrix) {
    const T *const_kernel;
    // Special path for regression problems since gradient is 2 * nrow
    if (mod == da_svm_model::svr || mod == da_svm_model::nusvr) {
        std::vector<T> add_to_gradient(nrow, 0);
        for (da_int i = 0; i < ncol; i++) {
            const_kernel = kernel_matrix.data() + i * nrow;
            da_blas::cblas_axpy(nrow, alpha_diff[i], const_kernel, 1,
                                add_to_gradient.data(), 1);
        }
        for (da_int i = 0; i < nrow * 2; i++) {
            gradient[i] += add_to_gradient[i % nrow];
        }
    } else {
        for (da_int i = 0; i < ncol; i++) {
            const_kernel = kernel_matrix.data() + i * nrow;
            da_blas::cblas_axpy(nrow, alpha_diff[i], const_kernel, 1, gradient.data(), 1);
        }
    }
};

// This function calculates highest power of 2, that is smaller n (number of rows in data)
// TODO: To be optimised
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
void base_svm<T>::wssi(std::vector<bool> &I_up, std::vector<T> &gradient, da_int &i,
                       T &min_grad) {
    // Start with very large value to find minimum and its index
    T min_grad_value = std::numeric_limits<T>::max();
    da_int min_grad_idx = -1;
    for (da_int iter = 0; iter < ws_size; iter++) {
        if (I_up[iter] && gradient[iter] < min_grad_value) {
            min_grad_value = gradient[iter];
            min_grad_idx = iter;
        }
    }
    i = min_grad_idx;
    min_grad = min_grad_value;
};

// Select j-th index for the local smo working set selection
// We pick argmax of (b^2)/a such that it is in I_low set, while at the same time tracking maximum gradient value in I_low set
// for the local_smo convergence test
template <typename T>
void base_svm<T>::wssj(std::vector<bool> &I_low, std::vector<T> &gradient, da_int &i,
                       T &min_grad, da_int &j, T &max_grad, std::vector<T> &kernel_matrix,
                       T &delta, T &max_fun) {
    // Start with very large negative value to find maximum and its index
    T max_grad_value = -std::numeric_limits<T>::max();
    T max_function_val = -std::numeric_limits<T>::max();
    da_int max_grad_idx = -1;
    T a, b, function_val, ratio, current_gradient;
    delta = 0;
    for (da_int iter = 0; iter < ws_size; iter++) {
        if (I_low[iter]) {
            current_gradient = gradient[iter];
            if (max_grad_value < current_gradient)
                max_grad_value = current_gradient;
            // b = y_t * gradient_t - y_i * gradient_i
            b = current_gradient - min_grad;
            if (b < 0)
                continue;
            // a = K_ii + K_tt - 2 * K_it
            a = kernel_matrix[i + i * ws_size] + kernel_matrix[iter + iter * ws_size] -
                2 * kernel_matrix[i + iter * ws_size];
            if (a <= 0)
                a = tau;
            ratio = b / a;
            function_val = ratio * b;
            if (function_val > max_function_val) {
                max_function_val = function_val;
                max_grad_idx = iter;
                delta = ratio;
            }
        }
    }
    max_grad = max_grad_value;
    max_fun = max_function_val;
    j = max_grad_idx;
};

} // namespace da_svm
#endif