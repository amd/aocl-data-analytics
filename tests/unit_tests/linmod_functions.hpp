/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <type_traits>

using namespace testing;

template <class T> struct option_t {
    std::string name = "";
    T value;
};

// Helper to define precision to which we expect the results to match
template <typename T> T expected_precision(T scale = (T)1.0);
template <> double expected_precision<double>(double scale) { return scale * 1.0e-3; }
template <> float expected_precision<float>(float scale) { return scale * 0.5f; }

// return precision as a string literal to set CSV options
template <typename T> constexpr const char *prec_name();
template <> constexpr const char *prec_name<float>() { return "single"; }
template <> constexpr const char *prec_name<double>() { return "double"; }

template <typename T> constexpr const char *type_opt_name();
template <> constexpr const char *type_opt_name<float>() { return "float"; }
template <> constexpr const char *type_opt_name<double>() { return "double"; }

template <typename T>
T objfun_mse(da_int n, da_int m, const T *x, const T *A, const T *b, bool intercept,
             T alpha, T lambda) {
    T eta{1}, beta{0};
    std::vector<T> y;
    y.resize(m);
    da_int aux = intercept ? 1 : 0;
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, m, n - aux, eta, A, m, x, 1, beta,
                        y.data(), 1);
    if (intercept) {
        for (da_int i = 0; i < m; i++)
            y[i] += x[n - 1];
    }
    eta = T(-1);
    da_blas::cblas_axpy(m, eta, b, 1, y.data(), 1);
    T rsq{0};
    T l1{0}, l2{0};
    for (da_int i = 0; i < m; ++i) {
        T res = (b[i] - y[i]);
        rsq += res * res;
        l1 += std::abs(x[i]);
        l2 += x[i] * x[i];
    }
    l1 *= (alpha * lambda);
    l2 *= ((T(1) - alpha) * lambda) / T(2);
    T loss = T(1) / T(2 * m) * rsq + l1 + l2;
}

// Get Loss value (MSE)
template <typename T>
T objfun(linmod_model mod, da_int n, da_int m, const T *x, const T *A, const T *b,
         bool intercept, T alpha, T lambda) {
    switch (mod) {
    case (linmod_model_mse):
        return objfun_mse(n, m, x, A, b, intercept, alpha, lambda);
        break;

    case (linmod_model_logistic):
        // return objfun_logistic(n, m, x, A, b, intercept, alpha, lambda);
        FAIL() << "not yet implemented";
        break;

    default:
        FAIL() << "unexpected gradient function";
    }
}

// T log_loss(T y, T p) { return -y * log(p) - (1 - y) * log(1 - p); }
// T logistic(T x) { return 1 / (1 + exp(-x)); }

template <typename T>
void objgrd_mse(da_int n, da_int m, T *x, std::vector<T> &grad, const T *A, const T *b,
                bool intercept) {

    T alpha = 1.0, beta = 0.0;
    std::vector<T> y;
    y.resize(m);
    da_int aux = intercept ? 1 : 0;
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, m, n - aux, alpha, A, m, x, 1, beta,
                        y.data(), 1);
    if (intercept) {
        for (da_int i = 0; i < m; i++)
            y[i] += x[n - 1];
    }
    alpha = -1.0;
    da_blas::cblas_axpy(m, alpha, b, 1, y.data(), 1);

    alpha = 2.0;
    da_blas::cblas_gemv(CblasColMajor, CblasTrans, m, n - aux, alpha, A, m, y.data(), 1,
                        beta, grad.data(), 1);
    if (intercept) {
        grad[n - 1] = 0.0;
        for (da_int i = 0; i < m; i++)
            grad[n - 1] += (T)2.0 * y[i];
    }
}

template <typename T>
void objgrd_logistic(da_int n, da_int m, T *x, std::vector<T> &grad, const T *A,
                     const T *b, bool intercept) {
    /* gradient of log loss of the logistic function
     * g_j = sum_i{A_ij*(b[i]-logistic(A_i^t x + x[n-1]))}
     */
    std::vector<T> y;

    // Compute A*x[0:n-2] = y
    da_int aux = intercept ? 1 : 0;
    T alpha = 1.0, beta = 0.0;
    y.resize(m);
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, m, n - aux, alpha, A, m, x, 1, beta,
                        y.data(), 1);

    std::fill(grad.begin(), grad.end(), 0);
    T lin_comb;
    for (da_int i = 0; i < m; i++) {
        lin_comb = intercept ? x[n - 1] + y[i] : y[i];
        for (da_int j = 0; j < n - aux; j++)
            grad[j] += (logistic(lin_comb) - b[i]) * A[m * j + i];
    }
    if (intercept) {
        grad[n - 1] = 0.0;
        for (da_int i = 0; i < m; i++) {
            lin_comb = x[n - 1] + y[i];
            grad[n - 1] += (logistic(lin_comb) - b[i]);
        }
    }
}

template <typename T>
void objgrd(linmod_model mod, da_int n, da_int m, T *x, std::vector<T> &grad, const T *A,
            const T *b, bool intercept) {
    switch (mod) {
    case (linmod_model_mse):
        objgrd_mse(n, m, x, grad, A, b, intercept);
        break;

    case (linmod_model_logistic):
        objgrd_logistic(n, m, x, grad, A, b, intercept);
        break;

    default:
        FAIL() << "unexpected gradient function";
    }
}
