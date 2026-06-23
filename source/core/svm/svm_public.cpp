/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "svm_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

using namespace svm_public;

template <typename T> da_status da_svm_select_model(da_handle handle, da_svm_model mod) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (svm_select_model<da_svm::svm<T>, T>(handle, mod)));
}

template <typename T>
da_status da_svm_set_data(da_handle handle, da_int n_samples, da_int n_features,
                          const T *X, da_int ldx_train, const T *y) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (svm_set_data<da_svm::svm<T>, T>(
                                handle, n_samples, n_features, X, ldx_train, y)));
}

template <typename T> da_status da_svm_compute(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (svm_compute<da_svm::svm<T>, T>(handle)));
}

template <typename T>
da_status da_svm_predict(da_handle handle, da_int n_samples, da_int n_features,
                         const T *X_test, da_int ldx_test, T *predictions) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (svm_predict<da_svm::svm<T>, T>(handle, n_samples, n_features,
                                                      X_test, ldx_test, predictions)));
}

template <typename T>
da_status da_svm_decision_function(da_handle handle, da_int n_samples, da_int n_features,
                                   const T *X_test, da_int ldx_test,
                                   da_svm_decision_function_shape shape,
                                   T *decision_values, da_int ldd) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (svm_decision_function<da_svm::svm<T>, T>(
                                handle, n_samples, n_features, X_test, ldx_test, shape,
                                decision_values, ldd)));
}

template <typename T>
da_status da_svm_score(da_handle handle, da_int n_samples, da_int n_features,
                       const T *X_test, da_int ldx_test, const T *y_test, T *score) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (svm_score<da_svm::svm<T>, T>(handle, n_samples, n_features, X_test,
                                                    ldx_test, y_test, score)));
}

template <typename T>
da_status da_svm_predict_proba(da_handle handle, da_int n_samples, da_int n_features,
                               const T *X_test, da_int ldx_test, T *y_proba, da_int ldy) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (svm_predict_proba<da_svm::svm<T>, T>(
                   handle, n_samples, n_features, X_test, ldx_test, y_proba, ldy)));
}

template <typename T>
da_status da_svm_predict_log_proba(da_handle handle, da_int n_samples, da_int n_features,
                                   const T *X_test, da_int ldx_test, T *y_log_proba,
                                   da_int ldy) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (svm_predict_log_proba<da_svm::svm<T>, T>(
                   handle, n_samples, n_features, X_test, ldx_test, y_log_proba, ldy)));
}

template da_status da_svm_select_model<float>(da_handle, da_svm_model);
template da_status da_svm_select_model<double>(da_handle, da_svm_model);
template da_status da_svm_set_data<float>(da_handle, da_int, da_int, const float *,
                                          da_int, const float *);
template da_status da_svm_set_data<double>(da_handle, da_int, da_int, const double *,
                                           da_int, const double *);
template da_status da_svm_compute<float>(da_handle);
template da_status da_svm_compute<double>(da_handle);
template da_status da_svm_predict<float>(da_handle, da_int, da_int, const float *, da_int,
                                         float *);
template da_status da_svm_predict<double>(da_handle, da_int, da_int, const double *,
                                          da_int, double *);
template da_status da_svm_decision_function<float>(da_handle, da_int, da_int,
                                                   const float *, da_int,
                                                   da_svm_decision_function_shape,
                                                   float *, da_int);
template da_status da_svm_decision_function<double>(da_handle, da_int, da_int,
                                                    const double *, da_int,
                                                    da_svm_decision_function_shape,
                                                    double *, da_int);
template da_status da_svm_score<float>(da_handle, da_int, da_int, const float *, da_int,
                                       const float *, float *);
template da_status da_svm_score<double>(da_handle, da_int, da_int, const double *, da_int,
                                        const double *, double *);
template da_status da_svm_predict_proba<float>(da_handle, da_int, da_int, const float *,
                                               da_int, float *, da_int);
template da_status da_svm_predict_proba<double>(da_handle, da_int, da_int, const double *,
                                                da_int, double *, da_int);
template da_status da_svm_predict_log_proba<float>(da_handle, da_int, da_int,
                                                   const float *, da_int, float *,
                                                   da_int);
template da_status da_svm_predict_log_proba<double>(da_handle, da_int, da_int,
                                                    const double *, da_int, double *,
                                                    da_int);