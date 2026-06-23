/*
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "decision_tree_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "macros.h"

using namespace decision_tree_public;

template <typename T>
da_status da_tree_set_training_data(da_handle handle, da_int n_samples, da_int n_features,
                                    da_int n_class, const T *X, da_int ldx,
                                    const da_int *y, const da_int *categorical_features) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(
        handle->err,
        return (decision_tree_set_data<da_decision_forest::decision_tree<T>, T>(
            handle, n_samples, n_features, n_class, X, ldx, y, categorical_features)));
}

template <typename T> da_status da_tree_fit(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(
        handle->err,
        return (decision_tree_fit<da_decision_forest::decision_tree<T>, T>(handle)));
}

template <typename T>
da_status da_tree_predict(da_handle handle, da_int n_obs, da_int n_features,
                          const T *X_test, da_int ldx_test, da_int *y_pred) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (decision_tree_predict<da_decision_forest::decision_tree<T>, T>(
                   handle, n_obs, n_features, X_test, ldx_test, y_pred)));
}

template <typename T>
da_status da_tree_predict_proba(da_handle handle, da_int n_obs, da_int n_features,
                                const T *X_test, da_int ldx_test, T *y_pred,
                                da_int n_class, da_int ldy) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(
        handle->err,
        return (decision_tree_predict_proba<da_decision_forest::decision_tree<T>, T>(
            handle, n_obs, n_features, X_test, ldx_test, y_pred, n_class, ldy)));
}

template <typename T>
da_status da_tree_predict_log_proba(da_handle handle, da_int n_obs, da_int n_features,
                                    const T *X_test, da_int ldx_test, T *y_pred,
                                    da_int n_class, da_int ldy) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(
        handle->err,
        return (decision_tree_predict_log_proba<da_decision_forest::decision_tree<T>, T>(
            handle, n_obs, n_features, X_test, ldx_test, y_pred, n_class, ldy)));
}

template <typename T>
da_status da_tree_score(da_handle handle, da_int n_samples, da_int n_features,
                        const T *X_test, da_int ldx_test, const da_int *y_test,
                        T *mean_accuracy) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(
        handle->err,
        return (decision_tree_score<da_decision_forest::decision_tree<T>, T>(
            handle, n_samples, n_features, X_test, ldx_test, y_test, mean_accuracy)));
}

template da_status da_tree_set_training_data<float>(da_handle, da_int, da_int, da_int,
                                                    const float *, da_int, const da_int *,
                                                    const da_int *);
template da_status da_tree_set_training_data<double>(da_handle, da_int, da_int, da_int,
                                                     const double *, da_int,
                                                     const da_int *, const da_int *);
template da_status da_tree_fit<float>(da_handle);
template da_status da_tree_fit<double>(da_handle);
template da_status da_tree_predict<float>(da_handle, da_int, da_int, const float *,
                                          da_int, da_int *);
template da_status da_tree_predict<double>(da_handle, da_int, da_int, const double *,
                                           da_int, da_int *);
template da_status da_tree_predict_proba<float>(da_handle, da_int, da_int, const float *,
                                                da_int, float *, da_int, da_int);
template da_status da_tree_predict_proba<double>(da_handle, da_int, da_int,
                                                 const double *, da_int, double *, da_int,
                                                 da_int);
template da_status da_tree_predict_log_proba<float>(da_handle, da_int, da_int,
                                                    const float *, da_int, float *,
                                                    da_int, da_int);
template da_status da_tree_predict_log_proba<double>(da_handle, da_int, da_int,
                                                     const double *, da_int, double *,
                                                     da_int, da_int);
template da_status da_tree_score<float>(da_handle, da_int, da_int, const float *, da_int,
                                        const da_int *, float *);
template da_status da_tree_score<double>(da_handle, da_int, da_int, const double *,
                                         da_int, const da_int *, double *);