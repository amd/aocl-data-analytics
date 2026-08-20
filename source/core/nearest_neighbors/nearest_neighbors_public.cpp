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

#include "nearest_neighbors_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"

using namespace neighbors_public;

template <typename T>
da_status da_nn_set_data(da_handle handle, da_int n_samples, da_int n_features,
                         const T *X_train, da_int ldx_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nn_set_data<da_neighbors::neighbors<T>, T>(
                                handle, n_samples, n_features, X_train, ldx_train)));
}

template <typename T>
da_status da_nn_set_labels(da_handle handle, da_int n_samples, const da_int *y_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nn_set_labels<da_neighbors::neighbors<T>, T>(
                                handle, n_samples, y_train)));
}

template <typename T>
da_status da_nn_set_targets(da_handle handle, da_int n_samples, const T *y_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nn_set_targets<da_neighbors::neighbors<T>, T>(
                                handle, n_samples, y_train)));
}

template <typename T>
da_status da_nn_kneighbors(da_handle handle, da_int n_queries, da_int n_features,
                           const T *X_test, da_int ldx_test, da_int *n_ind, T *n_dist,
                           da_int k, da_int return_distance) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nn_kneighbors<da_neighbors::neighbors<T>, T>(
                                handle, n_queries, n_features, X_test, ldx_test, n_ind,
                                n_dist, k, return_distance)));
}

template <typename T>
da_status da_nn_classes(da_handle handle, da_int *n_classes, da_int *classes) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nn_classes<da_neighbors::neighbors<T>, T>(
                                handle, n_classes, classes)));
}

template <typename T>
da_status da_nn_classifier_predict_proba(da_handle handle, da_int n_queries,
                                         da_int n_features, const T *X_test,
                                         da_int ldx_test, T *proba,
                                         da_nn_search_mode search_mode) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (nn_classifier_predict_proba<da_neighbors::neighbors<T>, T>(
                   handle, n_queries, n_features, X_test, ldx_test, proba, search_mode)));
}

template <typename T>
da_status da_nn_classifier_predict(da_handle handle, da_int n_queries, da_int n_features,
                                   const T *X_test, da_int ldx_test, da_int *y_test,
                                   da_nn_search_mode search_mode) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nn_classifier_predict<da_neighbors::neighbors<T>, T>(
                                handle, n_queries, n_features, X_test, ldx_test, y_test,
                                search_mode)));
}

template <typename T>
da_status da_nn_regressor_predict(da_handle handle, da_int n_queries, da_int n_features,
                                  const T *X_test, da_int ldx_test, T *y_test,
                                  da_nn_search_mode search_mode) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nn_regressor_predict<da_neighbors::neighbors<T>, T>(
                                handle, n_queries, n_features, X_test, ldx_test, y_test,
                                search_mode)));
}

template <typename T>
da_status da_nn_radius_neighbors(da_handle handle, da_int n_queries, da_int n_features,
                                 const T *X_test, da_int ldx_test, T radius,
                                 da_int return_distance, da_int sort_results) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nn_radius_neighbors<da_neighbors::neighbors<T>, T>(
                                handle, n_queries, n_features, X_test, ldx_test, radius,
                                return_distance, sort_results)));
}

template da_status da_nn_set_data<float>(da_handle, da_int, da_int, const float *,
                                         da_int);
template da_status da_nn_set_data<double>(da_handle, da_int, da_int, const double *,
                                          da_int);
template da_status da_nn_set_labels<float>(da_handle, da_int, const da_int *);
template da_status da_nn_set_labels<double>(da_handle, da_int, const da_int *);
template da_status da_nn_set_targets<float>(da_handle, da_int, const float *);
template da_status da_nn_set_targets<double>(da_handle, da_int, const double *);
template da_status da_nn_kneighbors<float>(da_handle, da_int, da_int, const float *,
                                           da_int, da_int *, float *, da_int, da_int);
template da_status da_nn_kneighbors<double>(da_handle, da_int, da_int, const double *,
                                            da_int, da_int *, double *, da_int, da_int);
template da_status da_nn_classes<float>(da_handle, da_int *, da_int *);
template da_status da_nn_classes<double>(da_handle, da_int *, da_int *);
template da_status da_nn_classifier_predict_proba<float>(da_handle, da_int, da_int,
                                                         const float *, da_int, float *,
                                                         da_nn_search_mode);
template da_status da_nn_classifier_predict_proba<double>(da_handle, da_int, da_int,
                                                          const double *, da_int,
                                                          double *, da_nn_search_mode);
template da_status da_nn_classifier_predict<float>(da_handle, da_int, da_int,
                                                   const float *, da_int, da_int *,
                                                   da_nn_search_mode);
template da_status da_nn_classifier_predict<double>(da_handle, da_int, da_int,
                                                    const double *, da_int, da_int *,
                                                    da_nn_search_mode);
template da_status da_nn_regressor_predict<float>(da_handle, da_int, da_int,
                                                  const float *, da_int, float *,
                                                  da_nn_search_mode);
template da_status da_nn_regressor_predict<double>(da_handle, da_int, da_int,
                                                   const double *, da_int, double *,
                                                   da_nn_search_mode);
template da_status da_nn_radius_neighbors<float>(da_handle, da_int, da_int, const float *,
                                                 da_int, float, da_int, da_int);
template da_status da_nn_radius_neighbors<double>(da_handle, da_int, da_int,
                                                  const double *, da_int, double, da_int,
                                                  da_int);