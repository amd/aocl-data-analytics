/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "da_handle.hpp"
#include "knn.hpp"

da_status da_knn_set_training_data_d(da_handle handle, da_int n_samples,
                                     da_int n_features, const double *X_train,
                                     da_int ldx_train, const da_int *y_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->knn_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");

    return handle->knn_d->set_training_data(n_samples, n_features, X_train, ldx_train,
                                            y_train);
}

da_status da_knn_set_training_data_s(da_handle handle, da_int n_samples,
                                     da_int n_features, const float *X_train,
                                     da_int ldx_train, const da_int *y_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->knn_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");

    return handle->knn_s->set_training_data(n_samples, n_features, X_train, ldx_train,
                                            y_train);
}

da_status da_knn_kneighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                              const double *X_test, da_int ldx_test, da_int *n_ind,
                              double *n_dist, da_int k, da_int return_distance) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->knn_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");
    return handle->knn_d->kneighbors(n_queries, n_features, X_test, ldx_test, n_ind,
                                     n_dist, k, return_distance);
}
da_status da_knn_kneighbors_s(da_handle handle, da_int n_queries, da_int n_features,
                              const float *X_test, da_int ldx_test, da_int *n_ind,
                              float *n_dist, da_int k, da_int return_distance) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->knn_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");
    return handle->knn_s->kneighbors(n_queries, n_features, X_test, ldx_test, n_ind,
                                     n_dist, k, return_distance);
}

da_status da_knn_classes_d(da_handle handle, da_int *n_classes, da_int *classes) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->knn_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");
    da_status status = da_status_success;
    if (*n_classes <= 0) { // Querying number of classes to allocate memory
        status = handle->knn_d->available_classes();
        if (status == da_status_success)
            *n_classes = da_int(handle->knn_d->classes.size());
    } else { // Now that the number of classes is known, return those values, sorted in ascending order
        if (classes == nullptr)
            return da_error_bypass(handle->err, da_status_invalid_pointer,
                                   "classes is not a valid pointer.");
        for (da_int i = 0; i < *n_classes; i++)
            classes[i] = handle->knn_d->classes[i];
    }
    return status;
}

da_status da_knn_classes_s(da_handle handle, da_int *n_classes, da_int *classes) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->knn_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");
    da_status status = da_status_success;
    if (*n_classes <= 0) { // Querying number of classes to allocate memory
        status = handle->knn_s->available_classes();
        if (status == da_status_success)
            *n_classes = da_int(handle->knn_s->classes.size());
    } else { // Now that the number of classes is known, return those values, sorted in ascending order
        if (classes == nullptr)
            return da_error_bypass(handle->err, da_status_invalid_pointer,
                                   "classes is not a valid pointer.");
        for (da_int i = 0; i < *n_classes; i++)
            classes[i] = handle->knn_s->classes[i];
    }
    return status;
}

da_status da_knn_predict_proba_d(da_handle handle, da_int n_queries, da_int n_features,
                                 const double *X_test, da_int ldx_test, double *proba) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->knn_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");

    return handle->knn_d->predict_proba(n_queries, n_features, X_test, ldx_test, proba);
}

da_status da_knn_predict_proba_s(da_handle handle, da_int n_queries, da_int n_features,
                                 const float *X_test, da_int ldx_test, float *proba) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->knn_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");

    return handle->knn_s->predict_proba(n_queries, n_features, X_test, ldx_test, proba);
}

da_status da_knn_predict_d(da_handle handle, da_int n_queries, da_int n_features,
                           const double *X_test, da_int ldx_test, da_int *y_test) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->knn_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");

    return handle->knn_d->predict(n_queries, n_features, X_test, ldx_test, y_test);
}

da_status da_knn_predict_s(da_handle handle, da_int n_queries, da_int n_features,
                           const float *X_test, da_int ldx_test, da_int *y_test) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->knn_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");

    return handle->knn_s->predict(n_queries, n_features, X_test, ldx_test, y_test);
}
