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

#include "knn_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"

using namespace knn_public;

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
    DISPATCHER(handle->err,
               return (knn_set_data<da_knn::knn<double>, double>(
                   handle, n_samples, n_features, X_train, ldx_train, y_train)));
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
    DISPATCHER(handle->err,
               return (knn_set_data<da_knn::knn<float>, float>(
                   handle, n_samples, n_features, X_train, ldx_train, y_train)));
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

    DISPATCHER(handle->err, return (knn_kneighbors<da_knn::knn<double>, double>(
                                handle, n_queries, n_features, X_test, ldx_test, n_ind,
                                n_dist, k, return_distance)));
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
    DISPATCHER(handle->err, return (knn_kneighbors<da_knn::knn<float>, float>(
                                handle, n_queries, n_features, X_test, ldx_test, n_ind,
                                n_dist, k, return_distance)));
}

da_status da_knn_classes_d(da_handle handle, da_int *n_classes, da_int *classes) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err, return (knn_classes<da_knn::knn<double>, double>(
                                handle, n_classes, classes)));
}

da_status da_knn_classes_s(da_handle handle, da_int *n_classes, da_int *classes) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err, return (knn_classes<da_knn::knn<float>, float>(
                                handle, n_classes, classes)));
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
    DISPATCHER(handle->err, return (knn_predict_proba<da_knn::knn<double>, double>(
                                handle, n_queries, n_features, X_test, ldx_test, proba)));
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
    DISPATCHER(handle->err, return (knn_predict_proba<da_knn::knn<float>, float>(
                                handle, n_queries, n_features, X_test, ldx_test, proba)));
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
    DISPATCHER(handle->err,
               return (knn_predict<da_knn::knn<double>, double>(
                   handle, n_queries, n_features, X_test, ldx_test, y_test)));
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
    DISPATCHER(handle->err,
               return (knn_predict<da_knn::knn<float>, float>(
                   handle, n_queries, n_features, X_test, ldx_test, y_test)));
}
