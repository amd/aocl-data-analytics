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

da_status da_nn_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                           const double *X_train, da_int ldx_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err, return (nn_set_data<da_neighbors::neighbors<double>, double>(
                                handle, n_samples, n_features, X_train, ldx_train)));
}

da_status da_nn_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                           const float *X_train, da_int ldx_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err, return (nn_set_data<da_neighbors::neighbors<float>, float>(
                                handle, n_samples, n_features, X_train, ldx_train)));
}

da_status da_nn_set_labels_d(da_handle handle, da_int n_samples, const da_int *y_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err,
               return (nn_set_labels<da_neighbors::neighbors<double>, double>(
                   handle, n_samples, y_train)));
}

da_status da_nn_set_labels_s(da_handle handle, da_int n_samples, const da_int *y_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err, return (nn_set_labels<da_neighbors::neighbors<float>, float>(
                                handle, n_samples, y_train)));
}

da_status da_nn_set_targets_d(da_handle handle, da_int n_samples, const double *y_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err,
               return (nn_set_targets<da_neighbors::neighbors<double>, double>(
                   handle, n_samples, y_train)));
}

da_status da_nn_set_targets_s(da_handle handle, da_int n_samples, const float *y_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err, return (nn_set_targets<da_neighbors::neighbors<float>, float>(
                                handle, n_samples, y_train)));
}

da_status da_nn_kneighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                             const double *X_test, da_int ldx_test, da_int *n_ind,
                             double *n_dist, da_int k, da_int return_distance) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    DISPATCHER(handle->err,
               return (nn_kneighbors<da_neighbors::neighbors<double>, double>(
                   handle, n_queries, n_features, X_test, ldx_test, n_ind, n_dist, k,
                   return_distance)));
}
da_status da_nn_kneighbors_s(da_handle handle, da_int n_queries, da_int n_features,
                             const float *X_test, da_int ldx_test, da_int *n_ind,
                             float *n_dist, da_int k, da_int return_distance) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err, return (nn_kneighbors<da_neighbors::neighbors<float>, float>(
                                handle, n_queries, n_features, X_test, ldx_test, n_ind,
                                n_dist, k, return_distance)));
}

da_status da_nn_classes_d(da_handle handle, da_int *n_classes, da_int *classes) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err, return (nn_classes<da_neighbors::neighbors<double>, double>(
                                handle, n_classes, classes)));
}

da_status da_nn_classes_s(da_handle handle, da_int *n_classes, da_int *classes) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err, return (nn_classes<da_neighbors::neighbors<float>, float>(
                                handle, n_classes, classes)));
}

da_status da_nn_classifier_predict_proba_d(da_handle handle, da_int n_queries,
                                           da_int n_features, const double *X_test,
                                           da_int ldx_test, double *proba,
                                           da_nn_search_mode search_mode) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(
        handle->err,
        return (nn_classifier_predict_proba<da_neighbors::neighbors<double>, double>(
            handle, n_queries, n_features, X_test, ldx_test, proba, search_mode)));
}

da_status da_nn_classifier_predict_proba_s(da_handle handle, da_int n_queries,
                                           da_int n_features, const float *X_test,
                                           da_int ldx_test, float *proba,
                                           da_nn_search_mode search_mode) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err,
               return (nn_classifier_predict_proba<da_neighbors::neighbors<float>, float>(
                   handle, n_queries, n_features, X_test, ldx_test, proba, search_mode)));
}

da_status da_nn_classifier_predict_d(da_handle handle, da_int n_queries,
                                     da_int n_features, const double *X_test,
                                     da_int ldx_test, da_int *y_test,
                                     da_nn_search_mode search_mode) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(
        handle->err,
        return (nn_classifier_predict<da_neighbors::neighbors<double>, double>(
            handle, n_queries, n_features, X_test, ldx_test, y_test, search_mode)));
}

da_status da_nn_classifier_predict_s(da_handle handle, da_int n_queries,
                                     da_int n_features, const float *X_test,
                                     da_int ldx_test, da_int *y_test,
                                     da_nn_search_mode search_mode) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(
        handle->err,
        return (nn_classifier_predict<da_neighbors::neighbors<float>, float>(
            handle, n_queries, n_features, X_test, ldx_test, y_test, search_mode)));
}

da_status da_nn_regressor_predict_d(da_handle handle, da_int n_queries, da_int n_features,
                                    const double *X_test, da_int ldx_test, double *y_test,
                                    da_nn_search_mode search_mode) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(
        handle->err,
        return (nn_regressor_predict<da_neighbors::neighbors<double>, double>(
            handle, n_queries, n_features, X_test, ldx_test, y_test, search_mode)));
}

da_status da_nn_regressor_predict_s(da_handle handle, da_int n_queries, da_int n_features,
                                    const float *X_test, da_int ldx_test, float *y_test,
                                    da_nn_search_mode search_mode) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(
        handle->err,
        return (nn_regressor_predict<da_neighbors::neighbors<float>, float>(
            handle, n_queries, n_features, X_test, ldx_test, y_test, search_mode)));
}

da_status da_nn_radius_neighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                                   const double *X_test, da_int ldx_test, double radius,
                                   da_int return_distance, da_int sort_results) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    DISPATCHER(handle->err,
               return (nn_radius_neighbors<da_neighbors::neighbors<double>, double>(
                   handle, n_queries, n_features, X_test, ldx_test, radius,
                   return_distance, sort_results)));
}
da_status da_nn_radius_neighbors_s(da_handle handle, da_int n_queries, da_int n_features,
                                   const float *X_test, da_int ldx_test, float radius,
                                   da_int return_distance, da_int sort_results) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err,
               return (nn_radius_neighbors<da_neighbors::neighbors<float>, float>(
                   handle, n_queries, n_features, X_test, ldx_test, radius,
                   return_distance, sort_results)));
}