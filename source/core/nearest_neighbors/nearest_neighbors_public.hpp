/* ************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

namespace neighbors_public {

template <typename neighbors_class, typename T>
da_status nn_set_data(da_handle handle, da_int n_samples, da_int n_features,
                      const T *X_train, da_int ldx_train) {
    neighbors_class *nn = dynamic_cast<neighbors_class *>(handle->get_alg_handle<T>());
    if (nn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nn or "
                        "handle is invalid.");

    return nn->set_data(n_samples, n_features, X_train, ldx_train);
}

template <typename neighbors_class, typename T>
da_status nn_set_labels(da_handle handle, da_int n_samples, const da_int *y_train) {
    neighbors_class *nn = dynamic_cast<neighbors_class *>(handle->get_alg_handle<T>());
    if (nn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nn or "
                        "handle is invalid.");

    return nn->set_labels(n_samples, y_train);
}

template <typename neighbors_class, typename T>
da_status nn_set_targets(da_handle handle, da_int n_samples, const T *y_train) {
    neighbors_class *nn = dynamic_cast<neighbors_class *>(handle->get_alg_handle<T>());
    if (nn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nn or "
                        "handle is invalid.");

    return nn->set_targets(n_samples, y_train);
}

template <typename neighbors_class, typename T>
da_status nn_kneighbors(da_handle handle, da_int n_queries, da_int n_features,
                        const T *X_test, da_int ldx_test, da_int *n_ind, T *n_dist,
                        da_int k, da_int return_distance) {
    neighbors_class *nn = dynamic_cast<neighbors_class *>(handle->get_alg_handle<T>());
    if (nn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nn or "
                        "handle is invalid.");

    return nn->kneighbors(n_queries, n_features, X_test, ldx_test, n_ind, n_dist, k,
                          return_distance);
}

template <typename neighbors_class, typename T>
da_status nn_classes(da_handle handle, da_int *n_classes, da_int *classes) {
    neighbors_class *nn = dynamic_cast<neighbors_class *>(handle->get_alg_handle<T>());
    if (nn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nn or "
                        "handle is invalid.");
    da_status status = da_status_success;
    if (*n_classes <= 0) { // Querying number of classes to allocate memory
        status = nn->available_classes();
        if (status == da_status_success)
            *n_classes = da_int(nn->classes.size());
    } else { // Now that the number of classes is known, return those values, sorted in ascending order
        if (classes == nullptr)
            return da_error_bypass(handle->err, da_status_invalid_pointer,
                                   "classes is not a valid pointer.");
        for (da_int i = 0; i < *n_classes; i++)
            classes[i] = nn->classes[i];
    }
    return status;
}

template <typename neighbors_class, typename T>
da_status nn_classifier_predict_proba(da_handle handle, da_int n_queries,
                                      da_int n_features, const T *X_test, da_int ldx_test,
                                      T *proba, da_nn_search_mode search_mode) {
    neighbors_class *nn = dynamic_cast<neighbors_class *>(handle->get_alg_handle<T>());
    if (nn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nn or "
                        "handle is invalid.");

    return nn->predict_proba(n_queries, n_features, X_test, ldx_test, proba, search_mode);
}

template <typename neighbors_class, typename T>
da_status nn_classifier_predict(da_handle handle, da_int n_queries, da_int n_features,
                                const T *X_test, da_int ldx_test, da_int *y_test,
                                da_nn_search_mode search_mode) {
    neighbors_class *nn = dynamic_cast<neighbors_class *>(handle->get_alg_handle<T>());
    if (nn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nn or "
                        "handle is invalid.");

    return nn->predict(n_queries, n_features, X_test, ldx_test, y_test, search_mode);
}

template <typename neighbors_class, typename T>
da_status nn_regressor_predict(da_handle handle, da_int n_queries, da_int n_features,
                               const T *X_test, da_int ldx_test, T *y_test,
                               da_nn_search_mode search_mode) {
    neighbors_class *nn = dynamic_cast<neighbors_class *>(handle->get_alg_handle<T>());
    if (nn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nn or "
                        "handle is invalid.");

    return nn->predict(n_queries, n_features, X_test, ldx_test, y_test, search_mode);
}

template <typename neighbors_class, typename T>
da_status nn_radius_neighbors(da_handle handle, da_int n_queries, da_int n_features,
                              const T *X_test, da_int ldx_test, T radius,
                              da_int return_distance, da_int sort_results) {
    neighbors_class *nn = dynamic_cast<neighbors_class *>(handle->get_alg_handle<T>());
    if (nn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nn or "
                        "handle is invalid.");

    return nn->radius_neighbors(n_queries, n_features, X_test, ldx_test, radius,
                                return_distance, sort_results);
}

} // namespace neighbors_public