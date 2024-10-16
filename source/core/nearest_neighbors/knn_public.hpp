/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

namespace knn_public {

template <typename knn_class, typename T>
da_status knn_set_data(da_handle handle, da_int n_samples, da_int n_features,
                       const T *X_train, da_int ldx_train, const da_int *y_train) {
    knn_class *knn = dynamic_cast<knn_class *>(handle->get_alg_handle<T>());
    if (knn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");

    return knn->set_training_data(n_samples, n_features, X_train, ldx_train, y_train);
}

template <typename knn_class, typename T>
da_status knn_kneighbors(da_handle handle, da_int n_queries, da_int n_features,
                         const T *X_test, da_int ldx_test, da_int *n_ind, T *n_dist,
                         da_int k, da_int return_distance) {
    knn_class *knn = dynamic_cast<knn_class *>(handle->get_alg_handle<T>());
    if (knn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");

    return knn->kneighbors(n_queries, n_features, X_test, ldx_test, n_ind, n_dist, k,
                           return_distance);
}

template <typename knn_class, typename T>
da_status knn_classes(da_handle handle, da_int *n_classes, da_int *classes) {
    knn_class *knn = dynamic_cast<knn_class *>(handle->get_alg_handle<T>());
    if (knn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");
    da_status status = da_status_success;
    if (*n_classes <= 0) { // Querying number of classes to allocate memory
        status = knn->available_classes();
        if (status == da_status_success)
            *n_classes = da_int(knn->classes.size());
    } else { // Now that the number of classes is known, return those values, sorted in ascending order
        if (classes == nullptr)
            return da_error_bypass(handle->err, da_status_invalid_pointer,
                                   "classes is not a valid pointer.");
        for (da_int i = 0; i < *n_classes; i++)
            classes[i] = knn->classes[i];
    }
    return status;
}

template <typename knn_class, typename T>
da_status knn_predict_proba(da_handle handle, da_int n_queries, da_int n_features,
                            const T *X_test, da_int ldx_test, T *proba) {
    knn_class *knn = dynamic_cast<knn_class *>(handle->get_alg_handle<T>());
    if (knn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");

    return knn->predict_proba(n_queries, n_features, X_test, ldx_test, proba);
}

template <typename knn_class, typename T>
da_status knn_predict(da_handle handle, da_int n_queries, da_int n_features,
                      const T *X_test, da_int ldx_test, da_int *y_test) {
    knn_class *knn = dynamic_cast<knn_class *>(handle->get_alg_handle<T>());
    if (knn == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_knn or "
                        "handle is invalid.");

    return knn->predict(n_queries, n_features, X_test, ldx_test, y_test);
}

} // namespace knn_public