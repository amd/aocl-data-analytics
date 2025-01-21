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

namespace kmeans_public {

template <typename kmeans_class, typename T>
da_status kmeans_set_data(da_handle handle, da_int n_samples, da_int n_features,
                          const T *A, da_int lda) {
    kmeans_class *kmeans = dynamic_cast<kmeans_class *>(handle->get_alg_handle<T>());
    if (kmeans == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    return kmeans->set_data(n_samples, n_features, A, lda);
}

template <typename kmeans_class, typename T>
da_status kmeans_set_init_centres(da_handle handle, const T *C, da_int ldc) {
    kmeans_class *kmeans = dynamic_cast<kmeans_class *>(handle->get_alg_handle<T>());
    if (kmeans == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    return kmeans->set_init_centres(C, ldc);
}

template <typename kmeans_class, typename T> da_status kmeans_compute(da_handle handle) {
    kmeans_class *kmeans = dynamic_cast<kmeans_class *>(handle->get_alg_handle<T>());
    if (kmeans == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    return kmeans->compute();
}

template <typename kmeans_class, typename T>
da_status kmeans_transform(da_handle handle, da_int m_samples, da_int m_features,
                           const T *X, da_int ldx, T *X_transform, da_int ldx_transform) {
    kmeans_class *kmeans = dynamic_cast<kmeans_class *>(handle->get_alg_handle<T>());
    if (kmeans == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    return kmeans->transform(m_samples, m_features, X, ldx, X_transform, ldx_transform);
}

template <typename kmeans_class, typename T>
da_status kmeans_predict(da_handle handle, da_int k_samples, da_int k_features,
                         const T *Y, da_int ldy, da_int *Y_labels) {
    kmeans_class *kmeans = dynamic_cast<kmeans_class *>(handle->get_alg_handle<T>());
    if (kmeans == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    return kmeans->predict(k_samples, k_features, Y, ldy, Y_labels);
}

} // namespace kmeans_public