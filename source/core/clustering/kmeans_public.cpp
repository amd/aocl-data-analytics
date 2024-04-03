/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include "kmeans.hpp"

da_status da_kmeans_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                               const double *A, da_int lda) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->kmeans_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    /*Initialize*/
    return handle->kmeans_d->set_data(n_samples, n_features, A, lda);
}

da_status da_kmeans_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                               const float *A, da_int lda) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->kmeans_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    /*Initialize*/
    return handle->kmeans_s->set_data(n_samples, n_features, A, lda);
}

da_status da_kmeans_set_init_centres_d(da_handle handle, const double *C, da_int ldc) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->kmeans_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    /*Initialize*/
    return handle->kmeans_d->set_init_centres(C, ldc);
}

da_status da_kmeans_set_init_centres_s(da_handle handle, const float *C, da_int ldc) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->kmeans_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    /*Initialize*/
    return handle->kmeans_s->set_init_centres(C, ldc);
}

da_status da_kmeans_compute_d(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->kmeans_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    return handle->kmeans_d->compute();
}

da_status da_kmeans_compute_s(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->kmeans_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    return handle->kmeans_s->compute();
}

da_status da_kmeans_transform_s(da_handle handle, da_int m_samples, da_int m_features,
                                const float *X, da_int ldx, float *X_transform,
                                da_int ldx_transform) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->kmeans_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    return handle->kmeans_s->transform(m_samples, m_features, X, ldx, X_transform,
                                       ldx_transform);
}

da_status da_kmeans_transform_d(da_handle handle, da_int m_samples, da_int m_features,
                                const double *X, da_int ldx, double *X_transform,
                                da_int ldx_transform) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->kmeans_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    /*Initialize*/
    return handle->kmeans_d->transform(m_samples, m_features, X, ldx, X_transform,
                                       ldx_transform);
}

da_status da_kmeans_predict_s(da_handle handle, da_int k_samples, da_int k_features,
                              const float *Y, da_int ldy, da_int *Y_labels) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->kmeans_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    /*Initialize*/
    return handle->kmeans_s->predict(k_samples, k_features, Y, ldy, Y_labels);
}

da_status da_kmeans_predict_d(da_handle handle, da_int k_samples, da_int k_features,
                              const double *Y, da_int ldy, da_int *Y_labels) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->kmeans_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_kmeans or "
                        "handle is invalid.");

    /*Initialize*/
    return handle->kmeans_d->predict(k_samples, k_features, Y, ldy, Y_labels);
}
