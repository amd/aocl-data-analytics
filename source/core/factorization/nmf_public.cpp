/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include "nmf_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"
#include "nmf/nmf.hpp"
#include <iostream>

using namespace nmf_public;

template <typename T>
da_status da_nmf_init(da_handle handle, da_int n_samples, da_int n_features, const T *A,
                      da_int lda) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear();

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nmf_init<da_nmf::nmf<T>, T>(handle, n_samples,
                                                                n_features, A, lda)))

    return da_status_success;
}

template <typename T> da_status da_nmf_compute(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear();

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nmf_compute<da_nmf::nmf<T>, T>(handle)))

    return da_status_success;
}

template <typename T>
da_status da_nmf_set_data(da_handle handle, da_int n_samples, da_int n_components,
                          const T *W, da_int ldw, da_int n_features, const T *H,
                          da_int ldh) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear();

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (nmf_set_data<da_nmf::nmf<T>, T>(handle, n_samples, n_components, W,
                                                       ldw, n_features, H, ldh)))

    return da_status_success;
}

template <typename T>
da_status da_nmf_transform(da_handle handle, da_int m_samples, da_int m_features,
                           const T *X, da_int ldx, T *Xt, da_int ldxt) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear();

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nmf_transform<da_nmf::nmf<T>, T>(
                                handle, m_samples, m_features, X, ldx, Xt, ldxt)))

    return da_status_success;
}

template <typename T>
da_status da_nmf_inverse_transform(da_handle handle, da_int k_samples,
                                   da_int k_components, const T *Xt, da_int ldxt, T *X,
                                   da_int ldx) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear();

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nmf_inverse_transform<da_nmf::nmf<T>, T>(
                                handle, k_samples, k_components, Xt, ldxt, X, ldx)))

    return da_status_success;
}

template da_status da_nmf_init<float>(da_handle, da_int, da_int, const float *, da_int);
template da_status da_nmf_init<double>(da_handle, da_int, da_int, const double *, da_int);
template da_status da_nmf_compute<float>(da_handle);
template da_status da_nmf_compute<double>(da_handle);
template da_status da_nmf_set_data<float>(da_handle, da_int, da_int, const float *,
                                          da_int, da_int, const float *, da_int);
template da_status da_nmf_set_data<double>(da_handle, da_int, da_int, const double *,
                                           da_int, da_int, const double *, da_int);
template da_status da_nmf_transform<float>(da_handle, da_int, da_int, const float *,
                                           da_int, float *, da_int);
template da_status da_nmf_transform<double>(da_handle, da_int, da_int, const double *,
                                            da_int, double *, da_int);
template da_status da_nmf_inverse_transform<float>(da_handle, da_int, da_int,
                                                   const float *, da_int, float *,
                                                   da_int);
template da_status da_nmf_inverse_transform<double>(da_handle, da_int, da_int,
                                                    const double *, da_int, double *,
                                                    da_int);
