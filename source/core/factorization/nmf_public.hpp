/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

namespace nmf_public {

template <typename nmf_class, typename T>
da_status nmf_init(da_handle handle, da_int n_samples, da_int n_features, const T *A,
                   da_int lda) {
    nmf_class *nmf = dynamic_cast<nmf_class *>(handle->get_alg_handle<T>());
    if (nmf == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nmf or "
                        "handle is invalid.");

    return nmf->init(n_samples, n_features, A, lda);
}

template <typename nmf_class, typename T>
da_status nmf_set_data(da_handle handle, da_int n_samples, da_int n_components,
                       const T *W, da_int ldw, da_int n_features, const T *H,
                       da_int ldh) {
    nmf_class *nmf = dynamic_cast<nmf_class *>(handle->get_alg_handle<T>());
    if (nmf == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nmf or "
                        "handle is invalid.");

    return nmf->nmf_set_data(n_samples, n_components, W, ldw, n_features, H, ldh);
}

template <typename nmf_class, typename T> da_status nmf_compute(da_handle handle) {
    nmf_class *nmf = dynamic_cast<nmf_class *>(handle->get_alg_handle<T>());
    if (nmf == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nmf or "
                        "handle is invalid.");

    return nmf->compute();
}

template <typename nmf_class, typename T>
da_status nmf_transform(da_handle handle, da_int m_samples, da_int m_features, const T *X,
                        da_int ldx, T *Xt, da_int ldxt) {
    nmf_class *nmf = dynamic_cast<nmf_class *>(handle->get_alg_handle<T>());
    if (nmf == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nmf or "
                        "handle is invalid.");

    return nmf->transform(m_samples, m_features, X, ldx, Xt, ldxt);
}

template <typename nmf_class, typename T>
da_status nmf_inverse_transform(da_handle handle, da_int k_samples, da_int k_components,
                                const T *Xt, da_int ldxt, T *X, da_int ldx) {
    nmf_class *nmf = dynamic_cast<nmf_class *>(handle->get_alg_handle<T>());
    if (nmf == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nmf or "
                        "handle is invalid.");

    return nmf->inverse_transform(k_samples, k_components, Xt, ldxt, X, ldx);
}

} // namespace nmf_public
