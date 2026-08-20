/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
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

namespace kernel_pca_public {

template <typename kpca_class, typename T>
da_status kernel_pca_init(da_handle handle, da_int n_samples, da_int n_features,
                          const T *A, da_int lda) {
    kpca_class *kpca = dynamic_cast<kpca_class *>(handle->get_alg_handle<T>());
    if (kpca == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_kernel_pca or handle is invalid.");

    return kpca->init(n_samples, n_features, A, lda);
}

template <typename kpca_class, typename T>
da_status kernel_pca_compute(da_handle handle) {
    kpca_class *kpca = dynamic_cast<kpca_class *>(handle->get_alg_handle<T>());
    if (kpca == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_kernel_pca or handle is invalid.");

    return kpca->compute();
}

template <typename kpca_class, typename T>
da_status kernel_pca_transform(da_handle handle, da_int m_samples, da_int m_features,
                               const T *X, da_int ldx, T *X_transform,
                               da_int ldx_transform) {
    kpca_class *kpca = dynamic_cast<kpca_class *>(handle->get_alg_handle<T>());
    if (kpca == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_kernel_pca or handle is invalid.");

    return kpca->transform(m_samples, m_features, X, ldx, X_transform, ldx_transform);
}

template <typename kpca_class, typename T>
da_status kernel_pca_inverse_transform(da_handle handle, da_int k_samples,
                                       da_int k_components, const T *Y, da_int ldy,
                                       T *Y_inv_transform, da_int ldy_inv_transform) {
    kpca_class *kpca = dynamic_cast<kpca_class *>(handle->get_alg_handle<T>());
    if (kpca == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_kernel_pca or handle is invalid.");

    return kpca->inverse_transform(k_samples, k_components, Y, ldy, Y_inv_transform,
                                   ldy_inv_transform);
}

} // namespace kernel_pca_public
