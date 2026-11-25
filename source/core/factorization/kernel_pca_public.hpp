/* ************************************************************************
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
