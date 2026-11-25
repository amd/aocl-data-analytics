/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

namespace pca_public {

template <typename pca_class, typename T>
da_status pca_init(da_handle handle, da_int n_samples, da_int n_features, const T *A,
                   da_int lda) {
    pca_class *pca = dynamic_cast<pca_class *>(handle->get_alg_handle<T>());
    if (pca == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_pca or "
                        "handle is invalid.");

    return pca->init(n_samples, n_features, A, lda);
}

template <typename pca_class, typename T> da_status pca_compute(da_handle handle) {
    pca_class *pca = dynamic_cast<pca_class *>(handle->get_alg_handle<T>());
    if (pca == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_pca or "
                        "handle is invalid.");

    return pca->compute();
}

template <typename pca_class, typename T>
da_status pca_transform(da_handle handle, da_int m_samples, da_int m_features, const T *X,
                        da_int ldx, T *X_transform, da_int ldx_transform) {
    pca_class *pca = dynamic_cast<pca_class *>(handle->get_alg_handle<T>());
    if (pca == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_pca or "
                        "handle is invalid.");

    return pca->transform(m_samples, m_features, X, ldx, X_transform, ldx_transform);
}

template <typename pca_class, typename T>
da_status pca_inverse_transform(da_handle handle, da_int k_samples, da_int k_features,
                                const T *X, da_int ldx, T *X_inv_transform,
                                da_int ldx_inv_transform) {
    pca_class *pca = dynamic_cast<pca_class *>(handle->get_alg_handle<T>());
    if (pca == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_pca or "
                        "handle is invalid.");

    return pca->inverse_transform(k_samples, k_features, X, ldx, X_inv_transform,
                                  ldx_inv_transform);
}

} // namespace pca_public