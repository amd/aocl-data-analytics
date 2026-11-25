/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "tsne_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

using namespace tsne_public;

template <typename T>
da_status da_tsne_set_data(da_handle handle, da_int n_samples, da_int n_features,
                           const T *X, da_int ldx) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear();

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (tsne_set_data<da_tsne::tsne<T>, T>(
                                handle, n_samples, n_features, X, ldx)));
}

template <typename T>
da_status da_tsne_set_init_embedding(da_handle handle, const T *Y, da_int ldy) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear();

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (tsne_set_init_embedding<da_tsne::tsne<T>, T>(handle, Y, ldy)));
}

template <typename T> da_status da_tsne_compute(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear();

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (tsne_compute<da_tsne::tsne<T>, T>(handle)));
}

template da_status da_tsne_set_data<float>(da_handle, da_int, da_int, const float *,
                                           da_int);
template da_status da_tsne_set_data<double>(da_handle, da_int, da_int, const double *,
                                            da_int);
template da_status da_tsne_set_init_embedding<float>(da_handle, const float *, da_int);
template da_status da_tsne_set_init_embedding<double>(da_handle, const double *, da_int);
template da_status da_tsne_compute<float>(da_handle);
template da_status da_tsne_compute<double>(da_handle);