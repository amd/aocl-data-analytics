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

namespace linmod_public {

template <typename linmod_class, typename T>
da_status linmod_select_model(da_handle handle, linmod_model mod) {
    linmod_class *linmod = dynamic_cast<linmod_class *>(handle->get_alg_handle<T>());
    if (linmod == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");

    return linmod->select_model(mod);
}

template <typename linmod_class, typename T>
da_status linmod_define_features(da_handle handle, da_int nsamples, da_int nfeat,
                                 const T *X, da_int ldX, const T *b) {
    linmod_class *linmod = dynamic_cast<linmod_class *>(handle->get_alg_handle<T>());
    if (linmod == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");

    return linmod->define_features(nfeat, nsamples, X, ldX, b);
}

template <typename linmod_class, typename T>
da_status linmod_fit_start(da_handle handle, da_int ncoefs, const T *coefs) {
    linmod_class *linmod = dynamic_cast<linmod_class *>(handle->get_alg_handle<T>());
    if (linmod == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");
    return linmod->fit(ncoefs, coefs);
}

template <typename linmod_class, typename T>
da_status linmod_evaluate_model(da_handle handle, da_int nsamples, da_int nfeat,
                                const T *Xeval, da_int ldXeval, T *predictions,
                                const T *observations, T *loss) {
    linmod_class *linmod = dynamic_cast<linmod_class *>(handle->get_alg_handle<T>());
    if (linmod == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");

    if (observations && loss)
        return linmod->evaluate_model(nfeat, nsamples, Xeval, ldXeval, predictions,
                                      observations, loss);
    else if (!observations && !loss) {
        return linmod->evaluate_model(nfeat, nsamples, Xeval, ldXeval, predictions,
                                      nullptr, nullptr);
    }
    return da_error(handle->err, da_status_invalid_input,
                    "Parameter `observations` should contain at least one single "
                    "observation. Parameter `loss` should point to a valid address.");
}

} // namespace linmod_public