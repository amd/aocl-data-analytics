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
                                const T *X, da_int ldX, T *predictions, T *observations,
                                T *loss) {
    linmod_class *linmod = dynamic_cast<linmod_class *>(handle->get_alg_handle<T>());
    if (linmod == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");

    if (observations && loss)
        return linmod->evaluate_model(nfeat, nsamples, X, ldX, predictions, observations,
                                      loss);
    else if (!observations || !loss) {
        return linmod->evaluate_model(nfeat, nsamples, X, ldX, predictions, nullptr,
                                      nullptr);
    }
    return da_error(handle->err, da_status_invalid_input,
                    "Parameter `observations` should contain at least one single "
                    "observation. Parameter `loss` should point to a valid address.");
}

} // namespace linmod_public