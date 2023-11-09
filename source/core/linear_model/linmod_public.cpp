/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
 */

#include "aoclda.h"
#include "da_handle.hpp"
#include "linear_model.hpp"

da_status da_linmod_select_model_d(da_handle handle, linmod_model mod) {
    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->linreg_d == nullptr)
        return da_status_invalid_pointer; // LCOV_EXCL_LINE

    return handle->linreg_d->select_model(mod);
}

da_status da_linmod_select_model_s(da_handle handle, linmod_model mod) {
    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->linreg_s == nullptr)
        return da_status_invalid_pointer; // LCOV_EXCL_LINE

    return handle->linreg_s->select_model(mod);
}

da_status da_linmod_define_features_d(da_handle handle, da_int m, da_int n, double *A,
                                      double *b) {
    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->linreg_d == nullptr)
        return da_status_invalid_pointer; // LCOV_EXCL_LINE

    return handle->linreg_d->define_features(n, m, A, b);
}

da_status da_linmod_define_features_s(da_handle handle, da_int m, da_int n, float *A,
                                      float *b) {
    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->linreg_s == nullptr)
        return da_status_invalid_pointer; // LCOV_EXCL_LINE

    return handle->linreg_s->define_features(n, m, A, b);
}

da_status da_linmod_fit_start_d(da_handle handle, da_int ncoefs, double *coefs) {
    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->linreg_d == nullptr)
        return da_status_invalid_pointer; // LCOV_EXCL_LINE

    return handle->linreg_d->fit(ncoefs, coefs);
}

da_status da_linmod_fit_d(da_handle handle) {
    // Call fit with no initial starting point
    return da_linmod_fit_start_d(handle, 0, nullptr);
}

da_status da_linmod_fit_start_s(da_handle handle, da_int ncoefs, float *coefs) {
    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->linreg_s == nullptr)
        return da_status_invalid_pointer; // LCOV_EXCL_LINE

    return handle->linreg_s->fit(ncoefs, coefs);
}

da_status da_linmod_fit_s(da_handle handle) {
    return da_linmod_fit_start_s(handle, 0, nullptr);
}

da_status da_linmod_evaluate_model_d(da_handle handle, da_int m, da_int n, double *X,
                                     double *predictions) {
    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->linreg_d == nullptr)
        return da_status_invalid_pointer; // LCOV_EXCL_LINE

    return handle->linreg_d->evaluate_model(n, m, X, predictions);
}

da_status da_linmod_evaluate_model_s(da_handle handle, da_int m, da_int n, float *X,
                                     float *predictions) {
    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->linreg_s == nullptr)
        return da_status_invalid_pointer; // LCOV_EXCL_LINE

    return handle->linreg_s->evaluate_model(n, m, X, predictions);
}
