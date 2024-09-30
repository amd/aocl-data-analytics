/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    da_linmod::linear_model<double> *linreg_d =
        dynamic_cast<da_linmod::linear_model<double> *>(handle->alg_handle_d);
    if (linreg_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_linmod or "
                        "handle is invalid.");

    return linreg_d->select_model(mod);
}

da_status da_linmod_select_model_s(da_handle handle, linmod_model mod) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    da_linmod::linear_model<float> *linreg_s =
        dynamic_cast<da_linmod::linear_model<float> *>(handle->alg_handle_s);
    if (linreg_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");

    return linreg_s->select_model(mod);
}

da_status da_linmod_define_features_d(da_handle handle, da_int nsamples, da_int nfeat,
                                      double *A, double *b) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    da_linmod::linear_model<double> *linreg_d =
        dynamic_cast<da_linmod::linear_model<double> *>(handle->alg_handle_d);
    if (linreg_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");

    return linreg_d->define_features(nfeat, nsamples, A, b);
}

da_status da_linmod_define_features_s(da_handle handle, da_int nsamples, da_int nfeat,
                                      float *A, float *b) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    da_linmod::linear_model<float> *linreg_s =
        dynamic_cast<da_linmod::linear_model<float> *>(handle->alg_handle_s);
    if (linreg_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");

    return linreg_s->define_features(nfeat, nsamples, A, b);
}

da_status da_linmod_fit_start_d(da_handle handle, da_int ncoefs, double *coefs) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    da_linmod::linear_model<double> *linreg_d =
        dynamic_cast<da_linmod::linear_model<double> *>(handle->alg_handle_d);
    if (linreg_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");

    return linreg_d->fit(ncoefs, coefs);
}

da_status da_linmod_fit_d(da_handle handle) {
    // Call fit with no initial starting point
    return da_linmod_fit_start_d(handle, 0, nullptr);
}

da_status da_linmod_fit_start_s(da_handle handle, da_int ncoefs, float *coefs) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    da_linmod::linear_model<float> *linreg_s =
        dynamic_cast<da_linmod::linear_model<float> *>(handle->alg_handle_s);
    if (linreg_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");

    return linreg_s->fit(ncoefs, coefs);
}

da_status da_linmod_fit_s(da_handle handle) {
    return da_linmod_fit_start_s(handle, 0, nullptr);
}

da_status da_linmod_evaluate_model_d(da_handle handle, da_int nsamples, da_int nfeat,
                                     double *X, double *predictions, double *observations,
                                     double *loss) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    da_linmod::linear_model<double> *linreg_d =
        dynamic_cast<da_linmod::linear_model<double> *>(handle->alg_handle_d);
    if (linreg_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");
    if (observations && loss)
        return linreg_d->evaluate_model(nfeat, nsamples, X, predictions, observations,
                                        loss);
    else if (!observations || !loss) {
        return linreg_d->evaluate_model(nfeat, nsamples, X, predictions, nullptr,
                                        nullptr);
    }
    return da_error(handle->err, da_status_invalid_input,
                    "Parameter `observations` should contain at least one single "
                    "observation. Parameter `loss` should point to a valid address.");
}

da_status da_linmod_evaluate_model_s(da_handle handle, da_int nsamples, da_int nfeat,
                                     float *X, float *predictions, float *observations,
                                     float *loss) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    da_linmod::linear_model<float> *linreg_s =
        dynamic_cast<da_linmod::linear_model<float> *>(handle->alg_handle_s);
    if (linreg_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_linmod or "
                        "handle is invalid.");
    if (observations && loss)
        return linreg_s->evaluate_model(nfeat, nsamples, X, predictions, observations,
                                        loss);
    else if (!observations || !loss) {
        return linreg_s->evaluate_model(nfeat, nsamples, X, predictions, nullptr,
                                        nullptr);
    }
    return da_error(handle->err, da_status_invalid_input,
                    "Parameter `observations` should contain at least one single "
                    "observation. Parameter `loss` should point to a valid address.");
}
