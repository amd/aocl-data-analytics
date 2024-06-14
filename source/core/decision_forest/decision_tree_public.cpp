/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "decision_tree.hpp"

da_status da_tree_set_training_data_d(da_handle handle, da_int n_samples,
                                      da_int n_features, da_int n_class, double *X,
                                      da_int ldx, da_int *y) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->dectree_d == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");

    return handle->dectree_d->set_training_data(n_samples, n_features, X, ldx, y,
                                                n_class);
}

da_status da_tree_set_training_data_s(da_handle handle, da_int n_samples,
                                      da_int n_features, da_int n_class, float *X,
                                      da_int ldx, da_int *y) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->dectree_s == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");

    return handle->dectree_s->set_training_data(n_samples, n_features, X, ldx, y,
                                                n_class);
}

da_status da_tree_fit_d(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->dectree_d == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");
    return handle->dectree_d->fit();
}
da_status da_tree_fit_s(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->dectree_s == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");
    return handle->dectree_s->fit();
}

da_status da_tree_predict_d(da_handle handle, da_int n_obs, da_int n_features,
                            double *X_test, da_int ldx_test, da_int *y_pred) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->dectree_d == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");
    return handle->dectree_d->predict(n_obs, n_features, X_test, ldx_test, y_pred);
}
da_status da_tree_predict_s(da_handle handle, da_int n_obs, da_int n_features,
                            float *X_test, da_int ldx_test, da_int *y_pred) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->dectree_s == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");
    return handle->dectree_s->predict(n_obs, n_features, X_test, ldx_test, y_pred);
}

da_status da_tree_score_d(da_handle handle, da_int n_samples, da_int n_features,
                          double *X_test, da_int ldx_test, da_int *y_test,
                          double *mean_accuracy) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->dectree_d == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");

    return handle->dectree_d->score(n_samples, n_features, X_test, ldx_test, y_test,
                                    mean_accuracy);
}

da_status da_tree_score_s(da_handle handle, da_int n_samples, da_int n_features,
                          float *X_test, da_int ldx_test, da_int *y_test,
                          float *mean_accuracy) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->dectree_s == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");

    return handle->dectree_s->score(n_samples, n_features, X_test, ldx_test, y_test,
                                    mean_accuracy);
}