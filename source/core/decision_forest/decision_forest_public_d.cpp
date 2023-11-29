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
#include "decision_forest.hpp"

da_status da_df_tree_set_training_data_d(da_handle handle, da_int n_obs,
                                         da_int n_features, double *x, da_int ldx,
                                         uint8_t *y) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->dt_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_decision_tree or handle is invalid.");

    return handle->dt_d->set_training_data(n_obs, n_features, x, ldx, y);
}

da_status da_df_tree_fit_d(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->dt_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_decision_tree or handle is invalid.");

    return handle->dt_d->fit();
}

da_status da_df_tree_predict_d(da_handle handle, da_int n_obs, double *x, da_int ldx,
                               uint8_t *y_pred) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->dt_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_decision_tree or handle is invalid.");

    return handle->dt_d->predict(n_obs, x, ldx, y_pred);
}

da_status da_df_tree_score_d(da_handle handle, da_int n_obs, double *x, da_int ldx,
                             uint8_t *y_test, double *score) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->dt_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_decision_tree or handle is invalid.");

    return handle->dt_d->score(n_obs, x, ldx, y_test, score);
}

da_status da_df_set_training_data_d(da_handle handle, da_int n_obs, da_int n_features,
                                    double *x, da_int ldx, uint8_t *y) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->df_d == nullptr)
        return da_status_invalid_pointer;

    return handle->df_d->set_training_data(n_obs, n_features, x, ldx, y);
}

da_status da_df_fit_d(da_handle handle) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->df_d == nullptr)
        return da_status_invalid_pointer;

    return handle->df_d->fit();
}

da_status da_df_predict_d(da_handle handle, da_int n_obs, double *x, da_int ldx,
                          uint8_t *y_pred) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->df_d == nullptr)
        return da_status_invalid_pointer;

    return handle->df_d->predict(n_obs, x, ldx, y_pred);
}

da_status da_df_score_d(da_handle handle, da_int n_obs, double *x, da_int ldx,
                        uint8_t *y_test, double *score) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->df_d == nullptr)
        return da_status_invalid_pointer;

    return handle->df_d->score(n_obs, x, ldx, y_test, score);
}
