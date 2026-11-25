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

namespace decision_tree_public {

template <typename decision_tree_class, typename T>
da_status get_decision_tree(da_handle handle, decision_tree_class *&decision_tree) {
    decision_tree = nullptr;
    if (handle->handle_type != da_handle_decision_tree)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "The handle was not initialized with da_handle_decision_tree.");
    decision_tree = dynamic_cast<decision_tree_class *>(handle->get_alg_handle<T>());
    if (decision_tree == nullptr)
        return da_error(
            handle->err, da_status_invalid_input,
            "Select a model with da_tree_select_model before using the tree.");
    return da_status_success;
}

template <typename decision_tree_class, typename T>
da_status decision_tree_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                 da_int n_class, const T *X, da_int ldx, const da_int *y,
                                 const da_int *categorical_features) {
    decision_tree_class *decision_tree = nullptr;
    da_status status = get_decision_tree<decision_tree_class, T>(handle, decision_tree);
    if (status != da_status_success)
        return status;

    return decision_tree->set_training_data(n_samples, n_features, X, ldx, y, n_class, 0,
                                            nullptr, categorical_features);
}

template <typename decision_tree_class, typename T>
da_status decision_tree_fit(da_handle handle) {
    decision_tree_class *decision_tree = nullptr;
    da_status status = get_decision_tree<decision_tree_class, T>(handle, decision_tree);
    if (status != da_status_success)
        return status;

    return decision_tree->fit();
}

template <typename decision_tree_class, typename T>
da_status decision_tree_predict(da_handle handle, da_int n_obs, da_int n_features,
                                const T *X_test, da_int ldx_test, da_int *y_pred) {
    decision_tree_class *decision_tree = nullptr;
    da_status status = get_decision_tree<decision_tree_class, T>(handle, decision_tree);
    if (status != da_status_success)
        return status;

    return decision_tree->predict(n_obs, n_features, X_test, ldx_test, y_pred);
}

template <typename decision_tree_class, typename T>
da_status decision_tree_predict_proba(da_handle handle, da_int n_obs, da_int n_features,
                                      const T *X_test, da_int ldx_test, T *y_pred,
                                      da_int n_class, da_int ldy) {
    decision_tree_class *decision_tree = nullptr;
    da_status status = get_decision_tree<decision_tree_class, T>(handle, decision_tree);
    if (status != da_status_success)
        return status;

    return decision_tree->predict_proba(n_obs, n_features, X_test, ldx_test, y_pred,
                                        n_class, ldy);
}

template <typename decision_tree_class, typename T>
da_status decision_tree_predict_log_proba(da_handle handle, da_int n_obs,
                                          da_int n_features, const T *X_test,
                                          da_int ldx_test, T *y_pred, da_int n_class,
                                          da_int ldy) {
    decision_tree_class *decision_tree = nullptr;
    da_status status = get_decision_tree<decision_tree_class, T>(handle, decision_tree);
    if (status != da_status_success)
        return status;

    return decision_tree->predict_log_proba(n_obs, n_features, X_test, ldx_test, y_pred,
                                            n_class, ldy);
}

template <typename decision_tree_class, typename T>
da_status decision_tree_score(da_handle handle, da_int n_samples, da_int n_features,
                              const T *X_test, da_int ldx_test, const da_int *y_test,
                              T *mean_accuracy) {
    decision_tree_class *decision_tree = nullptr;
    da_status status = get_decision_tree<decision_tree_class, T>(handle, decision_tree);
    if (status != da_status_success)
        return status;

    return decision_tree->score(n_samples, n_features, X_test, ldx_test, y_test,
                                mean_accuracy);
}

} // namespace decision_tree_public