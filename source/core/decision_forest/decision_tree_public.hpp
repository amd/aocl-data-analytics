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

namespace decision_tree_public {

template <typename decision_tree_class, typename T>
da_status decision_tree_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                 da_int n_class, const T *X, da_int ldx,
                                 const da_int *y) {
    decision_tree_class *decision_tree =
        dynamic_cast<decision_tree_class *>(handle->get_alg_handle<T>());
    if (decision_tree == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");

    return decision_tree->set_training_data(n_samples, n_features, X, ldx, y, n_class);
}

template <typename decision_tree_class, typename T>
da_status decision_tree_fit(da_handle handle) {
    decision_tree_class *decision_tree =
        dynamic_cast<decision_tree_class *>(handle->get_alg_handle<T>());
    if (decision_tree == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");

    return decision_tree->fit();
}

template <typename decision_tree_class, typename T>
da_status decision_tree_predict(da_handle handle, da_int n_obs, da_int n_features,
                                const T *X_test, da_int ldx_test, da_int *y_pred) {
    decision_tree_class *decision_tree =
        dynamic_cast<decision_tree_class *>(handle->get_alg_handle<T>());
    if (decision_tree == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");

    return decision_tree->predict(n_obs, n_features, X_test, ldx_test, y_pred);
}

template <typename decision_tree_class, typename T>
da_status decision_tree_predict_proba(da_handle handle, da_int n_obs, da_int n_features,
                                      const T *X_test, da_int ldx_test, T *y_pred,
                                      da_int n_class, da_int ldy) {
    decision_tree_class *decision_tree =
        dynamic_cast<decision_tree_class *>(handle->get_alg_handle<T>());
    if (decision_tree == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");

    return decision_tree->predict_proba(n_obs, n_features, X_test, ldx_test, y_pred,
                                        n_class, ldy);
}

template <typename decision_tree_class, typename T>
da_status decision_tree_predict_log_proba(da_handle handle, da_int n_obs,
                                          da_int n_features, const T *X_test,
                                          da_int ldx_test, T *y_pred, da_int n_class,
                                          da_int ldy) {
    decision_tree_class *decision_tree =
        dynamic_cast<decision_tree_class *>(handle->get_alg_handle<T>());
    if (decision_tree == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");

    return decision_tree->predict_log_proba(n_obs, n_features, X_test, ldx_test, y_pred,
                                            n_class, ldy);
}

template <typename decision_tree_class, typename T>
da_status decision_tree_score(da_handle handle, da_int n_samples, da_int n_features,
                              const T *X_test, da_int ldx_test, const da_int *y_test,
                              T *mean_accuracy) {
    decision_tree_class *decision_tree =
        dynamic_cast<decision_tree_class *>(handle->get_alg_handle<T>());
    if (decision_tree == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_decision_tree or "
            "handle is invalid.");

    return decision_tree->score(n_samples, n_features, X_test, ldx_test, y_test,
                                mean_accuracy);
}

} // namespace decision_tree_public