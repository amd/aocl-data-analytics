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

namespace random_forest_public {

template <typename random_forest_class, typename T>
da_status random_forest_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                 da_int n_class, const T *X, da_int ldx,
                                 const da_int *y) {
    random_forest_class *random_forest =
        dynamic_cast<random_forest_class *>(handle->get_alg_handle<T>());
    if (random_forest == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_random_forest or "
            "handle is invalid.");

    return random_forest->set_training_data(n_samples, n_features, X, ldx, y, n_class);
}

template <typename random_forest_class, typename T>
da_status random_forest_fit(da_handle handle) {
    random_forest_class *random_forest =
        dynamic_cast<random_forest_class *>(handle->get_alg_handle<T>());
    if (random_forest == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_random_forest or "
            "handle is invalid.");

    return random_forest->fit();
}

template <typename random_forest_class, typename T>
da_status random_forest_predict(da_handle handle, da_int n_obs, da_int n_features,
                                const T *X_test, da_int ldx_test, da_int *y_pred) {
    random_forest_class *random_forest =
        dynamic_cast<random_forest_class *>(handle->get_alg_handle<T>());
    if (random_forest == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_random_forest or "
            "handle is invalid.");

    return random_forest->predict(n_obs, n_features, X_test, ldx_test, y_pred);
}

template <typename random_forest_class, typename T>
da_status random_forest_predict_proba(da_handle handle, da_int n_obs, da_int n_features,
                                      const T *X_test, da_int ldx_test, T *y_pred,
                                      da_int n_class, da_int ldy) {
    random_forest_class *random_forest =
        dynamic_cast<random_forest_class *>(handle->get_alg_handle<T>());
    if (random_forest == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_random_forest or "
            "handle is invalid.");

    return random_forest->predict_proba(n_obs, n_features, X_test, ldx_test, y_pred,
                                        n_class, ldy);
}

template <typename random_forest_class, typename T>
da_status random_forest_predict_log_proba(da_handle handle, da_int n_obs,
                                          da_int n_features, const T *X_test,
                                          da_int ldx_test, T *y_pred, da_int n_class,
                                          da_int ldy) {
    random_forest_class *random_forest =
        dynamic_cast<random_forest_class *>(handle->get_alg_handle<T>());
    if (random_forest == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_random_forest or "
            "handle is invalid.");

    return random_forest->predict_log_proba(n_obs, n_features, X_test, ldx_test, y_pred,
                                            n_class, ldy);
}

template <typename random_forest_class, typename T>
da_status random_forest_score(da_handle handle, da_int n_samples, da_int n_features,
                              const T *X_test, da_int ldx_test, const da_int *y_test,
                              T *mean_accuracy) {
    random_forest_class *random_forest =
        dynamic_cast<random_forest_class *>(handle->get_alg_handle<T>());
    if (random_forest == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_random_forest or "
            "handle is invalid.");

    return random_forest->score(n_samples, n_features, X_test, ldx_test, y_test,
                                mean_accuracy);
}

} // namespace random_forest_public