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

namespace svm_public {

template <typename svm_class, typename T>
da_status svm_select_model(da_handle handle, da_svm_model mod) {
    svm_class *svm = dynamic_cast<svm_class *>(handle->get_alg_handle<T>());
    if (svm == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm->select_model(mod);
}

template <typename svm_class, typename T>
da_status svm_set_data(da_handle handle, da_int n_samples, da_int n_features, const T *X,
                       da_int ldx_train, const T *y) {
    svm_class *svm = dynamic_cast<svm_class *>(handle->get_alg_handle<T>());
    if (svm == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm->set_data(n_samples, n_features, X, ldx_train, y);
}

template <typename svm_class, typename T> da_status svm_compute(da_handle handle) {
    svm_class *svm = dynamic_cast<svm_class *>(handle->get_alg_handle<T>());
    if (svm == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm->compute();
}

template <typename svm_class, typename T>
da_status svm_predict(da_handle handle, da_int n_samples, da_int n_features,
                      const T *X_test, da_int ldx_test, T *predictions) {
    svm_class *svm = dynamic_cast<svm_class *>(handle->get_alg_handle<T>());
    if (svm == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm->predict(n_samples, n_features, X_test, ldx_test, predictions);
}

template <typename svm_class, typename T>
da_status svm_decision_function(da_handle handle, da_int n_samples, da_int n_features,
                                const T *X_test, da_int ldx_test,
                                da_svm_decision_function_shape shape, T *decision_values,
                                da_int ldd) {
    svm_class *svm = dynamic_cast<svm_class *>(handle->get_alg_handle<T>());
    if (svm == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm->decision_function(n_samples, n_features, X_test, ldx_test, shape,
                                  decision_values, ldd);
}

template <typename svm_class, typename T>
da_status svm_score(da_handle handle, da_int n_samples, da_int n_features,
                    const T *X_test, da_int ldx_test, const T *y_test, T *score) {
    svm_class *svm = dynamic_cast<svm_class *>(handle->get_alg_handle<T>());
    if (svm == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm->score(n_samples, n_features, X_test, ldx_test, y_test, score);
}
} // namespace svm_public