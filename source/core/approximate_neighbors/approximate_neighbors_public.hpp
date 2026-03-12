/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

namespace approx_nn_public {

template <typename approx_nn_class, typename T>
da_status approx_nn_set_training_data(da_handle handle, da_int n_samples,
                                      da_int n_features, const T *X_train,
                                      da_int ldx_train) {
    approx_nn_class *ann = dynamic_cast<approx_nn_class *>(handle->get_alg_handle<T>());
    if (ann == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_approx_nn or "
            "handle is invalid.");

    return ann->set_training_data(n_samples, n_features, X_train, ldx_train);
}

template <typename approx_nn_class, typename T>
da_status approx_nn_train(da_handle handle) {
    approx_nn_class *ann = dynamic_cast<approx_nn_class *>(handle->get_alg_handle<T>());
    if (ann == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_approx_nn or "
            "handle is invalid.");

    return ann->train();
}

template <typename approx_nn_class, typename T>
da_status approx_nn_add(da_handle handle, da_int n_samples_add, da_int n_features,
                        const T *X_add, da_int ldX_add) {
    approx_nn_class *ann = dynamic_cast<approx_nn_class *>(handle->get_alg_handle<T>());
    if (ann == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_approx_nn or "
            "handle is invalid.");

    return ann->add(n_samples_add, n_features, X_add, ldX_add);
}

template <typename approx_nn_class, typename T>
da_status approx_nn_train_and_add(da_handle handle) {
    approx_nn_class *ann = dynamic_cast<approx_nn_class *>(handle->get_alg_handle<T>());
    if (ann == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_approx_nn or "
            "handle is invalid.");

    return ann->train_and_add();
}

template <typename approx_nn_class, typename T>
da_status approx_nn_kneighbors(da_handle handle, da_int n_queries, da_int n_features,
                               const T *X_test, da_int ldx_test, da_int *n_ind, T *n_dist,
                               da_int k, da_int return_distance) {
    approx_nn_class *ann = dynamic_cast<approx_nn_class *>(handle->get_alg_handle<T>());
    if (ann == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_approx_nn or "
            "handle is invalid.");

    return ann->kneighbors(n_queries, n_features, X_test, ldx_test, n_ind, n_dist, k,
                           return_distance);
}

} // namespace approx_nn_public