/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
                               da_int k, bool return_distance) {
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