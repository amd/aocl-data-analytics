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

#include "approximate_neighbors_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"

using namespace approx_nn_public;

template <typename T>
da_status da_approx_nn_set_training_data(da_handle handle, da_int n_samples,
                                         da_int n_features, const T *X_train,
                                         da_int ldx_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(
        handle->err,
        return (approx_nn_set_training_data<da_approx_nn::approximate_neighbors<T>, T>(
            handle, n_samples, n_features, X_train, ldx_train)));
}

template <typename T> da_status da_approx_nn_train(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(
        handle->err,
        return (approx_nn_train<da_approx_nn::approximate_neighbors<T>, T>(handle)));
}

template <typename T>
da_status da_approx_nn_add(da_handle handle, da_int n_samples_add, da_int n_features,
                           const T *X_add, da_int ldX_add) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (approx_nn_add<da_approx_nn::approximate_neighbors<T>, T>(
                   handle, n_samples_add, n_features, X_add, ldX_add)));
}

template <typename T> da_status da_approx_nn_train_and_add(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (approx_nn_train_and_add<da_approx_nn::approximate_neighbors<T>, T>(
                   handle)));
}

template <typename T>
da_status da_approx_nn_kneighbors(da_handle handle, da_int n_queries, da_int n_features,
                                  const T *X_test, da_int ldx_test, da_int *n_ind,
                                  T *n_dist, da_int k, bool return_distance) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (approx_nn_kneighbors<da_approx_nn::approximate_neighbors<T>, T>(
                   handle, n_queries, n_features, X_test, ldx_test, n_ind, n_dist, k,
                   return_distance)));
}

template da_status da_approx_nn_set_training_data<float>(da_handle, da_int, da_int,
                                                         const float *, da_int);
template da_status da_approx_nn_set_training_data<double>(da_handle, da_int, da_int,
                                                          const double *, da_int);
template da_status da_approx_nn_train<float>(da_handle);
template da_status da_approx_nn_train<double>(da_handle);
template da_status da_approx_nn_add<float>(da_handle, da_int, da_int, const float *,
                                           da_int);
template da_status da_approx_nn_add<double>(da_handle, da_int, da_int, const double *,
                                            da_int);
template da_status da_approx_nn_train_and_add<float>(da_handle);
template da_status da_approx_nn_train_and_add<double>(da_handle);
template da_status da_approx_nn_kneighbors<float>(da_handle, da_int, da_int,
                                                  const float *, da_int, da_int *,
                                                  float *, da_int, bool);
template da_status da_approx_nn_kneighbors<double>(da_handle, da_int, da_int,
                                                   const double *, da_int, da_int *,
                                                   double *, da_int, bool);