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