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

da_status da_approx_nn_set_training_data_d(da_handle handle, da_int n_samples,
                                           da_int n_features, const double *X_train,
                                           da_int ldx_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err, return (approx_nn_set_training_data<
                                    da_approx_nn::approximate_neighbors<double>, double>(
                                handle, n_samples, n_features, X_train, ldx_train)));
}

da_status da_approx_nn_set_training_data_s(da_handle handle, da_int n_samples,
                                           da_int n_features, const float *X_train,
                                           da_int ldx_train) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err, return (approx_nn_set_training_data<
                                    da_approx_nn::approximate_neighbors<float>, float>(
                                handle, n_samples, n_features, X_train, ldx_train)));
}

da_status da_approx_nn_train_d(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(
        handle->err,
        return (approx_nn_train<da_approx_nn::approximate_neighbors<double>, double>(
            handle)));
}

da_status da_approx_nn_train_s(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err,
               return (approx_nn_train<da_approx_nn::approximate_neighbors<float>, float>(
                   handle)));
}

da_status da_approx_nn_add_d(da_handle handle, da_int n_samples_add, da_int n_features,
                             const double *X_add, da_int ldX_add) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err,
               return (approx_nn_add<da_approx_nn::approximate_neighbors<double>, double>(
                   handle, n_samples_add, n_features, X_add, ldX_add)));
}

da_status da_approx_nn_add_s(da_handle handle, da_int n_samples_add, da_int n_features,
                             const float *X_add, da_int ldX_add) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err,
               return (approx_nn_add<da_approx_nn::approximate_neighbors<float>, float>(
                   handle, n_samples_add, n_features, X_add, ldX_add)));
}

da_status da_approx_nn_train_and_add_d(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(
        handle->err,
        return (
            approx_nn_train_and_add<da_approx_nn::approximate_neighbors<double>, double>(
                handle)));
}

da_status da_approx_nn_train_and_add_s(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(
        handle->err,
        return (
            approx_nn_train_and_add<da_approx_nn::approximate_neighbors<float>, float>(
                handle)));
}

da_status da_approx_nn_kneighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                                    const double *X_test, da_int ldx_test, da_int *n_ind,
                                    double *n_dist, da_int k, da_int return_distance) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(
        handle->err,
        return (approx_nn_kneighbors<da_approx_nn::approximate_neighbors<double>, double>(
            handle, n_queries, n_features, X_test, ldx_test, n_ind, n_dist, k,
            return_distance)));
}

da_status da_approx_nn_kneighbors_s(da_handle handle, da_int n_queries, da_int n_features,
                                    const float *X_test, da_int ldx_test, da_int *n_ind,
                                    float *n_dist, da_int k, da_int return_distance) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(
        handle->err,
        return (approx_nn_kneighbors<da_approx_nn::approximate_neighbors<float>, float>(
            handle, n_queries, n_features, X_test, ldx_test, n_ind, n_dist, k,
            return_distance)));
}