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

#include "dbscan_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

using namespace dbscan_public;

da_status da_dbscan_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                               const double *A, da_int lda) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err, return (dbscan_set_data<da_dbscan::dbscan<double>, double>(
                                handle, n_samples, n_features, A, lda)));
}

da_status da_dbscan_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                               const float *A, da_int lda) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err, return (dbscan_set_data<da_dbscan::dbscan<float>, float>(
                                handle, n_samples, n_features, A, lda)));
}

da_status da_dbscan_compute_d(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(handle->err, da_status_wrong_type,
                        "The handle was initialized with a different precision type "
                        "than double.");
    DISPATCHER(handle->err,
               return (dbscan_compute<da_dbscan::dbscan<double>, double>(handle)));
}

da_status da_dbscan_compute_s(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(handle->err, da_status_wrong_type,
                        "The handle was initialized with a different precision type "
                        "than single.");
    DISPATCHER(handle->err,
               return (dbscan_compute<da_dbscan::dbscan<float>, float>(handle)));
}