/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include "pca.hpp"

da_status da_pca_d_init(da_handle handle, da_int n, da_int p, double *dataX) {
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->pca_d == nullptr)
        return da_status_invalid_pointer;

    /*Initialize*/
    handle->pca_d->init(n, p, dataX);
    return da_status_success;
}

da_status da_pca_s_init(da_handle handle, da_int n, da_int p, float *dataX) {
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->pca_s == nullptr)
        return da_status_invalid_pointer;

    /*Initialize*/
    handle->pca_s->init(n, p, dataX);
    return da_status_success;
}

da_status da_pca_set_method(da_handle handle, pca_comp_method method) {
    if (handle != nullptr) {
        if (handle->precision != da_single) {
            if (handle->pca_d != nullptr)
                handle->pca_d->set_pca_compute_method(method);
        } else {
            if (handle->pca_s != nullptr)
                handle->pca_s->set_pca_compute_method(method);
        }
        return da_status_success;
    } else {
        return da_status_memory_error;
    }
}

da_status da_pca_set_num_components(da_handle handle, da_int num_components) {
    if (handle != nullptr) {
        if (handle->precision != da_single) {
            if (handle->pca_d != nullptr)
                handle->pca_d->set_pca_components(num_components);
        } else {
            if (handle->pca_s != nullptr)
                handle->pca_s->set_pca_components(num_components);
        }
        return da_status_success;
    } else {
        return da_status_memory_error;
    }
}

da_status da_pca_d_compute(da_handle handle) {
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->pca_d == nullptr)
        return da_status_invalid_pointer;

    return handle->pca_d->compute();
}

da_status da_pca_s_compute(da_handle handle) {
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->pca_s == nullptr)
        return da_status_invalid_pointer;

    return handle->pca_s->compute();
}