/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "da_error.hpp"
#include "da_handle.hpp"
#include "parser.hpp"

/* Create (and populate with defaults) */
da_status da_handle_init_d(da_handle *handle, da_handle_type handle_type) {
    try {
        *handle = new _da_handle;
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }

    try {
        (*handle)->err = new da_errors::da_error_t(da_errors::action_t::DA_RECORD);
    } catch (...) {
        return da_status_internal_error;
    }

    (*handle)->handle_type = handle_type;
    da_status error = da_status_success;
    (*handle)->precision = da_double;

    switch (handle_type) {
    case da_handle_uninitialized:
        error = da_status_success;
        break;
    case da_handle_linmod:
        try {
            (*handle)->linreg_d = new linear_model<double>(*(*handle)->err);
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }
        break;
    case da_handle_pca:
        try {
            (*handle)->pca_d = new da_pca::da_pca<double>(*(*handle)->err);
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }
        break;
    case da_handle_decision_tree:
        try {
            (*handle)->dt_d = new decision_tree<double>(*(*handle)->err);
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }
    case da_handle_decision_forest:
        try {
            (*handle)->df_d = new decision_forest<double>(*(*handle)->err);
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }
    default:
        error = da_status_success;
        break;
    }

    if (!(error == da_status_success)) {
        da_handle_destroy(handle);
    }

    return error;
}

da_status da_handle_init_s(da_handle *handle, da_handle_type handle_type) {
    //TODO FIXME: rename error -> status
    // add da_error(...) to all non-successful sets. Same for _d
    try {
        *handle = new _da_handle;
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }

    try {
        (*handle)->err = new da_errors::da_error_t(da_errors::action_t::DA_RECORD);
    } catch (...) {
        return da_status_internal_error;
    }

    (*handle)->handle_type = handle_type;
    da_status error = da_status_success;
    (*handle)->precision = da_single;

    switch (handle_type) {
    case da_handle_uninitialized:
        error = da_status_success;
        break;
    case da_handle_linmod:
        try {
            (*handle)->linreg_s = new linear_model<float>(*(*handle)->err);
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }
        break;
    case da_handle_pca:
        try {
            (*handle)->pca_s = new da_pca::da_pca<float>(*(*handle)->err);
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }
        break;
    case da_handle_decision_tree:
        try {
            (*handle)->dt_s = new decision_tree<float>(*(*handle)->err);
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }
    case da_handle_decision_forest:
        try {
            (*handle)->df_s = new decision_forest<float>(*(*handle)->err);
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }
    default:
        error = da_status_success;
        break;
    }

    if (!(error == da_status_success)) {
        da_handle_destroy(handle);
    }

    return error;
}

da_status da_handle_print_error_message(da_handle handle) {
    // check to see if we have a valid handle
    if (handle) {
        handle->err->print();
        return da_status_success;
    }
    return da_status_invalid_input;
}

/* Destroy the da_handle struct */
void da_handle_destroy(da_handle *handle) {

    if (handle) {
        if (*handle) {
            if ((*handle)->linreg_d)
                delete (*handle)->linreg_d;
            if ((*handle)->linreg_s)
                delete (*handle)->linreg_s;
            if ((*handle)->csv_parser)
                delete (*handle)->csv_parser;
            if ((*handle)->pca_d)
                delete (*handle)->pca_d;
            if ((*handle)->pca_s)
                delete (*handle)->pca_s;
            if ((*handle)->dt_d)
                delete (*handle)->dt_d;
            if ((*handle)->dt_s)
                delete (*handle)->dt_s;
            if ((*handle)->df_d)
                delete (*handle)->df_d;
            if ((*handle)->df_s)
                delete (*handle)->df_s;
            if ((*handle)->err)
                delete (*handle)->err;
        }
        delete (*handle);
        *handle = nullptr;
    }
}

/* Get results out of the handle
 * Defines are in aoclda_result.h
 */

da_status da_handle_get_result_d(da_handle handle, da_result query, da_int *dim,
                                 double *result) {
    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    // Currently there can only be a SINGLE valid internal handle pointer,
    // so we cycle through them and query to see if the result is
    // provided by it.
    if (handle->linreg_d != nullptr)
        return handle->linreg_d->get_result(query, dim, result);
    else if (handle->pca_d != nullptr)
        return handle->pca_d->get_result(query, dim, result);

    // handle was not initialized with
    return da_error(handle->err, da_status_handle_not_initialized,
                    "The handle does not have any results to export. Have you "
                    "initialized the handle and performed any calculation?");
}

da_status da_handle_get_result_s(da_handle handle, da_result query, da_int *dim,
                                 float *result) {
    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than float.");

    // Currently there can only be a SINGLE valid internal handle pointer,
    // so we cycle through them and query to see if the result is
    // provided by it.
    if (handle->linreg_s != nullptr)
        return handle->linreg_s->get_result(query, dim, result);
    else if (handle->pca_s != nullptr)
        return handle->pca_s->get_result(query, dim, result);

    // handle was not initialized
    return da_error(handle->err, da_status_handle_not_initialized,
                    "The handle does not have any results to export. Have you "
                    "initialized the handle and performed any calculation?");
}

da_status da_handle_get_result_int(da_handle handle, da_result query, da_int *dim,
                                   da_int *result) {
    if (!handle)
        return da_status_invalid_pointer;

    // Currently there can only be a SINGLE valid internal handle pointer,
    // so we cycle through them and query to see if the result is
    // provided by it.
    if (handle->linreg_d != nullptr)
        return handle->linreg_d->get_result(query, dim, result);
    else if (handle->linreg_s != nullptr)
        return handle->linreg_s->get_result(query, dim, result);
    else if (handle->pca_d != nullptr)
        return handle->pca_d->get_result(query, dim, result);
    else if (handle->pca_s != nullptr)
        return handle->pca_s->get_result(query, dim, result);
    // handle was not initialized
    return da_error(handle->err, da_status_handle_not_initialized,
                    "The handle does not have any results to export. Have you "
                    "initialized the handle and performed any calculation?");
}
