/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "macros.h"
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
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }

    (*handle)->handle_type = handle_type;
    (*handle)->precision = da_double;
    da_status status = da_status_success;

    try {
        switch (handle_type) {
            break;
        case da_handle_linmod:
            DISPATCHER((*handle)->err,
                       (*handle)->alg_handle_d =
                           new da_linmod::linear_model<double>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        case da_handle_pca:
            DISPATCHER((*handle)->err, (*handle)->alg_handle_d =
                                           new da_pca::pca<double>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_d = nullptr;
                return status;
            }
            break;
        case da_handle_kmeans:
            DISPATCHER((*handle)->err,
                       (*handle)->alg_handle_d =
                           new da_kmeans::kmeans<double>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_d = nullptr;
                return status;
            }
            break;
        case da_handle_dbscan:
            DISPATCHER((*handle)->err,
                       (*handle)->alg_handle_d =
                           new da_dbscan::dbscan<double>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_d = nullptr;
                return status;
            }
            break;
        case da_handle_dbscan:
            (*handle)->alg_handle_d = new da_dbscan::da_dbscan<double>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_d = nullptr;
                return status;
            }
            break;
        case da_handle_decision_tree:
            DISPATCHER((*handle)->err, (*handle)->alg_handle_d =
                                           new da_decision_forest::decision_tree<double>(
                                               *(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_d = nullptr;
                return status;
            }
            break;
        case da_handle_decision_forest:
            DISPATCHER((*handle)->err, (*handle)->alg_handle_d =
                                           new da_decision_forest::random_forest<double>(
                                               *(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_d = nullptr;
                return status;
            }
            break;
        case da_handle_nlls:
            DISPATCHER((*handle)->err,
                       (*handle)->alg_handle_d =
                           new da_nlls::nlls<double>(status, *(*handle)->err));
            // status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_d = nullptr;
                return status;
            }
            break;
        case da_handle_knn:
            DISPATCHER((*handle)->err, (*handle)->alg_handle_d =
                                           new da_knn::knn<double>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_d = nullptr;
                return status;
            }
            break;
        case da_handle_svm:
            DISPATCHER((*handle)->err, (*handle)->alg_handle_d =
                                           new da_svm::svm<double>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_d = nullptr;
                return status;
            }
            break;
        default:
            break;
        }
    } catch (std::bad_alloc &) {
        return da_error((*handle)->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");             // LCOV_EXCL_LINE
    }
    return da_status_success;
}

da_status da_handle_init_s(da_handle *handle, da_handle_type handle_type) {

    try {
        *handle = new _da_handle;
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }
    try {
        (*handle)->err = new da_errors::da_error_t(da_errors::action_t::DA_RECORD);
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }

    (*handle)->handle_type = handle_type;
    (*handle)->precision = da_single;
    da_status status = da_status_success;

    try {
        switch (handle_type) {
        case da_handle_linmod:
            DISPATCHER((*handle)->err,
                       (*handle)->alg_handle_s =
                           new da_linmod::linear_model<float>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        case da_handle_pca:
            DISPATCHER((*handle)->err,
                       (*handle)->alg_handle_s = new da_pca::pca<float>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        case da_handle_kmeans:
            DISPATCHER((*handle)->err, (*handle)->alg_handle_s =
                                           new da_kmeans::kmeans<float>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        case da_handle_dbscan:
            DISPATCHER((*handle)->err, (*handle)->alg_handle_s =
                                           new da_dbscan::dbscan<float>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        case da_handle_dbscan:
            (*handle)->alg_handle_s = new da_dbscan::da_dbscan<float>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        case da_handle_decision_tree:
            DISPATCHER((*handle)->err,
                       (*handle)->alg_handle_s =
                           new da_decision_forest::decision_tree<float>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        case da_handle_decision_forest:
            DISPATCHER((*handle)->err,
                       (*handle)->alg_handle_s =
                           new da_decision_forest::random_forest<float>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        case da_handle_nlls:
            DISPATCHER((*handle)->err, (*handle)->alg_handle_s = new da_nlls::nlls<float>(
                                           status, *(*handle)->err));
            // status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        case da_handle_knn:
            DISPATCHER((*handle)->err,
                       (*handle)->alg_handle_s = new da_knn::knn<float>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        case da_handle_svm:
            DISPATCHER((*handle)->err,
                       (*handle)->alg_handle_s = new da_svm::svm<float>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->alg_handle_s = nullptr;
                return status;
            }
            break;
        default:
            break;
        }
    } catch (std::bad_alloc &) {
        return da_error((*handle)->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");             // LCOV_EXCL_LINE
    }
    return da_status_success;
}

da_status da_handle_print_error_message(da_handle handle) {
    // check to see if we have a valid handle
    if (handle) {
        if (handle->err) {
            handle->err->print();
            return da_status_success;
        } else {
            return da_status_internal_error;
        }
    }
    return da_status_invalid_input;
}

/* Destroy the da_handle struct */
void da_handle_destroy(da_handle *handle) {

    if (handle) {
        if (*handle) {
            if ((*handle)->alg_handle_d)
                delete (*handle)->alg_handle_d;
            if ((*handle)->alg_handle_s)
                delete (*handle)->alg_handle_s;
            if ((*handle)->csv_parser)
                delete (*handle)->csv_parser;
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
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(handle->err, da_status_wrong_type,
                        "The handle was initialized with a different precision type than "
                        "double precision floating point type.");

    if (dim == nullptr)
        return da_error(handle->err, da_status_invalid_input, "dim has not been defined");
    else if (result == nullptr)
        return da_error(handle->err, da_status_invalid_input,
                        "The result array has not been allocated");

    // Currently there can only be a SINGLE valid internal handle pointer,
    // so we cycle through them and query to see if the result is
    // provided by it.
    if (handle->alg_handle_d != nullptr)
        return handle->alg_handle_d->get_result(query, dim, result);

    // handle was not initialized with
    return da_error(handle->err, da_status_handle_not_initialized,
                    "The handle does not have any results to export. Have you "
                    "initialized the handle and performed any calculation?");
}

da_status da_handle_get_result_s(da_handle handle, da_result query, da_int *dim,
                                 float *result) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(handle->err, da_status_wrong_type,
                        "The handle was initialized with a different precision type than "
                        "single precision floating point type.");

    if (dim == nullptr)
        return da_error(handle->err, da_status_invalid_input, "dim has not been defined");
    else if (result == nullptr)
        return da_error(handle->err, da_status_invalid_input,
                        "The result array has not been allocated");

    // Currently there can only be a SINGLE valid internal handle pointer,
    // so we cycle through them and query to see if the result is
    // provided by it.
    if (handle->alg_handle_s != nullptr)
        return handle->alg_handle_s->get_result(query, dim, result);

    // handle was not initialized
    return da_error(handle->err, da_status_handle_not_initialized,
                    "The handle does not have any results to export. Have you "
                    "initialized the handle and performed any calculation?");
}

da_status da_handle_get_result_int(da_handle handle, da_result query, da_int *dim,
                                   da_int *result) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    if (dim == nullptr)
        return da_error(handle->err, da_status_invalid_input, "dim has not been defined");
    else if (result == nullptr)
        return da_error(handle->err, da_status_invalid_input,
                        "The result array has not been allocated");

    // Currently there can only be a SINGLE valid internal handle pointer,
    // so we cycle through them and query to see if the result is
    // provided by it.
    if (handle->alg_handle_d != nullptr)
        return handle->alg_handle_d->get_result(query, dim, result);
    else if (handle->alg_handle_s != nullptr)
        return handle->alg_handle_s->get_result(query, dim, result);

    // handle was not initialized
    return da_error(handle->err, da_status_handle_not_initialized,
                    "The handle does not have any results to export. Have you "
                    "initialized the handle and performed any calculation?");
}

da_status da_handle_get_error_message(da_handle handle, char **message) {
    // Check to see if we have a valid handle
    if (handle) {
        return handle->err->get_mesg_char(message);
    }
    return da_status_invalid_input;
}

da_status da_handle_get_error_severity(da_handle handle, da_severity *severity) {
    // Check to see if we have a valid handle
    if (handle) {
        *severity = handle->err->get_severity();
        return da_status_success;
    }
    return da_status_invalid_input;
}

void da_handle_refresh(da_handle handle) {
    if (handle) {
        if (handle->alg_handle_s != nullptr)
            handle->alg_handle_s->refresh();
        if (handle->alg_handle_d != nullptr)
            handle->alg_handle_d->refresh();
    }
}
