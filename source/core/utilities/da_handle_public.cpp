/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
            (*handle)->linreg_d = new da_linmod::linear_model<double>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->linreg_d = nullptr;
                return status;
            }
            break;
        case da_handle_pca:
            (*handle)->pca_d = new da_pca::da_pca<double>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->pca_d = nullptr;
                return status;
            }
            break;
        case da_handle_kmeans:
            (*handle)->kmeans_d = new da_kmeans::da_kmeans<double>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->kmeans_d = nullptr;
                return status;
            }
            break;
        case da_handle_decision_tree:
            (*handle)->dectree_d =
                new da_decision_tree::decision_tree<double>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->dectree_d = nullptr;
                return status;
            }
            break;
        case da_handle_decision_forest:
            (*handle)->forest_d =
                new da_random_forest::random_forest<double>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->forest_d = nullptr;
                return status;
            }
            break;
        case da_handle_nlls:
            (*handle)->nlls_d = new da_nlls::nlls<double>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->nlls_d = nullptr;
                return status;
            }
            break;
        case da_handle_knn:
            (*handle)->knn_d = new da_knn::da_knn<double>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->knn_d = nullptr;
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
            (*handle)->linreg_s = new da_linmod::linear_model<float>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->linreg_s = nullptr;
                return status;
            }
            break;
        case da_handle_pca:
            (*handle)->pca_s = new da_pca::da_pca<float>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->pca_s = nullptr;
                return status;
            }
            break;
        case da_handle_kmeans:
            (*handle)->kmeans_s = new da_kmeans::da_kmeans<float>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->kmeans_s = nullptr;
                return status;
            }
            break;
        case da_handle_decision_tree:
            (*handle)->dectree_s =
                new da_decision_tree::decision_tree<float>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->dectree_s = nullptr;
                return status;
            }
            break;
        case da_handle_decision_forest:
            (*handle)->forest_s =
                new da_random_forest::random_forest<float>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->forest_s = nullptr;
                return status;
            }
            break;
        case da_handle_nlls:
            (*handle)->nlls_s = new da_nlls::nlls<float>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->nlls_s = nullptr;
                return status;
            }
            break;
        case da_handle_knn:
            (*handle)->knn_s = new da_knn::da_knn<float>(*(*handle)->err);
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                (*handle)->knn_s = nullptr;
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
            if ((*handle)->kmeans_d)
                delete (*handle)->kmeans_d;
            if ((*handle)->kmeans_s)
                delete (*handle)->kmeans_s;
            if ((*handle)->dectree_d)
                delete (*handle)->dectree_d;
            if ((*handle)->dectree_s)
                delete (*handle)->dectree_s;
            if ((*handle)->forest_d)
                delete (*handle)->forest_d;
            if ((*handle)->forest_s)
                delete (*handle)->forest_s;
            if ((*handle)->nlls_d)
                delete (*handle)->nlls_d;
            if ((*handle)->nlls_s)
                delete (*handle)->nlls_s;
            if ((*handle)->knn_d)
                delete (*handle)->knn_d;
            if ((*handle)->knn_s)
                delete (*handle)->knn_s;
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
    if (handle->linreg_d != nullptr)
        return handle->linreg_d->get_result(query, dim, result);
    else if (handle->pca_d != nullptr)
        return handle->pca_d->get_result(query, dim, result);
    else if (handle->kmeans_d != nullptr)
        return handle->kmeans_d->get_result(query, dim, result);
    else if (handle->dectree_d != nullptr)
        return handle->dectree_d->get_result(query, dim, result);
    else if (handle->forest_d != nullptr)
        return handle->forest_d->get_result(query, dim, result);
    if (handle->nlls_d != nullptr)
        return handle->nlls_d->get_result(query, dim, result);
    if (handle->knn_d != nullptr)
        return handle->knn_d->get_result(query, dim, result);

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
    if (handle->linreg_s != nullptr)
        return handle->linreg_s->get_result(query, dim, result);
    else if (handle->pca_s != nullptr)
        return handle->pca_s->get_result(query, dim, result);
    else if (handle->kmeans_s != nullptr)
        return handle->kmeans_s->get_result(query, dim, result);
    else if (handle->dectree_s != nullptr)
        return handle->dectree_s->get_result(query, dim, result);
    else if (handle->forest_s != nullptr)
        return handle->forest_s->get_result(query, dim, result);
    else if (handle->nlls_s != nullptr)
        return handle->nlls_s->get_result(query, dim, result);
    else if (handle->knn_s != nullptr)
        return handle->knn_s->get_result(query, dim, result);

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
    if (handle->linreg_d != nullptr)
        return handle->linreg_d->get_result(query, dim, result);
    else if (handle->linreg_s != nullptr)
        return handle->linreg_s->get_result(query, dim, result);
    else if (handle->pca_d != nullptr)
        return handle->pca_d->get_result(query, dim, result);
    else if (handle->pca_s != nullptr)
        return handle->pca_s->get_result(query, dim, result);
    else if (handle->dectree_d != nullptr)
        return handle->dectree_d->get_result(query, dim, result);
    else if (handle->dectree_s != nullptr)
        return handle->dectree_s->get_result(query, dim, result);
    else if (handle->forest_d != nullptr)
        return handle->forest_d->get_result(query, dim, result);
    else if (handle->forest_s != nullptr)
        return handle->forest_s->get_result(query, dim, result);

    else if (handle->kmeans_d != nullptr)
        return handle->kmeans_d->get_result(query, dim, result);
    else if (handle->kmeans_s != nullptr)
        return handle->kmeans_s->get_result(query, dim, result);
    if (handle->knn_d != nullptr)
        return handle->knn_d->get_result(query, dim, result);
    else if (handle->knn_s != nullptr)
        return handle->knn_s->get_result(query, dim, result);
    if (handle->nlls_d != nullptr)
        return handle->nlls_d->get_result(query, dim, result);
    else if (handle->nlls_s != nullptr)
        return handle->nlls_s->get_result(query, dim, result);

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
