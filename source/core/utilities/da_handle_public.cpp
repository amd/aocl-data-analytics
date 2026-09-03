/*
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
template <typename T>
da_status da_handle_init(da_handle *handle, da_handle_type handle_type) {

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

    constexpr bool is_double = std::is_same_v<T, double>;
    (*handle)->precision = is_double ? da_double : da_single;

    basic_handle<T> *alg_handle = nullptr;

    da_status status = da_status_success;

    try {
        switch (handle_type) {
        case da_handle_linmod:
            DISPATCHER((*handle)->err,
                       alg_handle = new da_linmod::linear_model<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_pca:
            DISPATCHER((*handle)->err, alg_handle = new da_pca::pca<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_tsne:
            DISPATCHER((*handle)->err,
                       alg_handle = new da_tsne::tsne<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_kmeans:
            DISPATCHER((*handle)->err,
                       alg_handle = new da_kmeans::kmeans<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_dbscan:
            DISPATCHER((*handle)->err,
                       alg_handle = new da_dbscan::dbscan<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_decision_tree:
            DISPATCHER((*handle)->err,
                       alg_handle =
                           new da_decision_forest::decision_tree<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_decision_forest:
            DISPATCHER((*handle)->err,
                       alg_handle =
                           new da_decision_forest::decision_forest<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_nlls:
#ifdef NO_FORTRAN
            return da_error((*handle)->err, da_status_not_implemented, // LCOV_EXCL_LINE
                            "The nonlinear least squares solver is not available in this "
                            "implementation");
#endif
            DISPATCHER((*handle)->err,
                       alg_handle = new da_nlls::nlls<T>(status, *(*handle)->err));
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_nn:
            DISPATCHER((*handle)->err,
                       alg_handle = new da_neighbors::neighbors<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_svm:
            DISPATCHER((*handle)->err, alg_handle = new da_svm::svm<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_interpolation:
            DISPATCHER((*handle)->err,
                       alg_handle =
                           new da_interpolation::interpolation_p<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_approx_nn:
            DISPATCHER((*handle)->err,
                       alg_handle =
                           new da_approx_nn::approximate_neighbors<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
                return status;
            }
            break;
        case da_handle_kernel_pca:
            DISPATCHER((*handle)->err,
                       alg_handle = new da_kernel_pca::kernel_pca<T>(*(*handle)->err));
            status = (*handle)->err->get_status();
            if (status != da_status_success) {
                alg_handle = nullptr;
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

    if constexpr (is_double)
        (*handle)->alg_handle_d = alg_handle;
    else
        (*handle)->alg_handle_s = alg_handle;

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
template <typename T>
da_status da_handle_get_result(da_handle handle, da_result query, da_int *dim,
                               T *result) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    if (dim == nullptr)
        return da_error(handle->err, da_status_invalid_input, "dim has not been defined");
    else if (result == nullptr)
        return da_error(handle->err, da_status_invalid_input,
                        "The result array has not been allocated");

    if constexpr (std::is_same_v<da_int, T>) {
        if (handle->alg_handle_d != nullptr)
            return handle->alg_handle_d->get_result(query, dim, result);
        else if (handle->alg_handle_s != nullptr)
            return handle->alg_handle_s->get_result(query, dim, result);
    } else {
        da_status status = handle->check_precision<T>();
        if (status != da_status_success)
            return da_error_trace(handle->err, status, "Wrong precision type.");

        basic_handle<T> *alg = handle->get_alg_handle<T>();
        if (alg != nullptr) {
            return alg->get_result(query, dim, result);
        }
    }

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

/* Save (serialize) model */
da_status da_handle_save_model(da_handle handle, const char *file_name) {
    if (handle == nullptr) {
        return da_status_invalid_pointer;
    }
    if (!file_name) {
        return da_error(handle->err, da_status_invalid_pointer,
                        "file_name cannot be null.");
    }
    handle->clear(); // Clean up handle logs

    return handle->save_handle(std::string(file_name));
}

da_status da_handle_save_model(da_handle handle, std::vector<char> &buffer) {
    if (handle == nullptr) {
        return da_status_invalid_pointer;
    }
    handle->clear(); // Clean up handle logs

    return handle->save_handle(buffer);
}

/* Load (deserialize) model */
da_status da_handle_load_model(da_handle *handle, const char *buffer_data,
                               const size_t data_size) {
    if (handle == nullptr)
        return da_status_invalid_pointer;
    if (*handle != nullptr)
        return da_status_invalid_pointer;

    if (buffer_data == nullptr)
        return da_status_invalid_pointer;

    if (data_size == 0)
        return da_status_invalid_input;

    return _da_handle::load_handle(*handle, buffer_data, data_size);
}

da_status da_handle_load_model(da_handle *handle, const char *file_name) {
    if (handle == nullptr)
        return da_status_invalid_pointer;
    if (*handle != nullptr)
        return da_status_invalid_pointer;
    if (!file_name) {
        return da_status_invalid_pointer;
    }

    return _da_handle::load_handle(*handle, std::string(file_name));
}

/* Print saved versions of AOCL-DA and model serialization. */
da_status da_handle_print_model_versions(da_handle handle) {
    if (handle == nullptr)
        return da_status_invalid_pointer;
    return handle->print_model_versions();
}

template da_status da_handle_init<float>(da_handle *, da_handle_type);
template da_status da_handle_init<double>(da_handle *, da_handle_type);
template da_status da_handle_get_result<float>(da_handle, da_result, da_int *, float *);
template da_status da_handle_get_result<double>(da_handle, da_result, da_int *, double *);
template da_status da_handle_get_result<da_int>(da_handle, da_result, da_int *, da_int *);