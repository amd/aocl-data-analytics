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
            (*handle)->pca_d = new da_pca<double>();
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }
        break;
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
            (*handle)->pca_s = new da_pca<float>();
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }
        break;
    default:
        error = da_status_success;
        break;
    }

    if (!(error == da_status_success)) {
        da_handle_destroy(handle);
    }

    return error;
}

da_status da_check_handle_type(da_handle handle, da_handle_type expected_handle_type) {

    if (handle == nullptr)
        return da_status_handle_not_initialized;

    if (handle->handle_type == da_handle_uninitialized) {
        return da_error(handle->err, da_status_handle_not_initialized,
                        "The handle must be initialized before calling this routine.");
    } else if (handle->handle_type != expected_handle_type) {
        //TODO: would be nice to have enum->string conversion at some point so more error info can be printed
        return da_error(handle->err, da_status_invalid_handle_type,
                        "The handle has been initialized to the incorrect type.");
    }

    return da_status_success;
}

void da_handle_print_error_message(da_handle handle) { handle->err->print(); }

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
        // -> enable return handle->pca_d->get_result(query, dim, result);
        return da_status_not_implemented;

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
        // -> enable return handle->pca_d->get_result(query, dim, result);
        return da_status_not_implemented;

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
        // -> enable return handle->pca_d->get_result(query, dim, result);
        return da_status_not_implemented;
    else if (handle->pca_s != nullptr)
        // -> enable return handle->pca_s->get_result(query, dim, result);
        return da_status_not_implemented;

    // handle was not initialized
    return da_error(handle->err, da_status_handle_not_initialized,
                    "The handle does not have any results to export. Have you "
                    "initialized the handle and performed any calculation?");
}