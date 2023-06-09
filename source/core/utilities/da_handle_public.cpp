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
