#include "da_handle.hpp"
#include "parser.hpp"

/* Create (and populate with defaults) */
da_status da_handle_init_d(da_handle *handle, da_handle_type handle_type) {
    try {
        *handle = new _da_handle;
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }

    (*handle)->handle_type = handle_type;
    da_status error = da_status_success;
    (*handle)->precision = da_double;

    switch (handle_type) {
    case da_handle_uninitialized:
        error = da_status_success;
        break;
    case da_handle_csv_opts:
        error = da_parser_init(&((*handle)->parser));
        break;
    case da_handle_linreg:
        try {
            (*handle)->linreg_d = new linear_model<double>();
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

    (*handle)->handle_type = handle_type;
    da_status error = da_status_success;
    (*handle)->precision = da_single;

    switch (handle_type) {
    case da_handle_uninitialized:
        error = da_status_success;
        break;
    case da_handle_csv_opts:
        error = da_parser_init(&((*handle)->parser));
        break;
    case da_handle_linreg:
        try {
            (*handle)->linreg_s = new linear_model<float>();
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

    if (handle == nullptr) return da_status_handle_not_initialized;

    if (handle->handle_type == da_handle_uninitialized) {
        snprintf(handle->error_message, ERR_MSG_LEN,
                 "The handle must be initialized before calling this routine.");
        return da_status_handle_not_initialized;
    } else if (handle->handle_type != expected_handle_type) {
        snprintf(handle->error_message, ERR_MSG_LEN,
                 "The handle has been initialized to the incorrect type.");
        //TODO: would be nice to have enum->string conversion at some point so more error info can be printed
        return da_status_invalid_handle_type;
    }

    return da_status_success;
}

void da_handle_print_error_message(da_handle handle) {

    printf("================ Error message stored in da_handle struct ================");
    printf("%s", handle->error_message);
    printf("==========================================================================");
}

/* Option setting routine */
da_status da_handle_set_option(da_handle handle, da_handle_option option, char *str) {

    da_status error = da_status_success;

    if (handle == nullptr) {
        return da_status_handle_not_initialized;
    }

    switch (handle->handle_type) {
    case da_handle_csv_opts:
        return da_parser_set_option(handle, option, str);
        break;
    case da_handle_linreg:
        error = da_status_invalid_handle_type;
        snprintf(handle->error_message, ERR_MSG_LEN,
                 "There are no options that can be set for this type of routine.");
        break;
    case da_handle_uninitialized:
        snprintf(handle->error_message, ERR_MSG_LEN,
                 "The handle must be initialized before calling this routine.");
        error = da_status_handle_not_initialized;
        break;
    }

    return error;
}

/* Destroy the da_handle struct */
void da_handle_destroy(da_handle *handle) {

    if (handle) {
        if (*handle) {
            da_parser_destroy(&(*handle)->parser);
            if ((*handle)->linreg_d)
                delete (*handle)->linreg_d;
            if ((*handle)->linreg_s)
                delete (*handle)->linreg_s;
        }
        delete (*handle);
        *handle = nullptr;
    }
}