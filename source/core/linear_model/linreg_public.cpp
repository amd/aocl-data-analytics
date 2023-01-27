#include "aoclda.h"
#include "linear_model_data.hpp"

da_status da_linreg_d_init(da_linreg *handle) {
    try {
        *handle = new _da_linreg;
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }

    try {
        (*handle)->linreg_d = new linear_model_data<double>();
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }
    (*handle)->prec = da_double;

    return da_status_success;
}

da_status da_linreg_s_init(da_linreg *handle) {
    try {
        *handle = new _da_linreg;
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }

    try {
        (*handle)->linreg_s = new linear_model_data<float>();
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }
    (*handle)->prec = da_single;

    return da_status_success;
}

void da_linreg_destroy(da_linreg *handle) {
    if (handle) {
        if (*handle) {
            if ((*handle)->linreg_d)
                delete (*handle)->linreg_d;
            if ((*handle)->linreg_s)
                delete (*handle)->linreg_s;
            delete *handle;
            *handle = nullptr;
        }
    }
}

da_status da_linreg_d_select_model(da_linreg handle, linreg_model mod) {
    if (handle == nullptr)
        return da_status_invalid_pointer;
    if (handle->prec != da_double)
        return da_status_wrong_type;

    return handle->linreg_d->select_model(mod);
}

da_status da_linreg_s_select_model(da_linreg handle, linreg_model mod) {
    if (handle == nullptr)
        return da_status_invalid_pointer;
    if (handle->prec != da_single)
        return da_status_wrong_type;

    return handle->linreg_s->select_model(mod);
}

da_status da_linreg_d_define_features(da_linreg handle, da_int n, da_int m, double *A,
                                      double *b) {
    if (handle == nullptr)
        return da_status_invalid_pointer;
    if (handle->prec != da_double)
        return da_status_wrong_type;

    return handle->linreg_d->define_features(n, m, A, b);
}

da_status da_linreg_s_define_features(da_linreg handle, da_int n, da_int m, float *A,
                                      float *b) {
    if (handle == nullptr)
        return da_status_invalid_pointer;
    if (handle->prec != da_single)
        return da_status_wrong_type;

    return handle->linreg_s->define_features(n, m, A, b);
}

da_status da_linreg_d_fit(da_linreg handle) {
    if (handle == nullptr)
        return da_status_invalid_pointer;
    if (handle->prec != da_double)
        return da_status_wrong_type;

    return handle->linreg_d->fit();
}

da_status da_linreg_s_fit(da_linreg handle) {
    if (handle == nullptr)
        return da_status_invalid_pointer;
    if (handle->prec != da_single)
        return da_status_wrong_type;
    return handle->linreg_s->fit();
}

da_status da_linreg_d_get_coef(da_linreg handle, da_int *nc, double *x) {
    if (handle == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_d->get_coef(*nc, x);
}

da_status da_linreg_s_get_coef(da_linreg handle, da_int *nc, float *x) {
    if (handle == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_s->get_coef(*nc, x);
}

da_status da_linreg_d_evaluate_model(da_linreg handle, da_int n, da_int m, double *X,
                                     double *predictions) {
    if (handle == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_d->evaluate_model(n, m, X, predictions);
}

da_status da_linreg_s_evaluate_model(da_linreg handle, da_int n, da_int m, float *X,
                                     float *predictions) {
    if (handle == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_s->evaluate_model(n, m, X, predictions);
}