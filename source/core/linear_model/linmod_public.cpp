#include "aoclda.h"
#include "da_handle.hpp"
#include "linear_model.hpp"

da_status da_linmod_d_select_model(da_handle handle, linmod_model mod) {
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->linreg_d == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_d->select_model(mod);
}

da_status da_linmod_s_select_model(da_handle handle, linmod_model mod) {
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->linreg_s == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_s->select_model(mod);
}

da_status da_linmod_d_define_features(da_handle handle, da_int n, da_int m, double *A,
                                      double *b) {
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->linreg_d == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_d->define_features(n, m, A, b);
}

da_status da_linmod_s_define_features(da_handle handle, da_int n, da_int m, float *A,
                                      float *b) {
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->linreg_s == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_s->define_features(n, m, A, b);
}

da_status da_linmod_d_fit(da_handle handle) {
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->linreg_d == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_d->fit();
}

da_status da_linmod_s_fit(da_handle handle) {
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->linreg_s == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_s->fit();
}

da_status da_linmod_d_get_coef(da_handle handle, da_int *nc, double *x) {
    if (handle->linreg_d == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_d->get_coef(*nc, x);
}

da_status da_linmod_s_get_coef(da_handle handle, da_int *nc, float *x) {
    if (handle->linreg_s == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_s->get_coef(*nc, x);
}

da_status da_linmod_d_evaluate_model(da_handle handle, da_int n, da_int m, double *X,
                                     double *predictions) {
    if (handle->linreg_d == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_d->evaluate_model(n, m, X, predictions);
}

da_status da_linmod_s_evaluate_model(da_handle handle, da_int n, da_int m, float *X,
                                     float *predictions) {
    if (handle->linreg_s == nullptr)
        return da_status_invalid_pointer;

    return handle->linreg_s->evaluate_model(n, m, X, predictions);
}

/* To be removed when option setters are done */
da_status da_linmod_d_set_intercept(da_handle handle, bool inter) {
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->linreg_d == nullptr)
        return da_status_invalid_pointer;

    handle->linreg_d->set_intercept(inter);
    return da_status_success;
}
da_status da_linmod_s_set_intercept(da_handle handle, bool inter) {
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->linreg_s == nullptr)
        return da_status_invalid_pointer;

    handle->linreg_s->set_intercept(inter);
    return da_status_success;
}
