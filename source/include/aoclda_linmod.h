#ifndef AOCLDA_LINREG
#define AOCLDA_LINREG

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum linmod_model_ {
    linmod_model_undefined = 0,
    linmod_model_mse,
    linmod_model_logistic,
} linmod_model;

da_status da_linmod_d_select_model(da_handle handle, linmod_model mod);

da_status da_linmod_d_define_features(da_handle handle, da_int n, da_int m, double *A,
                                      double *b);

da_status da_linmod_d_fit(da_handle handle);

da_status da_linmod_d_get_coef(da_handle handle, da_int *nc, double *x);

da_status da_linmod_d_evaluate_model(da_handle handle, da_int n, da_int m, double *X,
                                     double *predictions);

da_status da_linmod_s_select_model(da_handle handle, linmod_model mod);

da_status da_linmod_s_define_features(da_handle handle, da_int n, da_int m, float *A,
                                      float *b);

da_status da_linmod_s_fit(da_handle handle);

da_status da_linmod_s_get_coef(da_handle handle, da_int *nc, float *x);

da_status da_linmod_s_evaluate_model(da_handle handle, da_int n, da_int m, float *X,
                                     float *predictions);

// Should become options
da_status da_linmod_d_set_intercept(da_handle handle, bool inter);
da_status da_linmod_s_set_intercept(da_handle handle, bool inter);

#ifdef __cplusplus
}
#endif

#endif