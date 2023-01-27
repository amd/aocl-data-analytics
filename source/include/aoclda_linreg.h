#ifndef AOCLDA_LINREG
#define AOCLDA_LINREG

#include "aoclda_error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum linreg_model_ {
    linreg_model_undefined = 0,
    linreg_model_mse,
    linreg_model_logistic,
} linreg_model;

typedef struct _da_linreg *da_linreg;

da_status da_linreg_d_init(da_linreg *handle);

void da_linreg_destroy(da_linreg *handle);

da_status da_linreg_d_select_model(da_linreg handle, linreg_model mod);

da_status da_linreg_d_define_features(da_linreg handle, da_int n, da_int m, double *A,
                                      double *b);

da_status da_linreg_d_fit(da_linreg handle);

da_status da_linreg_d_get_coef(da_linreg handle, da_int *nc, double *x);

da_status da_linreg_d_evaluate_model(da_linreg handle, da_int n, da_int m, double *X,
                                     double *predictions);

da_status da_linreg_s_init(da_linreg *handle);

void da_linreg_destroy(da_linreg *handle);

da_status da_linreg_s_select_model(da_linreg handle, linreg_model mod);

da_status da_linreg_s_define_features(da_linreg handle, da_int n, da_int m, float *A, float *b);

da_status da_linreg_s_fit(da_linreg handle);

da_status da_linreg_s_get_coef(da_linreg handle, da_int *nc, float *x);

da_status da_linreg_s_evaluate_model(da_linreg handle, da_int n, da_int m, float *X,
                                     float *predictions);

#ifdef __cplusplus
}
#endif

#endif