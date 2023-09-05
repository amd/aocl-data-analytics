#ifndef AOCLDA_LINREG
#define AOCLDA_LINREG

/**
 * \file
 * \anchor chapter_c
 * \brief Chapter C - Linear Models
 *
 * \todo add further description of this chapter
 *
 * \section chpc_intro Introduction
 * \section chc_reg Regression
 * \subsection chc_linmod Linear models
 * \subsubsection chc_mse Mean Square Error
 * \todo Minimize square error
 * \subsubsection chc_logistic Logistic regression
 * \todo
 */

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Linear models
 */
typedef enum linmod_model_ {
    linmod_model_undefined = 0, ///< No linear model set
    linmod_model_mse, ///< fit based of mimizing mean square error (using norm \f$L_2\f$).
    linmod_model_logistic, ///< Logistic regression
} linmod_model;

da_status da_linmod_d_select_model(da_handle handle, linmod_model mod);

da_status da_linmod_d_define_features(da_handle handle, da_int n, da_int m, double *A,
                                      double *b);

/**
 * @brief Fit linear model
 *
 * @param handle
 * @return da_status
 */
da_status da_linmod_d_fit_start(da_handle handle, da_int ncoefs, double *coefs);
da_status da_linmod_d_fit(da_handle handle);

da_status da_linmod_d_evaluate_model(da_handle handle, da_int n, da_int m, double *X,
                                     double *predictions);

da_status da_linmod_s_select_model(da_handle handle, linmod_model mod);

da_status da_linmod_s_define_features(da_handle handle, da_int n, da_int m, float *A,
                                      float *b);

da_status da_linmod_s_fit_start(da_handle handle, da_int ncoefs, float *coefs);
da_status da_linmod_s_fit(da_handle handle);

da_status da_linmod_s_evaluate_model(da_handle handle, da_int n, da_int m, float *X,
                                     float *predictions);

#ifdef __cplusplus
}
#endif

#endif
