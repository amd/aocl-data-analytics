/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
