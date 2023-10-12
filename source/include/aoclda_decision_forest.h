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

#ifndef AOCLDA_DF
#define AOCLDA_DF

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

da_status da_df_tree_set_training_data_d(da_handle handle, da_int n_obs, da_int n_features,
                                    double *x, da_int ldx, uint8_t *y);

da_status da_df_tree_fit_d(da_handle handle);
da_status da_df_tree_predict_d(da_handle handle, da_int n_obs,  double *x,
                          uint8_t *y_pred);

da_status da_df_tree_score_d(da_handle handle, da_int n_obs, double *x,
                        uint8_t *y_test, double *score);

da_status da_df_tree_set_training_data_s(da_handle handle, da_int n_obs, da_int n_features,
    float *x, da_int ldx, uint8_t *y);

da_status da_df_tree_fit_s(da_handle handle);
da_status da_df_tree_predict_s(da_handle handle, da_int n_obs,  float *x,
                          uint8_t *y_pred);

da_status da_df_tree_score_s(da_handle handle, da_int n_obs, float *x,
                        uint8_t *y_test, float *score);

// da_status da_df_sample_features_s(da_handle handle, da_int d);

// da_status da_df_bootstrap_obs_s(da_handle handle, da_int n_trees, da_int d);

da_status da_df_set_training_data_s(da_handle handle, da_int n_obs, da_int n_features,
                                    float *x, da_int ldx, uint8_t *y);

da_status da_df_fit_s(da_handle handle);

#endif
