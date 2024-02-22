/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AOCLDA_NLLS
#define AOCLDA_NLLS

/**
 * \file
 */

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

// Function callbacks

typedef da_int da_resfun_t_d(da_int, da_int, void *, double const *, double *);
typedef da_int da_resgrd_t_d(da_int, da_int, void *, double const *, double *);
typedef da_int da_reshes_t_d(da_int, da_int, void *, double const *, double const *,
                             double *);
typedef da_int da_reshp_t_d(da_int, da_int, const double *, const double *, double *,
                            void *);

typedef da_int da_resfun_t_s(da_int, da_int, void *, float const *, float *);
typedef da_int da_resgrd_t_s(da_int, da_int, void *, float const *, float *);
typedef da_int da_reshes_t_s(da_int, da_int, void *, float const *, float const *,
                             float *);
typedef da_int da_reshp_t_s(da_int, da_int, const float *, const float *, float *,
                            void *);

da_status da_nlls_define_residuals_d(da_handle handle, da_int n_coef, da_int nres,
                                     da_resfun_t_d *resfun, da_resgrd_t_d *resgrd,
                                     da_reshes_t_d *reshes, da_reshp_t_d *reshp);
da_status da_nlls_define_residuals_s(da_handle handle, da_int n_coef, da_int nres,
                                     da_resfun_t_s *resfun, da_resgrd_t_s *resgrd,
                                     da_reshes_t_s *reshes, da_reshp_t_s *reshp);

da_status da_nlls_define_bounds_d(da_handle handle, da_int n_coef, double *lower,
                                  double *upper);
da_status da_nlls_define_bounds_s(da_handle handle, da_int n_coef, float *lower,
                                  float *upper);

da_status da_nlls_define_weights_d(da_handle handle, da_int nres, double *weights);
da_status da_nlls_define_weights_s(da_handle handle, da_int nres, float *weights);

da_status da_nlls_fit_d(da_handle handle, da_int n_coef, double *coef, void *udata);
da_status da_nlls_fit_s(da_handle handle, da_int n_coef, float *coef, void *udata);

#endif
