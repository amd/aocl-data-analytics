/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * The purpose of this test is to check C compatibility of the aocl-da interfaces.
 * The test should be compiled using a C compiler and linked with a C++ compiler.
 * The contents of the test are largely irrelevant - the important check is that with a C
 * compiler we can successfully include aoclda.h.
 */

#include "aoclda.h"
#include "math.h"

da_status test_linmod(void) {
    // problem data
    da_int m = 5, n = 2;
    double Al[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bl[5] = {1, 1, 1, 1, 1};

    /* Initialize a linear regression */
    da_handle handle = NULL;
    da_status status;
    status = da_handle_init_d(&handle, da_handle_linmod);
    if (status != da_status_success) {
        da_handle_destroy(&handle);
        return da_status_handle_not_initialized;
    }
    status = da_linmod_select_model_d(handle, linmod_model_mse);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }
    status = da_linmod_define_features_d(handle, m, n, Al, m, bl);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }
    /* Compute regression */
    status = da_linmod_fit_d(handle);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }
    da_handle_destroy(&handle);
    return da_status_success;
}

struct c_cb_params_type {
    double *t; /* The m data points t_i */
    double *y; /* The m data points y_i */
};

/* Calculate r_i(x; t_i, y_i) = x_1 e^(x_2 * t_i) - y_i */
da_int c_cb_eval_r(da_int n, da_int m, void *params, double const *x, double *r) {
    double x1 = x[0];
    double x2 = x[1];
    double const *t = ((struct c_cb_params_type *)params)->t;
    double const *y = ((struct c_cb_params_type *)params)->y;

    for (da_int i = 0; i < m; i++)
        r[i] = x1 * exp(x2 * t[i]) - y[i];

    return 0;
}

/* Calculate:
 * J_i1 = e^(x_2 * t_i)
 * J_i2 = t_i x_1 e^(x_2 * t_i)
 */
da_int c_cb_eval_J(da_int n, da_int m, void *params, double const *x, double *J) {
    double x1 = x[0];
    double x2 = x[1];
    double const *t = ((struct c_cb_params_type *)params)->t;

    for (da_int i = 0; i < m; i++) {
        J[0 * m + i] = exp(x2 * t[i]);             /* J_i1 */
        J[1 * m + i] = t[i] * x1 * exp(x2 * t[i]); /* J_i2 */
    }

    return 0;
}

/* Calculate:
 * HF = sum_i r_i H_i
 * Where H_i = [ 1                t_i e^(x_2 t_i)    ]
 *             [ t_i e^(x_2 t_i)  t_i^2 e^(x_2 t_i)  ]
 */
da_int c_cb_eval_HF(da_int n, da_int m, void *params, double const *x, double const *r,
                    double *HF) {
    double x1 = x[0];
    double x2 = x[1];
    double const *t = ((struct c_cb_params_type *)params)->t;

    for (da_int i = 0; i < n * n; i++)
        HF[i] = 0.0;
    for (da_int i = 0; i < m; i++) {
        HF[0] += 0;                                                /* H_11 */
        HF[1] += r[i] * t[i] * exp(x2 * t[i]);                     /* H_21 */
        HF[1 * n + 1] += r[i] * t[i] * t[i] * x1 * exp(x2 * t[i]); /* H_22 */
    }
    HF[1 * n + 0] = HF[1]; /* H_12 by symmetry of Hessian */

    return 0;
}

da_status test_nlls(void) {
    da_handle handle = NULL;
    da_status status = da_status_success;
    double lower_bounds[2] = {0.0, 1.0};
    double upper_bounds[2] = {1.0, 10.0};
    double x[2] = {0.001, 0.9}; // Initial guess
    double info[100];
    da_int dim = 100;
    struct c_cb_params_type params = {.t = (double[]){1.0, 2.0, 4.0, 5.0, 8.0},
                                      .y = (double[]){3.0, 4.0, 6.0, 11.0, 20.0}};

    status = da_handle_init_d(&handle, da_handle_nlls);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }
    status = da_nlls_define_residuals_d(handle, 2, 5, c_cb_eval_r, c_cb_eval_J,
                                        c_cb_eval_HF, NULL);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }
    status = da_nlls_define_bounds_d(handle, 2, lower_bounds, upper_bounds);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }
    status = da_options_set_int(handle, "print level", (da_int)2);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }
    status = da_options_set_string(handle, "Storage Order", "Fortran");
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }
    status = da_options_set_real_d(handle, "ralfit convergence abs tol grd", 2.0e-4);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }
    status = da_options_set_real_d(handle, "ralfit convergence rel tol grd", 2.0e-4);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }
    status = da_nlls_fit_d(handle, 2, x, &params);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }

    status = da_handle_get_result_d(handle, da_rinfo, &dim, info);
    if (status != da_status_success) {
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
        return status;
    }

    if (info[2] < 2 || info[0] > 90 || info[1] > 1) {
        status = da_status_incorrect_output;
    }

    da_handle_destroy(&handle);
    return status;
}

int main(void) {
    da_status status;

    status = test_linmod();
    if (status != da_status_success) {
        return 1;
    }

#ifndef NO_FORTRAN
    status = test_nlls();
    if (status != da_status_success) {
        return 2;
    }
#endif
    return 0;
}
