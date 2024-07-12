/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#include "aoclda.h"
#include <math.h>
#include <vector>

namespace template_nlls_cb_errors {
template <typename T>
da_int eval_r_fail(da_int n, da_int m, void *params, T const *x, T *r) {
    return 1; // fail...
}
} // namespace template_nlls_cb_errors

namespace template_nlls_example_box_c {

template <typename T> struct params_type {
    T *t; // The m data points t_i
    T *y; // The m data points y_i
    da_int fcnt{100000}, jcnt{100000};
};

// Calculate r_i(x; t_i, y_i) = x_1 e^(x_2 * t_i) - y_i
template <typename T> da_int eval_r(da_int n, da_int m, void *params, T const *x, T *r) {
    T x1 = x[0];
    T x2 = x[1];
    T const *t = ((struct params_type<T> *)params)->t;
    T const *y = ((struct params_type<T> *)params)->y;
    static da_int count_down{0};
    da_int fcnt = ((struct params_type<T> *)params)->fcnt;

    if (fcnt >= 0) {
        count_down = fcnt;
        ((struct params_type<T> *)params)->fcnt = -1;
    }
    if (count_down-- <= 0) {
        return 1;
    }
    for (da_int i = 0; i < m; i++)
        r[i] = x1 * exp(x2 * t[i]) - y[i];

    return 0; // Success
}

// Calculate:
// J_i1 = e^(x_2 * t_i)
// J_i2 = t_i x_1 e^(x_2 * t_i)
template <typename T> da_int eval_J(da_int n, da_int m, void *params, T const *x, T *J) {
    T x1 = x[0];
    T x2 = x[1];
    T const *t = ((struct params_type<T> *)params)->t;

    for (da_int i = 0; i < m; i++) {
        J[0 * m + i] = exp(x2 * t[i]);             // J_i1
        J[1 * m + i] = t[i] * x1 * exp(x2 * t[i]); // J_i2
    }

    return 0; // Success
}

// User Stop...
template <typename T>
da_int eval_J_wrong(da_int n, da_int m, void *params, T const *x, T *J) {
    T x1 = x[0];
    T x2 = x[1];
    T const *t = ((struct params_type<T> *)params)->t;
    static da_int count_down{0};
    da_int jcnt = ((struct params_type<T> *)params)->jcnt;

    if (jcnt >= 0) {
        count_down = jcnt;
        ((struct params_type<T> *)params)->jcnt = -1;
    }
    if (count_down-- <= 0) {
        return 1;
    }
    for (da_int i = 0; i < m; i++) {
        J[0 * m + i] = exp(x2 * t[i]);
        J[1 * m + i] = t[i] * x1 * exp(x2 * t[i]);
    }
    return 0; // Success
}

// Num difficulties...
template <typename T>
da_int eval_J_bad(da_int n, da_int m, void *params, T const *x, T *J) {
    T x1 = x[0];
    T x2 = x[1];
    T const *t = ((struct params_type<T> *)params)->t;

    for (da_int i = 0; i < m; i++) {
        J[0 * m + i] = exp(x2 * t[i]) + x2 * x2;
        J[1 * m + i] = t[i] * x1 * exp(x2 * t[i]) + x1 * x2;
    }
    return 0; // Success
}

// Calculate:
// HF = sum_i r_i H_i
// Where H_i = [ 1                t_i e^(x_2 t_i)    ]
//             [ t_i e^(x_2 t_i)  t_i^2 e^(x_2 t_i)  ]
template <typename T>
da_int eval_HF(da_int n, da_int m, void *params, T const *x, T const *r, T *HF) {
    T x1 = x[0];
    T x2 = x[1];
    T const *t = ((struct params_type<T> *)params)->t;

    for (da_int i = 0; i < n * n; i++)
        HF[i] = T(0);
    for (da_int i = 0; i < m; i++) {
        HF[0] += T(0);                                             // H_11
        HF[1] += r[i] * t[i] * exp(x2 * t[i]);                     // H_21
        HF[1 * n + 1] += r[i] * t[i] * t[i] * x1 * exp(x2 * t[i]); // H_22
    }
    HF[1 * n + 0] = HF[1]; // H_12 by symmetry of Hessian

    return 0; // Success
}

} // namespace template_nlls_example_box_c

namespace template_lm_example_c {
struct usertype {
    double *sigma;
    double *y;
};

da_int eval_r(da_int n, da_int m, void *params, double const *x, double *r) {
    double *y = ((struct usertype *)params)->y;
    double *sigma = ((struct usertype *)params)->sigma;
    double A{x[0]};
    double lambda{x[1]};
    double b{x[2]};

    for (da_int i = 0; i < m; i++) {
        /* Model Yi = A * exp(-lambda * i) + b */
        double t = i;
        double Yi = A * exp(-lambda * t) + b;
        r[i] = (Yi - y[i]) / sigma[i];
    }
    return 0;
}

da_int eval_J(da_int n, da_int m, void *params, double const *x, double *J) {
    double *sigma = ((struct usertype *)params)->sigma;
    double A{x[0]};
    double lambda{x[1]};

    for (da_int i = 0; i < m; i++) {
        /* Jacobian matrix J(i,j) = dfi / dxj, */
        /* where fi = (Yi - yi)/sigma[i],      */
        /*       Yi = A * exp(-lambda * i) + b  */
        /* and the xj are the parameters (A,lambda,b) */
        double t = i;
        double s = sigma[i];
        double e = exp(-lambda * t);
        J[n * i + 0] = e / s;
        J[n * i + 1] = -t * A * e / s;
        J[n * i + 2] = 1 / s;
    }
    return 0;
}

da_int eval_J_bad(da_int n, da_int m, void *params, double const *x, double *J) {
    double *sigma = ((struct usertype *)params)->sigma;
    double A{x[0]};
    double lambda{x[1]};

    for (da_int i = 0; i < m; i++) {
        /* Jacobian matrix J(i,j) = dfi / dxj, */
        /* where fi = (Yi - yi)/sigma[i],      */
        /*       Yi = A * exp(-lambda * i) + b  */
        /* and the xj are the parameters (A,lambda,b) */
        double t = i;
        double s = sigma[i];
        double e = exp(-lambda * t);
        J[n * i + 0] = -e / s;
        J[n * i + 1] = -t * A * e / s;
        J[n * i + 2] = 1 / s;
    }
    return 0;
}
} // namespace template_lm_example_c

namespace template_nlls_example_box_fortran {
struct udata_t {
    const double *t;
    const double *y;
};

da_int eval_r(da_int n_coef, da_int n_res, void *udata, double const *x, double *r) {
    double x1 = x[0];
    double x2 = x[n_coef - 1];
    double const *t = ((struct udata_t *)udata)->t;
    double const *y = ((struct udata_t *)udata)->y;

    for (da_int i = 0; i < n_res; i++)
        r[i] = x1 * exp(x2 * t[i]) - y[i];

    return 0;
}
} // namespace template_nlls_example_box_fortran