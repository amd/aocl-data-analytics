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

// Example to fit a reduced Lanczos model f(x1, x2) = x1 exp(-t x2)

#include "aoclda.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

const da_int n_res = 5;
const da_int n_coef = 2;

struct udata_t {
    const float *t;
    const float *y;
};
const float t[n_res]{0.25f, 0.5f, 0.75f, 0.4f, 0.66f};
const float y[n_res]{0.60f, 0.368f, 0.22f, 0.45f, 0.26f};
const struct udata_t udata = {t, y};

da_int eval_r(da_int n_coef, da_int n_res, void *udata, float const *x, float *r) {
    float const *t = ((struct udata_t *)udata)->t;
    float const *y = ((struct udata_t *)udata)->y;

    // r_i = y_i - x_1 e^(-x_2 t_i)
    for (da_int i = 0; i < n_res; i++)
        r[i] = y[i] - x[0] * exp(-x[n_coef - 1] * t[i]);

    return 0;
}

int main(void) {
    std::cout << "--------------------------------------------------------------------"
              << std::endl;
    std::cout << " Nonlinear Least-Squares basic (reduced precision) example"
              << std::endl;
    std::cout << "--------------------------------------------------------------------"
              << std::endl;

    float coef[n_coef]{0.f, 1.f};
    const float coef_exp[n_coef]{1.0f, 2.0f};

    const float tol{2.0e-2};

    // Initialize handle for nonlinear regression
    da_handle handle = nullptr;

    bool pass = true;
    pass &= da_handle_init_s(&handle, da_handle_nlls) == da_status_success;
    pass &= da_nlls_define_residuals_s(handle, n_coef, n_res, eval_r, nullptr, nullptr,
                                       nullptr) == da_status_success;
    if (!pass) {
        std::cout << "Something unexpected happened in the model definition\n";
        da_handle_destroy(&handle);
        return 1;
    }

    pass &= da_options_set_string(handle, "ralfit globalization method",
                                  "regularization") == da_status_success;
    pass &=
        da_options_set_string(handle, "storage order", "row-major") == da_status_success;
    pass &= da_options_set_int(handle, "ralfit iteration limit", (da_int)200) ==
            da_status_success;
    pass &= da_options_set_real_s(handle, "finite differences step", 2e-4f) ==
            da_status_success;
    pass &= da_options_set_real_s(handle, "ralfit convergence abs tol grd", 1e-5f) ==
            da_status_success;
    pass &= da_options_set_real_s(handle, "ralfit convergence rel tol grd", 1e-8f) ==
            da_status_success;
    if (!pass) {
        std::cout << "Something unexpected happened while setting options\n";
        da_handle_destroy(&handle);
        return 2;
    }

    // Compute regression
    da_status status;
    status = da_nlls_fit_s(handle, n_coef, coef, (void *)&udata);
    bool ok{false};
    if (status == da_status_success) {
        const auto default_precision{std::cout.precision()};
        std::cout << "Regression computed successfully!" << std::endl;
        std::cout << "Coefficients: Idx           x            x*\n";
        ok = true;
        for (auto i = 0; i < n_coef; ++i) {
            float gap = std::abs(coef[i] - coef_exp[i]);
            bool oki = gap < tol;
            ok &= oki;
            std::cout << std::setprecision(4);
            std::cout << "                " << i << std::setw(12) << coef[i] << " "
                      << std::setw(1) << " " << std::setw(12) << coef_exp[i]
                      << std::setw(8) << (oki ? "PASS (" : "FAIL (") << std::setw(8)
                      << std::setprecision(3) << gap << std::setw(1) << ")\n";
        }
        std::cout << std::setprecision(3);
        std::cout << std::setprecision(default_precision);
    } else {
        std::cout << "Something wrong happened during the fit. Terminating. Message:"
                  << std::endl;
        char *mesg{nullptr};
        da_handle_get_error_message(handle, &mesg);
        std::cout << mesg << std::endl;
        free(mesg);
        da_handle_destroy(&handle);
        return 3;
    }

    // Get info out of handle
    std::vector<float> info(1);
    da_int size = info.size();
    status = da_handle_get_result_s(handle, da_result::da_rinfo, &size, info.data());
    if (status == da_status_operation_failed) {
        info.resize(size);
    }
    status = da_handle_get_result_s(handle, da_result::da_rinfo, &size, info.data());
    if (status == da_status_success) {
        std::cout << "Fit error                      : " << info[0] << std::endl;
        std::cout << "Norm of residual gradient      : " << info[1] << std::endl;
        std::cout << "Objective fun calls            : " << info[4] << std::endl;
        std::cout << "Objective fun calls (fin diff) : " << info[12] << std::endl;
    }

    da_handle_destroy(&handle);

    return ok ? 0 : 9;
}
