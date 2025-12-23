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

/* Example on how to call internally the NLP solver(s) */

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "da_error.hpp"
#include "optimization.hpp"
#include <cmath>
#include <iostream>

#ifndef NO_FORTRAN

using namespace TEST_ARCH;

da_int objfun(da_int n [[maybe_unused]], double *x, double *val, void *usrdata) {
    double *params = (double *)usrdata;
    double a = params[0];
    double b = params[1];

    double ax2 = (a - x[0]) * (a - x[0]);
    double xy = x[1] - x[0] * x[0];

    *val = ax2 + b * xy * xy;

    return 0;
}

da_int objgrd(da_int n [[maybe_unused]], double *x, double *val, void *usrdata,
              da_int xnew [[maybe_unused]]) {
    double *params = (double *)usrdata;
    double a = params[0];
    double b = params[1];

    double m2ax = 2.0 * (x[0] - a);
    double xy = x[0] * x[0] - x[1];

    *(val + 0) = m2ax + 4.0 * b * x[0] * xy;
    *(val + 1) = -2.0 * b * xy;
    return 0;
}

da_int monit(da_int n, double *x, double *val, double *info,
             void *usrdata [[maybe_unused]]) {
    if (info[da_linmod_info_t::linmod_info_iter] <= 1)
        std::cout << "Iter objective gradient x[0] g[n]" << std::endl;
    std::cout << (size_t)info[da_linmod_info_t::linmod_info_iter] << " "
              << info[da_linmod_info_t::linmod_info_objective] << " "
              << info[da_linmod_info_t::linmod_info_grad_norm] << " " << x[0] << " "
              << val[n - 1] << std::endl;

    if (info[da_linmod_info_t::linmod_info_iter] == 3)
        return -1;
    return 0;
}

int main(void) {
    da_int exit_status = 1;
    da_status status;
    double params[2] = {1.0, 100.0}; // parameter to pass to the call-backs
    const da_int n = 2;
    std::vector<double> l(n, -5.0);
    std::vector<double> u(n, 5.0);
    double tol = 1.0e-7;
    std::vector<double> x(2);
    const std::vector<double> xref(2, 1.0);
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    const da_int mon[2] = {10, 1};

    da_optim::da_optimization<double> *pd =
        new da_optim::da_optimization<double>(status, err);
    da_options::OptionRegistry opts = pd->get_opts();

    if (status != da_status_success) {
        pd = nullptr;
        goto abort;
    }
    status = da_status_internal_error;
    // build the problem to solve
    // Add variables
    if (pd->add_vars(n) != da_status_success)
        goto abort;
    // Add bound constraints
    if (pd->add_bound_cons(l, u) != da_status_success)
        goto abort;
    // Add Objective function call-back, returns f(x);
    if (pd->add_objfun(objfun) != da_status_success)
        goto abort;
    // Add Objective gradient function call-back, returns grad f(x);
    if (pd->add_objgrd(objgrd) != da_status_success)
        goto abort;
    // Optionally add monitor
    if (pd->add_monit(monit) != da_status_success)
        goto abort;
    // Optionally set up options
    if (pd->opts.set("Print Options", "yes") != da_status_success)
        goto abort;
    if (pd->opts.set("Print Level", (da_int)0) != da_status_success)
        goto abort;
    if (pd->opts.set("LBFGSB Convergence Tol", tol) != da_status_success)
        goto abort;
    if (pd->opts.set("LBfgSB Iteration Limit", (da_int)31) != da_status_success)
        goto abort;
    if (pd->opts.set("time limit", 100.1) != da_status_success)
        goto abort;
    if (pd->opts.set("LBfgSB memory Limit", (da_int)12) != da_status_success)
        goto abort;

    for (int i = 0; i < 2; ++i) {
        if (pd->opts.set("Monitoring Frequency", mon[i]) != da_status_success)
            goto abort;
        for (int j = 0; j < 2; ++j)
            x[i] = 0.0;
        // Ready to solve
        status = pd->solve(x, (void *)params);
        // make sure to check the return status (pd.err error structure contains the details)
        // some return codes provide a usable solution.

        // status that provide usable solutions:
        if (status == da_status_success || status == da_status_optimization_usrstop ||
            status == da_status_numerical_difficulties) {
            // solution is potentially OK, check...
            bool ok = true;
            for (da_int i = 0; i < n; i++)
                ok &= std::fabs(x[i] - xref[i]) <= 10.0 * tol;
            if (ok) {
                // operation was successful, x holds the solution
                std::cout << "Solution found: " << x[0] << ", " << x[1] << std::endl;
                exit_status = 0; // operation was successful
            } else {
                // fill error trace (e.g. unexpected)
                status = da_error(pd->err, da_status_internal_error,
                                  "Expecting the correct solution point");
            }
        }
    }

abort:
    if (status != da_status_success)
        std::cout << "status: " << status << std::endl;
    if (pd) {
        (*pd).err->print(); // print error trace
        // delete data
        delete pd;
    }

    return exit_status;
}

#endif