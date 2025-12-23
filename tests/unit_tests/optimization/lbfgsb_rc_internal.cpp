/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../utest_utils.hpp"
#include "lbfgsb_driver.hpp"
#include <cmath>
#include <iostream>

#ifndef NO_FORTRAN

using namespace TEST_ARCH;

template <typename T> int solve(void) {
    da_int n = 2, m = 2, iprint = 0;
    T factr = static_cast<T>(1.0e-1), pgtol = static_cast<T>(1.0e-05);

    da_int itask, lsavei[4], isave[44];
    T f, dsave[29];
    da_int *nbd = nullptr, *iwa = nullptr;
    T *x = nullptr, *l = nullptr, *u = nullptr, *g = nullptr, *wa = nullptr;

    da_int i;

    nbd = (da_int *)malloc(n * sizeof(da_int));
    x = (T *)malloc(n * sizeof(T));
    g = (T *)malloc(n * sizeof(T));
    l = (T *)malloc(n * sizeof(T));
    u = (T *)malloc(n * sizeof(T));
    iwa = (da_int *)malloc(3 * n * sizeof(da_int));
    da_int nwa = 2 * m * n + 5 * n + 11 * m * m + 8 * m;
    wa = (T *)malloc(nwa * sizeof(T));

    // Set bounds
    for (i = 0; i < n; i += 2) {
        nbd[i] = 2;
        l[i] = 1.0;
        u[i] = 100.0;
    }
    for (i = 1; i < n; i += 2) {
        nbd[i] = 2;
        l[i] = -100.0;
        u[i] = 100.0;
    }

    // Starting point
    for (i = 0; i < n; i++)
        x[i] = 3.0;

    std::cout << "Solving sample problem." << std::endl
              << "(f = 0.0 at the optimal solution x = [1, ..., 1].)" << std::endl;

    itask = 2;

    /* task = 'START => itask = 2
     * TASK = 'NEW_X' => itask = 1
     */
    bool compute_fg = true;
    while (itask == 2 || itask == 1 || compute_fg) {
        lbfgsb_rcomm(&n, &m, x, l, u, nbd, &f, g, &factr, &pgtol, wa, iwa, &itask,
                     &iprint, lsavei, isave, dsave);
        compute_fg = itask == 4 ||  // 'FG'
                     itask == 21 || // 'FG_START'
                     itask == 20;   // 'FG_LNSRCH
        if (compute_fg) {
            f = pow((T)1.0 - x[0], (T)2.0);
            f += 100 * pow(x[1] - x[0] * x[0], (T)2.0);

            // compute gradient
            g[0] = 2 * (x[0] - 1) - 400 * (x[1] - x[0] * x[0]) * x[0];
            g[1] = 200 * (x[1] - x[0] * x[0]);
        }
    }
    std::cout << std::endl;
    std::cout << "Solver working precision ID: " << isave[43] << std::endl;

    std::cout << "Final solution, f = " << f << std::endl;
    std::cout << "x = [" << x[0] << ", " << x[1] << "]" << std::endl;

    // Check for solution
    T lierr = 0, err;
    for (i = 0; i < n; i++) {
        err = fabs(x[i] - 1);
        if (lierr < err)
            lierr = err;
    }
    da_int status = lierr > 500 * pgtol;

    if (nbd)
        free(nbd);
    if (iwa)
        free(iwa);
    if (wa)
        free(wa);
    if (x)
        free(x);
    if (l)
        free(l);
    if (u)
        free(u);
    if (g)
        free(g);

    return status;
}

int main(void) {
    int status;
    status = solve<double>();
    status += solve<float>();
    return status;
}

#endif