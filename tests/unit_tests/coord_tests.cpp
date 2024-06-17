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

#include "algorithm"
#include "aoclda.h"
#include "da_handle.hpp"
#include "options.hpp"
#include "utest_utils.hpp"
#include "gtest/gtest.h"

#include "callbacks.hpp"
#include "coord.hpp"
#include "da_error.hpp"
#include "optimization_options.hpp"

using T = double;
da_int stepfun_cycleend(da_int n, T *x, T *newxk, da_int k, T *f, void *usrdata,
                        da_int action, [[maybe_unused]] T kdiff) {
    /* Actions regarding feature matrix evaluation
     * action < 0 means that feature matrix was previously called and that only a low rank
     *            update is requested and -(action+1) contains the previous k that changed
     *            kold = -(action+1);
     * action = 0 means not to evaluate the feature matrix (restore matvec from aux)
     * action > 0 evaluate the matrix.
     *
     * This function:
     * calls = usrdata
     * for calls < nfeat => "good progress" => x[k] <- x[k]/2
     * for calls >= nfeat => "poor progress"
     * for action > 0 => calls is reset to 0
     */

    if (f) {
        *f = 0;
        for (auto i = 0; i < n; i++) {
            *f += std::abs(x[i]);
        }
        return 0;
    }

    da_int *calls = nullptr;

    if (!action)
        return 0;

    if (!usrdata)
        return 1;

    calls = static_cast<da_int *>(usrdata);

    if (action > 0)
        *calls = 0U;

    // if (*calls < n - 3 || *calls >= 50 - 3)
    *newxk = x[k] / 2;
    // else
    //   *newxk = x[k] - static_cast<T>(1.0e-7);

    ++(*calls);

    calls = nullptr;
    return 0;
}

// Verify that the end-cycle logic is correct
TEST(Coord, CycleEnd) {
    da_errors::da_error_t err(da_errors::DA_THROW);
    da_options::OptionRegistry opts;
    da_status status;
    status = register_optimization_options<T>(err, opts);
    EXPECT_EQ(status, da_status_success) << "error from register_optiomization_options()";
    const da_int n = 10;
    std::vector<double> x(n, 10.0), l(0, 0.0), u(0, 0.0), info(100, 0.0);
    std::transform(x.begin(), x.begin() + 5, x.begin(), [](double y) { return y / 2; });
    da_int calls = 0; // must match cb calls type
    void *usrdata = &calls;
    monit_t<T> monit = nullptr;
    const T tol = da_numeric::tolerance<T>::safe_tol();
    const T inorm_init{10.0};
    const T ftol = da_numeric::tolerance<T>::tol(10);

    status = opts.set("print level", da_int(5), da_options::setby_t::user);
    EXPECT_EQ(status, da_status_success) << "error setting print level";
    status = opts.set("coord skip tol", 1.0e-6, da_options::setby_t::user);
    EXPECT_EQ(status, da_status_success) << "error setting coord skip tol";
    status = opts.set("coord skip min", da_int(2), da_options::setby_t::user);
    EXPECT_EQ(status, da_status_success) << "error setting coord skip min";
    status = opts.set("coord convergence tol", 1.0e-8, da_options::setby_t::user);
    EXPECT_EQ(status, da_status_success) << "error setting coord convergence tol";
    status = opts.set("coord restart", da_int(10), da_options::setby_t::user);
    EXPECT_EQ(status, da_status_success) << "error setting coord restart";
    status = opts.set("coord iteration limit", da_int(1500), da_options::setby_t::user);
    EXPECT_EQ(status, da_status_success) << "error setting coord iteration limit";
    opts.print_options();
    status = coord::coord(opts, n, x, l, u, info, stepfun_cycleend, monit, usrdata, err);
    EXPECT_EQ(status, da_status_success) << "error from coord";

    // Check info array
    // time
    EXPECT_GT(info[info_t::info_time], T(0));
    // iter
    EXPECT_GT(info[info_t::info_iter], T(28));
    EXPECT_LT(info[info_t::info_iter], T(32));
    // expensive
    EXPECT_GT(info[info_t::info_nevalf], T(28));
    EXPECT_LT(info[info_t::info_nevalf], T(32));
    // cheap
    EXPECT_GT(info[info_t::info_ncheap], T(28 * (n - 1)));
    EXPECT_LT(info[info_t::info_ncheap], T(32 * (n - 1)));
    // objective
    EXPECT_LT(info[info_t::info_objective], ftol);
    // gradient infinity norm
    EXPECT_EQ(info[info_t::info_grad_norm], T(0));
    // delta from two iterates in infinity norm
    EXPECT_LT(info[info_t::info_inorm], tol);
    // infinity-norm of the initial iterate
    EXPECT_EQ(info[info_t::info_inorm_init], inorm_init);

    // Second call at solution
    status = coord::coord(opts, n, x, l, u, info, stepfun_cycleend, monit, usrdata, err);
    EXPECT_EQ(status, da_status_success) << "error from 2nd call to coord";

    EXPECT_LE(info[info_t::info_iter], 1);
}
