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

// Deal with some Windows compilation issues regarding max/min macros
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "basic_handle.hpp"
#include "da_error.hpp"
#include "lbfgsb_driver.hpp"
#include "macros.h"
#include "optim_types.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <bitset>
#endif

namespace ARCH {

namespace da_optim {

using namespace da_optim_types;

template <typename T> class da_optimization : public basic_handle<T> {
  protected:
    // Lock for solver
    bool locked = false;

    // True if the model has been successfully trained
    bool model_trained{false};

    // Number of variables
    da_int nvar = 0;
    // Number of residuals
    da_int nres = 0;

    // model coefficients
    std::vector<T> coef;

    // Which type constraints are defined (only bound constraints are allowed for now)
    std::bitset<8> constraint_types{0};
    // Bound constraints (allocated only if constraint_types[cons_bound] is set)
    std::vector<T> l, u;
    // Alternatively, if user provided data, store location
    T *l_usrptr{nullptr};
    T *u_usrptr{nullptr};
    // Pointer to weights
    T *w_usrptr{nullptr};
    da_int lw_usrptr{0};

    // Pointers to callbacks
    objfun_t<T> objfun = nullptr;
    objgrd_t<T> objgrd = nullptr;
    stepfun_t<T> stepfun = nullptr;
    stepchk_t<T> stepchk = nullptr;
    monit_t<T> monit = nullptr;
    resfun_t<T> resfun = nullptr;
    resgrd_t<T> resgrd = nullptr;
    reshes_t<T> reshes = nullptr;
    reshp_t<T> reshp = nullptr;

    // Last iterate information
    // Objective function value
    T f = 0.0;
    // Objective function gradient
    std::vector<T> g;
    // Information vector
    std::vector<T> info;

    // Pointer to user data
    void *udata{nullptr};

  public:
    da_optimization(da_status &status, da_errors::da_error_t &err);
    ~da_optimization();

    /* This function is called when data in the handle has changed, e.g. options
     * changed. We mark the model untrained and prepare the handle in a way that
     * it is suitable to solve again.
     */
    void refresh();

    // Build model to solve
    da_status add_vars(da_int nvar);
    da_status add_res(da_int nres);
    da_status add_bound_cons(std::vector<T> &l, std::vector<T> &u);
    da_status add_bound_cons(da_int nvar, T *l, T *u);
    da_status add_weights(da_int lw, T *w);
    da_status add_objfun(objfun_t<T> usrfun);
    da_status add_objgrd(objgrd_t<T> usrgrd);
    da_status add_stepfun(stepfun_t<T> usrstep);
    da_status add_stepchk(stepchk_t<T> usrstepchk);
    da_status add_monit(monit_t<T> monit);
    da_status add_resfun(resfun_t<T> resfun);
    da_status add_resgrd(resgrd_t<T> resgrd);
    da_status add_reshes(reshes_t<T> reshes);
    da_status add_reshp(reshp_t<T> reshp);

    // Solver interfaces (only lbfgsb for now)
    da_status solve(std::vector<T> &x, void *usrdata);

    // Update info
    da_status set_info(da_int idx, const T value);

    // Retrieve data from solver
    da_status get_info(da_int &dim, T info[]);

    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] T *result);

    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
};

} // namespace da_optim

} // namespace ARCH
