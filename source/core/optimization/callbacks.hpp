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

#ifndef CALLBACKS_HPP
#define CALLBACKS_HPP
#include "aoclda.h"
#include <cmath>
#include <functional>

/* Generic function pointers to user callbacks for
 * optimization function / gradient and monitoring
 */

/** Objective function declaration meta_objcb
 *  -----------------------------------------
 * Input: n>0, x[n] iterate vector
 * Output val = f(x) if return status is 0, otherwise undefined
 * Must return 0 on successfull eval and nonzero to indicate that function could 
 * not be evaluated, some solvers don't have recovery capability.
 */
template <typename T> struct meta_objcb {
    static_assert(std::is_floating_point<T>::value,
                  "Objective function arguments must be floating point");
    using type = std::function<da_int(da_int n, T *x, T *val, void *usrdata)>;
};
template <typename T> using objfun_t = typename meta_objcb<T>::type;

/** Objective gradient (function) declaration meta_grdcb
 *  ----------------------------------------------------
 * Input: n>0, x[n] iterate vector, xnew to indicate that a new iterate is being provided and that
 *        objfun(x) was NOT called previously. Some iterative methods by design ALWAYS call objfun(x) 
 *        before objgrd (on the same iterate) an some common computation can be done only once at the 
 *        objfun(x) call. The latter call to objgrd would reused the computed values continue with the
 *        gradient calculation. E.g. linear models with constant feature matrix needs to be evaluated
 *        for computing f(x) and f'(x) and when using iterative solvers that evaluate first f(x) then
 *        g(x), then it is possible to indicate to objgrd to skip the matrix evaluation. If unsure,
 *        then set xnew = true and this will perform all the necessary calculations to correctly evaluate
 *        the objective gratient.
 * Output val = f'(x) = \nabla f(x) if return status is 0, otherwise val is untouched
 * Must return 0 on successfull eval and nonzero to indicate that function could 
 * not be evaluated, some solvers don't have recovery capability.
 * Not yet implemented: is *usrdata->fd == true then estimate the gradient using
 * a finite-difference method (forwards, bacbwards, center, cheap, etc).
 */
template <typename T> struct meta_grdcb {
    static_assert(std::is_floating_point<T>::value,
                  "Objective gradient function arguments must be floating point");
    using type =
        std::function<da_int(da_int n, T *x, T *val, void *usrdata, da_int xnew)>;
};
template <typename T> using objgrd_t = typename meta_grdcb<T>::type;

/** function declaration meta_stepcb
 *  --------------------------------
 * Input: n>0, x[n] iterate vector, n>k>=0 k-th coord,
 *        usrdata: pointer to user data, and
 *        action: action to take (implementation dependent).
 * Output: *s step to take along the k-th coord, *f objective value at x      
 * Must return 0 on successfull eval and nonzero to indicate that function could 
 * not be evaluated, some solvers don't have recovery capability.
 */
template <typename T> struct meta_stepcb {
    static_assert(std::is_floating_point<T>::value,
                  "Step function arguments must be floating point");
    using type = std::function<da_int(da_int n, T *x, T *s, da_int k, T *f, void *usrdata,
                                      da_int action)>;
};
template <typename T> using stepfun_t = typename meta_stepcb<T>::type;

/** User monitoring call-back declaration meta_moncb
 *  ------------------------------------------------
 * Input: n>0, x[n] iterate vector
 *        val = f(x), info[100] information vector
 * Must return 0 to intidate the solver to continue, otherwise by returning nonzero it 
 * request to interrupt the process and exit.
 */
template <typename T> struct meta_moncb {
    static_assert(std::is_floating_point<T>::value,
                  "Monitor function arguments must be floating point");
    using type = std::function<da_int(da_int n, T *x, T *val, T *info, void *usrdata)>;
};
template <typename T> using monit_t = typename meta_moncb<T>::type;

#endif