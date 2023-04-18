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
template <typename T> using objfun_t = typename meta_objcb<T>::type;

template <typename T> struct meta_grdcb {
    static_assert(std::is_floating_point<T>::value,
                  "Objective gradient function arguments must be floating point");
    using type =
        std::function<da_int(da_int n, T *x, T *val, void *usrdata, da_int xnew)>;
};
template <typename T> using objgrd_t = typename meta_grdcb<T>::type;

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