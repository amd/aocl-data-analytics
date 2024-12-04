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

#ifndef COORD_HPP
#define COORD_HPP

#include "aoclda_error.h"
#include "aoclda_result.h"
#include "da_error.hpp"
#include "linmod_nln_optim.hpp"
#include "options.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#define HDRCNT 30 // how frequently to print iteration banner
// Scaling for the convergence check
// 0 - absolute: ||Delta_W||_inf < tol
// 1 - relative: ||Delta_W||_inf < tol * ||W||_inf <------ sklearn / oneDAL
// 2 - safe:     ||Delta_W||_inf < tol * max(1, ||W||_inf) <--- safe-guard version for the cases where
// |Delta_W||_ing / ||W||_inf converges to a constant far from zero, say 1, and
// relative tolerance check is never satisfied. Set to 1 when benchmarking "well-behaved" problems
// Also can happen for problems where solution is zero (or when projecting to the positive cone)
#define __DA_COORD_SCALE_CONV_TOL 1

// LCOV_EXCL_START
// Excluding bound constraint class, this feature is for
// future use once bounds are supported in Linear Models
namespace constraints {
enum bound_t { none = 0, lower = 1, both = 2, upper = 3 };

/** Class for Constraint Bounds
  *  ===========================
  * if constrained = false
  *  * btyp = not allocated
  *  * lptr and uptr are not set
  * if constrained = true
  *  * vector btyp(n) = {none|lower|upper|both}
  *  * lptr => user l (readonly)
  *  * uptr => user u (readonly)
  */
template <typename T> class bound_constr {

  public:
    bool constrained = false;
    std::vector<enum bound_t> btyp;
    std::vector<T> *lptr = nullptr;
    std::vector<T> *uptr = nullptr;

    da_status proj(const size_t i, T &x) {

        if (!constrained)
            return da_status_success;
        switch (btyp[i]) {
        case none:
            return da_status_success;
        case both:
            x = std::max(std::min(x, (*uptr)[i]), (*lptr)[i]);
            break;
        case lower:
            x = std::max((*lptr)[i], x);
            break;
        case upper:
            x = std::min(x, (*uptr)[i]);
            break;
        }
        return da_status_success;
    }

    da_status proj(std::vector<T> &x) {

        if (!constrained)
            return da_status_success;
        const size_t n = x.size();
        for (size_t i = 0; i < n; i++) {
            switch (btyp[i]) {
            case none:
                continue;
            case both:
                x[i] = std::max(std::min(x[i], (*uptr)[i]), (*lptr)[i]);
                break;
            case lower:
                x[i] = std::max((*lptr)[i], x[i]);
                break;
            case upper:
                x[i] = std::min(x[i], (*uptr)[i]);
                break;
            }
        }
        return da_status_success;
    }

    da_status add(size_t n, std::vector<T> &l, std::vector<T> &u, T bigbnd,
                  da_errors::da_error_t &err) {
        if ((l.size() != 0 && l.size() != n) || (u.size() != 0 && u.size() != n)) {
            return da_error(&err, da_status_invalid_input,
                            "Bound constraint vectors need to be of size either 0 or " +
                                std::to_string(n) + ".");
        }

        if (l.size() == 0 && u.size() == 0) {
            this->constrained = false;
            this->lptr = nullptr;
            this->uptr = nullptr;
            this->btyp.resize(0);
            return da_status_success;
        }

        this->constrained = true;
        this->lptr = &l;
        this->uptr = &u;
        // Count how many bounds are +/-inf
        size_t ibnd = 0;
        try {
            this->btyp.resize(n);
        } catch (std::bad_alloc &) {
            return da_error(&err, da_status_memory_error,
                            "Could not allocate memory for solver.");
        }

        if (l.size() == 0 && u.size() != 0) {
            // Only upper bounds, set lower to -infinity
            for (size_t i = 0; i < n; i++) {
                if (u[i] <= -bigbnd) {
                    return da_error(&err, da_status_invalid_input,
                                    "Upper bound constraints cannot be -Infinity.");
                }
                if (u[i] < bigbnd)
                    this->btyp[i] = upper;
                else {
                    this->btyp[i] = none;
                    ibnd++;
                }
            }
        } else if (l.size() != 0 && u.size() == 0) {
            // Only lower bounds, set upper to +infinity
            for (size_t i = 0; i < n; i++) {
                if (l[i] >= bigbnd) {
                    return da_error(&err, da_status_invalid_input,
                                    "Lower bound constraints cannot be +Infinity.");
                }
                if (l[i] > -bigbnd)
                    this->btyp[i] = lower;
                else {
                    this->btyp[i] = none;
                    ibnd++;
                }
            }
        } else {
            // Both bounds present
            for (size_t i = 0; i < n; i++) {
                if (l[i] >= bigbnd || u[i] <= -bigbnd || l[i] > u[i]) {
                    return da_error(&err, da_status_invalid_input,
                                    "Lower bound constraints must be less that +Infinity "
                                    "and cannot be greater that upper bound constraints "
                                    "and these must be greater that -Infinity.");
                }
                if (l[i] > -bigbnd && u[i] < bigbnd)
                    this->btyp[i] = both;
                else if (l[i] > -bigbnd)
                    this->btyp[i] = lower;
                else if (u[i] < bigbnd)
                    this->btyp[i] = upper;
                else {
                    this->btyp[i] = none;
                    ibnd++;
                }
            }
        }
        // Check if user provided bounds are all +/-infinity
        if (ibnd == n) {
            this->constrained = false;
            this->lptr = nullptr;
            this->uptr = nullptr;
            this->btyp.resize(0);
        }
        return da_status_success;
    }
};
} // namespace constraints
// LCOV_EXCL_STOP

namespace coord {

enum solver_tasks {
    START = 1,
    NEWX = 2, // Full cycle of coordinates complete
    EVAL = 3, // Evaluate step function, get new x[k]
    STOP = 4,
    OPTIMCHK = 5, // Check optimality condition
};

template <typename T> class coord_slv {
  public:
    // kdiff = x[k] - x_old[k]
    T kdiff;

    // Ledger

    // Restart the skip ledger every <restart> iterations
    // restart = MAXINT disables periodic restarts
    // restart = 0 forces every iteration to be expensive
    size_t restart;
    // Minimum times a coordinate change must be less than skiptol
    // before it can start to be skipped
    // skipmin = max(1, skipmin); % needs to be at least 1
    size_t skipmin;
    // Initial maximum time a coordinate can be skipped, after this
    // the coordinate is checked.
    // skipmax_reset = max(skipmin+3, skipmax_reset); Needs to be bigger that skipmin
    size_t skipmax_reset;
    // Tolerance to skip a coordinate
    T skiptol;
    // Skip counter for each coordinate
    std::vector<size_t> skip; // (0U);
    // Maximum times a coordinate can be skipped, after this
    // the coordinate is checked. This count is customized per coordinate,
    // so coordinates that are fixed or active are checked less frequently
    std::vector<size_t> skipmax; //(0U);
    // Solver flag information
    size_t flags{0};
    // Infinity-norm of the current beta (coefficient vector)
    T inormbeta{T(0)};

    // Assumes all inputs are valid
    coord_slv(size_t restart, size_t skipmin, size_t skipmax_reset, T skiptol)
        : restart{restart}, skipmin{skipmin}, skipmax_reset{skipmax_reset},
          skiptol{skiptol} {};
    ~coord_slv(){};

    /* Reset the ledger stored in the work array
    * - the ledger set to 0
    * - skipmax for each coordinate, set to skipmax_reset
    */
    void reset_skip_ledger(void) {
        skipmax.assign(skipmax.size(), skipmax_reset);
        skip.assign(skip.size(), 0U);
    };

    // Check ledger to see if all coordinate skip counters are less than skipmin.
    // True indicates that all the coordinates have been checked and none have been
    // skipped.
    bool check_skip_ledger(void) {
        bool ok = true;
        for (size_t &iskip : skip) {
            ok &= iskip <= skipmin;
            if (!ok)
                break;
        }
        return ok;
    }

    // Resize (and reset) ledger vectors (assume input is valid)
    void resize_ledger(size_t n) {
        skip.resize(n);
        skipmax.resize(n);
        // reset ledger
        reset_skip_ledger();
    }
};

template <typename T>
da_status coord_rcomm(da_int n, std::vector<T> &x, constraints::bound_constr<T> &bc,
                      T factr, T tol, coord::solver_tasks &itask, da_int &k, T &newxk,
                      da_int &iter, T &inorm, T &optim, da_int &action,
                      da_errors::da_error_t &err, coord_slv<T> &w);

/** Coordinate Descent Method - Forward Communication <templated>
 *
 * COORD Coordinate Descent solver for Generalized Linear Models with Elastic Net
 *
 * Problem to solve is
 *           min f(x) subject to l <= x <= u
 *           x \in R^n
 * f(x) should be C1 inside the bounding box
 * The solver requires a user call-back that specifies the next iterate for a specific coordinate, say k \in {1..n},
 * and satisfies \nabla x_k f(x) = 0, that is, the next iterate x_k minimizes f with respect to the k-th coordinate.
 * Regularization is taken care implicitly in f(x)
 */
template <typename T>
da_status coord(da_options::OptionRegistry &opts, da_int n, std::vector<T> &x,
                std::vector<T> &l, std::vector<T> &u, std::vector<T> &info,
                const stepfun_t<T> stepfun, const monit_t<T> monit, void *usrdata,
                da_errors::da_error_t &err, const stepchk_t<T> stepchk) {

    if (!stepfun)
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Solver requires a valid pointer to the step function "
                        "call-back.");
    T bigbnd;
    if (opts.get("infinite bound size", bigbnd))
        return da_error( // LCOV_EXCL_LINE
            &err, da_status_internal_error,
            "expected option not found: <infinite bound size>.");
    T tol;
    if (opts.get("coord convergence tol", tol))
        return da_error( // LCOV_EXCL_LINE
            &err, da_status_internal_error,
            "expected option not found: <coord convergence tol>.");
    T factr;
    if (opts.get("coord progress factor", factr))
        return da_error( // LCOV_EXCL_LINE
            &err, da_status_internal_error,
            "expected option not found: <coord progress factor>.");
    T maxtime;
    if (opts.get("time limit", maxtime))
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "expected option not found: <time limit>.");
    da_int prnlvl;
    if (opts.get("print level", prnlvl))
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "expected option not found: <print level>.");
    da_int maxit;
    if (opts.get("coord iteration limit", maxit))
        return da_error( // LCOV_EXCL_LINE
            &err, da_status_internal_error,
            "expected option not found: <coord iteration limit>.");
    da_int mon = 0;
    if (monit) // monitor provided
        if (opts.get("monitoring frequency", mon))
            return da_error( // LCOV_EXCL_LINE
                &err, da_status_internal_error,
                "expected option not found: <monitoring frequency>.");

    /* Active-set ledger
     * =================
     *
     * The active set default behaviour is set for Linear LSQ + Elastic Net
     * problems:
     *
     *  Each coordinate is tested for "progress" skip_min times at the beginning
     *  before marking it as inactive then tested again "skip_max" (which is
     *  updated during solve). If there is movement along the coordinate, it
     *  is tested again for "progress" skip_min times and the cycle continues
     *  until convergence. If not, skip_max is doubled and the coordinate
     *  is kept inactive.
     */

    // Tolerance to consider skipping the coordinate
    T skiptol;
    if (opts.get("coord skip tol", skiptol))
        return da_error( // LCOV_EXCL_LINE
            &err, da_status_internal_error,
            "expected option not found: <coord skip tolerance>.");
    // Minimum times a coordinate change is smaller than skiptol to start skipping
    // Needs to be at least 1.
    da_int skipmin;
    if (opts.get("coord skip min", skipmin))
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "expected option not found: <coord skip min>.");

    // Initial max times a coordinate can be skipped after this the coordinate is checked
    // Expected to be greater that skipmin+3
    da_int skipmax_reset;
    if (opts.get("coord skip max", skipmax_reset))
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "expected option not found: <coord skip max>.");
    if (skipmax_reset != std::max(skipmin + 3, skipmax_reset)) {
        skipmax_reset = std::max(skipmin + 3, skipmax_reset);
        opts.set("coord skip max", skipmax_reset, da_options::solver);
    }

    // Restart (force expensive iteration every <restart>)
    da_int restart;
    if (opts.get("coord restart", restart))
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "expected option not found: <coord restart>.");

    if (n <= 0) {
        return da_error(&err, da_status_invalid_input, // LCOV_EXCL_LINE
                        "Number of variables needs to be positive.");
    }

    if (x.size() != (size_t)n) {
        return da_error(&err, da_status_invalid_input, // LCOV_EXCL_LINE
                        "Vector x needs to be of size n=" + std::to_string(n) + ".");
    }

    // Workspace
    coord_slv<T> w(restart, skipmin, skipmax_reset, skiptol);
    try {
        w.resize_ledger(n);
    } catch (std::bad_alloc const &) {
        return da_error( // LCOV_EXCL_LINE
            &err, da_status_memory_error,
            "Could not initialize work space for the coord solver");
    }

    da_status status;
    constraints::bound_constr<T> bc;

    // Check and add bound constraints
    status = bc.add(n, l, u, bigbnd, err);
    if (status != da_status_success)
        return status; // Error message already loaded

    da_int hdr{0}, fcnt{0}, lowrk{0}, iter{0}, k{0}, action{0}, chkcnt{0};
    T *f = &info[da_optim_info_t::info_objective];
    T *time = &info[da_optim_info_t::info_time];
    T newxk{T(0)};
    T inorm{std::numeric_limits<T>::infinity()};
    *f = std::numeric_limits<T>::infinity();
    T optim{T(-1)}; // Optimality condition measure (duality gap, KKT, ...)
    da_int cbflag{0};
    bool cbstop = false;

    solver_tasks itask = START;
    auto clock = std::chrono::system_clock::now();
    const size_t coutprec = std::cout.precision();

    if (prnlvl >= 5) {
        std::cout << "Initial coefficients:" << std::endl;
        for (da_int i = 0; i < n; ++i) {
            std::cout << " x[" << i << "] = " << x[i] << std::endl;
        }
    }

    while (itask == NEWX || itask == START || itask == EVAL || itask == OPTIMCHK) {
        coord_rcomm<T>(n, x, bc, factr, tol, itask, k, newxk, iter, inorm, optim, action,
                       err, w);

        switch (itask) {
        case EVAL: // Compute new iterate for x[k]
            if (iter == 0) {
                info[da_optim_info_t::info_inorm_init] =
                    std::max(info[da_optim_info_t::info_inorm_init], std::abs(x[k]));
            }
            if (action > 0)
                fcnt++;
            else if (action < 0) {
                lowrk++;
            }

            cbflag = stepfun(n, &x[0], &newxk, k, nullptr, usrdata, action, w.kdiff);
            if (cbflag != 0) {
                // Step could not be evaluated, set newxk = xk, f = fold;
                // and signal to stop at end of the cycle (full iteration)
                newxk = x[k];
                cbstop = true;
            }
            if (prnlvl >= 4) {
                bool skipmax = skipmax_reset < std::numeric_limits<T>::max();
                std::string flagss;
                da_int restartk = w.flags & 1U;   // Requested restart
                da_int reset = w.flags & 2U;      // Tolerance check requested restart
                da_int reqskip = w.flags & 4U;    // In skip regime
                da_int exhaustion = w.flags & 8U; // Search-space exhausted
                da_int optim = w.flags & 16U;     // Optimality condition not met
                da_int activate = w.flags & 32U;  // Movement was detected
                flagss += (reqskip) ? "S" : "";
                flagss += (action < 1 ? "C" : "E");
                flagss += (restartk) ? "R" : "";
                flagss += (reset) ? "T" : "";
                flagss += (exhaustion) ? "!" : "";
                flagss += (cbflag) ? "X" : "";
                flagss += (optim) ? "D" : "";
                flagss += (activate) ? "A" : "";

                /*
                 * Iteration banner
                 * ================
                 * Low detail
                 * ----------
                 * After each outer iteration this banner is printed
                 *
                 * ------------------------------------------------------
                 *  iteration objective maxchange       neval       lowrk
                 * ------------------------------------------------------
                 *         20 2.920e+08 8.844e-06           4         115
                 * Where
                 * * maxchange is the infinity-norm of (xk - xk-1)
                 * * neval are the number of calls to step function
                 * * lowrk are the number of calls to step function hinting
                 *   to use a low rank update
                 * Total number of function calls is neval+lowrk
                 *
                 * Detailed output
                 * ---------------
                 * for each inner iteration this banner is printed
                 *
                 * iteration coordinate   current       new    change      skip/  skipmax
                 *        19          9 +1.25e+00 +1.25e+00 +5.43e-06#        0/        8 ACDERSTX!
                 * Where
                 * * current is the value of xold[k]
                 * * new is the value of x[k]
                 * * change is xold[k] - x[k] if post-fixed with "#" then it indicates that skip ledger will increment
                 * * skip is the ledger entry for coordinate k
                 * * skipmax is the next count for when the coordinate k will be checked
                 * The "flags" at the end of the line inform about:
                 *  * "A" ledger indicates that a coordinate was re-activated for exploration
                 *  * "C" or "E" hinting to cheap or expensive step function evaluation
                 *  * "D" iterate is close to the previous one but optimality condition still not within tolerance
                 *  * "R" a user-restart (reset of ledger and expensive evaluation) is requested
                 *  * "S" ledger indicates that there are previous coordinates that skipped exploration
                 *  * "T" iterate seems to be a solution (Tolerance ok) but some coordinates were skipped, resetting
                 *        ledger to evaluate them.
                 *  * "X" callback returned a failed status
                 *  * "!" the previous iteration exhausted the seach-space (ledger skipped over all the coordinates)
                 *        and the ledger was reset.
                 */

                std::cout.precision(2);
                std::cout << std::setw(10) << "iteration" << std::setw(1) << ""
                          << std::setw(10) << "coordinate" << std::setw(1) << ""
                          << std::setw(9) << "current" << std::setw(1) << ""
                          << std::setw(9) << "new" << std::setw(1) << "" << std::setw(9)
                          << "change" << std::setw(1) << "" << std::setw(9) << "skip";
                if (skipmax)
                    std::cout << "/" << std::setw(9) << "skipmax";
                std::cout << std::endl;

                std::cout << std::setw(10) << iter << std::setw(1) << "" << std::setw(10)
                          << k << std::setw(1) << "" << std::scientific << std::showpos
                          << std::setw(9) << x[k] << std::setw(1) << "" << std::setw(9)
                          << newxk << std::setw(1) << "" << std::setw(9) << x[k] - newxk
                          << std::setw(1) << (std::abs(x[k] - newxk) > skiptol ? "" : "#")
                          << std::setw(9) << w.skip[k];
                if (skipmax)
                    std::cout << "/" << std::setw(9) << w.skipmax[k];
                std::cout << " " << flagss << std::endl << std::noshowpos;
                std::cout.precision(coutprec);
            }
            break;
        case OPTIMCHK:
            ++chkcnt;
            cbflag = stepchk(n, &x[0], usrdata, &optim);
            if (cbflag) {
                status = da_error(
                    &err, da_status_numerical_difficulties,
                    "Optimality check call-back returned error at current iterate.");
                cbstop = true;
            }
            break;
        case NEWX:
        case STOP: // Copy and print for the last time
            if ((itask == STOP || prnlvl > 1) && (!cbstop)) {
                // Get the objective value of the scaled problem.
                cbflag = stepfun(n, &x[0], &newxk, k, f, usrdata, action, w.kdiff);
            }
            if (prnlvl > 1) {
                if (hdr == 0 || prnlvl >= 4) {
                    hdr = HDRCNT;
                    std::cout << std::setw(65) << std::setfill('-') << "" << std::endl
                              << std::setfill(' ');
                    std::cout << std::setw(10) << "iteration" << std::setw(1) << ""
                              << std::setw(9) << std::scientific << "objective"
                              << std::setw(1) << "" << std::setw(9) << "maxchange"
                              << std::setw(12) << "neval" << std::setw(12) << "lowrk"
                              << std::setw(1) << "" << std::setw(9) << "optim"
                              << std::endl;
                    std::cout << std::setw(65) << std::setfill('-') << "" << std::endl
                              << std::setfill(' ');
                }
                --hdr;
                std::cout.precision(3);
                std::cout << std::setw(10) << iter << std::setw(1) << ""
                          << std::scientific << std::setw(9) << *f << std::setw(1) << ""
                          << std::setw(9) << inorm << std::setw(12) << fcnt
                          << std::setw(12) << lowrk;
                if (optim < std::numeric_limits<T>::infinity()) {
                    std::cout << std::setw(1) << "" << std::setw(9) << optim << std::endl;
                }
                std::cout << std::endl;
                std::cout.precision(coutprec);
                if (prnlvl >= 5) {
                    std::cout << "Current coefficients:" << std::endl;
                    for (da_int i = 0; i < n; i++) {
                        std::cout << " x[" << i << "] = " << x[i] << std::endl;
                    }
                }
            }

            // Copy all metrics to info
            info[da_optim_info_t::info_nevalf] = (T)fcnt;
            info[da_optim_info_t::info_ncheap] = (T)lowrk;
            info[da_optim_info_t::info_inorm] = inorm;
            info[da_optim_info_t::info_iter] = (T)iter;
            info[da_optim_info_t::info_optim] = optim;
            info[da_optim_info_t::info_optimcnt] = (T)chkcnt;
            *time = std::chrono::duration<T>(std::chrono::system_clock::now() - clock)
                        .count();

            if (cbstop) {
                if (iter == 1)
                    status = da_error(
                        &err, da_status_numerical_difficulties,
                        "Initial iterate is unusable. One or more coordinate steps could "
                        "not be computed by the callback.");
                else if (status == da_status_success) {
                    status = da_warn(&err, da_status_numerical_difficulties,
                                     "One or more coordinate steps could not be computed "
                                     "by the callback.");
                } // Otherwise status already filled.
                break;
            }

            if (itask == STOP)
                break;

            if (iter >= maxit) {
                itask = STOP;
                status = da_warn(
                    &err, da_status_maxit,
                    "Iteration limit reached without converging to set tolerance.");
                break;
            }

            if (mon != 0) {
                if (iter % mon == 0) {
                    // Call monitor
                    if (monit(n, &x[0], nullptr, &info[0], usrdata) != 0) {
                        // User request to stop
                        itask = STOP;
                        status = da_warn(&err, da_status_optimization_usrstop,
                                         "User requested to stop optimization process.");
                        break;
                    }
                }
            }

            if (maxtime > 0) {
                if (*time > maxtime) {
                    // Run out of time
                    itask = STOP;
                    status = da_warn(
                        &err, da_status_maxtime,
                        "Time limit reached without converging to set tolerance.");
                    break;
                }
            }
            break;
        default:
            itask = STOP;
            status = da_error(&err, da_status_internal_error,
                              "Unknown task requested for coordinate descent RCI.");
            break;
        }
    }
    if (prnlvl > 0) {
        // Exit summary message
        std::cout << std::endl << "Solver summary" << std::endl;
        std::cout << " Objective value (scaled problem): " << *f << std::endl;
        if (optim < std::numeric_limits<T>::infinity()) {
            std::cout << " Optimality measure:           " << optim << std::endl;
        } else {
            std::cout << " Optimality measure:           Infinity" << std::endl;
        }
        std::cout << " Number of optimality checks:  " << chkcnt << std::endl;
        std::cout << " Total number of step calls (cheap):   " << fcnt + lowrk << " ("
                  << lowrk << ")" << std::endl;
        std::cout << " Total solve time: " << *time << " sec" << std::endl;
        std::cout << " Total number of iterations: " << iter << std::endl;
        if (status == da_status_success) {
            std::cout
                << " Convergence status: "
                << "distance between two consecutive iterates is less than tolerance."
                << std::endl;
        } else {
            std::string errmsg;
            err.print(errmsg);
            std::cout << " Exit status: " << errmsg << std::endl;
        }
        std::cout << std::endl;
    }
    return status; // Error message already loaded
}

/**
 * @brief Coordinate Descent Method RCI
 *
 * Reverse Communication Interface (solver) for the Coordinate Descent Method
 * The solver communicates back to the user the required task to complete, this
 * is using TASK parameter
 *
 * TASK
 * ====
 * START: initial call to rci indicating to initialize data.
 * NEWX: solver found the next iterate and information can be either printed or
 *       inspected.
 * EVAL: evaluate the step function (see ACTION)
 * OPTIMCHK: check for optimality conditions, e.g. duality gap size, KKT, ...
 * STOP: search terminated, either found a solution or error occurred,
 *       also check the exit status.
 *
 * ACTION
 * ======
 * Action to perform by the callback when TASK=EVAL.
 * This parameter adds hints on how to evaluate the step function
 * action < 0, perform a (cheap iteration) low-rank update from kold to k
 *             element. kold = -(action+1).
 * action = 0, no change in x don't evaluate matrix.
 * action > 0, evaluate the feature matrix, perform MV on x.
 */
template <typename T>
da_status coord_rcomm(const da_int n, std::vector<T> &x, constraints::bound_constr<T> &bc,
                      [[maybe_unused]] T factr, T tol, coord::solver_tasks &itask,
                      da_int &k, T &newxk, da_int &iter, T &inorm, T &optim,
                      da_int &action, da_errors::da_error_t &err,
                      coord::coord_slv<T> &w) {
    // kchange = abs(kdiff)
    T kchange;
    bool endcycle, cheap;
    da_int kold{-1};

    // Quick check of work spaces
    if ((w.skip.size() != (size_t)n) || (w.skipmax.size() != (size_t)n)) {
        itask = STOP;
        return da_error(&err, da_status_invalid_array_dimension, // LCOV_EXCL_LINE
                        "Work array not initialized with the correct size. Vectors need "
                        "to be of size " +
                            std::to_string(n));
    }

    switch (itask) {
    case START:
        w.reset_skip_ledger();
        w.flags = 0U;
        iter = 0;
        // Make initial X feasible
        bc.proj(x);
        // Set initial coordinate to move on
        k = 0;
        // Request to evaluate objective and new update.
        action = 1;
        itask = EVAL;
        w.kdiff = 0;
        // Infinity-norm of the distance of two iterates
        inorm = T(0);
        // Infinity-norm of the coefficient vector
        w.inormbeta = T(0);
        // reset optimality measure
        optim = std::numeric_limits<T>::infinity();
        return da_status_success;
        break;
    case EVAL:
        // reset flags
        w.flags = 0;
        bc.proj(k, newxk);
        w.kdiff = newxk - x[k];
        kchange = std::abs(w.kdiff);
        inorm = std::max(inorm, kchange);
        w.inormbeta = std::max(w.inormbeta, std::abs(newxk));
        x[k] = newxk;

        if (kchange == (T)0)
            action = 0; // Iterate did not change
        else {
            kold = k;
            action = -(k + 1); // Inform the last coordinate to use for cheap iter
        }

        if (kchange > w.skiptol) {
            if (w.skip[k] > 0) {
                // Reset skip counter only for k
                w.skip[k] = 0;
                w.skipmax[k] = w.skipmax_reset;
                w.flags |= 32U; // Mark k coord reset;
            }
        } else {
            if (w.skip[k] >= w.skipmax[k]) {
                const size_t big{std::numeric_limits<size_t>::max()};
                const size_t half{big / 2};
                if (w.skipmax[k] >= half)
                    w.skipmax[k] = big;
                else
                    w.skipmax[k] *= 2; // Double the threshold;
            }
            // increment ledger
            ++(w.skip[k]);
        }

        // Check that one full cycle was performed
        if (k < n - 1) {
            // Keep cycling
            ++k;
            endcycle = false;
        } else {
            // Mark end of cycle
            k = 0;
            endcycle = true;
        }

        // Find the next k to use...
        while (w.skipmin < w.skip[k] && w.skip[k] < w.skipmax[k] && k != kold) {
            ++(w.skip[k]);
            w.flags |= 4U; // "S" mark iter as one or more coords were skipped;
            ++k;
            if (k >= n) {
                k = 0;
                // A full circle done: mark end-of-cycle reached
                endcycle = true;
            }
        }

        if (endcycle) {
            // Completed a full cycle
            ++iter;
            T itol = tol;
            if (__DA_COORD_SCALE_CONV_TOL >= 1)
                itol *= w.inormbeta;
            if (__DA_COORD_SCALE_CONV_TOL >= 2)
                itol = std::max(itol, tol);

            // Should it also safe-guard and check for
            // error < machine epsilon and inorm < machine epsilon?

            // Check for convergence or search-space exhaustion
            if (w.inormbeta == T(0) || inorm <= itol || k == kold) {
                if (w.check_skip_ledger()) {
                    // No coordinates were skipped and tolerance reached.
                    // Now check for optimality condition before declaring convergence
                    itask = OPTIMCHK;
                    // Step tolerance met, possibly close to solution,
                    // Bump skipmin to avoid skipping
                    w.skipmin = std::max(w.skipmin, size_t(10U));
                } else {
                    // Iterate distance tolerance met but
                    // there is at least one skipped coordinate, reset ledger
                    // and recheck the coordinates
                    w.flags |= 2U; // "T"
                    w.reset_skip_ledger();
                    itask = NEWX;

                    // Corner case where no new coordinate is available to explore...
                    // set k to be the next in line
                    if (k == kold) {
                        ++k;
                        if (k >= n)
                            k = 0;
                        w.flags |= 8U; // "!" Mark search space exhaustion
                    }
                }
            } else {
                // Indicate that point x can be printed, monitored, etc.
                itask = NEWX;
            }
        }
        // Check to see if a cheap iteration can be performed and check that
        // it is not time to restart... iter=k=0 also does a MV eval.
        // if restart=1 then all iters are requested as expensive.
        cheap = action < 1;
        if (cheap) {
            da_int rst = ((iter * n + k) % w.restart);
            w.flags |= (rst == 0);
            cheap = !(rst == 0);
        }
        if (!(cheap)) {
            // Request to perform MV operation (expensive evaluation)
            action = 1;
        }

        return da_status_success;
        break;

    case OPTIMCHK:
        // A full cycle has been completed, step is within tolerance
        // check for optimality condition, duality gap size, KKT, ...
        if (optim <= tol) {
            itask = STOP;
        } else {
            // Step is small but optimality not reached, e.g. duality gap still big
            w.flags |= 16U; // D
            itask = NEWX;
        }
        return da_status_success;
        break;

    case NEWX:
        // User did not stop, continue...
        itask = EVAL;
        // Reset inorm and inormbeta
        inorm = static_cast<T>(0);
        w.inormbeta = static_cast<T>(0);
        return da_status_success;
        break;
    default:
        itask = STOP;
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected taskid provided?");
        break;
    }
}

} // namespace coord
#endif
