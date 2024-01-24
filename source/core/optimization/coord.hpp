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

#ifndef COORD_HPP
#define COORD_HPP

#include "aoclda_error.h"
#include "aoclda_result.h"
#include "da_error.hpp"
#include "info.hpp"
#include "linmod_nln_optim.hpp"
#include "options.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#undef max
#undef min
#define HDRCNT 30 // how frequent to print iteration banner

// LCOV_EXCL_START
// Excluding bound constraint class, this feature is for
// future use once bounds are supported in Linear Models
namespace constraints {
enum bound_t { none = 0, lower = 1, both = 2, upper = 3 };

/** Class for Constraint Bounds
  *  ===========================
  * if constrained = false
  *  * btyp = not allocated
  *  * lptr and uptr and not set
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

enum solver_tasks { START = 1, NEWX = 2, EVAL = 3, STOP = 4 };

template <typename T>
da_status coord_rcomm(da_int n, std::vector<T> &x, constraints::bound_constr<T> &bc,
                      T factr, T tol, coord::solver_tasks &itask, da_int &k, T &newxk,
                      da_int &iter, T &inorm, da_int &action, da_errors::da_error_t &err,
                      std::vector<T> &w, std::vector<uint32_t> &iw);

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
                da_errors::da_error_t &err) {

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
    T factr; // FIXME currently not used
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

    // Active-set ledger
    // =================

    // tolerance to consider skipping the coordinate
    T skiptol;
    if (opts.get("coord skip tol", skiptol))
        return da_error( // LCOV_EXCL_LINE
            &err, da_status_internal_error,
            "expected option not found: <coord skip tolerance>.");
    // minimum times a coordinate change is smaller than skiptol to start skipping
    // needs to be at least 1.
    da_int skipmin;
    if (opts.get("coord skip min", skipmin))
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "expected option not found: <coord skip min>.");

    // initial max times a coordinate can be skipped after this the coordinate is checked
    // expected to be greater that skipmin+3
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

    // Work space
    std::vector<T> rw(10, 0);
    std::vector<size_t> iw;
    try {
        iw.resize(2 * n + 10);
    } catch (std::bad_alloc const &) {
        return da_error( // LCOV_EXCL_LINE
            &err, da_status_memory_error,
            "Could not allocate work space for the coord solver");
    }

    // Copy params into workspace
    iw[0] = restart;
    iw[1] = skipmin;
    iw[2] = skipmax_reset;
    iw[3] = true;
    iw[4] = 0;
    rw[0] = skiptol;
    rw[1] = 0;
    rw[2] = 0;
    rw[3] = 0;
    rw[4] = 0;

    da_status status;
    constraints::bound_constr<T> bc;

    // Check and add bound constraints
    status = bc.add(n, l, u, bigbnd, err);
    if (status != da_status_success)
        return status; // Error message already loaded

    // info
    info.resize(da_optim::info_number);
    std::fill(info.begin(), info.end(), 0);

    da_int hdr = 0, fcnt = 0, lowrk = 0, iter = 0, k, action;
    T *f = &info[da_optim::info_t::info_objective];
    T *time = &info[da_optim::info_t::info_time];
    T newxk, inorm;
    T kdiff = 0.0;

    // Convenience pointers
    size_t *cheapk = nullptr;
    size_t *skipk = nullptr;
    size_t *skipmaxk = nullptr;
    size_t *flags = nullptr;

    solver_tasks itask = START;

    const size_t coutprec = std::cout.precision();

    if (prnlvl >= 5) {
        std::cout << "Initial coefficients:" << std::endl;
        for (da_int i = 0; i < n; ++i) {
            std::cout << " x[" << i << "] = " << x[i] << std::endl;
        }
    }

    while (itask == NEWX || itask == START || itask == EVAL) {
        coord_rcomm<T>(n, x, bc, factr, tol, itask, k, newxk, iter, inorm, action, err,
                       rw, iw);

        switch (itask) {
        case EVAL: // Compute objective and step-length from current X
            if (action > 0)
                fcnt++;
            else if (action < 0) {
                lowrk++;
                kdiff = rw[5];
            }
            if (stepfun(n, &x[0], &newxk, k, f, usrdata, action, kdiff) != 0) {
                // This solver does not have recovery, stop
                // FIXME restore last valid x (and stats?)
                return da_warn(&err, da_status_optimization_num_difficult,
                               "Could not evaluate step at current point.");
            }
            if (prnlvl >= 4) {
                std::string flagss;
                cheapk = &iw[6];                 // cheap iteration
                skipk = &iw[7];                  // the current value of skip[k]
                skipmaxk = &iw[8];               // the current value of skipmax[k]
                flags = &iw[9];                  // flags
                da_int restartk = (*flags) & 1U; // requested restart
                da_int reset = (*flags) & 2U;    // tolerance check requested restart
                da_int reqskip = (*flags) & 4U;  // in skip regime
                flagss += (reqskip) ? "S" : "";
                flagss += (*cheapk ? "c" : "e");
                flagss += (restartk) ? "z" : "";
                flagss += (reset) ? "R" : "";

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
                 * * maxchange is the infitity-norm of xk and xk-1
                 * * neval are the number of calls to step function
                 * * lowrk are the number of calls to step function hinting
                 *   to use a low rank update is possible
                 * Total number of functions calls is neval+lowrk
                 *
                 * Detailed output
                 * ---------------
                 * for each inner iteration this banner is printed
                 *
                 * iteration coordinate   current       new    change      skip/  skipmax
                 *        19          9 +1.25e+00 +1.25e+00 +5.43e-06         0/        8 cSzR
                 * Where
                 * * current is the value of xold[k]
                 * * new is the value of x[k]
                 * * change is xold[k] - x[k]
                 * * skip is the ledged entry for coordinate k
                 * * skipmax is the next count for when the coordinate k will be checked
                 * The "flags" at the end of the line inform about:
                 *  * "c" or "e" hinting to cheap or expensive step function evaluation
                 *  * "S" ledger indicates that the current coordiante is skipped
                 *  * "z" a restart (reset of ledger and expensive evaliation) is requested
                 *  * "R" iterate seems to be a solution but some coordinates where skipt, resetting
                 *    ledger to evaluate them.
                 */

                std::cout.precision(2);
                std::cout << std::setw(10) << "iteration" << std::setw(1) << ""
                          << std::setw(10) << "coordinate" << std::setw(1) << ""
                          << std::setw(9) << "current" << std::setw(1) << ""
                          << std::setw(9) << "new" << std::setw(1) << "" << std::setw(9)
                          << "change" << std::setw(1) << "" << std::setw(9) << "skip"
                          << "/" << std::setw(9) << "skipmax" << std::endl;

                std::cout << std::setw(10) << iter << std::setw(1) << "" << std::setw(10)
                          << k << std::setw(1) << "" << std::scientific << std::showpos
                          << std::setw(9) << x[k] << std::setw(1) << "" << std::setw(9)
                          << newxk << std::setw(1) << "" << std::setw(9) << x[k] - newxk
                          << std::setw(1) << "" << std::setw(9) << *skipk << "/"
                          << std::setw(9) << *skipmaxk << " " << flagss << std::endl
                          << std::noshowpos;
                std::cout.precision(coutprec);
            }
            break;
        case NEWX:
        case STOP: // Copy and print for the last time
            if (prnlvl > 1) {
                if (hdr == 0 || prnlvl >= 4) {
                    hdr = HDRCNT;
                    std::cout << std::setw(54) << std::setfill('-') << "" << std::endl
                              << std::setfill(' ');
                    std::cout << std::setw(10) << "iteration" << std::setw(1) << ""
                              << std::setw(9) << std::scientific << "objective"
                              << std::setw(1) << "" << std::setw(9) << "maxchange"
                              << std::setw(12) << "neval" << std::setw(12) << "lowrk"
                              << std::endl;
                    std::cout << std::setw(54) << std::setfill('-') << "" << std::endl
                              << std::setfill(' ');
                }
                --hdr;
                std::cout.precision(3);
                std::cout << std::setw(10) << iter << std::setw(1) << ""
                          << std::scientific << std::setw(9) << *f << std::setw(1) << ""
                          << std::setw(9) << inorm << std::setw(12) << fcnt
                          << std::setw(12) << lowrk << std::endl;
                std::cout.precision(coutprec);
                if (prnlvl >= 5) {
                    std::cout << "Current coefficients:" << std::endl;
                    for (da_int i = 0; i < n; i++) {
                        std::cout << " x[" << i << "] = " << x[i] << std::endl;
                    }
                }
            }

            // Copy all metrics to info
            info[da_optim::info_t::info_nevalf] = (T)fcnt;
            info[da_optim::info_t::info_ncheap] = (T)lowrk;
            info[da_optim::info_t::info_inorm] = inorm;
            info[da_optim::info_t::info_iter] = (T)iter;

            if (itask == STOP)
                break;

            if (iter >= maxit) {
                return da_warn(
                    &err, da_status_maxit,
                    "Iteration limit reached without converging to set tolerance.");
            }

            if (mon != 0) {
                if (iter % mon == 0) {
                    // call monitor
                    if (monit(n, &x[0], nullptr, &info[0], usrdata) != 0) {
                        // user request to stop
                        return da_warn(&err, da_status_optimization_usrstop,
                                       "User requested to stop optimization process.");
                    }
                }
            }

            if (maxtime > 0) {
                if (*time > maxtime) {
                    // run out of time
                    return da_warn(
                        &err, da_status_maxtime,
                        "Time limit reached without converging to set tolerance.");
                }
            }
            break;
        default:
            break;
        }
    }
    return status; // Error message already loaded
}

/* Reset the ledger stored in the work array
 * iw[0:n-1]  - the ledger set to 0
 * iw[n:2n-1] - skipmax for each coordiante, set to skipmax_reset
 */
inline void reset_skip_ledger(const da_int n, std::vector<size_t> &iw) {
    size_t *skip = &iw[10];
    size_t *skipmax = &iw[10 + n];
    const size_t skipmax_reset = iw[2];
    for (da_int i = 0; i < n; ++i) {
        skipmax[i] = skipmax_reset;
        skip[i] = 0U;
    }
}

// Check ledger to see if all coordinate skip counters are less than skipmin.
// True indicates that all the coordinates have been checked and none as been
// skipped.
inline bool check_skip_ledger(const da_int n, std::vector<size_t> &iw) {
    size_t *skip = &iw[10];
    const size_t skipmin = iw[1];
    bool ok = true;
    for (da_int i = 0; i < n; ++i) {
        ok &= skip[i] <= skipmin;
    }
    return ok;
}

/**
 * @brief Coordinate Descent Method RCI
 *
 * Reverse Comumication Interface (solver) for the Coordinate Descent Method
 * The solver comunicates back to the user the required task to complete, this
 * is using TASK parameter
 *
 * TASK
 * ====
 * START: initial call to rci indicating to initialize data.
 * NEWX: solver found the next iterate and information can be either printed or
 *       inspected.
 * EVAL: evaluate the step function (see ACTION)
 * STOP: search terminated, either found a solution or error ocurred,
 *       also check the exit status.
 *
 * ACTION
 * ======
 * Action to perform by the callback when TASK=EVAL.
 * This parameter adds hints on how to evaluate the step function
 * action < 0, perform a (cheap iteration) low-rank update from kold to k
 *             elements kold = -(action+1).
 * action = 0, no change in x don't evaluate matrix.
 * action > 0, evaluate the feature matrix, perform MV on x.
 *
 * COORD Work arrays
 * =================
 * IW Integer work array
 * iw[0] IN restart
 * iw[1] IN skipmin
 * iw[2] IN skipmax_reset
 * iw[3] IN cheap - allow cheap iterations
 * iw[4] IN reserved
 * iw[5] OUT kold
 * iw[6] OUT cheap
 * iw[7] OUT skip[k]
 * iw[8] OUT skipmax[k]
 * iw[9] OUT reserved
 *
 * RW Real work array
 * rw[0] IN skiptol
 * rw[1] IN reserved
 * rw[2] IN reserved
 * rw[3] IN reserved
 * rw[4] IN reserved
 * rw[5] OUT kdiff
 * rw[6] OUT reserved
 * rw[7] OUT reserved
 * rw[8] OUT reserved
 * rw[9] OUT reserved
 */
template <typename T>
da_status coord_rcomm(const da_int n, std::vector<T> &x, constraints::bound_constr<T> &bc,
                      [[maybe_unused]] T factr, T tol, coord::solver_tasks &itask,
                      da_int &k, T &newxk, da_int &iter, T &inorm, da_int &action,
                      da_errors::da_error_t &err, std::vector<T> &rw,
                      std::vector<size_t> &iw) {
    // kchange = abs(kdiff)
    T kchange;
    // Restart the skip ledger every <restart> iterations
    // restart = MAXINT disables periodic restarts
    // restart = 0 forces every iteration to be expensive
    size_t restart; // = iw[0];
    // Tolerance to skip a coordinate
    T skiptol; // = rw[0];
    // kdiff = x[k] - x_old[k]
    T *kdiff; // = rw[5];
    // Minimum times a coordinate change must be less than skiptol
    // before it can start to be skipped
    // skipmin = max(1, skipmin); % needs to be at least 1
    size_t skipmin; // = iw[1];
    // Initial maximum time a coordinate can be skiped, after this
    // the coordinate is checked.
    // skipmax_reset = max(skipmin+3, skipmax_reset); % needs to be bigger that skipmin
    size_t skipmax_reset; // = iw[2];
    // Maximum times a coordinate can be skiped, after this
    // the coordinate is checked. This count is customized per coordinate,
    // so coordinates that are fixed or active are less frequently checked
    // [ARRAY:p]
    size_t *skipmax; // = &iw[10] + n; // counter vector of size n
    // Skip counter for each coordiante [ARRAY:p]
    size_t *skip; // = &iw[10]; // the ledger of size n
    // Metrics (move to RINFO)
    size_t *kold;     // = iw[5] old k user for low rank update (cheap iteration)
    size_t *cheap;    // = iw[6] allow cheap iter
    size_t *skipk;    // = iw[7] the current value of skip[k]
    size_t *skipmaxk; // = iw[8] the current value of skipmax[k]
    size_t *flags;    // information flags

    bool endcycle;

    // Quick check of work spaces
    if ((iw.size() < (size_t)(10 + 2 * n)) || (rw.size() < 10U)) {
        itask = STOP;
        return da_error(&err, da_status_invalid_array_dimension, // LCOV_EXCL_LINE
                        "Integer work space vector needs to be at least of size " +
                            std::to_string(2 * n + 10) + " and real work space " +
                            "vector needs to be at least of size " + std::to_string(10));
    }

    // get parameter values
    skiptol = rw[0];
    restart = iw[0];
    skipmin = iw[1];
    skipmax_reset = iw[2];

    // link aliases to work array
    kdiff = &rw[5];
    kold = &iw[5];
    cheap = &iw[6];
    skipk = &iw[7];    // the current value of skip[k]
    skipmaxk = &iw[8]; // the current value of skipmax[k]
    flags = &iw[9];
    skip = &iw[10];        // the ledger of size n
    skipmax = &iw[10] + n; // counter vector of size n

    switch (itask) {
    case START:
        // reset non-input work entries
        for (size_t i = 5; i < 10; ++i) {
            iw[i] = 0;
            rw[i] = 0;
        }
        reset_skip_ledger(n, iw);

        iter = 0;
        // Make initial X feasible
        bc.proj(x);
        // Set initial coordianate to move on
        k = 0;
        // Request to evaluate objective and new update.
        action = 1;
        *cheap = false;
        itask = EVAL;
        // Update stats
        *skipk = skip[k];
        *skipmaxk = skipmax[k];
        inorm = static_cast<T>(0);
        return da_status_success;
        break;
    case EVAL:
        // reset flags
        *flags = 0;
        // try to allow cheap iteration
        *cheap = true;
        bc.proj(k, newxk);
        *kdiff = newxk - x[k];
        kchange = std::abs(*kdiff);
        inorm = std::max(inorm, kchange);
        x[k] = newxk;

        if (kchange == (T)0)
            action = 0; // iterate did not change
        else {
            *kold = k;
            action = -(k + 1); // inform the last coordinate to use for cheap iter
        }

        if (kchange > skiptol) {
            if (skip[k] > 0) {
                // Reset skip counter only for k
                skip[k] = 0;
                skipmax[k] = skipmax_reset;
                *flags |= 8U; // mark k coord reset;
            }
        } else {
            if (skip[k] >= skipmax[k])
                skipmax[k] *= 2; // double the threshold;
            // increment ledger
            ++skip[k];
        }

        // Check that one full cycle was performed
        if (k < n - 1) {
            // keep cycling
            ++k;
            endcycle = false;
        } else {
            // mark end of cycle
            k = 0;
            endcycle = true;
        }

        while (skipmin <= skip[k] && skip[k] < skipmax[k]) {
            ++skip[k];
            *flags |= 4U; // mark iter as skipped;
            ++k;
            if (k >= n) {
                // last coord was also skipped, mark end-of-cycle
                k = 0; // FIXME 0 may not be the next coord not to skip
                // This requires exploring the next candidate.
                // Split RCI: EVAL+CHECKSKIP
                endcycle = true;
                break;
            }
        }

        if (endcycle) {
            // completed a full cycle
            ++iter;
            // check for convergence
            if (inorm <= tol) {
                if (check_skip_ledger(n, iw)) {
                    // No coordinates where skipped and tol reached
                    itask = STOP;
                } else {
                    // There is at least one skipped coordinate, reset ledger
                    // and check the coordiante(s)
                    reset_skip_ledger(n, iw);
                    *cheap = false;
                    itask = NEWX;
                    *flags |= 2U;
                }
            } else {
                // indicate that point x can be printed, monitored, etc.
                itask = NEWX;
            }
        }
        // Check to see if a cheap iteration can be performed and check that
        // it is not time to restart... iter=k=0 also does a MV eval.
        // if restart=1 then all iters are MV eval.
        if (*cheap) {
            da_int rst = ((iter * n + k) % restart);
            *flags |= (rst == 0);
            *cheap = !(rst == 0);
        }
        if (!(*cheap)) {
            // request to perform MV operation (expensive iteration)
            action = 1;
        }

        // Update stats
        *skipk = skip[k];
        *skipmaxk = skipmax[k];
        return da_status_success;
        break;
    case NEWX:
        // User did not stop, continue...
        itask = EVAL;
        // reset inorm
        inorm = static_cast<T>(0);
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
