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

#include "aoclda_error.h"
#include "coord.hpp"
#include "da_error.hpp"
#include "lbfgsb.hpp"
#include "macros.h"
#include "options.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace ARCH {

// L-BFGS-B Reverse Communication <overloaded>
inline void lbfgsb_rcomm([[maybe_unused]] da_int *n, [[maybe_unused]] da_int *m,
                         [[maybe_unused]] double *x, [[maybe_unused]] double *l,
                         [[maybe_unused]] double *u, [[maybe_unused]] da_int *nbd,
                         [[maybe_unused]] double *f, [[maybe_unused]] double *g,
                         [[maybe_unused]] double *factr, [[maybe_unused]] double *pgtol,
                         [[maybe_unused]] double *wa, [[maybe_unused]] da_int *iwa,
                         [[maybe_unused]] da_int *itask, [[maybe_unused]] da_int *iprint,
                         [[maybe_unused]] da_int *lsavei, [[maybe_unused]] da_int *isave,
                         [[maybe_unused]] double *dsave) {
#ifndef NO_FORTRAN
    DLBFGSB_SOLVER(n, m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, itask, iprint, lsavei,
                   isave, dsave);
#endif
}

inline void lbfgsb_rcomm([[maybe_unused]] da_int *n, [[maybe_unused]] da_int *m,
                         [[maybe_unused]] float *x, [[maybe_unused]] float *l,
                         [[maybe_unused]] float *u, [[maybe_unused]] da_int *nbd,
                         [[maybe_unused]] float *f, [[maybe_unused]] float *g,
                         [[maybe_unused]] float *factr, [[maybe_unused]] float *pgtol,
                         [[maybe_unused]] float *wa, [[maybe_unused]] da_int *iwa,
                         [[maybe_unused]] da_int *itask, [[maybe_unused]] da_int *iprint,
                         [[maybe_unused]] da_int *lsavei, [[maybe_unused]] da_int *isave,
                         [[maybe_unused]] float *dsave) {
#ifndef NO_FORTRAN
    SLBFGSB_SOLVER(n, m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, itask, iprint, lsavei,
                   isave, dsave);
#endif
}

/* Internal memory for lbfgsb */
template <typename T> class lbfgsb_work {
  public:
    da_int *nbd = nullptr;
    da_int *iwa = nullptr;
    T *wa = nullptr;

    lbfgsb_work(size_t mem, size_t nvar, da_status &status);
    ~lbfgsb_work();

    da_status add_bounds(size_t nvar, const std::vector<T> &l, const std::vector<T> &u,
                         T bigbnd);
};

template <typename T>
lbfgsb_work<T>::lbfgsb_work(size_t mem, size_t nvar, da_status &status) {
    if (mem < 1) {
        status = da_status_invalid_input;
        return;
    }
    try {
        this->iwa = new da_int[3 * nvar];
    } catch (std::bad_alloc &) {
        status = da_status_memory_error;
        return;
    }
    size_t nwa = 2 * mem * nvar + 5 * nvar + 11 * mem * mem + 8 * mem;
    try {
        this->wa = new T[nwa];
    } catch (std::bad_alloc &) {
        status = da_status_memory_error;
        return;
    }
    try {
        this->nbd = new da_int[nvar];
    } catch (std::bad_alloc &) {
        status = da_status_memory_error;
        return;
    }

    status = da_status_success;
    return;
}

template <typename T> lbfgsb_work<T>::~lbfgsb_work() {
    if (this->iwa)
        delete[] this->iwa;
    if (this->wa)
        delete[] this->wa;
    if (this->nbd)
        delete[] this->nbd;
}

template <typename T>
da_status lbfgsb_work<T>::add_bounds(size_t nvar, const std::vector<T> &l,
                                     const std::vector<T> &u, T bigbnd) {
    if (nvar < 1 || std::isnan(bigbnd) || bigbnd <= 0) {
        return da_status_invalid_input;
    }
    if ((l.size() != 0 && l.size() != nvar) || (u.size() != 0 && u.size() != nvar)) {
        return da_status_invalid_input;
    }
    if (!(this->nbd))
        return da_status_memory_error;

    size_t i;
    if (l.size() == 0 && u.size() == 0) {
        // No bounds defined
        for (i = 0; i < nvar; i++)
            this->nbd[i] = 0;
    } else if (l.size() == 0) {
        // Only upper bounds, set lower to -infinity
        for (i = 0; i < nvar; i++) {
            if (u[i] < bigbnd)
                this->nbd[i] = 3;
            else
                this->nbd[i] = 0;
        }
    } else if (u.size() == 0) {
        // Only lower bounds, set lower to -infinity
        for (i = 0; i < nvar; i++) {
            if (l[i] > -bigbnd)
                this->nbd[i] = 1;
            else
                this->nbd[i] = 0;
        }
    } else {
        // Both bounds present
        for (i = 0; i < nvar; i++) {
            if (l[i] > -bigbnd && u[i] < bigbnd)
                this->nbd[i] = 2;
            else if (l[i] > -bigbnd)
                this->nbd[i] = 1;
            else if (u[i] < bigbnd)
                this->nbd[i] = 3;
            else
                this->nbd[i] = 0;
        }
    }
    return da_status_success;
}

/* L-BFGS-B Forward Communication <templated>
 * This is the main entry point for the solver
 * It expects to have lbfgsb_data already initialized and
 * all the rest of input to have been validated. ????
 *
 * Requirements: mem: memory size, factr, pgtol, iprint
 */
template <typename T>
da_status lbfgsb_fcomm(da_options::OptionRegistry &opts, da_int nvar, std::vector<T> &x,
                       std::vector<T> &l, std::vector<T> &u, std::vector<T> &info,
                       std::vector<T> &g, objfun_t<T> objfun, objgrd_t<T> objgrd,
                       monit_t<T> monit, void *usrdata, da_errors::da_error_t &err) {

    if (!objfun)
        return da_error(
            &err, da_status_invalid_pointer,
            "NLP solver requires a valid pointer to the objective function call-back");
    if (!objgrd)
        return da_error(&err, da_status_not_implemented,
                        "NLP solver requires a valid pointer to the objective gradient "
                        "function call-back");
    da_int m;
    if (opts.get("lbfgsb memory limit", m))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: lbfgsb memory limit");
    T bigbnd;
    if (opts.get("infinite bound size", bigbnd))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: infinite bound size");
    T pgtol;
    if (opts.get("lbfgsb convergence tol", pgtol))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: lbfgsb convergence tol");
    T factr;
    if (opts.get("lbfgsb progress factor", factr))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: lbfgsb progress factor");
    T maxtime;
    if (opts.get("time limit", maxtime))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: time limit");
    da_int prnlvl;
    if (opts.get("print level", prnlvl))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: print level");
    da_int maxit;
    if (opts.get("lbfgsb iteration limit", maxit))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: lbfgsb iteration limit");
    da_int mon = 0;
    if (monit) // Monitor provided
        if (opts.get("monitoring frequency", mon))
            return da_error(&err, da_status_internal_error,
                            "expected option not found: monitoring frequency");

    da_status status;
    lbfgsb_work<T> *w = new lbfgsb_work<T>(m, nvar, status);
    if (status != da_status_success) {
        delete w;
        return da_error(&err, da_status_memory_error,
                        "unable to allocate memory for solving the problem");
    }

    if (w->add_bounds(nvar, l, u, bigbnd) != da_status_success) {
        delete w;
        return da_error(&err, da_status_internal_error,
                        "add_bounds() did not return success");
    }

    da_int n = nvar;
    da_int iprint;
    da_int iter = 0;
    T *f = &info[da_linmod_info_t::linmod_info_objective];
    da_int itask = 2; // 'START'
    bool compute_fg = true;
    da_int lsavei[4], isave[44];
    T dsave[29];

    switch (prnlvl) {
    case 0:
        // No output
        iprint = -1;
        break;
    case 1:
        // Summary only
        iprint = 0;
        break;
    case 2:
        // 1-liner every 30 iters
        iprint = 30;
        break;
    case 3:
        // 1-liner at each iter
        iprint = 1;
        break;
    case 4:
    case 5:
        iprint = 100;
        break;
    default:
        delete w;
        return da_error(&err, da_status_internal_error, "print level is out of range");
    }

    while (itask == 2 || itask == 1 || compute_fg) {
        lbfgsb_rcomm(&n, &m, x.data(), l.data(), u.data(), &(w->nbd[0]), &f[0], &g[0],
                     &factr, &pgtol, &w->wa[0], &w->iwa[0], &itask, &iprint, &lsavei[0],
                     &isave[0], &dsave[0]);
        if (itask == 1) { // NEW_X
            iter++;
            info[da_linmod_info_t::linmod_info_iter] = static_cast<T>(iter);
            info[da_linmod_info_t::linmod_info_grad_norm] = dsave[12]; // sbgnrm
            info[da_linmod_info_t::linmod_info_time] = dsave[6] + dsave[7] + dsave[8];

            if (iter > maxit) {
                itask = 100;
            }

            // Copy to info all relevant metrics
            if (mon != 0) {
                if (iter % mon == 0) {
                    // Call monitor
                    if (monit(n, &x[0], &g[0], &info[0], usrdata) != 0) {
                        // User request to stop
                        itask = 3;
                    }
                }
            }

            if (maxtime > 0) {
                if (info[da_linmod_info_t::linmod_info_time] > maxtime) {
                    // Run out of time
                    itask = 101;
                }
            }
        }
        compute_fg = itask == 4 ||  // 'FG'
                     itask == 21 || // 'FG_START'
                     itask == 20;   // 'FG_LNSRCH
        if (compute_fg) {
            ++info[da_linmod_info_t::linmod_info_nevalf];
            if (objfun(n, &x[0], f, usrdata) != 0) {
                // This solver does not have recovery, stop
                itask = 120;
            }
            if (objgrd(n, &x[0], &g[0], usrdata, 0)) {
                // This solver does not have recovery, stop
                itask = 121;
            }
        }
    }

    delete w;

    // Select correct exit status
    switch (itask) {
    case 6:
        // 'CONVERGENCE', 6 out:success
    case 7:
        // 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL', 7 out:success
    case 8:
        // 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH', 8 out:success
        return da_status_success;
        break;
    case 3:
        // 'STOP', 3 action: stop => user_request_stop
        return da_warn(&err, da_status_optimization_usrstop,
                       "User requested to stop optimization process");
        break;
    case 14:
        return da_error(&err, da_status_optimization_empty_space,
                        "No variables defined in the problem");
        break;
    case 15:
        return da_error(&err, da_status_optimization_infeasible, "Problem is infeasible");
        break;
    case 5:
        return da_warn(&err, da_status_numerical_difficulties,
                       "ABNORMAL_TERMINATION_IN_LNSRCH");
        break;
    case 9:
        return da_warn(&err, da_status_numerical_difficulties, "RESTART_FROM_LNSRCH");
        break;
    case 16:
        return da_warn(&err, da_status_numerical_difficulties, "ERROR: STP .GT. STPMAX");
        break;
    case 17:
        return da_warn(&err, da_status_numerical_difficulties, "ERROR: STP .LT. STPMIN");
        break;
    case 18:
        return da_warn(&err, da_status_numerical_difficulties,
                       "ERROR: STPMAX .LT. STPMIN");
        break;
    case 19:
        return da_warn(&err, da_status_numerical_difficulties, "ERROR: STPMIN .LT. ZERO");
        break;
    case 23:
        return da_warn(&err, da_status_numerical_difficulties,
                       "WARNING: ROUNDING ERRORS PREVENT PROGRESS");
        break;
    case 24:
        return da_warn(&err, da_status_numerical_difficulties, "WARNING: STP = STPMAX");
        break;
    case 25:
        return da_warn(&err, da_status_numerical_difficulties, "WARNING: STP = STPMIN");
        break;
    case 26:
        return da_warn(&err, da_status_numerical_difficulties,
                       "WARNING: XTOL TEST SATISFIED");
        break;
    case 10: // This can't happen due to options range check
        return da_error(&err, da_status_internal_error, "ERROR: FTOL < ZERO");
        break;
    case 11: // This can't happen due to options range check
        return da_error(&err, da_status_internal_error, "ERROR: GTOL .LT. ZERO");
        break;
    case 12:
        return da_error(&err, da_status_internal_error, "ERROR: INITIAL G .GE. ZERO");
        break;
    case 13: // This can't happen due to options range check
        return da_error(&err, da_status_internal_error, "ERROR: INVALID NBD");
        break;
    case 27: // This can't happen due to options range check
        return da_error(&err, da_status_internal_error, "ERROR: FACTR .LT. 0");
        break;
    case 28: // This can't happen due to options range check
        return da_error(
            &err, da_status_internal_error,
            "Limited memory amount must be zero or more. Recommended limit is 11");
        break;
    case 100: // This is external to LBFGSB: max it
        return da_warn(&err, da_status_maxit,
                       "Iteration limit reached without converging to set tolerance");
        break;
    case 101: // This is external to LBFGSB: max time limit
        return da_warn(&err, da_status_maxtime,
                       "Time limit reached without converging to set tolerance");
        break;
    case 120: // This is external to LBFGSB: Objective function callback return status != 0
        // no recovery implemented
        return da_warn(&err, da_status_option_invalid_value,
                       "User objective function could not be evaluated at a latest trial "
                       "point and no recovery process is implemented. ");
        break;
    case 121: // This is external to LBFGSB: Objective gradient callback return status != 0
        // no recovery implemented
        return da_warn(&err, da_status_option_invalid_value,
                       "User objective gradient could not be evaluated at a latest trial "
                       "point and no recovery process is implemented. ");
        break;

    case 1:  // 'NEW_X', 1  action: monitor => possible user request to stop
    case 2:  // 'START', 2  action: 1st iteration
    case 4:  // 'FG',    4  action: evaluate f+g
    case 20: // 'FG_LNSRCH', 20 action: evaluate f+g
    case 21: // 'FG_START', 21 action: evaluate f+g
    case 22: // 'ERROR: XTOL .LT. ZERO', 22 internal use only -> NEW_X
    default:
        return da_error(&err, da_status_internal_error,
                        "Unknown optimization task id at exit: " + std::to_string(itask));
        break;
    }

    return da_status_success;
}

} // namespace ARCH
