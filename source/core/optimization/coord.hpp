#ifndef COORD_HPP
#define COORD_HPP

#include "aoclda_error.h"
#include "aoclda_result.h"
#include "da_error.hpp"
#include "info.hpp"
#include "options.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#undef max
#undef min
#define HDRCNT 30 // how frequent to print iteration banner

namespace constraints {
enum bound_t { none = 0, lower = 1, both = 2, upper = 3 };

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

namespace coord {

/**
 * 
 * COORD Coordinate Descent solver for Generalized Linear Models with Elastic Net
 * 
 * Problem to solve is
 *   min f(x) subject to l <= x <= u
 * x \in R^n
 * f(x) should be C1 inside the bounding box
 * The solver requires a user call-back that specifies the next iterate for a specific coordinate, say k \in {1..n},
 * and satisfies \nabla x_k f(x) = 0, that is, the next iterate x_k minimizes f with respect to the k-th coordinate.
 */

/* Internal notes
 *
 * Bounds
 * ======
 * After check_bounds
 * if constrained = false
 *  * btyp = not allocated
 *  * lptr and uptr and not set
 * if constrained = true
 *  * vector btyp(n) = {none|lower|upper|both}
 *  * lptr => user l (readonly)
 *  * uptr => user u (readonly)
 */

/* COORD Forward Communication <templated>
 * This is the main entry point for the solver
 * It expects to have coord_data already initialized and
 * all the rest of input to have been validated.
 * 
 * This solver assumes that the data matrix is standardized, that is
 * Standardize so for each column j = 1:n we have
 *   zero-mean, and
 *   sum xij^2 = 1, i=1:m
 *
 * TODO
 * Coord solver
 * ============
 * - [ ] Recovery, restore last iterate and exit.
 * - [ ] Progress tolerance check.
 * - [ ] Add active constraint logic (ledger).
 * - [ ] Test the box bounds.
 * - [ ] Add spare matrix evaluation. PAPER! We have a POC in the octave file
 */

enum solver_tasks { START = 1, NEWX = 2, EVAL = 3, STOP = 4 };

template <typename T>
da_status coord_rcomm(da_int n, std::vector<T> &x, constraints::bound_constr<T> &bc,
                      T factr, T tol, coord::solver_tasks &itask, da_int &k, T &newxk,
                      da_int &iter, T &inorm, da_int &action, da_errors::da_error_t &err,
                      std::vector<T> &w, std::vector<uint32_t> &iw);

template <typename T>
da_status coord(da_options::OptionRegistry &opts, da_int n, std::vector<T> &x,
                std::vector<T> &l, std::vector<T> &u, std::vector<T> &info,
                const stepfun_t<T> stepfun, const monit_t<T> monit, void *usrdata,
                da_errors::da_error_t &err) {

    if (!stepfun)
        return da_error(
            &err, da_status_invalid_pointer,
            "Solver requires a valid pointer to the step function call-back.");
    T bigbnd;
    if (opts.get("infinite bound size", bigbnd))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: <infinite bound size>.");
    T tol;
    if (opts.get("coord convergence tol", tol))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: <coord convergence tol>.");
    T factr;
    if (opts.get("coord progress factor", factr))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: <coord progress factor>.");
    T maxtime;
    if (opts.get("time limit", maxtime))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: <time limit>.");
    da_int prnlvl;
    if (opts.get("print level", prnlvl))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: <print level>.");
    da_int maxit;
    if (opts.get("coord iteration limit", maxit))
        return da_error(&err, da_status_internal_error,
                        "expected option not found: <coord iteration limit>.");
    da_int mon = 0;
    if (monit) // monitor provided
        if (opts.get("monitoring frequency", mon))
            return da_error(&err, da_status_internal_error,
                            "expected option not found: <monitoring frequency>.");

    if (n <= 0) {
        return da_error(&err, da_status_invalid_input,
                        "Number of variables needs to be positive.");
    }

    if (x.size() != (size_t)n) {
        return da_error(&err, da_status_invalid_input,
                        "Vector x needs to be of size n=" + std::to_string(n) + ".");
    }

    da_status status;
    constraints::bound_constr<T> bc;

    // Check and add bound constraints
    status = bc.add(n, l, u, bigbnd, err);
    if (status != da_status_success)
        return status;

    // Work vectors
    std::vector<uint32_t> iw;
    std::vector<T> w;

    // info
    info.resize(optim::info_number);
    std::fill(info.begin(), info.end(), 0);

    da_int iter;
    T *f = &info[optim::info_t::info_objective];
    T *time = &info[optim::info_t::info_time];
    T newxk, inorm;
    da_int k, action;
    size_t hdr = 0, fcnt = 0;

    solver_tasks itask = START;

    const size_t coutprec = std::cout.precision();

    if (prnlvl >= 5) {
        std::cout << "Initial coefficients:" << std::endl;
        for (da_int i = 0; i < n; i++) {
            std::cout << " x[" << i << "] = " << x[i] << std::endl;
        }
    }

    while (itask == NEWX || itask == START || itask == EVAL) {
        coord_rcomm<T>(n, x, bc, factr, tol, itask, k, newxk, iter, inorm, action, err, w, iw);

        switch (itask) {
        case EVAL: // Compute objective and step-length from current X
            // This solver does not have recovery, stop
            // FIXME-FUTURE: restore last valid x (and stats?)
            // FIXME: action > 0 => full eval action < 0 low rank update,
            // action = 0 no feature matrix eval, just step calculation
            if (stepfun(n, &x[0], &newxk, k, f, usrdata, action) != 0) {
                // step could not be evaluated?
                return da_warn(&err, da_status_optimization_num_difficult,
                               "Could not evaluate step at current point.");
            }
            if (action)
                fcnt++;
            // FIXME
            // else if action < 0
            //   lowrk++;
            if (prnlvl >= 4) {
                std::cout.precision(2);
                std::cout << std::setw(10) << "iteration" << std::setw(1) << ""
                          << std::setw(10) << "coordinate" << std::setw(1) << ""
                          << std::setw(9) << "current" << std::setw(1) << ""
                          << std::setw(9) << "new" << std::setw(1) << "" << std::setw(9)
                          << "change" << std::setw(1) << "" << std::setw(9) << std::endl;
                std::cout << std::setw(10) << iter << std::setw(1) << "" << std::setw(10)
                          << k << std::setw(1) << "" << std::scientific << std::showpos
                          << std::setw(9) << x[k] << std::setw(1) << "" << std::setw(9)
                          << newxk << std::setw(1) << "" << std::setw(9) << x[k] - newxk
                          << std::endl
                          << std::noshowpos;
                std::cout.precision(coutprec);
            }
            break;
        case NEWX:
        case STOP: // Copy and print for the last time
            if (prnlvl > 1) {
                if (hdr == 0 || prnlvl >= 4) {
                    hdr = HDRCNT;
                    std::cout << std::setw(42) << std::setfill('-') << "" << std::endl
                              << std::setfill(' ');
                    std::cout << std::setw(10) << "iteration" << std::setw(1) << ""
                              << std::setw(9) << std::scientific << "objective"
                              << std::setw(1) << "" << std::setw(9) << "maxchange"
                              << std::setw(12) << "neval" << std::endl;
                    std::cout << std::setw(42) << std::setfill('-') << "" << std::endl
                              << std::setfill(' ');
                }
                --hdr;
                std::cout.precision(3);
                std::cout << std::setw(10) << iter << std::setw(1) << ""
                          << std::scientific << std::setw(9) << *f << std::setw(1) << ""
                          << std::setw(9) << inorm << std::setw(12) << fcnt << std::endl;
                std::cout.precision(coutprec);
                if (prnlvl >= 5) {
                    std::cout << "Current coefficients:" << std::endl;
                    for (da_int i = 0; i < n; i++) {
                        std::cout << " x[" << i << "] = " << x[i] << std::endl;
                    }
                }
            }

            // Copy all metrics to info
            info[optim::info_t::info_nevalf] = fcnt;
            info[optim::info_t::info_inorm] = inorm;
            info[optim::info_t::info_iter] = iter;

            if (itask == STOP)
                break;

            if (iter >= maxit) {
                return da_warn(
                    &err, da_status_optimization_maxit,
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
                        &err, da_status_optimization_maxtime,
                        "Time limit reached without converging to set tolerance.");
                }
            }
            break;
        default:
            break;
        }
    }
    return status;
}

template <typename T>
da_status coord_rcomm(da_int n, std::vector<T> &x, constraints::bound_constr<T> &bc,
                      [[maybe_unused]] T factr, T tol, coord::solver_tasks &itask,
                      da_int &k, T &newxk, da_int &iter, T &inorm, da_int &action,
                      da_errors::da_error_t &err, std::vector<T> &w,
                      std::vector<uint32_t> &iw) {
    T change;
    switch (itask) {
    case START:

        // Allocate space for the solver
        // integer ledger (size n) + skipchk (size n) + change

        iter = 0;
        // Make initial X feasible
        bc.proj(x);
        // Set initial coordianate to move on
        k = 0;
        // Request to evaluate objective and new update.
        action = 1;
        itask = EVAL;
        return da_status_success;
        break;
    case EVAL:
        if (k == 0)
            inorm = static_cast<T>(0);
        bc.proj(k, newxk);
        change = std::abs(x[k] - newxk);
        // Set callback action to perform
        // action < 0, low rank update using change and kold = -action
        // action = 0, no change in x, don't evaluate feature matrix
        // action > 0, evaluate feature matrix
        // FIXME for now we don't request low rank updates
        action = 1; // FOR NOW ONLY >0 allowed, see step callback. change != 0;
        inorm = std::max(inorm, change);
        // w.ledger[k] = kchange <= factr ? w.ledger[k]++ : 0;
        x[k] = newxk;

        if (k < n - 1) {
            // keep cycling
            k++;
        } else {
            // completed a full cycle
            iter++;
            // check for convergence
            if (inorm <= tol) {
                itask = STOP;
            } else {
                // indicate that point x can be printed, monitored, etc.
                itask = NEWX;
            }
            k = 0;
        }
        return da_status_success;
        break;
    case NEWX:
        // user did not stop, continue
        itask = EVAL;
        return da_status_success;
        break;
    default:
        itask = STOP;
        return da_error(&err, da_status_internal_error, "Unexpected taskid provided?");
        break;
    }
}

} // namespace coord
#endif
