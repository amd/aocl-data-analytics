/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

// Deal with some Windows compilation issues regarding max/min macros
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "basic_handle.hpp"
#include "callbacks.hpp"
#include "coord.hpp"
#include "da_error.hpp"
#include "lbfgsb_driver.hpp"
#include "optimization_options.hpp"
#include "ralfit_driver.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <bitset>
#endif

namespace da_optim {

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
    void refresh() { model_trained = false; };

    // Build model to solve
    da_status add_vars(da_int nvar);
    da_status add_res(da_int nres);
    da_status add_bound_cons(std::vector<T> &l, std::vector<T> &u);
    da_status add_bound_cons(da_int nvar, T *l, T *u);
    da_status add_weights(da_int lw, T *w);
    da_status add_objfun(objfun_t<T> usrfun);
    da_status add_objgrd(objgrd_t<T> usrgrd);
    da_status add_stepfun(stepfun_t<T> usrstep);
    da_status add_monit(monit_t<T> monit);
    da_status add_resfun(resfun_t<T> resfun);
    da_status add_resgrd(resgrd_t<T> resgrd);
    da_status add_reshes(reshes_t<T> reshes);
    da_status add_reshp(reshp_t<T> reshp);

    // Solver interfaces (only lbfgsb for now)
    da_status solve(std::vector<T> &x, void *usrdata);

    // Update info
    da_status set_info(da_int idx, const T value) {
        if (0 <= idx && idx < (da_int)info.size()) {
            info[idx] = value;
            return da_status_success;
        }
        return da_error(this->err, da_status_internal_error, "info index out-of-bounds?");
    }

    // Retrieve data from solver
    da_status get_info(da_int &dim, T info[]);

    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] T *result) {
        if (!this->model_trained)
            return da_warn(this->err, da_status_unknown_query,
                           "Handle does not contain data relevant to this query. Was the "
                           "last call to the solver successful?");
        switch (query) {
        case da_result::da_rinfo:
            return this->get_info(*dim, result);
            break;
        default:
            return da_warn(this->err, da_status_unknown_query,
                           "The requested result could not be queried by this handle.");
        }
    }

    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result) {
        return da_error( // LCOV_EXCL_LINE
            this->err, da_status_unknown_query,
            "Handle does not contain data relevant to this query.");
    }
};

template <typename T> da_status da_optimization<T>::get_info(da_int &dim, T info[]) {
    // Blind copy-out of elements in da_optimization
    const da_int ilen{(da_int)this->info.size()};
    const da_int mlen{std::max(ilen, da_int(100))};
    if (dim < mlen) {
        dim = mlen;
        return da_warn(
            this->err, da_status_operation_failed,
            "Failed to copy info array, make sure info is of length at least " +
                std::to_string(mlen));
    }
    da_int i;
    for (i = 0; i < ilen; ++i)
        info[i] = this->info[i];
    for (; i < mlen; ++i)
        info[i] = T(0);

    return da_status_success;
};

template <typename T>
da_optimization<T>::da_optimization(da_status &status, da_errors::da_error_t &err) {
    // Assuming that err is valid
    this->err = &err;
    try {
        this->info.resize(da_optim_info_t::info_number);
    } catch (...) {
        status = da_error(&err, da_status_memory_error,
                          "could not resize solver information vector");
    }
    this->info.assign(da_optim_info_t::info_number, 0);
    status = register_optimization_options<T>(*this->err, this->opts);
};

template <typename T> da_optimization<T>::~da_optimization(){};

// Add variables to the problem
template <typename T> da_status da_optimization<T>::add_vars(da_int nvar) {
    if (nvar <= 0) {
        return da_error(this->err, da_status_invalid_input,
                        "Search space dimension must be positive, set nvar > 0");
    }

    this->nvar = nvar;
    return da_status_success;
}

// Add equation or residual number to the problem
template <typename T> da_status da_optimization<T>::add_res(da_int nres) {
    if (nres <= 0) {
        return da_error(this->err, da_status_invalid_input,
                        "Number of residuals must be positive, set nres > 0");
    }

    this->nres = nres;
    return da_status_success;
}

// Add bound constraints to the problem (copy into opt handle)
template <typename T>
da_status da_optimization<T>::add_bound_cons(std::vector<T> &l, std::vector<T> &u) {

    if (l.size() != (size_t)(this->nvar) || u.size() != (size_t)(this->nvar)) {
        return da_error(this->err, da_status_invalid_input,
                        "Constraint vectors l or u are of the wrong size.");
    }
    try {
        this->l.resize(this->nvar);
        this->u.resize(this->nvar);
    } catch (const std::bad_alloc &) {
        return da_error(this->err, da_status_memory_error, "Memory allocation failed");
    } catch (...) {
        return da_error(this->err, da_status_internal_error,
                        "Resize of bound vectors failed");
    }

    // Quick check on bounds
    for (da_int i = 0; i < this->nvar; i++) {
        if (std::isnan(l[i])) {
            return da_error(this->err, da_status_option_invalid_bounds,
                            "Constraint l[" + std::to_string(i) + "] is NaN.");
        }
        if (std::isnan(u[i])) {
            return da_error(this->err, da_status_option_invalid_bounds,
                            "Constraint u[" + std::to_string(i) + "] is NaN.");
        }
        if (l[i] > u[i]) {
            return da_error(this->err, da_status_option_invalid_bounds,
                            "Constraint l[" + std::to_string(i) + "] > u[" +
                                std::to_string(i) + "].");
        }
        this->l[i] = l[i];
        this->u[i] = u[i];
    }

    // All checks passed: mark that there are bound constraints
    this->constraint_types[size_t(cons_bounds)] = true;

    return da_status_success;
}

// Add bound constraints to the problem (get pointer to user data)
template <typename T>
da_status da_optimization<T>::add_bound_cons(da_int nvar, T *l, T *u) {
    if (nvar == 0) {
        this->l_usrptr = nullptr;
        this->u_usrptr = nullptr;
    } else if (this->nvar == nvar) {
        this->l_usrptr = l;
        this->u_usrptr = u;
    } else {
        return da_error(this->err, da_status_invalid_input,
                        "Invalid size of nvar, it must match zero or the number of "
                        "variables defined: " +
                            std::to_string(this->nvar) + ".");
    }
    return da_status_success;
}

// Add vector of weights to the problem (get pointer to user data)
template <typename T> da_status da_optimization<T>::add_weights(da_int lw, T *w) {
    if (lw == 0)
        this->w_usrptr = nullptr;
    else if (w == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "w must be a valid pointer");
    else if (lw == this->nres)
        this->w_usrptr = w;
    else
        return da_error(this->err, da_status_invalid_input,
                        "Invalid size of lw, it must match zero or the "
                        "number of residuals defined: " +
                            std::to_string(this->nres) + ".");
    return da_status_success;
}

template <typename T> da_status da_optimization<T>::add_objfun(objfun_t<T> usrfun) {
    if (!usrfun) {
        return da_status_invalid_pointer;
    }
    objfun = usrfun;
    return da_status_success;
}

template <typename T> da_status da_optimization<T>::add_objgrd(objgrd_t<T> usrgrd) {
    if (!usrgrd) {
        return da_status_invalid_pointer;
    }
    objgrd = usrgrd;
    return da_status_success;
}

template <typename T> da_status da_optimization<T>::add_stepfun(stepfun_t<T> usrstep) {
    if (!usrstep) {
        return da_status_invalid_pointer;
    }
    stepfun = usrstep;
    return da_status_success;
}

template <typename T> da_status da_optimization<T>::add_monit(monit_t<T> monit) {
    if (!monit) {
        return da_status_invalid_pointer;
    }
    this->monit = monit;
    return da_status_success;
}

template <typename T> da_status da_optimization<T>::add_resfun(resfun_t<T> resfun) {
    if (!resfun) {
        return da_status_invalid_pointer;
    }
    this->resfun = resfun;
    return da_status_success;
}

template <typename T> da_status da_optimization<T>::add_resgrd(resgrd_t<T> resgrd) {
    this->resgrd = resgrd;
    return da_status_success;
}

template <typename T> da_status da_optimization<T>::add_reshes(reshes_t<T> reshes) {
    this->reshes = reshes;
    return da_status_success;
}

template <typename T> da_status da_optimization<T>::add_reshp(reshp_t<T> reshp) {
    this->reshp = reshp;
    return da_status_success;
}

template <typename T>
da_status da_optimization<T>::solve(std::vector<T> &x, void *usrdata) {

    da_status status;

    // Check if solver is locked (protect against recursive call)
    if (this->locked)
        return da_error(this->err, da_status_internal_error,
                        "method solve() was called within itself");
    // Note that this->nvar == 0 is checked down the line in the solver driver

    if (x.size() != 0 && x.size() != (size_t)nvar)
        return da_error(this->err, da_status_invalid_input,
                        "initial starting point x0 is of wrong length, must be of either "
                        "length 0 or nvar=" +
                            std::to_string(this->nvar));

    if (x.size() == 0) {
        // No initial point provided, resize and set to zero.
        try {
            x.resize(this->nvar);
        } catch (std::bad_alloc &) {
            return da_error(this->err, da_status_memory_error,
                            "Could not allocate memory for initial iterate x");
        }
        x.assign(this->nvar, 0);
    }

    // Lock solver
    this->locked = true;

    // Print welcome banner
    da_int prnlvl;
    if (this->opts.get("print level", prnlvl) != da_status_success)
        return da_error(this->err, da_status_internal_error,
                        "expected option not found: print options");

    // Print options
    std::string prn;
    if (this->opts.get("print options", prn) != da_status_success)
        return da_error(this->err, da_status_internal_error,
                        "expected option not found: print options");

    // Select_solver based on problem and options
    da_int solver;
    std::string solvname;
    if (this->opts.get("optim method", solvname, solver) != da_status_success)
        return da_error(this->err, da_status_internal_error,
                        "expected option not found: optim method");

    switch (solver) {
    case solver_lbfgsb:
        if (prnlvl > 0) {
            std::cout << "-----------------------------------------------------\n"
                      << "    AOCL-DA L-BFGS-B Nonlinear Programming Solver\n"
                      << "-----------------------------------------------------\n";
        }
        if (prn == "yes")
            this->opts.print_options();
        // Derivative based solver, allocate gradient memory
        try {
            this->g.resize(this->nvar);
        } catch (std::bad_alloc &) {
            return da_error(this->err, da_status_memory_error,
                            "Could not allocate memory for gradient vector");
        }
        status =
            lbfgsb_fcomm(this->opts, this->nvar, x, this->l, this->u, this->info, this->g,
                         this->objfun, this->objgrd, this->monit, usrdata, *this->err);

        break;
    case solver_coord:
        if (prnlvl > 0) {
            std::cout << "-----------------------------------------------------------\n"
                      << " AOCL-DA COORD Generalized Linear Model Elastic Net Solver\n"
                      << "-----------------------------------------------------------\n";
        }
        if (prn == "yes")
            this->opts.print_options();
        status = coord::coord(this->opts, this->nvar, x, this->l, this->u, this->info,
                              this->stepfun, this->monit, usrdata, *this->err);
        break;
    case solver_ralfit:
        if (prnlvl > 0) {
            std::cout << " ------------------------------------------------------\n"
                      << "     AOCL-DA NLP Solver for Nonlinear Least-Squares    \n"
                      << " ------------------------------------------------------\n";
        }
        if (prn == "yes")
            this->opts.print_options();

        status = ralfit::ralfit_driver(this->opts, this->nvar, this->nres, x.data(),
                                       this->resfun, this->resgrd, this->reshes,
                                       this->reshp, this->l_usrptr, this->u_usrptr,
                                       this->w_usrptr, usrdata, this->info, *this->err);
        break;
    case solver_undefined:
        status = da_error(
            this->err, da_status_internal_error,
            "No NLP solver compatible with the problem type and selected options");
        break;
    default:
        status = da_error(this->err, da_status_internal_error,
                          "No NLP solver with id: " + std::to_string(solver) +
                              "is implemented");
        break;
    }

    // Unlock solver
    this->locked = false;
    return status; // Error message already loaded
}

} // namespace da_optim

#endif
