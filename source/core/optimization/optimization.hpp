#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

#include "callbacks.hpp"
#include "da_error.hpp"
#include "info.hpp"
#include "lbfgsb_driver.hpp"
#include "optimization_options.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#if defined(_WIN32)
#include <bitset>
#endif

namespace optim {

using namespace optim;

/* all constraint types
 * mainly used to check wether a specific type of constraint is defined in a bool array
 */
enum cons_type { cons_bounds = 0, cons_linear = 1 };

enum solvers { solver_undefined = 0, solver_lbfgsb };

template <typename T> class da_optimization {
  private:
    // Lock for solver
    bool locked = false;

    // Number of variable
    da_int nvar = 0;

    // Which type constraints are defined (only bound constraint are allowed for now)
    std::bitset<8> constraint_types{0};

    // Bound constraints (allocated only if constraint_types[cons_bound] is set)
    std::vector<T> l, u;

    // Pointers to callbacks
    objfun_t<T> objfun = nullptr;
    objgrd_t<T> objgrd = nullptr;
    monit_t<T> monit = nullptr;

    // Last iterate information
    // Objective function value
    T f = 0.0;
    // Objective function gradient
    std::vector<T> g;
    // Information vector
    std::vector<T> info;

  public:
    // Options
    da_options::OptionRegistry opts;

    // Pointer to error handler
    da_errors::da_error_t *err = nullptr;

    da_optimization(da_status &status, da_errors::da_error_t &err);
    ~da_optimization();
    // Build model to solve
    da_status add_vars(da_int nvar);
    da_status add_bound_cons(std::vector<T> &l, std::vector<T> &u);
    da_status add_objfun(objfun_t<T> usrfun);
    da_status add_objgrd(objgrd_t<T> usrgrd);
    da_status add_monit(monit_t<T> monit);

    // Solver interfaces (only lbfgsb for now)
    da_status solve(std::vector<T> &x, void *usrdata);

    // Retrieve data from solver
    da_status get_info(std::vector<T> &info, std::vector<T> &g);
};

template <typename T>
da_status da_optimization<T>::get_info(std::vector<T> &g, std::vector<T> &info) {
    // blind copy-out of elements in da_optimization
    try {
        g.resize(0);
        // copy
        std::copy(this->g.begin(), this->g.end(), std::back_inserter(g));

        info.resize(0);
        // copy
        std::copy(this->info.begin(), this->info.end(), std::back_inserter(info));
    } catch (...) {
        return da_error(this->err, da_status_operation_failed,
                        "Failed to resize or copy input vectors g or info.");
    }

    return da_status_success;
};

template <typename T>
da_optimization<T>::da_optimization(da_status &status, da_errors::da_error_t &err) {
    // Assuming that err is valid
    this->err = &err;
    try {
        this->info.resize(info_t::info_number);
    } catch (...) {
        status = da_error(&err, da_status_memory_error,
                          "could not resize solver information vector");
    }
    this->info.assign(info_t::info_number, 0);
    status = register_optimization_options<T>(err, opts);
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

// Add bound constraints to the problem
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

template <typename T> da_status da_optimization<T>::add_monit(monit_t<T> monit) {
    if (!monit) {
        return da_status_invalid_pointer;
    }
    this->monit = monit;
    return da_status_success;
}

template <typename T>
da_status da_optimization<T>::solve(std::vector<T> &x, void *usrdata) {

    da_status status;

    // Check if solver is locked (protect agains recursive call)
    if (this->locked)
        return da_error(this->err, da_status_internal_error,
                        "method solve() was called within itself");
    // note that this->nvar == 0 is checked down the line in the solver driver

    if (x.size() != 0 && x.size() != (size_t)nvar)
        return da_error(this->err, da_status_invalid_input,
                        "initial starting point x0 is of wrong length, must be of either "
                        "length 0 or nvar=" +
                            std::to_string(this->nvar));

    if (x.size() == 0) {
        // no initial point provided, resize and set to zero.
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
    if (opts.get("print level", prnlvl) != da_status_success)
        return da_error(this->err, da_status_internal_error,
                        "expected option not found: print options");

    // Print options
    std::string s;
    if (opts.get("print options", s) != da_status_success)
        return da_error(this->err, da_status_internal_error,
                        "expected option not found: print options");

    // Select_solver based on problem and options
    // Decision tree
    optim::solvers solver = solver_lbfgsb;

    switch (solver) {
    case solver_lbfgsb:
        if (prnlvl > 0) {
            std::cout << "-----------------------------------------------------\n"
                      << "    AOCL-DA L-BFGS-B Nonlinear Programming Solver\n"
                      << "-----------------------------------------------------\n";
        }
        if (s == "yes")
            opts.print_options();
        // derivative based solver, allocate gradient memory
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
    return status;
}

} // namespace optim

#endif