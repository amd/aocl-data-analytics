/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef OPTIMIZATION_OPTIONS_HPP
#define OPTIMIZATION_OPTIONS_HPP

#include "da_error.hpp"
#include "optim_types.hpp"
#include "options.hpp"
#include <limits>

// Needed for windows build
#undef min
#undef max

template <class T>
inline da_status register_optimization_options(da_errors::da_error_t &err,
                                               da_options::OptionRegistry &opts) {
    using namespace da_options;

    try {
        // ===========================================================================
        // INTEGER OPTIONS
        // ===========================================================================
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("coord skip min",
                                  "Minimum times a coordinate change is smaller than "
                                  "\"coord skip tol\" to start skipping",
                                  1, da_options::lbound_t::greaterequal, max_da_int,
                                  da_options::ubound_t::p_inf, 5));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("coord skip max",
                                  "Initial max times a coordinate can be skipped after "
                                  "this the coordinate is checked",
                                  4, da_options::lbound_t::greaterequal, max_da_int,
                                  da_options::ubound_t::p_inf, 8));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "coord restart",
            "Number of inner iteration to perform before requesting to perform a full "
            "evaluation of the step function",
            0, da_options::lbound_t::greaterequal, max_da_int,
            da_options::ubound_t::p_inf, max_da_int, "\\infty"));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "coord iteration limit", "Maximum number of iterations to perform", 1,
            da_options::lbound_t::greaterequal, max_da_int, da_options::ubound_t::p_inf,
            100000));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "lbfgsb iteration limit", "Maximum number of iterations to perform", 1,
            da_options::lbound_t::greaterequal, max_da_int, da_options::ubound_t::p_inf,
            10000));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "lbfgsb memory limit",
            "Number of vectors to use for approximating the Hessian", 1,
            da_options::lbound_t::greaterequal, 1000, da_options::ubound_t::lessequal,
            11));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "monitoring frequency",
            "How frequent to call the user-supplied monitor function", 0,
            da_options::lbound_t::greaterequal, max_da_int, da_options::ubound_t::p_inf,
            0));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("print level",
                                  "set level of verbosity for the solver 0 indicates no "
                                  "output while 5 is a very verbose printing",
                                  0, da_options::lbound_t::greaterequal, 5,
                                  da_options::ubound_t::lessequal, 1));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "debug", "set debug level (internal use)", 0,
            da_options::lbound_t::greaterequal, 3, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        // ===========================================================================
        // REAL OPTIONS
        // ===========================================================================
        // Tolerance based on sqrt(safe_epsilon)
        da_options::safe_tol<T> tol;
        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("time limit", "maximum time allowed to run", 0.0,
                             da_options::lbound_t::greaterthan, 0,
                             da_options::ubound_t::p_inf, static_cast<T>(1.0e6), "10^6"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "infinite bound size", "threshold value to take for +/- infinity", 1000,
            da_options::lbound_t::greaterthan, 0, da_options::ubound_t::p_inf,
            static_cast<T>(1.0e20), "10^{20}"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "lbfgsb convergence tol",
            "tolerance of the projected gradient infinity norm to "
            "declare convergence",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            tol.safe_eps(), tol.safe_eps_latex()));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "lbfgsb progress factor",
            "the iteration stops when (f^k - f{k+1})/max{abs(fk);abs(f{k+1});1} <= "
            "factr*epsmch"
            " where epsmch is the machine precision. Typical values for type double: "
            "10e12 for"
            " low accuracy; 10e7 for moderate accuracy; 10 for extremely"
            " high accuracy.",
            0.0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            tol.safe_inveps((T)10, (T)1), tol.safe_inveps_latex((T)10, (T)1)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "coord convergence tol",
            "tolerance of the projected gradient infinity norm to declare convergence",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            tol.safe_eps(), tol.safe_eps_latex()));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "coord skip tol", "Coordinate skip tolerance", 0.0,
            da_options::lbound_t::greaterthan, 0, da_options::ubound_t::p_inf,
            tol.safe_eps(), tol.safe_eps_latex()));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "coord progress factor",
            "the iteration stops when (fk - f{k+1})/max{abs(fk);abs(f{k+1});1} <= "
            "factr*epsmch"
            " where epsmch is the machine precision. Typical values for type double: "
            "10e12 for"
            " low accuracy; 10e7 for moderate accuracy; 10 for extremely"
            " high accuracy.",
            0.0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            tol.safe_inveps((T)10, (T)1), tol.safe_inveps_latex((T)10, (T)1)));
        opts.register_opt(oT);

        // ===========================================================================
        // STRING OPTIONS
        // ===========================================================================
        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(OptionString(
            "print options", "Print options list", {{"yes", 1}, {"no", 0}}, "no"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(
            OptionString("optim method", "Select optimization solver to use",
                         {{"lbfgsb", da_optim::solvers::solver_lbfgsb},
                          {"lbfgs", da_optim::solvers::solver_lbfgsb},
                          {"bfgs", da_optim::solvers::solver_lbfgsb},
                          {"coord", da_optim::solvers::solver_coord}},
                         "lbfgsb"));
        opts.register_opt(os);

    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error,
                        "Memory allocation failed"); // LCOV_EXCL_LINE
    } catch (...) {                                  // LCOV_EXCL_LINE
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_error(&err, da_status_internal_error,
                        "Unexpected internal error"); // LCOV_EXCL_LINE
    }

    return da_status_success;
}

#endif //OPTIMIZATION_OPTIONS_HPP