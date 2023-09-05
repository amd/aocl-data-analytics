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
#include "options.hpp"
#include <limits>

// Needed for windows build
#undef min
#undef max

namespace optim {
enum solvers { solver_undefined = 0, solver_lbfgsb = 1, solver_qr = 2, solver_coord = 3 };
}

template <class T>
inline da_status register_optimization_options(da_errors::da_error_t &err,
                                               da_options::OptionRegistry &opts) {
    using namespace da_options;
    T safe_eps = (T)2.0 * std::numeric_limits<T>::epsilon();
    T safe_tol = std::sqrt(safe_eps);
    [[maybe_unused]] T max_real = std::numeric_limits<T>::max();
    da_int imax = std::numeric_limits<da_int>::max();

    try {
        // ===========================================================================
        // INTEGER OPTIONS
        // ===========================================================================
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "coord iteration limit", "Maximum number of iterations to perform", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::lessequal,
            imax));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "lbfgsb iteration limit", "Maximum number of iterations to perform", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::lessequal,
            imax));
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
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::lessequal,
            0));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "print level", "set level of verbosity for the solver TODO explain levels", 0,
            da_options::lbound_t::greaterequal, 5, da_options::ubound_t::lessequal, 1));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "debug", "set debug level (internal use)", 0,
            da_options::lbound_t::greaterequal, 3, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        // ===========================================================================
        // REAL OPTIONS
        // ===========================================================================
        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("time limit", "maximum time allowed to run", 0.0,
                             da_options::lbound_t::greaterthan, 0,
                             da_options::ubound_t::p_inf, static_cast<T>(1.0e6)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "infinite bound size", "threshold value to take for +/- infinity", 1000,
            da_options::lbound_t::greaterthan, 0, da_options::ubound_t::p_inf,
            static_cast<T>(1.0e20)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "lbfgsb convergence tol",
            "tolerance of the projected gradient infinity norm to declare convergence",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            safe_tol));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "lbfgsb progress factor",
            "the iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= "
            "factr*epsmch"
            " where epsmch is the machine precision. Typical values for type double: "
            "1.e+12 for"
            " low accuracy; 1.e+7 for moderate accuracy; 1.e+1 for extremely"
            " high accuracy.",
            0.0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<T>(10.0) / safe_tol));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "coord convergence tol",
            "tolerance of the projected gradient infinity norm to declare convergence",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            safe_tol));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "coord progress factor",
            "the iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= "
            "factr*epsmch"
            " where epsmch is the machine precision. Typical values for type double: "
            "1.e+12 for"
            " low accuracy; 1.e+7 for moderate accuracy; 1.e+1 for extremely"
            " high accuracy.",
            0.0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<T>(10.0) / safe_tol));
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
                         {{"lbfgsb", optim::solvers::solver_lbfgsb},
                          {"lbfgs", optim::solvers::solver_lbfgsb},
                          {"bfgs", optim::solvers::solver_lbfgsb},
                          {"coord", optim::solvers::solver_coord}},
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