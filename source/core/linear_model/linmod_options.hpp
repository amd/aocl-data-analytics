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

#ifndef LINMOD_OPTIONS_HPP
#define LINMOD_OPTIONS_HPP

#include "optimization.hpp"
#include "options.hpp"
#include <limits>

// Needed for windows build
#undef min
#undef max

template <class T>
inline da_status register_linmod_options(da_options::OptionRegistry &opts) {
    using namespace da_options;
    T safe_eps = (T)2 * std::numeric_limits<T>::epsilon();
    T safe_tol = std::sqrt(safe_eps);
    T max_real = std::numeric_limits<T>::max();
    da_int imax = std::numeric_limits<da_int>::max();

    try {

        std::shared_ptr<OptionNumeric<da_int>> oi;

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "print level", "set level of verbosity for the solver", 0,
            da_options::lbound_t::greaterequal, 5, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "linmod intercept", "Add intercept variable to the model", 0,
            da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "linmod optim iteration limit",
            "Maximum number of iterations to perform in the optimization phase. Valid "
            "only for iterative solvers, e.g. L-BFGS-B, Coordinate Descent, etc.",
            1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::lessequal,
            imax));
        opts.register_opt(oi);

        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("linmod alpha",
                             "coefficient of alpha in the regularization terms: lambda( "
                             "(1-alpha) L2 + alpha L1 )",
                             0.0, da_options::lbound_t::greaterequal, 1.0,
                             da_options::ubound_t::lessequal, 0.0));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("linmod lambda",
                             "penalty coefficient for the regularization terms: lambda( "
                             "(1-alpha) L2 + alpha L1 )",
                             0.0, da_options::lbound_t::greaterequal, max_real,
                             da_options::ubound_t::p_inf, 0.0));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "linmod optim convergence tol",
            "tolerance to declare convergence for the iterative optimization step. See "
            "option in the corresponding optimization solver documentation.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            safe_tol));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "linmod optim progress factor",
            "factor used to detect convergence of the iterative optimization step. See "
            "option in the corresponding optimization solver documentation.",
            0.0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<T>(10.0) / safe_tol));
        opts.register_opt(oT);

        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(
            OptionString("linmod optim method", "Select optimization method to use.",
                         {{"auto", optim::solvers::solver_undefined},
                          {"lbfgs", optim::solvers::solver_lbfgsb},
                          {"qr", optim::solvers::solver_qr},
                          {"coord", optim::solvers::solver_coord}},
                         "auto"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "print options", "Print options.", {{"no", 0}, {"yes", 2}}, "no"));
        opts.register_opt(os);

    } catch (std::bad_alloc &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    } catch (...) {                    // LCOV_EXCL_LINE
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

#endif //LINMOD_OPTIONS_HPP