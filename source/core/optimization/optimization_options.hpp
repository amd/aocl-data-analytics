/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

#include "da_error.hpp"
#include "macros.h"
#include "optim_types.hpp"
#include "options.hpp"
#include <limits>

namespace ARCH {

template <class T>
inline da_status register_optimization_options(da_errors::da_error_t &err,
                                               da_options::OptionRegistry &opts) {
    using namespace da_options;
    using namespace da_optim_types;

    const T rmax = std::numeric_limits<T>::max();

    try {
        // ===========================================================================
        // INTEGER OPTIONS
        // ===========================================================================
        std::shared_ptr<OptionNumeric<da_int>> oi;
        // ---
        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("coord skip min",
                                  "Minimum times a coordinate change is smaller than "
                                  "coord skip tol to start skipping.",
                                  2, da_options::lbound_t::greaterequal, max_da_int,
                                  da_options::ubound_t::p_inf, 2));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("coord skip max",
                                  "Maximum times a coordinate can be skipped, after "
                                  "this the coordinate is checked.",
                                  10, da_options::lbound_t::greaterequal, max_da_int,
                                  da_options::ubound_t::p_inf, 100));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("coord restart",
                                  "Number of inner iterations to perform before "
                                  "requesting to perform a full "
                                  "evaluation of the step function.",
                                  0, da_options::lbound_t::greaterequal, max_da_int,
                                  da_options::ubound_t::p_inf, max_da_int, "\\infty"));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "coord iteration limit", "Maximum number of iterations to perform.", 1,
            da_options::lbound_t::greaterequal, max_da_int, da_options::ubound_t::p_inf,
            100000));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "lbfgsb iteration limit", "Maximum number of iterations to perform.", 1,
            da_options::lbound_t::greaterequal, max_da_int, da_options::ubound_t::p_inf,
            10000));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "lbfgsb memory limit",
            "Number of vectors to use for approximating the Hessian.", 1,
            da_options::lbound_t::greaterequal, 1000, da_options::ubound_t::lessequal,
            11));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "monitoring frequency",
            "How frequently to call the user-supplied monitor function.", 0,
            da_options::lbound_t::greaterequal, max_da_int, da_options::ubound_t::p_inf,
            0));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("print level",
                                  "Set level of verbosity for the solver: from 0, "
                                  "indicating no output, to 5, which is very verbose.",
                                  0, da_options::lbound_t::greaterequal, 5,
                                  da_options::ubound_t::lessequal, 1));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "debug", "Set debug level (internal use).", 0,
            da_options::lbound_t::greaterequal, 3, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "ralfit iteration limit", "Maximum number of iterations to perform.", 1,
            da_options::lbound_t::greaterequal, max_da_int, da_options::ubound_t::p_inf,
            100));
        opts.register_opt(oi);

        // ===========================================================================
        // REAL OPTIONS
        // ===========================================================================
        // Tolerance based on sqrt(safe_epsilon)
        da_options::safe_tol<T> tol;
        std::shared_ptr<OptionNumeric<T>> oT;
        // ---
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "derivative test tol",
            "Tolerance used to check user-provided derivatives by finite-differences. "
            "If <print level> is 1, then only the entries with larger discrepancy are "
            "reported, and if print level is greater than or equal to 2, "
            "then all entries are printed.",
            0.0, da_options::lbound_t::greaterthan, 10.0, da_options::ubound_t::lessequal,
            1.0e-4, "10^{-4}"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "finite differences step",
            "Size of step to use for estimating derivatives using finite-differences.",
            0.0, da_options::lbound_t::greaterthan, 10.0, da_options::ubound_t::lessthan,
            tol.safe_eps(10), tol.safe_eps_latex(10)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("time limit", "Maximum time allowed to run (in seconds).",
                             0.0, da_options::lbound_t::greaterthan, 0,
                             da_options::ubound_t::p_inf, static_cast<T>(1.0e6), "10^6"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "infinite bound size", "Threshold value to take for +/- infinity.", 1000,
            da_options::lbound_t::greaterthan, 0, da_options::ubound_t::p_inf,
            static_cast<T>(1.0e20), "10^{20}"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "lbfgsb convergence tol",
            "Tolerance of the projected gradient infinity norm to "
            "declare convergence.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            tol.safe_eps(), tol.safe_eps_latex()));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "lbfgsb progress factor",
            "The iteration stops when (f^k - f{k+1})/max{abs(fk);abs(f{k+1});1} <= "
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
            "Tolerance of the projected gradient infinity norm to declare convergence.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            tol.safe_eps((T)50, (T)1), tol.safe_eps_latex((T)50, (T)1)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "coord skip tol",
            "Coordinate skip tolerance, a given coordinate could be skipped if the "
            "change between two consecutive iterates is less than tolerance. Any "
            "negative value disables the skipping scheme.",
            -1.0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            tol.safe_eps((T)50, (T)1), tol.safe_eps_latex((T)50, (T)1)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "coord optimality tol",
            "Tolerance to declare optimality, e.g. dual-gap, KKT conditions, etc.", 0.0,
            da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            tol.safe_eps((T)50, (T)1), tol.safe_eps_latex((T)50, (T)1)));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "ralfit convergence abs tol fun",
            "Absolute tolerance to declare convergence for the "
            "iterative optimization step. See "
            "details in optimization solver documentation.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            tol.safe_eps((T)10, (T)21), tol.safe_eps_latex((T)10, (T)21)));

        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "ralfit convergence rel tol fun",
            "Relative tolerance to declare convergence for the "
            "iterative optimization step. See "
            "details in optimization solver documentation.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            tol.safe_eps((T)10, (T)21), tol.safe_eps_latex((T)10, (T)21)));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "ralfit convergence abs tol grd",
            "Absolute tolerance on the gradient norm to declare "
            "convergence for the iterative optimization step. See "
            "details in optimization solver documentation.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            tol.safe_eps((T)500), tol.safe_eps_latex((T)500)));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "ralfit convergence rel tol grd",
            "Relative tolerance on the gradient norm to declare convergence for the "
            "iterative optimization step. See "
            "details in optimization solver documentation.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            tol.safe_eps((T)10, (T)21), tol.safe_eps_latex((T)10, (T)21)));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "ralfit convergence step size",
            "Absolute tolerance over the step size to declare "
            "convergence for the iterative optimization step. See "
            "details in optimization solver documentation.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            tol.mcheps(1, 2), tol.mcheps_latex(1, 2)));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "regularization term",
            "Value of the regularization term. A value of 0 disables regularization.",
            0.0, da_options::lbound_t::greaterequal, rmax, da_options::ubound_t::p_inf,
            0.0));
        opts.register_opt(oT);

        // ===========================================================================
        // STRING OPTIONS
        // ===========================================================================
        std::shared_ptr<OptionString> os;
        // ---
        os = std::make_shared<OptionString>(OptionString(
            "print options", "Print options list.", {{"yes", 1}, {"no", 0}}, "no"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(
            OptionString("check derivatives",
                         "Check user-provided derivatives using finite-differences.",
                         {{"yes", 1}, {"no", 0}}, "no"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(
            OptionString("optim method", "Select optimization solver to use.",
                         {{"lbfgsb", da_optim_types::solvers::solver_lbfgsb},
                          {"lbfgs", da_optim_types::solvers::solver_lbfgsb},
                          {"bfgs", da_optim_types::solvers::solver_lbfgsb},
                          {"coord", da_optim_types::solvers::solver_coord},
                          {"ralfit", da_optim_types::solvers::solver_ralfit}},
                         "lbfgsb"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString("ralfit model",
                                                         "NLLS model to solve.",
                                                         {{"gauss-newton", 1},
                                                          {"quasi-newton", 2},
                                                          {"hybrid", 3},
                                                          {"tensor-newton", 4}},
                                                         "hybrid"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString("ralfit nlls method",
                                                         "NLLS solver to use.",
                                                         {{"powell-dogleg", 1},
                                                          {"aint", 2},
                                                          {"more-sorensen", 3},
                                                          {"linear solver", 3},
                                                          {"galahad", 4}},
                                                         "galahad"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "ralfit globalization method",
            "Globalization method to use. This parameter makes use of the regularization "
            "term and power option values.",
            {{"trust-region", 1}, {"tr", 1}, {"regularization", 2}, {"reg", 2}},
            "trust-region"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "regularization power", "Value of the regularization power term.",
            {{"quadratic", regularization::quadratic}, {"cubic", regularization::cubic}},
            "quadratic"));
        opts.register_opt(os);

    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error,
                        "Memory allocation failed"); // LCOV_EXCL_LINE
    } catch (...) {                                  // LCOV_EXCL_LINE
        // Invalid use of the constructor, shouldn't happen (invalid_argument)
        return da_error(&err, da_status_internal_error,
                        "Unexpected internal error"); // LCOV_EXCL_LINE
    }

    return da_status_success;
}

} // namespace ARCH
