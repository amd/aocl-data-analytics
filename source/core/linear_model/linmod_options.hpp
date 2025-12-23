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
#include "linmod_types.hpp"
#include "macros.h"
#include "options.hpp"
#include <limits>

namespace ARCH {

namespace da_linmod {

using namespace da_linmod_types;

template <class T>
inline da_status register_linmod_options(da_options::OptionRegistry &opts,
                                         da_errors::da_error_t &err) {
    using namespace da_options;
    T rmax = std::numeric_limits<T>::max();
    // Tolerance based on sqrt(safe_epsilon)
    da_options::safe_tol<T> tol;
    try {

        std::shared_ptr<OptionNumeric<da_int>> oi;

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "print level", "Set level of verbosity for the solver.", 0,
            da_options::lbound_t::greaterequal, 5, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "intercept", "Add intercept variable to the model.", 0,
            da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "optim iteration limit",
            "Maximum number of iterations to perform in the optimization phase. Valid "
            "only for iterative solvers, e.g. L-BFGS-B, Coordinate Descent, etc.",
            1, da_options::lbound_t::greaterequal, max_da_int,
            da_options::ubound_t::p_inf, 10000));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("optim coord skip min",
                                  "Minimum times a coordinate change is smaller than "
                                  "coord skip tol to start skipping.",
                                  2, da_options::lbound_t::greaterequal, max_da_int,
                                  da_options::ubound_t::p_inf, 2));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("optim coord skip max",
                                  "Maximum times a coordinate can be skipped, after "
                                  "this the coordinate is checked.",
                                  10, da_options::lbound_t::greaterequal, max_da_int,
                                  da_options::ubound_t::p_inf, 100));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "debug", "Set debug level (internal use).", 0,
            da_options::lbound_t::greaterequal, 3, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        std::shared_ptr<OptionNumeric<T>> oT;

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "optim time limit",
            "Maximum time limit (in seconds). Solver will exit with a warning "
            "after this limit. Valid only for iterative solvers, e.g. L-BFGS-B, "
            "Coordinate Descent, etc.",
            0, da_options::lbound_t::greaterthan, rmax, da_options::ubound_t::p_inf,
            1000000, "10^6"));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("alpha",
                             "Coefficient of alpha in the regularization terms: lambda( "
                             "(1-alpha)/2 L2 + alpha L1 ).",
                             0.0, da_options::lbound_t::greaterequal, 1.0,
                             da_options::ubound_t::lessequal, 0.0));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("lambda",
                             "Penalty coefficient for the regularization terms: lambda( "
                             "(1-alpha)/2 L2 + alpha L1 ).",
                             0.0, da_options::lbound_t::greaterequal, rmax,
                             da_options::ubound_t::p_inf, 0.0));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "optim convergence tol",
            "Tolerance to declare convergence for the iterative optimization step. See "
            "option in the corresponding optimization solver documentation.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            1.e-4, "10^{-4}"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "optim progress factor",
            "Factor used to detect convergence of the iterative optimization step. See "
            "option in the corresponding optimization solver documentation.",
            0.0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            tol.safe_inveps((T)10, (T)1), tol.safe_inveps_latex((T)10, (T)1)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "optim dual gap tol",
            "Tolerance to declare convergence based on the estimate of dual gap size. "
            "See option in the corresponding optimization solver documentation.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::p_inf,
            1.e-4, "10^{-4}"));
        opts.register_opt(oT);

        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(
            OptionString("optim method", "Select optimization method to use.",
                         {{"auto", linmod_method::undefined},
                          {"bfgs", linmod_method::lbfgsb},
                          {"lbfgs", linmod_method::lbfgsb},
                          {"lbfgsb", linmod_method::lbfgsb},
                          {"qr", linmod_method::qr},
                          {"coord", linmod_method::coord},
                          {"svd", linmod_method::svd},
                          {"sparse_cg", linmod_method::cg},
                          {"cg", linmod_method::cg},
                          {"cholesky", linmod_method::cholesky},
                          {"chol", linmod_method::cholesky}},
                         "auto"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "print options", "Print options.", {{"no", 0}, {"yes", 2}}, "no"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "scaling",
            "Scale or standardize feature matrix and response vector. Matrix is copied "
            "and then rescaled. Option key value auto indicates that rescaling type is "
            "chosen by the solver (this also includes no scaling).",
            {{"no", scaling_t::none},
             {"none", scaling_t::none},
             {"auto", scaling_t::automatic},
             {"scale", scaling_t::scale_only},
             {"scale only", scaling_t::scale_only},
             {"standardize", scaling_t::standardize},
             {"standardise", scaling_t::standardize},
             {"centering", scaling_t::centering}},
            "auto"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "logistic constraint",
            "Affects only multinomial logistic regression. Type of constraint put on "
            "coefficients. This will affect number of coefficients returned. RSC - means "
            "we choose a reference category whose coefficients will be set to all 0. "
            "This results in K-1 class coefficients for problems with K classes. SSC - "
            "means "
            "the sum of coefficients class-wise for each feature is 0. It will result in "
            "K class coefficients for problems with K classes.",
            {{"rsc", logistic_constraint::rsc},
             {"reference category", logistic_constraint::rsc},
             {"ssc", logistic_constraint::ssc},
             {"symmetric side", logistic_constraint::ssc},
             {"symmetric", logistic_constraint::ssc}},
            "ssc"));
        opts.register_opt(os);
    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    } catch (...) { // LCOV_EXCL_LINE
        // Invalid use of the constructor, shouldn't happen (invalid_argument)
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected error while registering options");
    }

    return da_status_success;
}

} // namespace da_linmod

} // namespace ARCH
