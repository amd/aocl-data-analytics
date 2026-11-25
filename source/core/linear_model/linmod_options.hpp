/* ************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
    // For _Float16, register options as float so set/get with user-facing
    // float/double works; _Float16 reads use the converting get() overload.
    using opt_T = std::conditional_t<std::is_same_v<T, _Float16>, float, T>;
    opt_T rmax = std::numeric_limits<opt_T>::max();
    // Tolerance based on sqrt(safe_epsilon)
    da_options::safe_tol<opt_T> tol;
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
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "low precision iteration limit",
            "If mixed precision iterative refinement is enabled, maximum number of "
            "iterations for the low precision phase.",
            1, da_options::lbound_t::greaterequal, max_da_int,
            da_options::ubound_t::p_inf, 5000));
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

        std::shared_ptr<OptionNumeric<opt_T>> oT;

        oT = std::make_shared<OptionNumeric<opt_T>>(OptionNumeric<opt_T>(
            "optim time limit",
            "Maximum time limit (in seconds). Solver will exit with a warning "
            "after this limit. Valid only for iterative solvers, e.g. L-BFGS-B, "
            "Coordinate Descent, etc.",
            0, da_options::lbound_t::greaterthan, rmax, da_options::ubound_t::p_inf,
            1000000, "10^6"));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<opt_T>>(OptionNumeric<opt_T>(
            "alpha",
            "Coefficient of alpha in the regularization terms: lambda( "
            "(1-alpha)/2 L2 + alpha L1 ).",
            0.0, da_options::lbound_t::greaterequal, 1.0, da_options::ubound_t::lessequal,
            0.0));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<opt_T>>(OptionNumeric<opt_T>(
            "lambda",
            "Penalty coefficient for the regularization terms: lambda( "
            "(1-alpha)/2 L2 + alpha L1 ).",
            0.0, da_options::lbound_t::greaterequal, rmax, da_options::ubound_t::p_inf,
            0.0));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<opt_T>>(OptionNumeric<opt_T>(
            "optim convergence tol",
            "Tolerance to declare convergence for the iterative optimization step. See "
            "option in the corresponding optimization solver documentation.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessthan,
            1.e-4, "10^{-4}"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<opt_T>>(OptionNumeric<opt_T>(
            "low precision convergence tol",
            "If mixed precision iterative refinement is enabled, convergence tolerance "
            "for the low precision phase.",
            0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<opt_T>(1.0e-3), "10^{-3}"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<opt_T>>(OptionNumeric<opt_T>(
            "optim progress factor",
            "Factor used to detect convergence of the iterative optimization step. See "
            "option in the corresponding optimization solver documentation.",
            0.0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            tol.safe_inveps((opt_T)10, (opt_T)1),
            tol.safe_inveps_latex((opt_T)10, (opt_T)1)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<opt_T>>(OptionNumeric<opt_T>(
            "optim dual gap tol",
            "Tolerance to declare convergence based on the estimate of dual gap size. "
            "See option in the corresponding optimization solver documentation.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::p_inf,
            1.e-4, "10^{-4}"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<opt_T>>(OptionNumeric<opt_T>(
            "optim coord skip tol",
            "Coordinate skip tolerance, a given coordinate could be skipped if the "
            "change between two consecutive iterates is less than tolerance. Any "
            "negative value disables the skipping scheme.",
            T(-1), da_options::lbound_t::m_inf, 0, da_options::ubound_t::p_inf, 1.0e-4,
            "10^{-4}"));
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
            "mixed precision",
            "Whether to use mixed precision iterative refinement, in which "
            "lower precision arithmetic is used before switching to the working "
            "precision for the final iterations.",
            {{"yes", 1}, {"no", 0}}, "no"));
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
