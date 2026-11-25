/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "macros.h"
#include "options.hpp"
#include "svm_types.hpp"

#include <limits>

namespace ARCH {

namespace da_svm {

using namespace da_svm_types;

template <class T>
inline da_status register_svm_options(da_options::OptionRegistry &opts,
                                      da_errors::da_error_t &err) {
    using namespace da_options;
    da_int imax = std::numeric_limits<da_int>::max();
    T rmax = std::numeric_limits<T>::max();
    T nrmax = -rmax;
    // Tolerance based on sqrt(safe_epsilon)
    da_options::safe_tol<T> tol;

    try {
        /* Integer options */
        std::shared_ptr<OptionNumeric<da_int>> oi;

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "degree", "Parameter for 'polynomial' kernel.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 3));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "max_iter",
            "Sets the maximum number of iterations. Use 0 to specify no limit.", 0,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            100000));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "max_ws_size",
            "Specifies the maximum working set size. A value divisible by 64 is "
            "recommended for optimal performance. Setting -1 automatically selects the "
            "optimal size based on the input data.",
            -1.0, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            -1.0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_folds",
            "Number of folds to use with cross validation. Only used when predict "
            "probabilities is enabled.",
            1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 5));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "seed",
            "Seed for random number generation; set to -1 for non-deterministic "
            "results.",
            -1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "predict probabilities",
            "Evaluate class probabilities (in addition to class predictions)."
            "Needs to be 1 if calls to predict_proba or predict_log_proba"
            "are made after fit.",
            0, lbound_t::greaterequal, 1, ubound_t::lessequal, 0));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "low precision max_iter",
            "If mixed precision iterative refinement is enabled, maximum number of "
            "iterations for the low precision phase.",
            0, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            80000));
        opts.register_opt(oi);

        /* Float options */
        std::shared_ptr<OptionNumeric<T>> oT;

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "c",
            "Regularization parameter. Controls the trade-off between maximizing the "
            "margin between classes and minimizing classification errors. A larger "
            "value means higher penalty to the loss function on misclassified "
            "observations. Applies to SVC, SVR and NuSVR.",
            0.0, da_options::lbound_t::greaterthan, rmax, da_options::ubound_t::p_inf,
            1.0));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("epsilon",
                             "Defines the tolerance for errors in predictions by "
                             "creating an acceptable margin (tube) "
                             "within which errors are not penalized. Applies to SVR",
                             0.0, da_options::lbound_t::greaterequal, rmax,
                             da_options::ubound_t::p_inf, 0.1));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "nu",
            "An upper bound on the fraction of margin errors and a "
            "lower bound of the fraction of support vectors. Applies to NuSVC and NuSVR.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessequal,
            0.5));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "tolerance", "Convergence tolerance.", 0.0, da_options::lbound_t::greaterthan,
            0, da_options::ubound_t::p_inf, static_cast<T>(1.0e-3), "10^{-3}"));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "gamma",
            "Parameter for 'rbf', 'polynomial', and 'sigmoid' kernels. If the value is "
            "less than 0, it is set to 1/(n_features * Var(X)).",
            -1.0, da_options::lbound_t::greaterequal, rmax, da_options::ubound_t::p_inf,
            -1.0));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "coef0", "Constant in 'polynomial' and 'sigmoid' kernels.", nrmax,
            da_options::lbound_t::m_inf, rmax, da_options::ubound_t::p_inf, 0.0));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "tau",
            "Numerical stability parameter used in working set selection when kernel is "
            "not positive semi definite.",
            0.0, da_options::lbound_t::greaterequal, rmax, da_options::ubound_t::p_inf,
            tol.mcheps(1, 1), tol.mcheps_latex(1, 1)));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "cache size",
            "Size of the kernel cache in MB. The default value is -1.0 "
            "which automatically sets it to a value which will enable storage of the "
            "sampled kernel matrix. Increasing value of this "
            "option will result in faster training time.",
            -1.0, da_options::lbound_t::greaterequal, rmax, da_options::ubound_t::p_inf,
            -1.0));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "low precision convergence tolerance",
            "If mixed precision iterative refinement is enabled, convergence tolerance "
            "for the low precision phase.",
            0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<T>(1.0e-2), "10^{-2}"));
        opts.register_opt(oT);

        /* String options */
        std::shared_ptr<OptionString> os;

        os = std::make_shared<OptionString>(
            OptionString("kernel", "Kernel function to use for the calculations.",
                         {{"rbf", svm_kernel::rbf},
                          {"linear", svm_kernel::linear},
                          {"polynomial", svm_kernel::polynomial},
                          {"poly", svm_kernel::polynomial},
                          {"sigmoid", svm_kernel::sigmoid}},
                         "rbf"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "mixed precision",
            "Whether to use mixed precision iterative refinement, in which "
            "lower precision arithmetic is used before switching to the working "
            "precision for the final iterations.",
            {{"yes", 1}, {"no", 0}}, "no"));
        opts.register_opt(os);

    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    } catch (...) {
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected error while registering options");
    }

    return da_status_success;
}

} // namespace da_svm

} // namespace ARCH
