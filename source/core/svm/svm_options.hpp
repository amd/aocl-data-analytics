/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

// Deal with some Windows compilation issues regarding max/min macros
#ifndef NOMINMAX
#define NOMINMAX
#endif

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

    try {
        /* Integer options */
        std::shared_ptr<OptionNumeric<da_int>> oi;

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "degree", "Parameter for 'polynomial' kernel.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 3));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "max_iter",
            "Maximum number of iterations. If the value is -1, it is set to infinity.",
            -1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            -1));
        opts.register_opt(oi);

        /* Float options */
        std::shared_ptr<OptionNumeric<T>> oT;

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "C",
            "Regularization parameter. Controls the trade-off between maximizing the "
            "margin between classes and minimizing classification errors. The larger "
            "value means higher penalty to the loss function on misclassified "
            "observations.",
            0.0, da_options::lbound_t::greaterthan, rmax, da_options::ubound_t::p_inf,
            1.0));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("epsilon",
                             "Epsilon in the SVR model. Defines the tolerance for errors "
                             "in predictions by creating an acceptable margin (tube) "
                             "within which errors are not penalized.",
                             0.0, da_options::lbound_t::greaterequal, rmax,
                             da_options::ubound_t::p_inf, 0.1));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("nu",
                             "An upper bound on the fraction of margin errors and a "
                             "lower bound of the fraction of support vectors.",
                             0.0, da_options::lbound_t::greaterthan, 1.0,
                             da_options::ubound_t::lessequal, 0.5));
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
            "tau", "Parameter used in working set selection.", 0.0,
            da_options::lbound_t::greaterequal, rmax, da_options::ubound_t::p_inf,
            static_cast<T>(1.0e-12), "10^{-12}"));
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
