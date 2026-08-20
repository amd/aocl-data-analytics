/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

#ifndef KERNEL_PCA_OPTIONS_HPP
#define KERNEL_PCA_OPTIONS_HPP

#include "aoclda_types.h"
#include "da_error.hpp"
#include "kernel_pca/kernel_pca_types.hpp"
#include "macros.h"
#include "options.hpp"
#include <limits>

namespace ARCH {

namespace da_kernel_pca {

using namespace da_kernel_pca_types;

template <class T>
inline da_status register_kernel_pca_options(da_options::OptionRegistry &opts,
                                             da_errors::da_error_t &err) {
    using namespace da_options;
    da_int imax = std::numeric_limits<da_int>::max();

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_components", "Number of kernel principal components to compute.", 0,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "degree", "Degree for the polynomial kernel.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 3));
        opts.register_opt(oi);

        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(
            OptionString("fit inverse transform", "Whether to fit the inverse transform.",
                         {{"yes", 1}, {"no", 0}}, "no"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "remove zero eig", "Whether to remove components whose eigenvalue is zero.",
            {{"yes", 1}, {"no", 0}}, "no"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "copy data", "Whether or not to store a copy of the training data.",
            {{"yes", 1}, {"no", 0}}, "yes"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(
            OptionString("eigensolver",
                         "Which method to use for computing the eigendecomposition of "
                         "the kernel matrix",
                         {{"auto", solver_auto},
                          {"syevd", solver_syevd},
                          {"randomized", solver_rand_syevd}},
                         "auto"));
        opts.register_opt(os);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_oversamples",
            "Extra columns added to the random sample to reduce approximation error. "
            "This option is only used in the randomized solver.",
            0, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            10));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "power iterations",
            "Number of power iterations used in the randomized solver.", -1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, -1));
        opts.register_opt(oi);
        os = std::make_shared<OptionString>(OptionString(
            "power normalization",
            "Normalization method used in the randomized solver power iteration.",
            {{"qr", 0}, {"lu", 1}, {"none", 2}}, "qr"));
        opts.register_opt(os);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "seed",
            "Seed for random number generation; set to -1 for non-deterministic results. "
            "This option is only used in the randomized solver.",
            -1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            0));
        opts.register_opt(oi);

        std::shared_ptr<OptionNumeric<T>> oT;
        T tmax = std::numeric_limits<T>::max();
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "gamma", "Kernel coefficient for rbf, poly, and sigmoid kernels.", -tmax,
            da_options::lbound_t::m_inf, tmax, da_options::ubound_t::p_inf,
            static_cast<T>(-1.0)));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "coef0", "Independent term for polynomial and sigmoid kernels.", -tmax,
            da_options::lbound_t::m_inf, tmax, da_options::ubound_t::p_inf,
            static_cast<T>(1.0)));
        opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "alpha",
            "Ridge regularization parameter for the inverse transform linear solve.",
            static_cast<T>(0.0), da_options::lbound_t::greaterthan, tmax,
            da_options::ubound_t::p_inf, static_cast<T>(1.0)));
        opts.register_opt(oT);

        os = std::make_shared<OptionString>(
            OptionString("kernel", "Kernel function to use.",
                         {{"linear", pca_kernel::linear},
                          {"poly", pca_kernel::polynomial},
                          {"rbf", pca_kernel::rbf},
                          {"sigmoid", pca_kernel::sigmoid},
                          {"precomputed", pca_kernel::precomputed}},
                         "linear"));
        opts.register_opt(os);

    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    } catch (...) {                                     // LCOV_EXCL_LINE
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected error while registering options");
    }

    return da_status_success;
}

/* Update the upper bound for n components once n_samples is known from set_data. */
template <class T>
inline da_status reregister_kernel_pca_n_components(da_options::OptionRegistry &opts,
                                                    da_int n_samples) {
    using namespace da_options;

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_components", "Number of kernel principal components to compute.", 0,
            da_options::lbound_t::greaterequal, n_samples,
            da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi, true);
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    } catch (...) {
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

} // namespace da_kernel_pca
} // namespace ARCH

#endif // KERNEL_PCA_OPTIONS_HPP
