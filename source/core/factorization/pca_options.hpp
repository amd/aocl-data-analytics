/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#ifndef PCA_OPTIONS_HPP
#define PCA_OPTIONS_HPP

#include "aoclda_types.h"
#include "da_error.hpp"
#include "options.hpp"
#include "pca_types.hpp"

#include <limits>

namespace da_pca {

template <class T>
inline da_status register_pca_options(da_options::OptionRegistry &opts,
                                      da_errors::da_error_t &err) {
    using namespace da_options;
    da_int imax = std::numeric_limits<da_int>::max();

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_components",
            "Number of principal components to compute. If 0, then all components will "
            "be kept.",
            0, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 1));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "store U", "Whether or not to store the matrix U from the SVD.", 0,
            da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);
        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(OptionString(
            "pca method", "Compute PCA based on the covariance or correlation matrix.",
            {{"covariance", pca_method_cov},
             {"correlation", pca_method_corr},
             {"svd", pca_method_svd}},
            "covariance"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("degrees of freedom",
                         "Whether to use biased or unbiased estimators for standard "
                         "deviations and variances.",
                         {{"biased", -1}, {"unbiased", 0}}, "unbiased"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("svd solver",
                         "Which LAPACK routine to use for the underlying singular value "
                         "decomposition.",
                         {{"auto", solver_auto},
                          {"gesvdx", solver_gesvdx},
                          {"gesvd", solver_gesvd},
                          {"gesdd", solver_gesdd},
                          {"syevd", solver_syevd}},
                         "auto"));
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

/* Special case of option registering: after data matrix is passed to handle we wish to update the default and bounds for the number of principal components */
template <class T>
inline da_status reregister_pca_option(da_options::OptionRegistry &opts, da_int p) {
    using namespace da_options;

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("n_components",
                                  "Number of principal components to compute. If 0, then "
                                  "all components will be kept.",
                                  0, da_options::lbound_t::greaterequal, p,
                                  da_options::ubound_t::lessequal, 1));
        opts.register_opt(oi, true);
        std::shared_ptr<OptionString> os;

    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    } catch (...) {
        // Invalid use of the constructor, shouldn't happen (invalid_argument)
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

} // namespace da_pca

#endif //PCA_OPTIONS_HPP
