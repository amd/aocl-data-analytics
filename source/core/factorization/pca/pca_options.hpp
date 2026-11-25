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

#ifndef PCA_OPTIONS_HPP
#define PCA_OPTIONS_HPP

#include "aoclda_types.h"
#include "da_error.hpp"
#include "macros.h"
#include "options.hpp"
#include "pca/pca_types.hpp"
#include <limits>

namespace ARCH {

namespace da_pca {

using namespace da_pca_types;

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
            "store u", "Whether or not to store the matrix U from the SVD.", 0,
            da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "whiten", "Whether or not we whiten when transforming the data.", 0,
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
                          {"syevd", solver_syevd},
                          {"randomized", solver_rand_svd}},
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
} // namespace ARCH

#endif //PCA_OPTIONS_HPP
