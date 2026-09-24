/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#ifndef NMF_OPTIONS_HPP
#define NMF_OPTIONS_HPP

#include "aoclda_types.h"
#include "da_error.hpp"
#include "macros.h"
#include "nmf/nmf_types.hpp"
#include "options.hpp"
#include <limits>

namespace ARCH {

namespace da_nmf {

template <class T>
inline da_status register_nmf_options(da_options::OptionRegistry &opts,
                                      da_errors::da_error_t &err) {
    using namespace da_options;
    da_int imax = std::numeric_limits<da_int>::max();
    T rmax = std::numeric_limits<T>::max();

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_components",
            "Number of components to compute. If 0 (default), all components are kept.",
            0, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 0));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "max_iter", "Maximum number of MU iterations.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 200));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "random_state", "Random seed used to initialize W and H.", -1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, -1));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "check_every",
            "Number of MU iterations between successive convergence checks. "
            "The convergence criterion (relative change of the Frobenius residual "
            "against tol) is evaluated once every check_every iterations. "
            "Larger values reduce the overhead of the residual computation at the "
            "cost of potentially running extra iterations after the threshold is met. "
            "Only effective when tol > 0.",
            1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            10));
        opts.register_opt(oi);

        std::shared_ptr<OptionNumeric<T>> orr;
        orr = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("tol",
                             "Tolerance for the stopping condition: iterations stop "
                             "when the relative improvement of the Frobenius residual "
                             "falls below this value. Set to 0 to disable early "
                             "stopping. The check frequency is controlled by the "
                             "'check_every' option.",
                             (T)0, da_options::lbound_t::greaterequal, rmax,
                             da_options::ubound_t::p_inf, (T)1.0e-4));
        opts.register_opt(orr);

        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(OptionString(
            "solver", "NMF solver to use. Currently only multiplicative update.",
            {{"mu", da_nmf_types::mu}}, "mu"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "beta_loss", "Loss function to minimize. Currently only Frobenius norm.",
            {{"frobenius", da_nmf_types::frobenius}}, "frobenius"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("init", "Procedure used to initialize W and H.",
                         {{"random", da_nmf_types::random},
                          {"nndsvd", da_nmf_types::nndsvd},
                          {"nndsvda", da_nmf_types::nndsvda},
                          {"nndsvdar", da_nmf_types::nndsvdar},
                          {"custom", da_nmf_types::custom},
                          {"auto", da_nmf_types::automatic}},
                         "auto"));
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

} // namespace da_nmf
} // namespace ARCH

#endif // NMF_OPTIONS_HPP
