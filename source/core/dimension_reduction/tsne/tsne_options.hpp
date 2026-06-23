/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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
#ifndef TSNE_OPTIONS_HPP
#define TSNE_OPTIONS_HPP

#include "aoclda_types.h"
#include "da_error.hpp"
#include "macros.h"
#include "options.hpp"
#include <limits>

namespace ARCH {

namespace da_tsne {

template <class T>
inline da_status register_tsne_options(da_options::OptionRegistry &opts,
                                       da_errors::da_error_t &err) {
    using namespace da_options;
    da_int imax = std::numeric_limits<da_int>::max();

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_components", "Number of embedding dimensions.", 1,
            da_options::lbound_t::greaterequal, 3, da_options::ubound_t::lessequal, 2));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "max_iter", "Maximum number of gradient descent iterations.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 1000));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "seed",
            "Seed for random number generation; set to -1 for non-deterministic results.",
            -1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            0));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_iter_without_progress",
            "Stop if no progress is made for this many iterations.", 0,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 300));
        opts.register_opt(oi);

        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "perplexity", "Target perplexity for conditional probabilities.", (T)1,
            da_options::lbound_t::greaterequal, (T)0, da_options::ubound_t::p_inf,
            static_cast<T>(30)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("learning rate",
                             "Gradient descent learning rate. Use any non-positive "
                             "value for auto: max(N / early_exaggeration / 4, 50).",
                             (T)0, da_options::lbound_t::m_inf, (T)0,
                             da_options::ubound_t::p_inf, static_cast<T>(-1)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "early exaggeration", "Exaggeration factor for early iterations.", (T)1,
            da_options::lbound_t::greaterequal, (T)0, da_options::ubound_t::p_inf,
            static_cast<T>(12)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "min_grad_norm", "Stop if the gradient norm is below this threshold.", (T)0,
            da_options::lbound_t::greaterequal, (T)0, da_options::ubound_t::p_inf,
            static_cast<T>(1e-7)));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("theta", "Barnes-Hut approximation parameter (0 for exact).",
                             (T)0, da_options::lbound_t::greaterequal, (T)1,
                             da_options::ubound_t::lessequal, static_cast<T>(0.5)));
        opts.register_opt(oT);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "low precision max_iter",
            "If mixed precision iterative refinement is enabled, maximum number of "
            "iterations for the low precision phase.",
            1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            200));
        opts.register_opt(oi);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "low precision min_grad_norm",
            "If mixed precision iterative refinement is enabled, gradient norm "
            "convergence threshold for the low precision phase.",
            (T)0, da_options::lbound_t::greaterequal, (T)0, da_options::ubound_t::p_inf,
            static_cast<T>(1e-4)));
        opts.register_opt(oT);

        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(
            OptionString("init", "Initialization method for the embedding.",
                         {{"pca", 0}, {"random", 1}, {"supplied", 2}}, "pca"));
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
    } catch (...) {                                     // LCOV_EXCL_LINE
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected error while registering options");
    }

    return da_status_success;
}

template <class T>
inline da_status reregister_tsne_options(da_options::OptionRegistry &opts,
                                         da_int n_samples, da_int n_features) {
    using namespace da_options;
    da_int max_components = std::min<da_int>(3, n_features);
    da_int max_perplexity = std::max<da_int>(1, n_samples - 1);

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_components", "Number of embedding dimensions.", 1,
            da_options::lbound_t::greaterequal, max_components,
            da_options::ubound_t::lessequal, std::min<da_int>(2, max_components)));
        opts.register_opt(oi, true);

        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "perplexity", "Target perplexity for conditional probabilities.", (T)1,
            da_options::lbound_t::greaterequal, static_cast<T>(max_perplexity),
            da_options::ubound_t::lessequal,
            static_cast<T>(std::min<da_int>(30, max_perplexity))));
        opts.register_opt(oT, true);

    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    } catch (...) {
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

} // namespace da_tsne

} // namespace ARCH

#endif
