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

#include "aoclda_types.h"
#include "da_error.hpp"
#include "kmeans_types.hpp"
#include "macros.h"
#include "options.hpp"

#include <limits>

namespace ARCH {

namespace da_kmeans {

using namespace da_kmeans_types;

template <class T>
inline da_status register_kmeans_options(da_options::OptionRegistry &opts,
                                         da_errors::da_error_t &err) {
    using namespace da_options;
    da_int imax = std::numeric_limits<da_int>::max();

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_clusters", "Number of clusters required.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 1));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("n_init",
                                  "Number of runs with different random seeds (ignored "
                                  "if you have specified initial cluster centres).",
                                  1, da_options::lbound_t::greaterequal, imax,
                                  da_options::ubound_t::p_inf, 10));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "max_iter", "Maximum number of iterations.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 300));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "low precision max_iter",
            "If mixed precision iterative refinement is enabled, maximum number of "
            "iterations for the low precision phase.",
            1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            200));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "seed",
            "Seed for random number generation; set to -1 for non-deterministic "
            "results.",
            -1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            0));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "afk-mc2 samples",
            "Number of samples to take for the AFK-MC2 initialization method.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 50));
        opts.register_opt(oi);
        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(OptionString(
            "initialization method", "How to determine the initial cluster centres.",
            {{"random", random_samples},
             {"k-means++", kmeanspp},
             {"supplied", supplied},
             {"random partitions", random_partitions},
             {"afk-mc2", afk_mcmc}},
            "k-means++"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("algorithm", "Choice of underlying k-means algorithm.",
                         {{"lloyd", lloyd},
                          {"elkan", elkan},
                          {"hartigan-wong", hartigan_wong},
                          {"macqueen", macqueen}},
                         "lloyd"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "mixed precision",
            "Whether to use mixed precision iterative refinement, in which "
            "lower precision arithmetic is used before switching to the working "
            "precision for the final iterations.",
            {{"yes", 1}, {"no", 0}}, "no"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "empty clusters",
            "How to deal with empty clusters at the end of a k-means iteration.",
            {{"ignore", ignore}, {"error", error}, {"split", split}}, "ignore"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "distance",
            "Distance metric used for clustering. Use 'euclidean' for standard "
            "k-means or 'cosine' for spherical k-means (not compatible with "
            "Hartigan-Wong).",
            {{"euclidean", 0}, {"cosine", 1}}, "euclidean"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("normalize data",
                         "Whether to normalize the input data before clustering. "
                         "This option is only used if distance is set to cosine.",
                         {{"yes", 1}, {"no", 0}}, "yes"));
        opts.register_opt(os);
        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "convergence tolerance", "Convergence tolerance.", 0,
            da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<T>(1.0e-4), "10^{-4}"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "low precision convergence tolerance",
            "If mixed precision iterative refinement is enabled, convergence tolerance "
            "for the low precision phase.",
            0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<T>(1.0e-2), "10^{-2}"));
        opts.register_opt(oT);

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

/* Special case of option registering: after data matrix is passed to handle we wish to update the default and bounds for the number of clusters */
template <class T>
inline da_status reregister_kmeans_option(da_options::OptionRegistry &opts, da_int p) {
    using namespace da_options;

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_clusters", "Number of clusters required.", 1,
            da_options::lbound_t::greaterequal, p, da_options::ubound_t::lessequal, p));
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

} // namespace da_kmeans

} // namespace ARCH
