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

#ifndef KMEANS_OPTIONS_HPP
#define KMEANS_OPTIONS_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "kmeans_types.hpp"
#include "options.hpp"

#include <limits>

namespace da_kmeans {

template <class T>
inline da_status register_kmeans_options(da_options::OptionRegistry &opts) {
    using namespace da_options;
    da_int imax = std::numeric_limits<da_int>::max();

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_clusters", "Number of clusters required", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 1));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("n_init",
                                  "Number of runs with different random seeds (ignored "
                                  "if you have specified initial cluster centres)",
                                  1, da_options::lbound_t::greaterequal, imax,
                                  da_options::ubound_t::p_inf, 10));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "max_iter", "Maximum number of iterations", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 300));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "seed",
            "Seed for random number generation; set to -1 for non-deterministic "
            "results",
            -1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            0));
        opts.register_opt(oi);
        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(OptionString(
            "initialization method", "How to determine the initial cluster centres",
            {{"random", random_samples},
             {"k-means++", kmeanspp},
             {"supplied", supplied},
             {"random partitions", random_partitions}},
            "random"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("algorithm", "Choice of underlying k-means algorithm",
                         {{"lloyd", lloyd},
                          {"elkan", elkan},
                          {"hartigan-wong", hartigan_wong},
                          {"macqueen", macqueen}},
                         "lloyd"));
        opts.register_opt(os);
        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "convergence tolerance", "Convergence tolerance", 0,
            da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<T>(1.0e-4), "10^{-4}"));
        opts.register_opt(oT);

    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    } catch (...) {
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

/* Special case of option registering: after data matrix is passed to handle we wish to update the default and bounds for the number of principal components */
template <class T>
inline da_status reregister_kmeans_option(da_options::OptionRegistry &opts, da_int p) {
    using namespace da_options;

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_clusters", "Number of clusters required", 1,
            da_options::lbound_t::greaterequal, p, da_options::ubound_t::lessequal, p));
        opts.register_opt(oi, true);
        std::shared_ptr<OptionString> os;

    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    } catch (...) {
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

} // namespace da_kmeans

#endif //KMEANS_OPTIONS_HPP
