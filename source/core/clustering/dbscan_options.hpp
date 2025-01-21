/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda_types.h"
#include "da_error.hpp"
#include "dbscan_types.hpp"
#include "macros.h"
#include "options.hpp"

#include <limits>

namespace ARCH {

namespace da_dbscan {

using namespace da_dbscan_types;

template <class T>
inline da_status register_dbscan_options(da_options::OptionRegistry &opts,
                                         da_errors::da_error_t &err) {
    using namespace da_options;
    da_int imax = std::numeric_limits<da_int>::max();

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "min samples", "Minimum number of neighborhood samples for a core point.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 5));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "leaf size", "Leaf size for KD tree or ball tree (reserved for future use).",
            1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            30));
        opts.register_opt(oi);
        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(
            OptionString("algorithm", "Choice of algorithm (reserved for future use).",
                         {{"brute", brute},
                          {"brute serial", brute_serial},
                          {"kd tree", kd_tree},
                          {"ball tree", ball_tree},
                          {"auto", automatic}},
                         "brute"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("metric",
                         "Choice of metric used to compute pairwise distances (reserved "
                         "for future use).",
                         {{"euclidean", euclidean},
                          {"sqeuclidean", sqeuclidean},
                          {"minkowski", minkowski},
                          {"manhattan", manhattan}},
                         "euclidean"));
        opts.register_opt(os);
        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "eps",
            "Maximum distance for two samples to be considered in each other's "
            "neighborhood.",
            0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<T>(1.0e-4), "10^{-4}"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "power", "The power of the Minkowski metric used (reserved for future use).",
            0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<T>(2.0), "2.0"));
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

} // namespace da_dbscan

} // namespace ARCH
