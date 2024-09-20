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

#ifndef kNN_OPTIONS_HPP
#define kNN_OPTIONS_HPP

#include "aoclda_knn.h"
#include "aoclda_metrics.h"
#include "aoclda_types.h"
#include "da_error.hpp"
#include "options.hpp"

namespace da_knn {

template <typename T>
inline da_status register_knn_options(da_options::OptionRegistry &opts,
                                      da_errors::da_error_t &err) {
    using namespace da_options;
    da_int imax = std::numeric_limits<da_int>::max();
    try {
        // Integer options
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "number of neighbors",
            "Number of neighbors considered for k-nearest neighbors.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 5));
        opts.register_opt(oi);
        // String options
        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(OptionString(
            "algorithm", "Algorithm used to compute the k-nearest neighbors.",
            {{"brute", da_brute_force}}, "brute"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("metric", "Metric used to compute the pairwise distance matrix.",
                         {
                             {"euclidean", da_euclidean},
                             {"sqeuclidean", da_sqeuclidean},
                         },
                         "euclidean"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "weights", "Weight function used to compute the k-nearest neighbors.",
            {{"uniform", da_knn_uniform}, {"distance", da_knn_distance}}, "uniform"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "storage order",
            "Whether data is supplied and returned in row- or column-major order.",
            {{"row-major", row_major}, {"column-major", column_major}}, "column-major"));
        opts.register_opt(os);
    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    } catch (...) {
        // Invalid use of the constructor, shouldn't happen (invalid_argument)
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected error while registering options");
    }

    return da_status_success;
};

} // namespace da_knn

#endif
