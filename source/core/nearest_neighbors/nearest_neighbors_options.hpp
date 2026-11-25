/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef NEAREST_NEIGHBORS_OPTIONS_HPP
#define NEAREST_NEIGHBORS_OPTIONS_HPP

#include "aoclda_metrics.h"
#include "aoclda_nearest_neighbors.h"
#include "aoclda_types.h"
#include "da_error.hpp"
#include "macros.h"
#include "nearest_neighbors_types.hpp"
#include "options.hpp"

namespace ARCH {

namespace da_neighbors {

using namespace da_neighbors_types;

template <typename T>
inline da_status register_neighbors_options(da_options::OptionRegistry &opts,
                                            da_errors::da_error_t &err) {
    using namespace da_options;
    da_int imax = std::numeric_limits<da_int>::max();
    da_int imin = std::numeric_limits<da_int>::min();
    T fpmax = std::numeric_limits<T>::max();
    T fpmin = std::numeric_limits<T>::lowest();
    try {
        // Integer options
        std::shared_ptr<OptionNumeric<da_int>> oi;
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "number of neighbors",
            "Number of neighbors considered for k-nearest neighbors.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 5));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "leaf size", "Leaf size for k-d tree.", 1, da_options::lbound_t::greaterequal,
            imax, da_options::ubound_t::p_inf, 30));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "outlier label",
            "Classification label for queries with no neighbors within the specified "
            "radius.",
            imin, da_options::lbound_t::m_inf, imax, da_options::ubound_t::p_inf, 0));
        opts.register_opt(oi);
        // fp options
        std::shared_ptr<OptionNumeric<T>> ofp;
        ofp = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("minkowski parameter",
                             "Minkowski parameter for metric used for the computation of "
                             "nearest neighbors.",
                             0.0, da_options::lbound_t::greaterthan, fpmax,
                             da_options::ubound_t::p_inf, 2.0));
        opts.register_opt(ofp);
        ofp = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "radius", "Maximum distance for the radius neighbors computation.", 0.0,
            da_options::lbound_t::greaterequal, fpmax, da_options::ubound_t::p_inf, 1.0));
        opts.register_opt(ofp);
        ofp = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "outlier target",
            "Regression target for queries with no neighbors within the specified "
            "radius.",
            fpmin, da_options::lbound_t::m_inf, fpmax, da_options::ubound_t::p_inf, 0.0));
        opts.register_opt(ofp);
        // String options
        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(OptionString(
            "algorithm", "Algorithm used to compute the k-nearest neighbors.",
            {{"auto", da_neighbors_types::automatic},
             {"brute", da_neighbors_types::brute},
             {"kd tree", da_neighbors_types::kd_tree},
             {"ball tree", da_neighbors_types::ball_tree}},
            "auto"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("metric", "Metric used to compute the pairwise distance matrix.",
                         {{"euclidean", da_euclidean},
                          {"l2", da_l2},
                          {"sqeuclidean", da_sqeuclidean},
                          {"manhattan", da_manhattan},
                          {"l1", da_l1},
                          {"cityblock", da_cityblock},
                          {"cosine", da_cosine},
                          {"minkowski", da_minkowski},
                          {"sqeuclidean_gemm", da_sqeuclidean_gemm},
                          {"euclidean_gemm", da_euclidean_gemm},
                          {"inner product", da_inner_product}},
                         "euclidean"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "weights", "Weight function used to compute the k-nearest neighbors.",
            {{"uniform", da_neighbors_types::uniform},
             {"distance", da_neighbors_types::distance}},
            "uniform"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "outlier handling",
            "How to handle samples with no neighbors within the specified radius.",
            {{"none", da_neighbors_types::none},
             {"manual", da_neighbors_types::manual},
             {"most frequent", da_neighbors_types::most_frequent}},
            "none"));
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

} // namespace da_neighbors

} // namespace ARCH

#endif
