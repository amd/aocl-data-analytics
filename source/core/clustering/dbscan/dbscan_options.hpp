/* ************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda_types.h"
#include "da_error.hpp"
#include "macros.h"
#include "nearest_neighbors_types.hpp"
#include "options.hpp"

#include <limits>

namespace ARCH {

namespace da_dbscan {

using namespace da_neighbors_types;

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
            "leaf size", "Leaf size for k-d tree or ball tree.", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 30));
        opts.register_opt(oi);
        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(OptionString("algorithm",
                                                         "Choice of algorithm.",
                                                         {{"brute", brute},
                                                          {"kd tree", kd_tree},
                                                          {"ball tree", ball_tree},
                                                          {"auto", automatic}},
                                                         "auto"));
        opts.register_opt(os);
        // da_euclidean_gemm will call syrk which is optimized for the case where dist(X,X) is required.
        // Since da_euclidean is optimized for generic dist(X,Y) only, when
        // "euclidean" is chosen as an option we call the syrk-based implementation through euclidean_gemm.
        // Once blocked distances are optimized for the cases where dist(X,X) is computed,
        // we will make available both euclidean_gemm and euclidean options.
        os = std::make_shared<OptionString>(
            OptionString("metric", "Choice of metric used to compute pairwise distances.",
                         {{"euclidean", da_euclidean_gemm},
                          {"l2", da_l2},
                          {"sqeuclidean", da_sqeuclidean_gemm},
                          {"manhattan", da_manhattan},
                          {"l1", da_l1},
                          {"cityblock", da_cityblock},
                          {"cosine", da_cosine},
                          {"minkowski", da_minkowski}},
                         "euclidean"));
        opts.register_opt(os);
        opts.register_opt(os);
        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "eps",
            "Maximum distance for two samples to be considered in each other's "
            "neighborhood.",
            0, da_options::lbound_t::greaterequal, 0, da_options::ubound_t::p_inf,
            static_cast<T>(1.0e-4), "10^{-4}"));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("power", "The power of the Minkowski metric used.", 0,
                             da_options::lbound_t::greaterequal, 0,
                             da_options::ubound_t::p_inf, static_cast<T>(2.0), "2.0"));
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
