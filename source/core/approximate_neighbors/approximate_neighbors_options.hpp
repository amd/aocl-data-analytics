/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 */

#ifndef APPROX_NN_OPTIONS
#define APPROX_NN_OPTIONS

#include "aoclda_metrics.h"
#include "aoclda_types.h"
#include "approximate_neighbors_types.hpp"
#include "da_error.hpp"
#include "macros.h"
#include "options.hpp"
#include "scalar_quantizer.hpp"

namespace ARCH {

namespace da_approx_nn {

using namespace da_approx_nn_types;

template <typename T>
inline da_status register_approximate_neighbors_options(da_options::OptionRegistry &opts,
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
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_list", "Number of lists to construct for inverted file indices", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 1));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_probe",
            "Number of lists to probe at search time for inverted file indices", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 1));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "k-means_iter",
            "Maximum number of k-means iterations to perform at train time", 1,
            da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf, 10));
        opts.register_opt(oi);
        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "seed",
            "Seed for random number generation; set to -1 for non-deterministic "
            "results.",
            -1, da_options::lbound_t::greaterequal, imax, da_options::ubound_t::p_inf,
            0));
        opts.register_opt(oi);
        // floating-point options
        std::shared_ptr<OptionNumeric<T>> ofp;
        ofp = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "train fraction", "Fraction of training data to use for k-means clustering.",
            0.0, da_options::lbound_t::greaterthan, 1.0, da_options::ubound_t::lessequal,
            1.0));
        opts.register_opt(ofp);
        // String options
        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(OptionString(
            "algorithm",
            "Algorithm used to compute the approximate nearest neighbors. Auto selects "
            "ivfflat.",
            {{"auto", approx_nn_algorithm::automatic},
             {"ivfflat", approx_nn_algorithm::ivfflat},
             {"ivfsq", approx_nn_algorithm::ivfsq}},
            "auto"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("metric", "Metric used to compute distances.",
                         {{"euclidean", euclidean},
                          {"sqeuclidean", sqeuclidean},
                          {"inner product", inner_product},
                          {"cosine", cosine}},
                         "sqeuclidean"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "quantization",
            "Storage format used for the indexed database when algorithm is ivfsq.",
            {{"int8", scalar_quantizer::int8},
             {"int4", scalar_quantizer::int4},
             {"bfloat16", scalar_quantizer::bfloat16}},
            "bfloat16"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "quantization_scheme",
            "Scheme used by ivfsq to map values to integer codes. "
            "The direct schemes assume values already lie in the format's "
            "reconstructed range: int4 uses [-8, 7] signed or [0, 15] unsigned; "
            "int8 uses [-128, 127] signed or [0, 255] unsigned. Direct unsigned "
            "is unsuitable for residual encoding because IVF residuals may be "
            "negative. bfloat16 requires 'direct_signed', which performs direct "
            "round-to-nearest-even conversion without scaling.",
            {{"per_dimension", scalar_quantizer::per_dimension},
             {"uniform", scalar_quantizer::uniform},
             {"direct_unsigned", scalar_quantizer::direct_unsigned},
             {"direct_signed", scalar_quantizer::direct_signed}},
            "per_dimension"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(OptionString(
            "quantization_residual",
            "Whether ivfsq codes encode residuals (x - nearest centroid) rather than "
            "raw vectors.",
            {{"yes", 1}, {"no", 0}}, "yes"));
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

} // namespace da_approx_nn

} // namespace ARCH

#endif // APPROX_NN_OPTIONS
