/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda_types.h"
#include "da_error.hpp"
#include "decision_tree_types.hpp"
#include "macros.h"
#include "options.hpp"

namespace ARCH {

namespace da_decision_forest {

using namespace da_decision_tree_types;

template <class T>
inline da_status register_forest_options(da_options::OptionRegistry &opts,
                                         da_errors::da_error_t &err) {
    da_status status = da_status_success;

    try {
        using namespace da_options;

        std::shared_ptr<OptionString> os;
        std::shared_ptr<OptionNumeric<da_int>> oi;
        std::shared_ptr<OptionNumeric<T>> oT;

        os = std::make_shared<OptionString>(
            OptionString("scoring function", "Select scoring function to use.",
                         {{"gini", gini},
                          {"cross-entropy", cross_entropy},
                          {"entropy", cross_entropy},
                          {"misclassification-error", misclassification},
                          {"misclassification", misclassification},
                          {"misclass", misclassification}},
                         "gini"));
        status = opts.register_opt(os);

        os = std::make_shared<OptionString>(
            OptionString("sorting method", "Select sorting method to use.",
                         {{"stl", stl_sort}, {"boost", boost_sort}}, "boost"));
        status = opts.register_opt(os);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "maximum depth", "Set the maximum depth of trees.", 0, lbound_t::greaterequal,
            (da_int)(std::numeric_limits<da_int>::digits) - 2, ubound_t::lessequal, 29));
        status = opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "seed",
            "Set random seed for the random number generator. If "
            "the value is -1, a random seed is automatically generated. In this case the "
            "resulting classification will create non-reproducible results.",
            -1, lbound_t::greaterequal, max_da_int, ubound_t::p_inf, -1));
        status = opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "number of trees", "Set the number of trees to compute. ", 1,
            lbound_t::greaterequal, max_da_int, ubound_t::p_inf, 100));
        status = opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "node minimum samples",
            "Minimum number of samples to consider a node for splitting.", 2,
            lbound_t::greaterequal, max_da_int, ubound_t::p_inf, 2));
        status = opts.register_opt(oi);

        os = std::make_shared<OptionString>(OptionString(
            "bootstrap", "Select whether to bootstrap the samples in the trees.",
            {{"yes", 1}, {"no", 0}}, "yes"));
        status = opts.register_opt(os);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "bootstrap samples factor",
            "Proportion of samples to draw from the data set to build each tree if "
            "'bootstrap' was set to 'yes'.",
            0., da_options::lbound_t::greaterthan, 1., da_options::ubound_t::lessequal,
            (T)0.8));
        opts.register_opt(oT);

        os = std::make_shared<OptionString>(OptionString(
            "tree building order", "Select in which order to explore the nodes.",
            {{"depth first", depth_first}, {"breadth first", breadth_first}},
            "depth first"));
        status = opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "features selection", "Select how many features to use for each split.",
            {{"all", feat_selection::all},
             {"sqrt", feat_selection::sqrt},
             {"log2", feat_selection::log2},
             {"custom", feat_selection::custom}},
            "sqrt"));
        status = opts.register_opt(os);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "maximum features",
            "Set the number of features to consider when splitting a node. 0 means "
            "take all the features.",
            0, lbound_t::greaterequal, max_da_int, ubound_t::p_inf, 0));
        status = opts.register_opt(oi);

        T rmax = std::numeric_limits<T>::max();
        T diff_thres_default = (T)1e-6;
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "feature threshold",
            "Minimum difference in feature value required for splitting.", 0.0,
            lbound_t::greaterequal, rmax, ubound_t::p_inf, diff_thres_default));
        status = opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "minimum split score",
            "Minimum score needed for a node to be considered for splitting.", 0.0,
            lbound_t::greaterequal, 1.0, ubound_t::lessequal, (T)0.03));
        status = opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "minimum split improvement",
            "Minimum score improvement needed to consider a split from the parent node.",
            0.0, lbound_t::greaterequal, rmax, ubound_t::p_inf, (T)0.03));
        status = opts.register_opt(oT);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "block size", "Set the size of the blocks for parallel computations.", 1,
            lbound_t::greaterequal, max_da_int, ubound_t::lessequal, DF_BLOCK_SIZE));
        status = opts.register_opt(oi);

    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    } catch (...) { // LCOV_EXCL_LINE
        // Invalid use of the constructor, shouldn't happen (invalid_argument)
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected error while registering options");
    }

    return status;
}
} // namespace da_decision_forest

} // namespace ARCH
