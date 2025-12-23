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
inline da_status register_decision_tree_options(da_options::OptionRegistry &opts,
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

        os = std::make_shared<OptionString>(OptionString(
            "predict probabilities",
            "evaluate class probabilities (in addition to class predictions). "
            "Needs to be set to 'yes' if calls to predict_proba or predict_log_proba "
            "are made after fit.",
            {{"yes", 1}, {"no", 0}}, "yes"));
        status = opts.register_opt(os);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "maximum depth", "Set the maximum depth of trees.", 0, lbound_t::greaterequal,
            (da_int)(std::numeric_limits<da_int>::digits) - 2, ubound_t::lessequal, 29));
        status = opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "seed",
            "Set the random seed for the random number generator. If "
            "the value is -1, a random seed is automatically generated. In this case the "
            "resulting classification will create non-reproducible results.",
            -1, lbound_t::greaterequal, max_da_int, ubound_t::p_inf, -1));
        status = opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "maximum features",
            "Set the number of features to consider when splitting a node. 0 means "
            "take all the features.",
            0, lbound_t::greaterequal, max_da_int, ubound_t::p_inf, 0));
        status = opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "node minimum samples",
            "The minimum number of samples required to split an internal node.", 1,
            lbound_t::greaterequal, max_da_int, ubound_t::p_inf, 2));
        status = opts.register_opt(oi);

        T rmax = std::numeric_limits<T>::max();
        T diff_thres_default = (T)1e-5;
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "feature threshold",
            "Minimum difference in feature value required for splitting.", 0.0,
            lbound_t::greaterequal, rmax, ubound_t::p_inf, diff_thres_default));
        status = opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "minimum split score",
            "Minimum score needed for a node to be considered for splitting.", 0.0,
            lbound_t::greaterequal, 1.0, ubound_t::lessequal, (T)0.0));
        status = opts.register_opt(oT);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "minimum impurity decrease",
            "Minimum score improvement needed to consider a split from the parent node.",
            0.0, lbound_t::greaterequal, rmax, ubound_t::p_inf, (T)0.0));
        status = opts.register_opt(oT);

        os = std::make_shared<OptionString>(OptionString(
            "detect categorical data",
            "Check if the data is categorical, encoded in [0, n_categories-1].",
            {{"yes", 1}, {"no", 0}}, "no"));
        status = opts.register_opt(os);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "maximum categories",
            "Number of distinct values required for each feature for it to be considered "
            "categorical.",
            2, lbound_t::greaterequal, max_da_int, ubound_t::p_inf, 50));
        status = opts.register_opt(oi);

        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "category tolerance",
            "How far data can be from an integer to be considered not categorical.", 0.0,
            lbound_t::greaterequal, (T)1.0, ubound_t::lessthan, (T)1.0e-05));
        status = opts.register_opt(oT);

        os = std::make_shared<OptionString>(
            OptionString("category split strategy",
                         "Strategy to split categorical features. For a given "
                         "categorical feature, 'one-vs-all' tries to "
                         "split each categorical value from all the the others while "
                         "'ordered' will try "
                         "to split the smaller categories from the bigger ones.",
                         {{"one-vs-all", (da_int)categorical_onevall},
                          {"ordered", (da_int)categorical_ordered}},
                         "ordered"));
        status = opts.register_opt(os);

        // Histogram options
        os = std::make_shared<OptionString>(OptionString(
            "histogram",
            "Choose whether to use histograms constructed from the data matrix X.",
            {{"yes", 1}, {"no", 0}}, "no"));
        status = opts.register_opt(os);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "maximum bins", "Maximum number of bins in histograms.", 2,
            lbound_t::greaterequal, 65535, ubound_t::lessequal, 256));
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
