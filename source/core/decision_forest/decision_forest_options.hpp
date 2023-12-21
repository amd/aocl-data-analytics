/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

template <class T>
inline da_status register_df_options(da_options::OptionRegistry &opts) {
    da_status status = da_status_success;

    try {
        // put following in global namespace for this try{ }
        // da_options::OptionString, da_options::OptionNumeric,
        // da_options::lbound_t,     da_options::ubound_t,
        using namespace da_options;

        std::shared_ptr<OptionString> os;
        std::shared_ptr<OptionNumeric<da_int>> oi;
        std::shared_ptr<OptionNumeric<T>> oT;

        os = std::make_shared<OptionString>(OptionString(
            "scoring function", "Select scoring function to use",
            {{"gini", 0}, {"cross-entropy", 1}, {"misclassification-error", 2}}, "gini"));
        status = opts.register_opt(os);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "depth", "set max depth of tree", -1, lbound_t::greaterequal, max_da_int,
            ubound_t::p_inf, -1));
        status = opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "seed", "set random seed for Mersenne Twister (64-bit) PRNG", -1,
            lbound_t::greaterequal, max_da_int, ubound_t::lessequal, -1));
        status = opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_obs_per_tree", "set number of observations in each tree", 0,
            lbound_t::greaterthan, max_da_int, ubound_t::p_inf, 1));
        status = opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "n_features_to_select", "set number of features in selection for splitting",
            0, lbound_t::greaterthan, max_da_int, ubound_t::p_inf, 1));
        status = opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(
            OptionNumeric<da_int>("n_trees", "set number of features in each tree", 0,
                                  lbound_t::greaterthan, max_da_int, ubound_t::p_inf, 1));
        status = opts.register_opt(oi);

        T rmax = std::numeric_limits<T>::max();
        T diff_thres_default = (T)1e-6;
        oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
            "diff_thres", "minimum difference in feature value required for splitting",
            0.0, lbound_t::greaterthan, rmax, ubound_t::p_inf, diff_thres_default));
        status = opts.register_opt(oT);

    } catch (std::bad_alloc &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    } catch (std::invalid_argument &e) {
        std::cerr << e.what() << std::endl; // LCOV_EXCL_LINE
        return da_status_internal_error;    // LCOV_EXCL_LINE
    } catch (...) {                         // LCOV_EXCL_LINE
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return status;
}
