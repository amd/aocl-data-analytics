/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef PCA_OPTIONS_HPP
#define PCA_OPTIONS_HPP

#include "options.hpp"
#include <limits>

template <class T>
inline da_status register_pca_options(da_options::OptionRegistry &opts) {
    using namespace da_options;
    T max_real = std::numeric_limits<T>::max();

    try {
        std::shared_ptr<OptionNumeric<da_int>> oi;

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "print level", "set level of verbosity for the solver", 0,
            da_options::lbound_t::greaterequal, 5, da_options::ubound_t::lessequal, 0));
        opts.register_opt(oi);

        oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
            "number of principal components", "Add intercept variable to the model", 3,
            da_options::lbound_t::greaterequal, 1, da_options::ubound_t::lessequal, 5));
        opts.register_opt(oi);

        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(
            OptionString("pca method", "Select SVD method to compute PCA",
                         {{"svd", 0}, {"corr", 1}}, "svd"));
        opts.register_opt(os);
        os = std::make_shared<OptionString>(
            OptionString("print options", "Print option list and set values.",
                         {{"no", 0}, {"yes", 2}}, "no"));
        opts.register_opt(os);

    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    } catch (...) {
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

#endif //LINMOD_OPTIONS_HPP