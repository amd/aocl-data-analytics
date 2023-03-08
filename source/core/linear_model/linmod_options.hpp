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

#ifndef LINMOD_OPTIONS_HPP
#define LINMOD_OPTIONS_HPP

#include "aoclda.h"
#include "linear_model.hpp"
#include "options.hpp"
#include <limits>

template <class T> da_status register_linmod_options(da_options::OptionRegistry &opts) {
    using namespace da_options;
    T safe_eps = (T)2.0 * std::numeric_limits<T>::epsilon();
    T safe_tol = std::sqrt(safe_eps);
    T max_real = std::numeric_limits<T>::max();

    try {
        std::shared_ptr<OptionNumeric<bool>> ob;
        ob = std::make_shared<OptionNumeric<bool>>(OptionNumeric<bool>(
            "linmod intercept", "Add intercept variable to the model", false));
        opts.register_opt(ob);

        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("linmod norm2 reg", "norm2 regularization term", 0.0,
                             da_options::lbound_t::greaterequal, max_real,
                             da_options::ubound_t::lessthan, 0.0));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("linmod norm1 reg", "norm1 regularization term", 0.0,
                             da_options::lbound_t::greaterequal, max_real,
                             da_options::ubound_t::lessthan, 0.0));
        opts.register_opt(oT);

        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(
            OptionString("linmod optim method", "Select optimization method to use",
                         {{"auto", 0}, {"lbfgs", 1}, {"qr", 2}}, "auto"));
        opts.register_opt(os);

    } catch (std::bad_alloc &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    } catch (...) { // LCOV_EXCL_LINE
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

#endif //LINMOD_OPTIONS_HPP