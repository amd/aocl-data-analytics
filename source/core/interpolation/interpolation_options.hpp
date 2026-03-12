/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef INTERPOLATION_OPTIONS_HPP
#define INTERPOLATION_OPTIONS_HPP

#include "aoclda_types.h"
#include "cubic_spline/cubic_spline_types.hpp"
#include "da_error.hpp"
#include "macros.h"
#include "options.hpp"

namespace ARCH {

namespace da_interpolation {

using namespace da_cubic_spline;

template <class T>
inline da_status register_interpolation_options(da_options::OptionRegistry &opts,
                                                da_errors::da_error_t &err) {
    using namespace da_options;

    try {
        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(OptionString(
            "cubic spline type",
            "Type of cubic spline to construct. Options: 'natural' (zero second "
            "derivatives at endpoints), 'clamped zero' (zero first derivatives at "
            "endpoints), "
            "'custom' (user-specified first or second derivatives at endpoints), "
            "'Hermite' (piecewise cubic Hermite interpolation).",
            {{"natural", da_cubic_spline::natural},
             {"clamped zero", da_cubic_spline::clamped},
             {"custom", da_cubic_spline::custom},
             {"hermite", da_cubic_spline::hermite}},
            "natural"));
        opts.register_opt(os);

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

} // namespace da_interpolation
} // namespace ARCH

#endif // INTERPOLATION_OPTIONS_HPP
