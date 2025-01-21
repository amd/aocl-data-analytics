/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "basic_statistics.hpp"
#include "macros.h"

namespace ARCH {

namespace da_basic_statistics {

/* Utility function for basic statistics routines:
    Inputs: order, axis, n_in, p_in, ldx
    If order is column_major, then we just check validity of n_in, p_in and ldx and copy to n and p
    If order is row_major, then we swap the axis, check validity of n_in, p_in and ldx and copy to p and n respectively.
    This ensures that the calling function can continue as if it was a column major order computation.
    */
da_status row_to_col_major(da_order order, da_axis axis_in, da_int n_in, da_int p_in,
                           da_int ldx, da_axis &axis, da_int &n, da_int &p) {
    if (n_in < 1 || p_in < 1)
        return da_status_invalid_array_dimension;

    if (order == column_major) {
        if (ldx < n_in)
            return da_status_invalid_leading_dimension;
        n = n_in;
        p = p_in;
        axis = axis_in;
        return da_status_success;
    } else {
        if (ldx < p_in)
            return da_status_invalid_leading_dimension;
        n = p_in;
        p = n_in;
        if (axis_in == da_axis_row)
            axis = da_axis_col;
        if (axis_in == da_axis_col)
            axis = da_axis_row;
        if (axis_in == da_axis_all)
            axis = da_axis_all;
        return da_status_success;
    }
}

} // namespace da_basic_statistics

} // namespace ARCH
