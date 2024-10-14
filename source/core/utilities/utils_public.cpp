/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "aoclda.h"
#include "da_utils.hpp"

da_status da_check_data_d(da_order order, da_int n_rows, da_int n_cols, const double *X,
                          da_int ldx) {
    return da_utils::check_data(order, n_rows, n_cols, X, ldx);
}

da_status da_check_data_s(da_order order, da_int n_rows, da_int n_cols, const float *X,
                          da_int ldx) {
    return da_utils::check_data(order, n_rows, n_cols, X, ldx);
}

da_status da_switch_order_copy_d(da_order order, da_int n_rows, da_int n_cols,
                                 const double *X, da_int ldx, double *Y, da_int ldy) {
    return da_utils::switch_order_copy(order, n_rows, n_cols, X, ldx, Y, ldy);
}
da_status da_switch_order_copy_s(da_order order, da_int n_rows, da_int n_cols,
                                 const float *X, da_int ldx, float *Y, da_int ldy) {
    return da_utils::switch_order_copy(order, n_rows, n_cols, X, ldx, Y, ldy);
}

da_status da_switch_order_in_place_d(da_order order_X_in, da_int n_rows, da_int n_cols,
                                     double *X, da_int ldx_in, da_int ldx_out) {
    return da_utils::switch_order_in_place(order_X_in, n_rows, n_cols, X, ldx_in,
                                           ldx_out);
}

da_status da_switch_order_in_place_s(da_order order_X_in, da_int n_rows, da_int n_cols,
                                     float *X, da_int ldx_in, da_int ldx_out) {
    return da_utils::switch_order_in_place(order_X_in, n_rows, n_cols, X, ldx_in,
                                           ldx_out);
}