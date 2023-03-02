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

#ifndef AOCLDA_OPTIONS_H
#define AOCLDA_OPTIONS_H

#include "aoclda_handle.h"
#include "aoclda_types.h"

da_status da_options_set_bool(da_handle handle, const char *option, bool value);
da_status da_options_set_int(da_handle handle, const char *option, int value);
da_status da_options_set_string(da_handle handle, const char *option, const char *value);
da_status da_options_set_s_real(da_handle handle, const char *option, float value);
da_status da_options_set_d_real(da_handle handle, const char *option, double value);

da_status da_options_get_bool(da_handle handle, const char *option, bool *value);
da_status da_options_get_int(da_handle handle, const char *option, int *value);
da_status da_options_get_string(da_handle handle, const char *option, char *value, size_t lvalue);
da_status da_options_get_s_real(da_handle handle, const char *option, float *value);
da_status da_options_get_d_real(da_handle handle, const char *option, double *value);

#endif //AOCLDA_OPTIONS_H