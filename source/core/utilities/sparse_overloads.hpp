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

#ifndef SPARSE_TEMPLATES_HPP
#define SPARSE_TEMPLATES_HPP
#include "aoclsparse.h"

template <class T>
inline aoclsparse_status aoclsparse_itsol_init(aoclsparse_itsol_handle *handle);
template <>
inline aoclsparse_status aoclsparse_itsol_init<double>(aoclsparse_itsol_handle *handle) {
    return aoclsparse_itsol_d_init(handle);
}
template <>
inline aoclsparse_status aoclsparse_itsol_init<float>(aoclsparse_itsol_handle *handle) {
    return aoclsparse_itsol_s_init(handle);
}

inline aoclsparse_status aoclsparse_itsol_rci_input(aoclsparse_itsol_handle handle,
                                                    aoclsparse_int n, const double *b) {
    return aoclsparse_itsol_d_rci_input(handle, n, b);
}
inline aoclsparse_status aoclsparse_itsol_rci_input(aoclsparse_itsol_handle handle,
                                                    aoclsparse_int n, const float *b) {
    return aoclsparse_itsol_s_rci_input(handle, n, b);
}

inline aoclsparse_status aoclsparse_itsol_rci_solve(aoclsparse_itsol_handle handle,
                                                    aoclsparse_itsol_rci_job *ircomm,
                                                    double **u, double **v, double *x,
                                                    double rinfo[100]) {
    return aoclsparse_itsol_d_rci_solve(handle, ircomm, u, v, x, rinfo);
}
inline aoclsparse_status aoclsparse_itsol_rci_solve(aoclsparse_itsol_handle handle,
                                                    aoclsparse_itsol_rci_job *ircomm,
                                                    float **u, float **v, float *x,
                                                    float rinfo[100]) {
    return aoclsparse_itsol_s_rci_solve(handle, ircomm, u, v, x, rinfo);
}

namespace da {} // namespace da
#endif
