/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 */

#ifndef THIN_SVD_HPP
#define THIN_SVD_HPP

#include "aoclda.h"
#include "da_error.hpp"
#include "macros.h"

namespace ARCH {

namespace da_svd {

/**
 * Compute the thin SVD of a column-major matrix A using LAPACK's gesdd
 * (jobz = 'S').
 *
 * On success: A = U * diag(S) * V^T with k = min(m, n) singular triplets.
 *
 * Buffers are caller-allocated. The contract mirrors LAPACK's gesdd, so no
 * data is copied beyond what gesdd itself needs.
 *
 *   A    - input matrix, column-major. **Destroyed on return** (gesdd
 *          overwrites it). Size: at least lda * n.
 *   m    - number of rows. m >= 1.
 *   n    - number of columns. n >= 1.
 *   lda  - leading dimension of A. lda >= m.
 *   U    - output left singular vectors, column-major; the leading m * k
 *          block is written. Size: at least ldu * k.
 *   ldu  - leading dimension of U. ldu >= m.
 *   S    - output singular values in decreasing order. Size: at least k.
 *   Vt   - output transposed right singular vectors, column-major; the
 *          leading k * n block is written. Size: at least ldvt * n.
 *   ldvt - leading dimension of Vt. ldvt >= k.
 *   err  - error registry for diagnostic messages.
 *
 * Returns:
 *   - da_status_success
 *   - da_status_invalid_input              (bad dimension or null pointer)
 *   - da_status_invalid_leading_dimension
 *   - da_status_memory_error               (workspace allocation failed)
 *   - da_status_numerical_difficulties     (gesdd failed to converge)
 *   - da_status_internal_error             (unexpected gesdd error)
 */
template <typename T>
da_status thin_svd(T *A, da_int m, da_int n, da_int lda, T *U, da_int ldu, T *S, T *Vt,
                   da_int ldvt, da_errors::da_error_t &err);

} // namespace da_svd
} // namespace ARCH

#endif // THIN_SVD_HPP
