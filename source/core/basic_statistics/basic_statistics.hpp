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
#include "macros.h"

namespace ARCH {

namespace da_basic_statistics {

/* Compute double/float raised to positive integer power efficiently by binary powering */
template <typename T> T power(T a, da_int exponent);

/* Arithmetic mean along specified axis */
template <typename T>
da_status mean(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
               da_int ldx, T *amean);

/* Geometric mean computed using log and exp to avoid overflow. Care needed to deal with negative or zero entries */
template <typename T>
da_status geometric_mean(da_order order, da_axis axis_in, da_int n_in, da_int p_in,
                         const T *x, da_int ldx, T *gmean);

/* Harmonic mean along a specified axis */
template <typename T>
da_status harmonic_mean(da_order order, da_axis axis_in, da_int n_in, da_int p_in,
                        const T *x, da_int ldx, T *hmean);

/* Mean and variance along specified axis */
template <typename T>
da_status variance(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
                   da_int ldx, da_int dof, T *amean, T *var);

/* Mean, variance and skewness along specified axis */
template <typename T>
da_status skewness(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
                   da_int ldx, T *amean, T *var, T *skew);

/* Mean, variance and kurtosis along specified axis */
template <typename T>
da_status kurtosis(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
                   da_int ldx, T *amean, T *var, T *kurt);

/* kth moment along specified axis. Optionally use precomputed mean. */
template <typename T>
da_status moment(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
                 da_int ldx, da_int k, da_int use_precomputed_mean, T *amean, T *mom);

/* Correlation or covariance matrix of x */
template <typename T>
da_status cov_corr_matrix(da_order order, da_int n, da_int p, const T *x, da_int ldx,
                          da_int dof, T *mat, da_int ldmat, bool compute_corr);

template <typename T>
da_status covariance_matrix(da_order order, da_int n, da_int p, const T *x, da_int ldx,
                            da_int dof, T *cov, da_int ldcov);

template <typename T>
da_status correlation_matrix(da_order order, da_int n, da_int p, const T *x, da_int ldx,
                             T *corr, da_int ldcorr);

/* This routine uses the partial sort routine std::nth_element to correctly place the kth element of x.
   It uses the index array to do the sorting, so x is not itself reordered. */
template <typename T>
da_status indexed_partial_sort(const T *x, da_int length, da_int stride, da_int *xindex,
                               da_int k, da_int dim1, bool two_d, T &stat);

/* Compute the qth quantile of x along the specified axis */
template <typename T>
da_status quantile(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
                   da_int ldx, T q, T *quant, da_quantile_type quantile_type);

/* Compute min/max, hinges and median along specified axis */
template <typename T>
da_status five_point_summary(da_order order, da_axis axis_in, da_int n_in, da_int p_in,
                             const T *x, da_int ldx, T *minimum, T *lower_hinge,
                             T *median, T *upper_hinge, T *maximum);

template <typename T>
da_status standardize(da_order order, da_axis axis_in, da_int n_in, da_int p_in, T *x,
                      da_int ldx, da_int dof, da_int mode, T *shift, T *scale);

da_status row_to_col_major(da_order order, da_axis axis_in, da_int n_in, da_int p_in,
                           da_int ldx, da_axis &axis, da_int &n, da_int &p);

} // namespace da_basic_statistics

} // namespace ARCH
