/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AOCLDA_BASICSTATS
#define AOCLDA_BASICSTATS

#include "aoclda_error.h"
#include "aoclda_types.h"

/**
 * \file
 */

/**
 * \brief Defines whether to compute statistical quantities by row, by column or for the whole data matrix
 **/
typedef enum da_axis_ {
    da_axis_col, ///< Compute statistics column wise.
    da_axis_row, ///< Compute statistics row wise.
    da_axis_all, ///< Compute statistics for the whole data matrix.
} da_axis;

/**
 * \brief Defines the method used to compute quantiles in \ref da_quantile_s and \ref da_quantile_d.
 *
 * @rst
 * The available quantile types correspond to the 9 different quantile types commonly used (see :cite:t:`hyfa96` for further details). It is recommended to use type 6 or type 7 as a default.
 * @endrst
 *
 * Notes about the available types:
 * - Types 1, 2 and 3 give discontinuous results.
 * - Type 8 is recommended if the sample distribution function is unknown.
 * - Type 9 is recommended if the sample distribution function is known to be normal.
 *
 * In each case a number @f$h@f$ is computed, corresponding to the approximate location in the data array of the required quantile, \p q @f$\in [0,1]@f$. Then the quantile is computed as follows:
 */
typedef enum da_quantile_type_ {
    da_quantile_type_1, ///< @f$h=n \times q@f$; return @f$\texttt{x[i]}@f$, where @f$i = \lceil h \rceil@f$.
    da_quantile_type_2, ///< @f$h=n \times q+0.5@f$; return @f$(\texttt{x[i]}+\texttt{x[j]})/2@f$, where @f$i = \lceil h-1/2\rceil@f$ and @f$j = \lfloor h+1/2\rfloor@f$.
    da_quantile_type_3, ///< @f$h=n \times q-0.5@f$; return @f$\texttt{x[i]}@f$, where @f$i@f$ is the nearest integer to @f$h@f$.
    da_quantile_type_4, ///< @f$h=n \times q@f$; return @f$\texttt{x[i]} + (h-\lfloor h \rfloor)(\texttt{x[j]}-\texttt{x[i]})@f$, where @f$i = \lfloor h\rfloor@f$ and @f$j = \lceil h \rceil@f$.
    da_quantile_type_5, ///< @f$h=n \times q+0.5@f$; return @f$\texttt{x[i]} + (h-\lfloor h \rfloor)(\texttt{x[j]}-\texttt{x[i]})@f$, where @f$i = \lfloor h\rfloor@f$ and @f$j = \lceil h \rceil@f$.
    da_quantile_type_6, ///< @f$h=(n+1) \times q@f$; return @f$\texttt{x[i]} + (h-\lfloor h \rfloor)(\texttt{x[j]}-\texttt{x[i]})@f$, where @f$i = \lfloor h\rfloor@f$ and @f$j = \lceil h \rceil@f$.
    da_quantile_type_7, ///< @f$h=(n-1) \times q+1@f$; return @f$\texttt{x[i]} + (h-\lfloor h \rfloor)(\texttt{x[j]}-\texttt{x[i]})@f$, where @f$i = \lfloor h\rfloor@f$ and @f$j = \lceil h \rceil@f$.
    da_quantile_type_8, ///< @f$h=(n+1/3) \times q + 1/3@f$; return @f$\texttt{x[i]} + (h-\lfloor h \rfloor)(\texttt{x[j]}-\texttt{x[i]})@f$, where @f$i = \lfloor h\rfloor@f$ and @f$j = \lceil h \rceil@f$.
    da_quantile_type_9, ///< @f$h=(n+1/4) \times q + 3/8@f$; return @f$\texttt{x[i]} + (h-\lfloor h \rfloor)(\texttt{x[j]}-\texttt{x[i]})@f$, where @f$i = \lfloor h\rfloor@f$ and @f$j = \lceil h \rceil@f$.
} da_quantile_type;

/** \{
 * \brief Arithmetic mean of a data matrix.
 *
 * For a dataset  @f$\{x_1, \dots, x_n\}@f$, the arithmetic mean,  @f$\bar{x}@f$, is defined as
 * \f[
 * \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i.
 * \f]
 *
 * \param[in] axis a \ref da_axis enumerated type, specifying whether means are computed by row, by column, or overall.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[out] mean the array which will hold the computed means. If \p axis = \ref da_axis_col the array must be at least of size @f$p@f$. If \p axis = \ref da_axis_row the array must be at least of size @f$n@f$. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint \p ldx @f$\ge@f$ \p n_rows was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X or \p mean is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 */
da_status da_mean_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                    da_int ldx, double *mean);
da_status da_mean_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                    da_int ldx, float *mean);
/** \} */

/** \{
 * \brief Geometric mean of a data matrix.
 *
 * For a dataset  @f$\{x_1, \dots, x_n\}@f$, the geometric mean,  @f$\bar{x}_{geom}@f$, is defined as
 * \f[
 * \bar{x}_{geom} = \left(\prod_{i=1}^n x_i\right)^{\frac{1}{n}} \equiv \exp\left(\frac{1}{n}\sum_{i=1}^n\ln x_i\right).
 * \f]
 * \param[in] axis a \ref da_axis enumerated type, specifying whether geometric means are computed by row, by column, or overall.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X. Note that \p X must contain non-negative data only.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[out] geometric_mean the array which will hold the computed geometric means. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint \p ldx @f$\ge@f$ \p n_rows was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X or \p geometric_mean is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 * - \ref da_status_negative_data - \p X contains negative data. The geometric mean is not defined.
 */
da_status da_geometric_mean_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                              da_int ldx, double *geometric_mean);
da_status da_geometric_mean_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                              da_int ldx, float *geometric_mean);
/** \} */

/** \{
 * \brief Harmonic mean of a data matrix.
 *
 *  For a dataset  @f$\{x_1, \dots, x_n\}@f$, the harmonic mean,  @f$\bar{x}_{harm}@f$, is defined as
 * \f[
 * \bar{x}_{harm} = \frac{n}{\sum_{i=1}^n \frac{1}{x_i}}.
 * \f]
 *
 * \param[in] axis a \ref da_axis enumerated type, specifying whether harmonic means are computed by row, by column, or overall.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[out] harmonic_mean the array which will hold the computed harmonic means. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint \p ldx @f$\ge@f$ \p n_rows was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X or \p harmonic_mean is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 */
da_status da_harmonic_mean_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                             da_int ldx, double *harmonic_mean);
da_status da_harmonic_mean_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                             da_int ldx, float *harmonic_mean);
/** \} */

/** \{
 * \brief Arithmetic mean and variance of a data matrix.
 *
 * For a dataset  @f$\{x_1, \dots, x_n\}@f$, the variance,  @f$s^2@f$, is defined as
 * \f[
 * s^2 = \frac{1}{\text{dof}}\sum_{i=1}^n(x_i-\bar{x})^2,
 * \f]
 * where dof is the number of <em>degrees of freedom</em>.
 * Setting  @f$\text{dof} = n @f$ gives the sample variance, whereas setting @f$\text{dof} = n -1 @f$ gives an unbiased estimate of the population variance. The argument \p dof is used to specify the number of degrees of freedom.
 *
 * \param[in] axis a \ref da_axis enumerated type, specifying whether statistics are computed by row, by column, or overall.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[in] dof the number of degrees of freedom used to compute the variance:
 * - \p dof < 0 - the degrees of freedom will be set to the number of observations, where the number of observations is \p n_rows for column-wise variances, \p n_cols for row-wise variances and \p n_rows @f$\times @f$ \p n_cols for the overall variance.
 * - \p dof = 0 - the degrees of freedom will be set to the number of observations - 1.
 * - \p dof > 0 - the degrees of freedom will be set to the specified value.
 * \param[out] mean the array which will hold the computed means. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[out] variance the array which will hold the computed variances. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint \p ldx @f$\ge@f$ \p n_rows was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X, \p mean or \p variance is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 */
da_status da_variance_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                        da_int ldx, da_int dof, double *mean, double *variance);
da_status da_variance_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                        da_int ldx, da_int dof, float *mean, float *variance);
/** \} */

/** \{
 * \brief Arithmetic mean, variance and skewness of a data matrix.
 *
 * @rst
 * The skewness is computed as the Fischer-Pearson coefficient of skewness (that is, with the central moments scaled by the number of observations, see :cite:t:`kozw2000`).
 * @endrst
 * Thus, for a dataset  @f$\{x_1, \dots, x_n\}@f$, the skewness, @f$g_1@f$, is defined as
 * \f[
 * g_1 = \frac{\frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^3}{\left[\frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^2\right]^{3/2}}.
 * \f]
 * The degrees of freedom used to compute the variance is given by the number of observations, where the number of observations is \p n_rows for column-wise variances, \p n_cols for row-wise variances and \p n_rows @f$\times @f$ \p n_cols for the overall variance.
 *
 * \param[in] axis a \ref da_axis enumerated type, specifying whether statistics are computed by row, by column, or overall.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[out] mean the array which will hold the computed means. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[out] variance the array which will hold the computed variances. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[out] skewness the array which will hold the computed skewnesses. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint \p ldx @f$\ge@f$ \p n_rows was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X, \p mean, \p variance or \p skewness is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 */
da_status da_skewness_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                        da_int ldx, double *mean, double *variance, double *skewness);
da_status da_skewness_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                        da_int ldx, float *mean, float *variance, float *skewness);
/** \} */

/** \{
 * \brief Arithmetic mean, variance and kurtosis of a data matrix.
 *
 * @rst
 * The kurtosis is computed using Fischer's coefficient of excess kurtosis (that is, with the central moments scaled by the number of observations and 3 subtracted to ensure normally distributed data gives a value of 0, see :cite:t:`kozw2000`).
 * @endrst
 * Thus, for a dataset  @f$\{x_1, \dots, x_n\}@f$, the kurtosis, @f$g_2@f$, is defined as
 * \f[
 * g_2 = \frac{\frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^4}{\left[\frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^2\right]^{2}}-3.
 * \f]
 * The degrees of freedom used to compute the variance is given by the number of observations, where the number of observations is \p n_rows for column-wise variances, \p n_cols for row-wise variances and \p n_rows @f$\times @f$ \p n_cols for the overall variance.
 *
 * \param[in] axis a \ref da_axis enumerated type, specifying whether statistics are computed by row, by column, or overall.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[out] mean the array which will hold the computed means. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n.  If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[out] variance the array which will hold the computed variances. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[out] kurtosis the array which will hold the computed kurtoses. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint \p ldx @f$\ge@f$ \p n_rows was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X, \p mean, \p variance or \p kurtosis is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 */
da_status da_kurtosis_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                        da_int ldx, double *mean, double *variance, double *kurtosis);
da_status da_kurtosis_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                        da_int ldx, float *mean, float *variance, float *kurtosis);
/** \} */

/** \{
 * \brief Central moment of a data matrix.
 *
 * For a dataset  @f$\{x_1, \dots, x_n\}@f$, the <i>k</i>th central moment, @f$m_k@f$, is defined as
 * \f[
 * m_k=\frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^k.
 * \f]
 * Here, the moments are scaled by the number of observations: \p n_rows for column-wise moments, \p n_cols for row-wise moments and \p n_rows @f$\times @f$ \p n_cols for the overall moment.
 * The function gives you the option of supplying precomputed means about which the moments are computed. Otherwise it will compute the means and return them along with the moments.
 *
 * \param[in] axis a \ref da_axis enumerated type, specifying whether moments are computed by row, by column, or overall.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[in] k the order of the moment to be computed. Constraint: k @f$>@f$ 0.
 * \param[in] use_precomputed_mean if nonzero, then means supplied by the calling program will be used. Otherwise means will be computed internally and returned to the calling program.
 * \param[in,out] mean the array which will hold the computed means. If use_precomputed_mean is zero then this array need not be set on entry. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n.  If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[out] moment the array which will hold the computed moments. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint \p ldx @f$\ge@f$ \p n_rows was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X or \p mean is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 * - \ref da_status_invalid_input - @f$k < 0@f$.
 */
da_status da_moment_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                      da_int ldx, da_int k, da_int use_precomputed_mean, double *mean,
                      double *moment);
da_status da_moment_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                      da_int ldx, da_int k, da_int use_precomputed_mean, float *mean,
                      float *moment);
/** \} */

/** \{
 * \brief Selected quantile of a data matrix.
 *
 * Computes the <i>q</i>th quantiles of a data array along the specified axis.
 * Note that there are multiple ways to define quantiles. These are specified using the \ref da_quantile_type enum.
 *
 * \param[in] axis a \ref da_axis enumerated type, specifying whether quantiles are computed by row, by column, or overall.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[in] q the quantile required. Constraint: q must lie in the interval [0,1].
 * \param[out] quantile the array which will hold the computed quantiles. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[in] quantile_type specifies the method used to compute the quantiles.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint \p ldx @f$\ge@f$ \p n_rows was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X or \p quantile is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 * - \ref da_status_invalid_input - \p q is not in the interval @f$[0,1]@f$.
 * - \ref da_status_memory_error - a memory allocation error occurred.
 */
da_status da_quantile_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                        da_int ldx, double q, double *quantile,
                        da_quantile_type quantile_type);
da_status da_quantile_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                        da_int ldx, float q, float *quantile,
                        da_quantile_type quantile_type);
/** \} */

/** \{
 * \brief Summary statistics of a data matrix.
 *
 * Computes the maximum, minimum, median and upper/lower hinges of a data array along the specified axis.
 *
 * \param[in] axis a \ref da_axis enumerated type, specifying whether statistics are computed by row, by column, or overall.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[out] minimum the array which will hold the computed minima. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[out] lower_hinge the array which will hold the computed lower_hinges. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n.  If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[out] median the array which will hold the computed medians. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[out] upper_hinge the array which will hold the computed upper_hinges. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n.  If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[out] maximum the array which will hold the computed maxima. If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint \p ldx @f$\ge@f$ \p n_rows was violated.
 * - \ref da_status_invalid_pointer - one of the array arguments is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 * - \ref da_status_memory_error - a memory allocation error occurred.
 */
da_status da_five_point_summary_d(da_axis axis, da_int n_rows, da_int n_cols,
                                  const double *X, da_int ldx, double *minimum,
                                  double *lower_hinge, double *median,
                                  double *upper_hinge, double *maximum);
da_status da_five_point_summary_s(da_axis axis, da_int n_rows, da_int n_cols,
                                  const float *X, da_int ldx, float *minimum,
                                  float *lower_hinge, float *median, float *upper_hinge,
                                  float *maximum);
/** \} */

/** \{
 * \brief Standardize a data matrix.
 *
 * This routine can be called in various different ways.
 * - If the arrays \p shift and \p scale are both null, then the mean and standard deviations will be computed along the appropriate axis and will be used to shift and scale the data.
 * - If the arrays \p shift and \p scale are both supplied, then the data matrix \p X will be shifted (by subtracting the values in \p shift) then scaled (by dividing by the values in \p scale) along the selected axis.
 * - If one of the arrays \p shift or \p scale contains only zeros, then the mean or standard deviations about the supplied means will be computed as appropriate and stored in that array before being used to standardize the data.
 * - If one of the arrays \p shift or \p scale is null then it will be ignored and only the other will be used (so that the data is only shifted or only scaled).
 *
 *  In each case, if a 0 scaling factor is encountered then it will not be used.
 *
 * An additional computational mode is available by setting \p mode = 1. In this case the standardization is reversed, so that the data matrix is multiplied by the values in \p scale before adding the values in \p shift.
 * This enables users to undo the standardization after the data has been used in another computation.
 *
 * \param[in] axis a \ref da_axis enumerated type, specifying whether statistics are computed by row, by column, or overall.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
  * \param[in] dof the number of degrees of freedom used to compute standard deviations:
 * - \p dof < 0 - the degrees of freedom will be set to the number of observations, where the number of observations is \p n_rows for the column-wise computation, \p n_cols for the row-wise computation and \p n_rows @f$\times @f$ \p n_cols for the overall computation.
 * - \p dof = 0 - the degrees of freedom will be set to the number of observations - 1.
 * - \p dof > 0 - the degrees of freedom will be set to the specified value.
 * \param[in] mode determines whether or not the standardization proceeds in reverse:
 * - \p mode = 0 - the data matrix will be shifted (by subtracting the values in \p shift) then scaled (by dividing by the values in \p scale).
 * - \p mode = 1 - the data matrix will be scaled (by multiplying by the values in \p scale) then shifted (by adding the values in \p shift).
 * \param[in] shift the array of values for shifting the data. Can be null (see above). If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \param[in] scale the array of values for scaling the data. Can be null (see above). If \p axis = \ref da_axis_col the array must be at least of size p. If \p axis = \ref da_axis_row the array must be at least of size n. If \p axis = \ref da_axis_all the array must be at least of size 1.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_input - \p mode must be either 0 or 1.
 * - \ref da_status_invalid_leading_dimension - the constraint \p ldx @f$\ge@f$ \p n_rows was violated.
 * - \ref da_status_invalid_pointer -\p X is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$..
 */
da_status da_standardize_d(da_axis axis, da_int n_rows, da_int n_cols, double *X,
                           da_int ldx, da_int dof, da_int mode, double *shift,
                           double *scale);
da_status da_standardize_s(da_axis axis, da_int n_rows, da_int n_cols, float *X,
                           da_int ldx, da_int dof, da_int mode, float *shift,
                           float *scale);
/** \} */

/** \{
 * \brief Covariance matrix of a data matrix, with the rows treated as observations and the columns treated as variables.
 *
 * For a dataset  @f$X = [\textbf{x}_1, \dots, \textbf{x}_{n_{\text{cols}}}]^T@f$ with column means @f$\{\bar{x}_1, \dots, \bar{x}_{n_{\text{cols}}}\}@f$, the @f$(i, j)@f$ element of the covariance matrix is given by the covariance between @f$\textbf{x}_i@f$ and @f$\textbf{x}_j@f$:
 * \f[
 * \text{cov}(i,j) = \frac{1}{\text{dof}}(\textbf{x}_i-\bar{x}_i)\cdot(\textbf{x}_j-\bar{x}_j),
 * \f]
 * where dof is the number of <em>degrees of freedom</em>.
 * Setting  @f$\text{dof} = n_{\text{cols}} @f$ gives the sample covariances, whereas setting @f$\text{dof} = n_{\text{cols}} -1 @f$ gives unbiased estimates of the population covariances. The argument \p dof is used to specify the number of degrees of freedom.
 *
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[in] dof the number of degrees of freedom used to compute the covariances:
 * - \p dof < 0 - the degrees of freedom will be set to \p n_rows.
 * - \p dof = 0 - the degrees of freedom will be set to \p n_rows - 1.
 * - \p dof > 0 - the degrees of freedom will be set to the specified value.
 * \param[out] cov the array which will hold the p &times; p covariance matrix. Must be of size at least p*ldcov. Data will be returned in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column is stored in the [(<i>j</i> - 1) &times; \a ldcov + <i>i</i> - 1]th entry of the array.
 * \param[in] ldcov the leading dimension of the covariance matrix. Constraint: \p ldcov @f$>@f$ \p n_cols.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - one of the constraints \p ldx @f$\ge@f$ \p n_rows or \p ldcov @f$>@f$ \p n_cols was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X or \p cov is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 * - \ref da_status_memory_error - a memory allocation error occurred.
 */
da_status da_covariance_matrix_d(da_int n_rows, da_int n_cols, const double *X,
                                 da_int ldx, da_int dof, double *cov, da_int ldcov);
da_status da_covariance_matrix_s(da_int n_rows, da_int n_cols, const float *X, da_int ldx,
                                 da_int dof, float *cov, da_int ldcov);
/** \} */

/** \{
 * \brief Correlation matrix of a data matrix, with the rows treated as observations and the columns treated as variables.
 *
 * For a dataset  @f$X = [\textbf{x}_1, \dots, \textbf{x}_{n_{\text{cols}}}]^T@f$ with column means @f$\{\bar{x}_1, \dots, \bar{x}_{n_{\text{cols}}}\}@f$ and column standard deviations @f$\{\sigma_1, \dots, \sigma_{n_{\text{cols}}}\}@f$, the @f$(i, j)@f$ element of the correlation matrix is given by the correlation between @f$\textbf{x}_i@f$ and @f$\textbf{x}_j@f$:
 * \f[
 * \text{corr}(i,j) = \frac{\text{cov}(i,j)}{\sigma_i\sigma_j}.
 * \f]
 * Note that the values in the correlation matrix are independent of the number of degrees of freedom used to compute the standard deviations and covariances.
 *
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n_rows @f$-1@f$ and @f$0 \le j \le@f$ \p n_cols @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows.
 * \param[out] corr the array which will hold the p &times; p correlation matrix. Must be of size at least p*ldcov. Data will be returned in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column is stored in the [(<i>j</i> - 1) &times; \a ldcov + <i>i</i> - 1]th entry of the array.
 * \param[in] ldcorr the leading dimension of the correlation matrix. Constraint: \p ldcorr @f$>@f$ \p n_cols.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - one of the constraints \p ldx @f$\ge@f$ \p n_rows or \p ldcorr @f$>@f$ \p n_cols was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X or \p corr is null.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 * - \ref da_status_memory_error - a memory allocation error occurred.
 */
da_status da_correlation_matrix_d(da_int n_rows, da_int n_cols, const double *X,
                                  da_int ldx, double *corr, da_int ldcorr);
da_status da_correlation_matrix_s(da_int n_rows, da_int n_cols, const float *X,
                                  da_int ldx, float *corr, da_int ldcorr);
/** \} */

#endif
