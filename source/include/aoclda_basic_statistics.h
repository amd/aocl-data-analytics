#ifndef AOCLDA_BASICSTATS
#define AOCLDA_BASICSTATS

#include "aoclda_error.h"
#include "aoclda_types.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Defines whether to compute statistical quantities by row, by column or for the whole data matrix
 */
typedef enum da_axis_ {
    da_axis_col, ///< Compute statistics column wise.
    da_axis_row, ///< Compute statistics row wise.
    da_axis_all, ///< Compute statistics for the whole data matrix.
} da_axis;

/**
 * \brief Defines the method used to compute quantiles in \ref da_quantile_s and \ref da_quantile_d.
 * The available quantile types correspond to the 9 different quantile types commonly used. It is recommended to use type 6 or type 7 as a default.
 * 
 * Notes about the available types:
 * - Types 1, 2 and 3 give discontinuous results.
 * - Type 8 is recommended if the sample distribution function is unknown.
 * - Type 9 is recommended if the sample distribution function is known to be normal.
 * 
 * In each case a number @f$h@f$ is computed, corresponding to the approximate location of the quantile within the data array, then the quantile is computed as follows:
 */
typedef enum da_quantile_type_ {
    da_quantile_type_1, ///< @f$h=n \times q@f$; return @f$\texttt{x[i]}@f$, where @f$i = \lceil h \rceil@f$.
    da_quantile_type_2, ///< @f$h=n \times q+0.5@f$; return @f$(\texttt{x[i]}+\texttt{x[j]})/2@f$, where @f$i = \lceil h-1/2\rceil@f$ and @f$j = \lfloor h+1/2\rfloor@f$.
    da_quantile_type_3, ///< @f$h=n \times q-0.5@f$; return @f$\texttt{x[i]}@f$, where @f$i@f$ is the nearest integer to @f$h@f$.
    da_quantile_type_4, ///< @f$h=n \times q@f$; return @f$\texttt{x[i]} + (h-\lfloor h \rfloor)(\texttt{x[j]}-\texttt{x[k]})@f$, where @f$i@f$ is the nearest integer to @f$h@f$, @f$j = \lceil h \rceil@f$ and @f$k = \lfloor h\rfloor@f$.
    da_quantile_type_5, ///< @f$h=n \times q+0.5@f$; return @f$\texttt{x[i]} + (h-\lfloor h\rfloor)(\texttt{x[j]}-\texttt{x[k]})@f$, where @f$i@f$ is the nearest integer to @f$h@f$, @f$j = \lceil h \rceil@f$, @f$k = \lfloor h\rfloor@f$.
    da_quantile_type_6, ///< @f$h=(n+1) \times q@f$; return @f$\texttt{x[i]} + (h-\lfloor h\rfloor)(\texttt{x[j]}-\texttt{x[k]})@f$, where @f$i@f$ is the nearest integer to @f$h@f$, @f$j = \lceil h \rceil@f$, @f$k = \lfloor h\rfloor@f$.
    da_quantile_type_7, ///< @f$h=(n-1) \times q+1@f$; return @f$\texttt{x[i]} + (h-\lfloor h\rfloor)(\texttt{x[j]}-\texttt{x[k]})@f$, where @f$i@f$ is the nearest integer to @f$h@f$, @f$j = \lceil h \rceil@f$, @f$k = \lfloor h\rfloor@f$.
    da_quantile_type_8, ///< @f$h=(n+1/3) \times q + 1/3@f$; return @f$\texttt{x[i]} + (h-\lfloor h\rfloor)(\texttt{x[j]}-\texttt{x[k]})@f$, where @f$i@f$ is the nearest integer to @f$h@f$, @f$j = \lceil h \rceil@f$, @f$k = \lfloor h\rfloor@f$.
    da_quantile_type_9, ///< @f$h=(n+1/4) \times q + 3/8@f$; return @f$\texttt{x[i]} + (h-\lfloor h\rfloor)(\texttt{x[j]}-\texttt{x[k]})@f$, where @f$i@f$ is the nearest integer to @f$h@f$, @f$j = \lceil h \rceil@f$, @f$k = \lfloor h\rfloor@f$.
} da_quantile_type;

/** \{
 * \brief Arithmetic mean of a data array.
 * 
 * \param[in] axis a \ref da_axis enumerated type, specifying whether means are computed by row, by column, or overall.
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[out] mean the array which will hold the computed means. If \ref axis = da_axis_col the array must be at least of size @f$p@f$. If \ref axis = da_axis_row the array must be at least of size @f$n@f$. If \ref axis = da_axis_all the array must be at least of size 1.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the arrays @f$\texttt{x}@f$ or @f$\texttt{mean}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 */
da_status da_mean_d(da_axis axis, da_int n, da_int p, const double *x, da_int ldx,
                    double *mean);
da_status da_mean_s(da_axis axis, da_int n, da_int p, const float *x, da_int ldx,
                    float *mean);
/** \} */

/** \{
 * \brief Geometric mean of a data array.
 * 
 * \param[in] axis a \ref da_axis enumerated type, specifying whether geometric means are computed by row, by column, or overall.
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array. Note that x must contain non-negative data only.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[out] geometric_mean the array which will hold the computed geometric means. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the arrays @f$\texttt{x}@f$ or @f$\texttt{geometric\_mean}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 * - \ref da_status_negative_data - @f$\texttt{x}@f$ contains negative data. The geometric mean is not defined.
 */
da_status da_geometric_mean_d(da_axis axis, da_int n, da_int p, const double *x,
                              da_int ldx, double *geometric_mean);
da_status da_geometric_mean_s(da_axis axis, da_int n, da_int p, const float *x,
                              da_int ldx, float *geometric_mean);
/** \} */

/** \{
 * \brief Harmonic mean of a data array.
 * 
 * \param[in] axis a \ref da_axis enumerated type, specifying whether harmonic means are computed by row, by column, or overall.
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[out] harmonic_mean the array which will hold the computed harmonic means. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the arrays @f$\texttt{x}@f$ or @f$\texttt{harmonic\_mean}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 */
da_status da_harmonic_mean_d(da_axis axis, da_int n, da_int p, const double *x,
                             da_int ldx, double *harmonic_mean);
da_status da_harmonic_mean_s(da_axis axis, da_int n, da_int p, const float *x, da_int ldx,
                             float *harmonic_mean);
/** \} */

/** \{
 * \brief Arithmetic mean and variance of a data array.
 * The variance is scaled by the number of degrees of freedom - 1, to give an unbiased estimate of the population variance, based on the sample in the data matrix. The number of degrees of freedom is n for column-wise variances, p for row-wise variances and n &times; p for the overall variance.
 * 
 * \param[in] axis a \ref da_axis enumerated type, specifying whether statistics are computed by row, by column, or overall.
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[out] mean the array which will hold the computed means. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[out] variance the array which will hold the computed variances. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the arrays @f$\texttt{x}@f$, @f$\texttt{mean}@f$ or @f$\texttt{variance}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 */
da_status da_variance_d(da_axis axis, da_int n, da_int p, const double *x, da_int ldx,
                        double *mean, double *variance);
da_status da_variance_s(da_axis axis, da_int n, da_int p, const float *x, da_int ldx,
                        float *mean, float *variance);
/** \} */

/** \{
 * \brief Arithmetic mean, variance and skewness of a data array.
 * The variance is scaled by the number of degrees of freedom - 1, to give an unbiased estimate of the population variance, based on the sample in the data matrix. The number of degrees of freedom is n for column-wise variances, p for row-wise variances and n &times; p for the overall variance.
 * The skewness is computed as the Fischer-Pearson coefficient of skewness (that is, with the central moments scaled by the number of degrees of freedom).
 * 
 * \param[in] axis a \ref da_axis enumerated type, specifying whether statistics are computed by row, by column, or overall.
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[out] mean the array which will hold the computed means. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[out] variance the array which will hold the computed variances. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[out] skewness the array which will hold the computed skewnesses. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the arrays @f$\texttt{x}@f$, @f$\texttt{mean}@f$, @f$\texttt{variance}@f$ or @f$\texttt{skewness}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 */
da_status da_skewness_d(da_axis axis, da_int n, da_int p, const double *x, da_int ldx,
                        double *mean, double *variance, double *skewness);
da_status da_skewness_s(da_axis axis, da_int n, da_int p, const float *x, da_int ldx,
                        float *mean, float *variance, float *skewness);
/** \} */

/** \{
 * \brief Arithmetic mean, variance and kurtosis of a data array.
 * The variance is scaled by the number of degrees of freedom - 1, to give an unbiased estimate of the population variance, based on the sample in the data matrix. The number of degrees of freedom is n for column-wise variances, p for row-wise variances and n &times; p for the overall variance.
 * The kurtosis is computed using Fischer's definition of excess kurtosis (that is, with the central moments scaled by the number of degrees of freedom and 3 subtracted to ensure normally distributed data gives a value of 0).
 * 
 * \param[in] axis a \ref da_axis enumerated type, specifying whether statistics are computed by row, by column, or overall.
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[out] mean the array which will hold the computed means. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n.  If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[out] variance the array which will hold the computed variances. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[out] kurtosis the array which will hold the computed skewnesses. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the arrays @f$\texttt{x}@f$, @f$\texttt{mean}@f$, @f$\texttt{variance}@f$ or @f$\texttt{kurtosis}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 */
da_status da_kurtosis_d(da_axis axis, da_int n, da_int p, const double *x, da_int ldx,
                        double *mean, double *variance, double *kurtosis);
da_status da_kurtosis_s(da_axis axis, da_int n, da_int p, const float *x, da_int ldx,
                        float *mean, float *variance, float *kurtosis);
/** \} */

/** \{
 * \brief Central moment a data array.
 * The <i>k</i>th central moments of a data array are computed along the specified axis.
 * The moments are scaled by the number of degrees of freedom: n for column-wise moments, p for row-wise moments and n &times; p for the overall moment.
 * The function gives you the option of supplying precomputed means about which the moments are computed. Otherwise it will compute the means and return them along with the moments.
 * 
 * \param[in] axis a \ref da_axis enumerated type, specifying whether moments are computed by row, by column, or overall.
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[in] k the order of the moment to be computed. Constraint: k >= 0.
 * \param[in] use_precomputed_mean if nonzero, then means supplied by the calling program will be used. Otherwise means will be computed internally and returned to the calling program.
 * \param[in,out] mean the array which will hold the computed means. If use_precomputed_mean is zero then this array need not be set on entry. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n.  If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[out] moment the array which will hold the computed moments. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the arrays @f$\texttt{x}@f$ or @f$\texttt{mean}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 * - \ref da_status_invalid_input - @f$k < 0@f$.
 */
da_status da_moment_d(da_axis axis, da_int n, da_int p, const double *x, da_int ldx,
                      da_int k, da_int use_precomputed_mean, double *mean,
                      double *moment);
da_status da_moment_s(da_axis axis, da_int n, da_int p, const float *x, da_int ldx,
                      da_int k, da_int use_precomputed_mean, float *mean, float *moment);
/** \} */

/** \{
 * \brief Selected quantile of a data array.
 * Computes the <i>q</i>th quantiles of a data array along the specified axis.
 * Note that there are multiple ways to define quantiles. These are specified using the \ref da_quantile_type enum.
 * 
 * \param[in] axis a \ref da_axis enumerated type, specifying whether quantiles are computed by row, by column, or overall.
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[in] q the quantile required. Constraint: q must lie in the interval [0,1].
 * \param[out] quantile the array which will hold the computed quantiles. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[in] quantile_type specifies the method used to compute the quantiles.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the arrays @f$\texttt{x}@f$ or @f$\texttt{quantile}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 * - \ref da_status_invalid_input - @f$q@f$ is not in the interval @f$[0,1]@f$.
 * - \ref da_status_memory_error - a memory allocation error occured.
 */
da_status da_quantile_d(da_axis axis, da_int n, da_int p, const double *x, da_int ldx,
                        double q, double *quantile, da_quantile_type quantile_type);
da_status da_quantile_s(da_axis axis, da_int n, da_int p, const float *x, da_int ldx,
                        float q, float *quantile, da_quantile_type quantile_type);
/** \} */

/** \{
 * \brief Summary statistics of a data array.
 * Computes the maximum/minumum, median and upper/lower hinges of a data array along the specified axis.
 * 
 * \param[in] axis a \ref da_axis enumerated type, specifying whether statistics are computed by row, by column, or overall.
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[out] minimum the array which will hold the computed minima. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[out] lower_hinge the array which will hold the computed lower_hinges. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n.  If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[out] median the array which will hold the computed medians. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[out] upper_hinge the array which will hold the computed upper_hinges. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n.  If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[out] maximum the array which will hold the computed maxima. If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the array arguments is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 * - \ref da_status_memory_error - a memory allocation error occured.
 */
da_status da_five_point_summary_d(da_axis axis, da_int n, da_int p, const double *x,
                                  da_int ldx, double *minimum, double *lower_hinge,
                                  double *median, double *upper_hinge, double *maximum);
da_status da_five_point_summary_s(da_axis axis, da_int n, da_int p, const float *x,
                                  da_int ldx, float *minimum, float *lower_hinge,
                                  float *median, float *upper_hinge, float *maximum);
/** \} */

/** \{
 * \brief Standardize a data array.
 * This routine can be called in various different ways:
 * - if the arrays shift and scale are both supplied, then the data matrix x will be shifted (by subtracting the values in shift) and scaled (by dividing by the values in scale) along the selected axis.
 * - if the arrays shift and scale are both null, then the mean and standard deviations will be computed along the appropriate axis and will be used to shift and scale the data.
 * - if one of the arrays (shift or scale) is null then it will be ignored and only the other will be used (so that the data is only shifted or only scaled).
 * In each case, if a 0 scaling factor is encountered then it will not be used.
 * 
 * \param[in] axis a \ref da_axis enumerated type, specifying whether statistics are computed by row, by column, or overall.
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[in] shift the array of values for shifting the data. Can be null (see above). If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \param[in] scale the array of values for scaling the data. Can be null (see above). If \ref axis = da_axis_col the array must be at least of size p. If \ref axis = da_axis_row the array must be at least of size n. If \ref axis = da_axis_all the array must be at least of size 1.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - @f$\texttt{x}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$..
 */
da_status da_standardize_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                           double *shift, double *scale);
da_status da_standardize_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                           float *shift, float *scale);
/** \} */

/** \{
 * \brief Covariance matrix of a data array.
 * Covariance matrix of a data array, with the rows treated as observations and the columns treated as variables. The scaling factor n - 1 is used in computing the covariances.
 * 
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[out] cov the array which will hold the p &times; p covariance matrix. Must be of size at least p*ldcov. Data will be returned in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) &times; \a ldcov + <i>i</i> - 1]th entry of the array.
 * \param[in] ldcov the leading dimension of the covariance matrix. Constraint: ldcov >= p.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the arrays @f$\texttt{x}@f$ or @f$\texttt{cov}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 * - \ref da_status_memory_error - a memory allocation error occured.
 */
da_status da_covariance_matrix_d(da_int n, da_int p, const double *x, da_int ldx,
                                 double *cov, da_int ldcov);
da_status da_covariance_matrix_s(da_int n, da_int p, const float *x, da_int ldx,
                                 float *cov, da_int ldcov);
/** \} */

/** \{
 * \brief Correlation matrix of a data array.
 * Correlation matrix of a data array, with the rows treated as observations and the columns treated as variables.
 * 
 * \param[in] n the number of rows (observations) in the data matrix. Constraint: @f$n \ge 1@f$.
 * \param[in] p the number of columns (variables) in the data matrix. Constraint: @f$p \ge 1@f$.
 * \param[in] x the @f$n \times p@f$ data matrix. Data is expected to be stored in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) @f$\times@f$ \a ldx + <i>i</i> - 1]th entry of the array.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: @f$ldx \ge n@f$.
 * \param[out] corr the array which will hold the p &times; p correlation matrix. Must be of size at least p*ldcov. Data will be returned in column major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>j</i> - 1) &times; \a ldcov + <i>i</i> - 1]th entry of the array.
 * \param[in] ldcorr the leading dimension of the correlation matrix. Constraint: ldcov >= p.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - the constraint @f$ldx \ge n@f$ was violated.
 * - \ref da_status_invalid_pointer - one of the arrays @f$\texttt{x}@f$ or @f$\texttt{corr}@f$ is null.
 * - \ref da_status_invalid_array_dimension - either @f$n < 1@f$ or @f$p < 1@f$.
 * - \ref da_status_memory_error - a memory allocation error occured.
 */
da_status da_correlation_matrix_d(da_int n, da_int p, const double *x, da_int ldx,
                                  double *corr, da_int ldcorr);
da_status da_correlation_matrix_s(da_int n, da_int p, const float *x, da_int ldx,
                                  float *corr, da_int ldcorr);
/** \} */

/**
 *\}
 */

#ifdef __cplusplus
}
#endif

#endif
