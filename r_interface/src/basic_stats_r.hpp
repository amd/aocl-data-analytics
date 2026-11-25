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

#ifndef BASIC_STATS_R_HPP
#define BASIC_STATS_R_HPP

#include <Rcpp.h>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

// Helper function to get matrix dimensions, axis enum and output size
inline void get_size_and_axis(const std::string &axis, da_axis &axis_enum,
                              const NumericMatrix &X, size_t &size, da_int &m,
                              da_int &n) {
    m = X.nrow();
    n = X.ncol();

    if (axis == "all") {
        axis_enum = da_axis_all;
    } else if (axis == "col") {
        axis_enum = da_axis_col;
    } else if (axis == "row") {
        axis_enum = da_axis_row;
    } else {
        stop("Invalid axis. Must be 'all', 'row', or 'col'");
    }

    if (n == 1 && axis_enum != da_axis_all) {
        axis_enum = da_axis_all;
        Rf_warning(
            "Axes 'col' and 'row' are undefined for 1D arrays. Defaulting to 'all'.");
    }

    switch (axis_enum) {
    case da_axis_all:
        size = 1;
        break;
    case da_axis_col:
        size = n;
        break;
    case da_axis_row:
        size = m;
        break;
    }
}

inline NumericVector da_mean_wrapper(NumericMatrix X, std::string axis = "col") {
    da_status status;
    da_int m, n;
    da_axis axis_enum;
    size_t mean_sz;
    get_size_and_axis(axis, axis_enum, X, mean_sz, m, n);

    NumericVector mean(mean_sz);

    status = da_mean_d(column_major, axis_enum, m, n, X.begin(), m, mean.begin());
    status_to_exception(status);

    return mean;
}

inline NumericVector da_harmonic_mean_wrapper(NumericMatrix X, std::string axis = "col") {
    da_status status;
    da_int m, n;
    da_axis axis_enum;
    size_t harmonic_mean_sz;
    get_size_and_axis(axis, axis_enum, X, harmonic_mean_sz, m, n);

    NumericVector harmonic_mean(harmonic_mean_sz);

    status = da_harmonic_mean_d(column_major, axis_enum, m, n, X.begin(), m,
                                harmonic_mean.begin());
    status_to_exception(status);

    return harmonic_mean;
}

inline NumericVector da_geometric_mean_wrapper(NumericMatrix X,
                                               std::string axis = "col") {
    da_status status;
    da_int m, n;
    da_axis axis_enum;
    size_t mean_sz;
    get_size_and_axis(axis, axis_enum, X, mean_sz, m, n);

    NumericVector geometric_mean(mean_sz);

    status = da_geometric_mean_d(column_major, axis_enum, m, n, X.begin(), m,
                                 geometric_mean.begin());
    status_to_exception(status);

    return geometric_mean;
}

inline NumericVector da_variance_wrapper(NumericMatrix X, da_int dof = 0,
                                         std::string axis = "col") {
    da_status status;
    da_int m, n;
    da_axis axis_enum;
    size_t variance_sz;
    get_size_and_axis(axis, axis_enum, X, variance_sz, m, n);

    NumericVector mean(variance_sz);
    NumericVector variance(variance_sz);

    status = da_variance_d(column_major, axis_enum, m, n, X.begin(), m, dof, mean.begin(),
                           variance.begin());
    status_to_exception(status);

    return variance;
}

inline NumericVector da_skewness_wrapper(NumericMatrix X, std::string axis = "col") {
    da_status status;
    da_int m, n;
    da_axis axis_enum;
    size_t skewness_sz;
    get_size_and_axis(axis, axis_enum, X, skewness_sz, m, n);

    NumericVector mean(skewness_sz);
    NumericVector variance(skewness_sz);
    NumericVector skewness(skewness_sz);

    status = da_skewness_d(column_major, axis_enum, m, n, X.begin(), m, mean.begin(),
                           variance.begin(), skewness.begin());
    status_to_exception(status);

    return skewness;
}

inline NumericVector da_kurtosis_wrapper(NumericMatrix x, std::string axis = "col") {
    da_status status;
    da_int m, n;
    da_axis axis_enum;
    size_t kurtosis_sz;
    get_size_and_axis(axis, axis_enum, x, kurtosis_sz, m, n);

    NumericVector mean(kurtosis_sz);
    NumericVector variance(kurtosis_sz);
    NumericVector kurtosis(kurtosis_sz);

    status = da_kurtosis_d(column_major, axis_enum, m, n, x.begin(), m, mean.begin(),
                           variance.begin(), kurtosis.begin());
    status_to_exception(status);

    return kurtosis;
}

inline NumericVector da_moment_wrapper(NumericMatrix X, da_int k,
                                       Nullable<NumericVector> mean,
                                       std::string axis = "col") {
    da_status status;
    da_int m, n;
    da_axis axis_enum;
    size_t moment_sz;
    get_size_and_axis(axis, axis_enum, X, moment_sz, m, n);

    NumericVector moment(moment_sz);

    if (mean.isNotNull()) {
        // Use direct pointer access, no copy
        SEXP mean_sexp = mean.get();
        if (Rf_length(mean_sexp) != moment_sz) {
            stop("Size of mean array does not match data size");
        }
        status = da_moment_d(column_major, axis_enum, m, n, X.begin(), m, k, 1,
                             REAL(mean_sexp), moment.begin());
    } else {
        NumericVector mean_aux(moment_sz);
        status = da_moment_d(column_major, axis_enum, m, n, X.begin(), m, k, 0,
                             mean_aux.begin(), moment.begin());
    }

    status_to_exception(status);

    return moment;
}

inline da_quantile_type quantile_string_to_method(const std::string &method) {
    if (method == "inverted_cdf")
        return da_quantile_type_1;
    else if (method == "averaged_inverted_cdf")
        return da_quantile_type_2;
    else if (method == "closest_observation")
        return da_quantile_type_3;
    else if (method == "interpolated_inverted_cdf")
        return da_quantile_type_4;
    else if (method == "hazen")
        return da_quantile_type_5;
    else if (method == "weibull")
        return da_quantile_type_6;
    else if (method == "linear")
        return da_quantile_type_7;
    else if (method == "median_unbiased")
        return da_quantile_type_8;
    else if (method == "normal_unbiased")
        return da_quantile_type_9;
    else
        stop("Invalid quantile method");
}

inline SEXP da_quantile_wrapper(NumericMatrix X, NumericVector q,
                                std::string method = "linear", std::string axis = "col") {
    da_status status;
    da_int m, n;
    da_axis axis_enum;
    size_t quantile_sz;
    get_size_and_axis(axis, axis_enum, X, quantile_sz, m, n);

    da_int n_q = q.size();
    da_quantile_type method_enum = quantile_string_to_method(method);

    NumericMatrix result(n_q, quantile_sz);
    status = da_quantile_d(column_major, axis_enum, m, n, X.begin(), m, q.begin(), n_q,
                           result.begin(), method_enum);
    status_to_exception(status);

    if (n_q == 1 || quantile_sz == 1)
        return NumericVector(result.begin(), result.end());
    return result;
}

inline List da_five_point_summary_wrapper(NumericMatrix X, std::string axis = "col") {
    da_status status;
    da_int m, n;
    da_axis axis_enum;
    size_t fps_sz;
    get_size_and_axis(axis, axis_enum, X, fps_sz, m, n);

    NumericVector min(fps_sz);
    NumericVector lq(fps_sz);
    NumericVector med(fps_sz);
    NumericVector uq(fps_sz);
    NumericVector max(fps_sz);

    status =
        da_five_point_summary_d(column_major, axis_enum, m, n, X.begin(), m, min.begin(),
                                lq.begin(), med.begin(), uq.begin(), max.begin());

    status_to_exception(status);

    return List::create(Named("min") = min, Named("lq") = lq, Named("med") = med,
                        Named("uq") = uq, Named("max") = max);
}

inline NumericMatrix da_standardize_wrapper(NumericMatrix X,
                                            Nullable<NumericVector> shift,
                                            Nullable<NumericVector> scale, da_int dof = 0,
                                            bool reverse = false, bool inplace = false,
                                            std::string axis = "col") {
    da_status status;
    da_int m, n;
    da_axis axis_enum;
    size_t standardize_sz;
    get_size_and_axis(axis, axis_enum, X, standardize_sz, m, n);

    double *shift_ptr = shift.isNotNull() ? REAL(shift.get()) : nullptr;
    double *scale_ptr = scale.isNotNull() ? REAL(scale.get()) : nullptr;

    if ((shift_ptr != nullptr && (size_t)Rf_length(shift.get()) != standardize_sz) ||
        (scale_ptr != nullptr && (size_t)Rf_length(scale.get()) != standardize_sz)) {
        stop("The size of shift or scale array does not match the expected size.");
    } else if (shift_ptr == nullptr && scale_ptr == nullptr && reverse) {
        stop("Reverse standardization only works if either shift or scale (or both) is "
             "(are) supplied.");
    }

    NumericMatrix output = inplace ? X : clone(X);

    status = da_standardize_d(column_major, axis_enum, m, n, output.begin(), m, dof,
                              (da_int)reverse, shift_ptr, scale_ptr);
    status_to_exception(status);

    return output;
}

inline NumericMatrix da_covariance_wrapper(NumericMatrix X, da_int dof = 0,
                                           da_int assume_centered = 0) {
    da_status status;
    da_int m = X.nrow();
    da_int n = X.ncol();

    NumericMatrix cov(n, n);

    status = da_covariance_matrix_d(column_major, m, n, X.begin(), m, dof, cov.begin(), n,
                                    assume_centered);

    status_to_exception(status);

    return cov;
}

inline NumericMatrix da_correlation_wrapper(NumericMatrix X) {
    da_status status;
    da_int m = X.nrow();
    da_int n = X.ncol();

    NumericMatrix corr(n, n);

    status = da_correlation_matrix_d(column_major, m, n, X.begin(), m, corr.begin(), n);

    status_to_exception(status);

    return corr;
}

#endif // BASIC_STATS_R_HPP
