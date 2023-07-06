#include "aoclda.h"
#include "moment_statistics.hpp"
#include "order_statistics.hpp"
#include "statistical_utilities.hpp"
#include "correlation_and_covariance.hpp"

da_status da_mean_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                    double *amean) {
    return da_basic_statistics::mean(axis, n, p, x, ldx, amean);
}

da_status da_mean_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx, float *amean) {
    return da_basic_statistics::mean(axis, n, p, x, ldx, amean);
}

da_status da_geometric_mean_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                              double *gmean) {
    return da_basic_statistics::geometric_mean(axis, n, p, x, ldx, gmean);
}

da_status da_geometric_mean_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                              float *gmean) {
    return da_basic_statistics::geometric_mean(axis, n, p, x, ldx, gmean);
}

da_status da_harmonic_mean_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                             double *hmean) {
    return da_basic_statistics::harmonic_mean(axis, n, p, x, ldx, hmean);
}

da_status da_harmonic_mean_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                             float *hmean) {
    return da_basic_statistics::harmonic_mean(axis, n, p, x, ldx, hmean);
}

da_status da_variance_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                        double *mean, double *var) {
    return da_basic_statistics::variance(axis, n, p, x, ldx, mean, var);
}

da_status da_variance_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                        float *mean, float *var) {
    return da_basic_statistics::variance(axis, n, p, x, ldx, mean, var);
}

da_status da_skewness_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                        double *mean, double *var, double *skew) {
    return da_basic_statistics::skewness(axis, n, p, x, ldx, mean, var, skew);
}

da_status da_skewness_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                        float *mean, float *var, float *skew) {
    return da_basic_statistics::skewness(axis, n, p, x, ldx, mean, var, skew);
}

da_status da_kurtosis_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                        double *mean, double *var, double *kurt) {
    return da_basic_statistics::kurtosis(axis, n, p, x, ldx, mean, var, kurt);
}

da_status da_kurtosis_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                        float *mean, float *var, float *kurt) {
    return da_basic_statistics::kurtosis(axis, n, p, x, ldx, mean, var, kurt);
}

da_status da_moment_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx, da_int k,
                      da_int use_precomputed_mean, double *mean, double *mom) {
    return da_basic_statistics::moment(axis, n, p, x, ldx, k, use_precomputed_mean,
                                          mean, mom);
}

da_status da_moment_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx, da_int k,
                      da_int use_precomputed_mean, float *mean, float *mom) {
    return da_basic_statistics::moment(axis, n, p, x, ldx, k, use_precomputed_mean,
                                          mean, mom);
}

da_status da_quantile_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx, double q,
                        double *quant, da_quantile_type quantile_type) {
    return da_basic_statistics::quantile(axis, n, p, x, ldx, q, quant,
                                            quantile_type);
}

da_status da_quantile_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx, float q,
                        float *quant, da_quantile_type quantile_type) {
    return da_basic_statistics::quantile(axis, n, p, x, ldx, q, quant,
                                            quantile_type);
}

da_status da_five_point_summary_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                                  double *minimum, double *lower_hinge, double *median,
                                  double *upper_hinge, double *maximum) {
    return da_basic_statistics::five_point_summary(
        axis, n, p, x, ldx, minimum, lower_hinge, median, upper_hinge, maximum);
}

da_status da_five_point_summary_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                                  float *minimum, float *lower_hinge, float *median,
                                  float *upper_hinge, float *maximum) {
    return da_basic_statistics::five_point_summary(
        axis, n, p, x, ldx, minimum, lower_hinge, median, upper_hinge, maximum);
}

/* Shift by a constant amount and scale */
da_status da_standardize_d(da_axis axis, da_int n, da_int p, double *x, da_int ldx, double *shift, double *scale){
    return da_basic_statistics::standardize(axis, n, p, x, ldx, shift, scale);
}

da_status da_standardize_s(da_axis axis, da_int n, da_int p, float *x, da_int ldx, float *shift, float *scale){
    return da_basic_statistics::standardize(axis, n, p, x, ldx, shift, scale);
}

da_status da_covariance_matrix_d(da_int n, da_int p, double *x, da_int ldx, double *cov, da_int ldcov){
    return da_basic_statistics::covariance_matrix(n, p, x, ldx, cov, ldcov);
}

da_status da_covariance_matrix_s(da_int n, da_int p, float *x, da_int ldx, float *cov, da_int ldcov){
    return da_basic_statistics::covariance_matrix(n, p, x, ldx, cov, ldcov);
}

da_status da_correlation_matrix_d(da_int n, da_int p, double *x, da_int ldx, double *corr, da_int ldcorr){
        return da_basic_statistics::correlation_matrix(n, p, x, ldx, corr, ldcorr);
}

da_status da_correlation_matrix_s(da_int n, da_int p, float *x, da_int ldx, float *corr, da_int ldcorr){
        return da_basic_statistics::correlation_matrix(n, p, x, ldx, corr, ldcorr);
}

da_status da_colmean_s(da_int n, da_int p, float *x, da_int incx, float *mean) {
    return da_colmean(n, p, x, incx, mean);
}

da_status da_colmean_d(da_int n, da_int p, double *x, da_int incx, double *mean) {
    return da_colmean(n, p, x, incx, mean);
}
