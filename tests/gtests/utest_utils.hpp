#ifndef UTEST_UTILS_HPP
#define UTEST_UTILS_HPP
#include "aoclda.h"
#include <iostream>

#define EXPECT_ARR_NEAR(n, x, y, abs_error)                                              \
    for (int j = 0; j < (n); j++)                                                        \
    EXPECT_NEAR((x[j]), (y[j]), abs_error)                                               \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

/* FIXME The tests should be able to include directly read_csv.hpp
 * This is a workaround for now
 */
inline da_status da_read_csv(da_csv_opts opts, const char *filename, double **a,
                             size_t *nrows, size_t *ncols) {
    return da_read_csv_d(opts, filename, a, nrows, ncols);
}

inline da_status da_read_csv(da_csv_opts opts, const char *filename, float **a,
                             size_t *nrows, size_t *ncols) {
    return da_read_csv_s(opts, filename, a, nrows, ncols);
}

inline da_status da_read_csv(da_csv_opts opts, const char *filename, int64_t **a,
                             size_t *nrows, size_t *ncols) {
    return da_read_csv_int64(opts, filename, a, nrows, ncols);
}

inline da_status da_read_csv(da_csv_opts opts, const char *filename, uint64_t **a,
                             size_t *nrows, size_t *ncols) {
    return da_read_csv_uint64(opts, filename, a, nrows, ncols);
}

inline da_status da_read_csv_uint8(da_csv_opts opts, const char *filename, uint8_t **a,
                                   size_t *nrows, size_t *ncols) {
    return da_read_csv_uint8(opts, filename, a, nrows, ncols);
}

/* linmod overloaded functions */
template <class T> da_status da_linreg_init(da_linreg *handle);
template <> da_status da_linreg_init<double>(da_linreg *handle) {
    return da_linreg_d_init(handle);
}
template <> da_status da_linreg_init<float>(da_linreg *handle) {
    return da_linreg_s_init(handle);
}

template <class T> da_status da_linreg_select_model(da_linreg handle, linreg_model mod);
template <> da_status da_linreg_select_model<double>(da_linreg handle, linreg_model mod) {
    return da_linreg_d_select_model(handle, mod);
}
template <> da_status da_linreg_select_model<float>(da_linreg handle, linreg_model mod) {
    return da_linreg_s_select_model(handle, mod);
}

inline da_status da_linreg_define_features(da_linreg handle, int n, int m, double *A,
                                           double *b) {
    return da_linreg_d_define_features(handle, n, m, A, b);
}
inline da_status da_linreg_define_features(da_linreg handle, int n, int m, float *A,
                                           float *b) {
    return da_linreg_s_define_features(handle, n, m, A, b);
}

template <class T> da_status da_linreg_fit(da_linreg handle);
template <> da_status da_linreg_fit<double>(da_linreg handle) {
    return da_linreg_d_fit(handle);
}
template <> da_status da_linreg_fit<float>(da_linreg handle) {
    return da_linreg_s_fit(handle);
}

inline da_status da_linreg_get_coef(da_linreg handle, int *nc, double *x) {
    return da_linreg_d_get_coef(handle, nc, x);
}
inline da_status da_linreg_get_coef(da_linreg handle, int *nc, float *x) {
    return da_linreg_s_get_coef(handle, nc, x);
}

inline da_status da_linreg_evaluate_model(da_linreg handle, int n, int m, double *X,
                                          double *predictions) {
    return da_linreg_d_evaluate_model(handle, n, m, X, predictions);
}
inline da_status da_linreg_evaluate_model(da_linreg handle, int n, int m, float *X,
                                          float *predictions) {
    return da_linreg_s_evaluate_model(handle, n, m, X, predictions);
}

#endif