#ifndef UTEST_UTILS_HPP
#define UTEST_UTILS_HPP
#include "aoclda.h"
#include <iostream>

#define EXPECT_ARR_NEAR(n, x, y, abs_error)                                              \
    for (da_int j = 0; j < (n); j++)                                                     \
    EXPECT_NEAR((x[j]), (y[j]), abs_error)                                               \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

/* FIXME The tests should be able to include directly read_csv.hpp
 * This is a workaround for now
 */
inline da_status da_read_csv(da_handle handle, const char *filename, double **a,
                             da_int *nrows, da_int *ncols) {
    return da_read_csv_d(handle, filename, a, nrows, ncols);
}

inline da_status da_read_csv(da_handle handle, const char *filename, float **a,
                             da_int *nrows, da_int *ncols) {
    return da_read_csv_s(handle, filename, a, nrows, ncols);
}

inline da_status da_read_csv(da_handle handle, const char *filename, int64_t **a,
                             da_int *nrows, da_int *ncols) {
    return da_read_csv_int64(handle, filename, a, nrows, ncols);
}

inline da_status da_read_csv(da_handle handle, const char *filename, uint64_t **a,
                             da_int *nrows, da_int *ncols) {
    return da_read_csv_uint64(handle, filename, a, nrows, ncols);
}

inline da_status da_read_csv(da_handle handle, const char *filename, uint8_t **a,
                                   da_int *nrows, da_int *ncols) {
    return da_read_csv_uint8(handle, filename, a, nrows, ncols);
}

inline da_status da_read_csv(da_handle handle, const char *filename, double **a,
                             da_int *nrows, da_int *ncols, char ***headings) {
    return da_read_csv_d_h(handle, filename, a, nrows, ncols, headings);
}

inline da_status da_read_csv(da_handle handle, const char *filename, float **a,
                             da_int *nrows, da_int *ncols, char ***headings) {
    return da_read_csv_s_h(handle, filename, a, nrows, ncols, headings);
}

inline da_status da_read_csv(da_handle handle, const char *filename, int64_t **a,
                             da_int *nrows, da_int *ncols, char ***headings) {
    return da_read_csv_int64_h(handle, filename, a, nrows, ncols, headings);
}

inline da_status da_read_csv(da_handle handle, const char *filename, uint64_t **a,
                             da_int *nrows, da_int *ncols, char ***headings) {
    return da_read_csv_uint64_h(handle, filename, a, nrows, ncols, headings);
}

inline da_status da_read_csv(da_handle handle, const char *filename, uint8_t **a,
                                   da_int *nrows, da_int *ncols, char ***headings) {
    return da_read_csv_uint8_h(handle, filename, a, nrows, ncols, headings);
}

/* linmod overloaded functions */
template <class T> da_status da_linreg_init(da_handle *handle);
template <> da_status da_linreg_init<double>(da_handle *handle) {
    return da_handle_init_d(handle, da_handle_linreg);
}
template <> da_status da_linreg_init<float>(da_handle *handle) {
    return da_handle_init_s(handle, da_handle_linreg);
}

template <class T> da_status da_linreg_select_model(da_handle handle, linreg_model mod);
template <> da_status da_linreg_select_model<double>(da_handle handle, linreg_model mod) {
    return da_linreg_d_select_model(handle, mod);
}
template <> da_status da_linreg_select_model<float>(da_handle handle, linreg_model mod) {
    return da_linreg_s_select_model(handle, mod);
}

inline da_status da_linreg_define_features(da_handle handle, da_int n, da_int m,
                                           double *A, double *b) {
    return da_linreg_d_define_features(handle, n, m, A, b);
}
inline da_status da_linreg_define_features(da_handle handle, da_int n, da_int m, float *A,
                                           float *b) {
    return da_linreg_s_define_features(handle, n, m, A, b);
}

template <class T> da_status da_linreg_fit(da_handle handle);
template <> da_status da_linreg_fit<double>(da_handle handle) {
    return da_linreg_d_fit(handle);
}
template <> da_status da_linreg_fit<float>(da_handle handle) {
    return da_linreg_s_fit(handle);
}

inline da_status da_linreg_get_coef(da_handle handle, da_int *nc, double *x) {
    return da_linreg_d_get_coef(handle, nc, x);
}
inline da_status da_linreg_get_coef(da_handle handle, da_int *nc, float *x) {
    return da_linreg_s_get_coef(handle, nc, x);
}

inline da_status da_linreg_evaluate_model(da_handle handle, da_int n, da_int m, double *X,
                                          double *predictions) {
    return da_linreg_d_evaluate_model(handle, n, m, X, predictions);
}
inline da_status da_linreg_evaluate_model(da_handle handle, da_int n, da_int m, float *X,
                                          float *predictions) {
    return da_linreg_s_evaluate_model(handle, n, m, X, predictions);
}

#endif