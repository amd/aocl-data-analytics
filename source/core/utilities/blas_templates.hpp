#ifndef BLAS_TEMPLATES_HPP
#define BLAS_TEMPLATES_HPP
#include "cblas.h"

inline void da_cblas_gemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
                          const int M, const int N, const double alpha, const double *A,
                          const int lda, const double *X, const int incX,
                          const double beta, double *Y, const int incY) {
    cblas_dgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

inline void da_cblas_gemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
                          const int M, const int N, const float alpha, const float *A,
                          const int lda, const float *X, const int incX, const float beta,
                          float *Y, const int incY) {

    cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
};

inline void da_cblas_axpy(const int N, const double alpha, const double *X,
                          const int incX, double *Y, const int incY) {
    cblas_daxpy(N, alpha, X, incX, Y, incY);
}

inline void da_cblas_axpy(const int N, const float alpha, const float *X, const int incX,
                          float *Y, const int incY) {
    cblas_saxpy(N, alpha, X, incX, Y, incY);
}

inline double da_cblas_dot(const int N, const double *X, const int incX, const double *Y,
                           const int incY) {
    return cblas_ddot(N, X, incX, Y, incY);
}

inline float da_cblas_dot(const int N, const float *X, const int incX, const float *Y,
                          const int incY) {
    return cblas_sdot(N, X, incX, Y, incY);
}

#endif