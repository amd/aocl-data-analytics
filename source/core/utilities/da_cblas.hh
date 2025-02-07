/******************************************************************************
* Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

/*  cblas.hh
 *  cblas.hh defines all the overloaded CPP functions to be invoked from
 *  template da_interfaces
 *  */
#ifndef DA_CBLAS_HH
#define DA_CBLAS_HH

extern "C" {
#define BLIS_ENABLE_CBLAS
#include "cblas.h"
/*
 * Adding those declarations because they do not exist in cblas.h
 */
void simatcopy_(char *trans, da_int *m, da_int *n, const float *alpha, float *A,
                da_int *lda_in, da_int *lda_out);
void dimatcopy_(char *trans, da_int *m, da_int *n, const double *alpha, double *A,
                da_int *lda_in, da_int *lda_out);
void somatcopy_(char *trans, da_int *m, da_int *n, const float *alpha, const float *A,
                da_int *lda_in, float *B, da_int *ldb_out);
void domatcopy_(char *trans, da_int *m, da_int *n, const double *alpha, const double *A,
                da_int *lda_in, double *B, da_int *ldb_out);
}

#include <complex>

namespace da_blas {

template <typename... Types> struct real_type_traits;

//define real_type<> type alias
template <typename... Types>
using real_type = typename real_type_traits<Types...>::real_t;

// For one type
template <typename T> struct real_type_traits<T> {
    using real_t = T;
};

// For one complex type, strip complex
template <typename T> struct real_type_traits<std::complex<T>> {
    using real_t = T;
};

// =============================================================================
// Level 1 BLAS
// -----------------------------------------------------------------------------
inline void cblas_rotg(float *a, float *b, float *c, float *s) {
    cblas_srotg(a, b, c, s);
}

inline void cblas_rotg(double *a, double *b, double *c, double *s) {
    cblas_drotg(a, b, c, s);
}

// -----------------------------------------------------------------------------
inline void cblas_rotmg(float *d1, float *d2, float *x1, float y1, float param[5]) {
    cblas_srotmg(d1, d2, x1, y1, param);
}

inline void cblas_rotmg(double *d1, double *d2, double *x1, double y1, double param[5]) {
    cblas_drotmg(d1, d2, x1, y1, param);
}

// -----------------------------------------------------------------------------
inline void cblas_rot(da_int n, float *x, da_int incx, float *y, da_int incy, float c,
                      float s) {
    cblas_srot(n, x, incx, y, incy, c, s);
}

inline void cblas_rot(da_int n, double *x, da_int incx, double *y, da_int incy, double c,
                      double s) {
    cblas_drot(n, x, incx, y, incy, c, s);
}

// -----------------------------------------------------------------------------
inline void cblas_rotm(da_int n, float *x, da_int incx, float *y, da_int incy,
                       const float p[5]) {
    cblas_srotm(n, x, incx, y, incy, p);
}

inline void cblas_rotm(da_int n, double *x, da_int incx, double *y, da_int incy,
                       const double p[5]) {
    cblas_drotm(n, x, incx, y, incy, p);
}

// -----------------------------------------------------------------------------
inline void cblas_swap(da_int n, float *x, da_int incx, float *y, da_int incy) {
    cblas_sswap(n, x, incx, y, incy);
}

inline void cblas_swap(da_int n, double *x, da_int incx, double *y, da_int incy) {
    cblas_dswap(n, x, incx, y, incy);
}

inline void cblas_swap(da_int n, std::complex<float> *x, da_int incx,
                       std::complex<float> *y, da_int incy) {
    cblas_cswap(n, x, incx, y, incy);
}

inline void cblas_swap(da_int n, std::complex<double> *x, da_int incx,
                       std::complex<double> *y, da_int incy) {
    cblas_zswap(n, x, incx, y, incy);
}

// -----------------------------------------------------------------------------
inline void cblas_scal(da_int n, float alpha, float *x, da_int incx) {
    cblas_sscal(n, alpha, x, incx);
}

inline void cblas_scal(da_int n, double alpha, double *x, da_int incx) {
    cblas_dscal(n, alpha, x, incx);
}

inline void cblas_scal(da_int n, std::complex<float> alpha, std::complex<float> *x,
                       da_int incx) {
    cblas_cscal(n, &alpha, x, incx);
}

inline void cblas_scal(da_int n, std::complex<double> alpha, std::complex<double> *x,
                       da_int incx) {
    cblas_zscal(n, &alpha, x, incx);
}

inline void cblas_scal(da_int n, float alpha, std::complex<float> *x, da_int incx) {
    cblas_csscal(n, alpha, x, incx);
}

inline void cblas_scal(da_int n, double alpha, std::complex<double> *x, da_int incx) {
    cblas_zdscal(n, alpha, x, incx);
}

// -----------------------------------------------------------------------------
inline void cblas_copy(da_int n, float const *x, da_int incx, float *y, da_int incy) {
    cblas_scopy(n, x, incx, y, incy);
}

inline void cblas_copy(da_int n, double const *x, da_int incx, double *y, da_int incy) {
    cblas_dcopy(n, x, incx, y, incy);
}

inline void cblas_copy(da_int n, std::complex<float> const *x, da_int incx,
                       std::complex<float> *y, da_int incy) {
    cblas_ccopy(n, x, incx, y, incy);
}

inline void cblas_copy(da_int n, std::complex<double> const *x, da_int incx,
                       std::complex<double> *y, da_int incy) {
    cblas_zcopy(n, x, incx, y, incy);
}

// -----------------------------------------------------------------------------
inline void cblas_axpy(da_int n, float alpha, float const *x, da_int incx, float *y,
                       da_int incy) {
    cblas_saxpy(n, alpha, x, incx, y, incy);
}

inline void cblas_axpy(da_int n, double alpha, double const *x, da_int incx, double *y,
                       da_int incy) {
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

inline void cblas_axpy(da_int n, std::complex<float> alpha, std::complex<float> const *x,
                       da_int incx, std::complex<float> *y, da_int incy) {
    cblas_caxpy(n, &alpha, x, incx, y, incy);
}

inline void cblas_axpy(da_int n, std::complex<double> alpha,
                       std::complex<double> const *x, da_int incx,
                       std::complex<double> *y, da_int incy) {
    cblas_zaxpy(n, &alpha, x, incx, y, incy);
}

// -----------------------------------------------------------------------------
inline float cblas_dot(da_int n, float const *x, da_int incx, float const *y,
                       da_int incy) {
    return cblas_sdot(n, x, incx, y, incy);
}

inline double cblas_dot(da_int n, double const *x, da_int incx, double const *y,
                        da_int incy) {
    return cblas_ddot(n, x, incx, y, incy);
}
// -----------------------------------------------------------------------------
inline std::complex<float> cblas_dotu(da_int n, std::complex<float> const *x, da_int incx,
                                      std::complex<float> const *y, da_int incy) {
    std::complex<float> result;
    cblas_cdotu_sub(n, x, incx, y, incy, &result);
    return result;
}

inline std::complex<double> cblas_dotu(da_int n, std::complex<double> const *x,
                                       da_int incx, std::complex<double> const *y,
                                       da_int incy) {
    std::complex<double> result;
    cblas_zdotu_sub(n, x, incx, y, incy, &result);
    return result;
}

// -----------------------------------------------------------------------------
inline std::complex<float> cblas_dotc(da_int n, std::complex<float> const *x, da_int incx,
                                      std::complex<float> const *y, da_int incy) {
    std::complex<float> result;
    cblas_cdotc_sub(n, x, incx, y, incy, &result);
    return result;
}

inline std::complex<double> cblas_dotc(da_int n, std::complex<double> const *x,
                                       da_int incx, std::complex<double> const *y,
                                       da_int incy) {
    std::complex<double> result;
    cblas_zdotc_sub(n, x, incx, y, incy, &result);
    return result;
}

// -----------------------------------------------------------------------------
inline da_int cblas_iamax(da_int n, float const *x, da_int incx) {
    return cblas_isamax(n, x, incx);
}

inline da_int cblas_iamax(da_int n, double const *x, da_int incx) {
    return cblas_idamax(n, x, incx);
}

inline da_int cblas_iamax(da_int n, std::complex<float> const *x, da_int incx) {
    return cblas_icamax(n, x, incx);
}

inline da_int cblas_iamax(da_int n, std::complex<double> const *x, da_int incx) {
    return cblas_izamax(n, x, incx);
}

// -----------------------------------------------------------------------------
inline float cblas_nrm2(da_int n, float const *x, da_int incx) {
    return cblas_snrm2(n, x, incx);
}

inline double cblas_nrm2(da_int n, double const *x, da_int incx) {
    return cblas_dnrm2(n, x, incx);
}

inline float cblas_nrm2(da_int n, std::complex<float> const *x, da_int incx) {
    return cblas_scnrm2(n, x, incx);
}

inline double cblas_nrm2(da_int n, std::complex<double> const *x, da_int incx) {
    return cblas_dznrm2(n, x, incx);
}

// -----------------------------------------------------------------------------
inline float cblas_asum(da_int n, float const *x, da_int incx) {
    return cblas_sasum(n, x, incx);
}

inline double cblas_asum(da_int n, double const *x, da_int incx) {
    return cblas_dasum(n, x, incx);
}

inline float cblas_asum(da_int n, std::complex<float> const *x, da_int incx) {
    return cblas_scasum(n, x, incx);
}

inline double cblas_asum(da_int n, std::complex<double> const *x, da_int incx) {
    return cblas_dzasum(n, x, incx);
}
// =============================================================================
// Level 2 BLAS

// -----------------------------------------------------------------------------
inline void cblas_gemv(CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, da_int m, da_int n,
                       float alpha, float const *A, da_int lda, float const *x,
                       da_int incx, float beta, float *y, da_int incy) {
    cblas_sgemv(layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void cblas_gemv(CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, da_int m, da_int n,
                       double alpha, double const *A, da_int lda, double const *x,
                       da_int incx, double beta, double *y, da_int incy) {
    cblas_dgemv(layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void cblas_gemv(CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, da_int m, da_int n,
                       std::complex<float> alpha, std::complex<float> const *A,
                       da_int lda, std::complex<float> const *x, da_int incx,
                       std::complex<float> beta, std::complex<float> *y, da_int incy) {
    cblas_cgemv(layout, trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

inline void cblas_gemv(CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, da_int m, da_int n,
                       std::complex<double> alpha, std::complex<double> const *A,
                       da_int lda, std::complex<double> const *x, da_int incx,
                       std::complex<double> beta, std::complex<double> *y, da_int incy) {
    cblas_zgemv(layout, trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}
inline void cblas_gbmv(CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, da_int m, da_int n,
                       da_int kl, da_int ku, float alpha, float const *A, da_int lda,
                       float const *x, da_int incx, float beta, float *y, da_int incy) {
    cblas_sgbmv(layout, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

inline void cblas_gbmv(CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, da_int m, da_int n,
                       da_int kl, da_int ku, double alpha, double const *A, da_int lda,
                       double const *x, da_int incx, double beta, double *y,
                       da_int incy) {
    cblas_dgbmv(layout, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

inline void cblas_gbmv(CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, da_int m, da_int n,
                       da_int kl, da_int ku, std::complex<float> alpha,
                       std::complex<float> const *A, da_int lda,
                       std::complex<float> const *x, da_int incx,
                       std::complex<float> beta, std::complex<float> *y, da_int incy) {
    cblas_cgbmv(layout, trans, m, n, kl, ku, &alpha, A, lda, x, incx, &beta, y, incy);
}

inline void cblas_gbmv(CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, da_int m, da_int n,
                       da_int kl, da_int ku, std::complex<double> alpha,
                       std::complex<double> const *A, da_int lda,
                       std::complex<double> const *x, da_int incx,
                       std::complex<double> beta, std::complex<double> *y, da_int incy) {
    cblas_zgbmv(layout, trans, m, n, kl, ku, &alpha, A, lda, x, incx, &beta, y, incy);
}

// -----------------------------------------------------------------------------
inline void cblas_hemv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n,
                       std::complex<float> alpha, std::complex<float> const *A,
                       da_int lda, std::complex<float> const *x, da_int incx,
                       std::complex<float> beta, std::complex<float> *y, da_int incy) {
    cblas_chemv(layout, uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

inline void cblas_hemv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n,
                       std::complex<double> alpha, std::complex<double> const *A,
                       da_int lda, std::complex<double> const *x, da_int incx,
                       std::complex<double> beta, std::complex<double> *y, da_int incy) {
    cblas_zhemv(layout, uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

// -----------------------------------------------------------------------------
inline void cblas_hbmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, da_int k,
                       std::complex<float> alpha, std::complex<float> const *A,
                       da_int lda, std::complex<float> const *x, da_int incx,
                       std::complex<float> beta, std::complex<float> *y, da_int incy) {
    cblas_chbmv(layout, uplo, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
}

inline void cblas_hbmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, da_int k,
                       std::complex<double> alpha, std::complex<double> const *A,
                       da_int lda, std::complex<double> const *x, da_int incx,
                       std::complex<double> beta, std::complex<double> *y, da_int incy) {
    cblas_zhbmv(layout, uplo, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
}

// -----------------------------------------------------------------------------
inline void cblas_hpmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n,
                       std::complex<float> alpha, std::complex<float> const *Ap,
                       std::complex<float> const *x, da_int incx,
                       std::complex<float> beta, std::complex<float> *y, da_int incy) {
    cblas_chpmv(layout, uplo, n, &alpha, Ap, x, incx, &beta, y, incy);
}

inline void cblas_hpmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n,
                       std::complex<double> alpha, std::complex<double> const *Ap,
                       std::complex<double> const *x, da_int incx,
                       std::complex<double> beta, std::complex<double> *y, da_int incy) {
    cblas_zhpmv(layout, uplo, n, &alpha, Ap, x, incx, &beta, y, incy);
}

// -----------------------------------------------------------------------------
inline void cblas_symv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, float alpha,
                       float const *A, da_int lda, float const *x, da_int incx,
                       float beta, float *y, da_int incy) {
    cblas_ssymv(layout, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void cblas_symv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, double alpha,
                       double const *A, da_int lda, double const *x, da_int incx,
                       double beta, double *y, da_int incy) {
    cblas_dsymv(layout, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

// -----------------------------------------------------------------------------
inline void cblas_sbmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, da_int k,
                       float alpha, float const *A, da_int lda, float const *x,
                       da_int incx, float beta, float *y, da_int incy) {
    cblas_ssbmv(layout, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

inline void cblas_sbmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, da_int k,
                       double alpha, double const *A, da_int lda, double const *x,
                       da_int incx, double beta, double *y, da_int incy) {
    cblas_dsbmv(layout, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

// -----------------------------------------------------------------------------
inline void cblas_spmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, float alpha,
                       float const *Ap, float const *x, da_int incx, float beta, float *y,
                       da_int incy) {
    cblas_sspmv(layout, uplo, n, alpha, Ap, x, incx, beta, y, incy);
}

inline void cblas_spmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, double alpha,
                       double const *Ap, double const *x, da_int incx, double beta,
                       double *y, da_int incy) {
    cblas_dspmv(layout, uplo, n, alpha, Ap, x, incx, beta, y, incy);
}

// -----------------------------------------------------------------------------
inline void cblas_trmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, float const *A, da_int lda, float *x,
                       da_int incx) {
    cblas_strmv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

inline void cblas_trmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, double const *A, da_int lda, double *x,
                       da_int incx) {
    cblas_dtrmv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

inline void cblas_trmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, std::complex<float> const *A,
                       da_int lda, std::complex<float> *x, da_int incx) {
    cblas_ctrmv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

inline void cblas_trmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, std::complex<double> const *A,
                       da_int lda, std::complex<double> *x, da_int incx) {
    cblas_ztrmv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

// -----------------------------------------------------------------------------
inline void cblas_tbmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, da_int k, float const *A, da_int lda,
                       float *x, da_int incx) {
    cblas_stbmv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}

inline void cblas_tbmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, da_int k, double const *A, da_int lda,
                       double *x, da_int incx) {
    cblas_dtbmv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}

inline void cblas_tbmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, da_int k, std::complex<float> const *A,
                       da_int lda, std::complex<float> *x, da_int incx) {
    cblas_ctbmv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}

inline void cblas_tbmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, da_int k, std::complex<double> const *A,
                       da_int lda, std::complex<double> *x, da_int incx) {
    cblas_ztbmv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}

// -----------------------------------------------------------------------------
inline void cblas_tpmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, float const *Ap, float *x,
                       da_int incx) {
    cblas_stpmv(layout, uplo, trans, diag, n, Ap, x, incx);
}

inline void cblas_tpmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, double const *Ap, double *x,
                       da_int incx) {
    cblas_dtpmv(layout, uplo, trans, diag, n, Ap, x, incx);
}

inline void cblas_tpmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, std::complex<float> const *Ap,
                       std::complex<float> *x, da_int incx) {
    cblas_ctpmv(layout, uplo, trans, diag, n, Ap, x, incx);
}

inline void cblas_tpmv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, std::complex<double> const *Ap,
                       std::complex<double> *x, da_int incx) {
    cblas_ztpmv(layout, uplo, trans, diag, n, Ap, x, incx);
}

// -----------------------------------------------------------------------------
inline void cblas_trsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, float const *A, da_int lda, float *x,
                       da_int incx) {
    cblas_strsv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

inline void cblas_trsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, double const *A, da_int lda, double *x,
                       da_int incx) {
    cblas_dtrsv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

inline void cblas_trsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, std::complex<float> const *A,
                       da_int lda, std::complex<float> *x, da_int incx) {
    cblas_ctrsv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

inline void cblas_trsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, std::complex<double> const *A,
                       da_int lda, std::complex<double> *x, da_int incx) {
    cblas_ztrsv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

// -----------------------------------------------------------------------------
inline void cblas_tbsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, da_int k, float const *A, da_int lda,
                       float *x, da_int incx) {
    cblas_stbsv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}

inline void cblas_tbsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, da_int k, double const *A, da_int lda,
                       double *x, da_int incx) {
    cblas_dtbsv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}

inline void cblas_tbsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, da_int k, std::complex<float> const *A,
                       da_int lda, std::complex<float> *x, da_int incx) {
    cblas_ctbsv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}

inline void cblas_tbsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, da_int k, std::complex<double> const *A,
                       da_int lda, std::complex<double> *x, da_int incx) {
    cblas_ztbsv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}

// -----------------------------------------------------------------------------
inline void cblas_tpsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, float const *Ap, float *x,
                       da_int incx) {
    cblas_stpsv(layout, uplo, trans, diag, n, Ap, x, incx);
}

inline void cblas_tpsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, double const *Ap, double *x,
                       da_int incx) {
    cblas_dtpsv(layout, uplo, trans, diag, n, Ap, x, incx);
}

inline void cblas_tpsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, std::complex<float> const *Ap,
                       std::complex<float> *x, da_int incx) {
    cblas_ctpsv(layout, uplo, trans, diag, n, Ap, x, incx);
}

inline void cblas_tpsv(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       CBLAS_DIAG diag, da_int n, std::complex<double> const *Ap,
                       std::complex<double> *x, da_int incx) {
    cblas_ztpsv(layout, uplo, trans, diag, n, Ap, x, incx);
}

// -----------------------------------------------------------------------------
inline void cblas_ger(CBLAS_ORDER layout, da_int m, da_int n, float alpha, float const *x,
                      da_int incx, float const *y, da_int incy, float *A, da_int lda) {
    cblas_sger(layout, m, n, alpha, x, incx, y, incy, A, lda);
}

inline void cblas_ger(CBLAS_ORDER layout, da_int m, da_int n, double alpha,
                      double const *x, da_int incx, double const *y, da_int incy,
                      double *A, da_int lda) {
    cblas_dger(layout, m, n, alpha, x, incx, y, incy, A, lda);
}

// -----------------------------------------------------------------------------
inline void cblas_geru(CBLAS_ORDER layout, da_int m, da_int n, std::complex<float> alpha,
                       std::complex<float> const *x, da_int incx,
                       std::complex<float> const *y, da_int incy, std::complex<float> *A,
                       da_int lda) {
    cblas_cgeru(layout, m, n, &alpha, x, incx, y, incy, A, lda);
}

inline void cblas_geru(CBLAS_ORDER layout, da_int m, da_int n, std::complex<double> alpha,
                       std::complex<double> const *x, da_int incx,
                       std::complex<double> const *y, da_int incy,
                       std::complex<double> *A, da_int lda) {
    cblas_zgeru(layout, m, n, &alpha, x, incx, y, incy, A, lda);
}

// -----------------------------------------------------------------------------
inline void cblas_gerc(CBLAS_ORDER layout, da_int m, da_int n, std::complex<float> alpha,
                       std::complex<float> const *x, da_int incx,
                       std::complex<float> const *y, da_int incy, std::complex<float> *A,
                       da_int lda) {
    cblas_cgerc(layout, m, n, &alpha, x, incx, y, incy, A, lda);
}

inline void cblas_gerc(CBLAS_ORDER layout, da_int m, da_int n, std::complex<double> alpha,
                       std::complex<double> const *x, da_int incx,
                       std::complex<double> const *y, da_int incy,
                       std::complex<double> *A, da_int lda) {
    cblas_zgerc(layout, m, n, &alpha, x, incx, y, incy, A, lda);
}

// -----------------------------------------------------------------------------
inline void cblas_her(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, float alpha,
                      std::complex<float> const *x, da_int incx, std::complex<float> *A,
                      da_int lda) {
    cblas_cher(layout, uplo, n, alpha, x, incx, A, lda);
}

inline void cblas_her(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, double alpha,
                      std::complex<double> const *x, da_int incx, std::complex<double> *A,
                      da_int lda) {
    cblas_zher(layout, uplo, n, alpha, x, incx, A, lda);
}

// -----------------------------------------------------------------------------
inline void cblas_hpr(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, float alpha,
                      std::complex<float> const *x, da_int incx,
                      std::complex<float> *Ap) {
    cblas_chpr(layout, uplo, n, alpha, x, incx, Ap);
}

inline void cblas_hpr(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, double alpha,
                      std::complex<double> const *x, da_int incx,
                      std::complex<double> *Ap) {
    cblas_zhpr(layout, uplo, n, alpha, x, incx, Ap);
}

// -----------------------------------------------------------------------------
inline void cblas_her2(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n,
                       std::complex<float> alpha, std::complex<float> const *x,
                       da_int incx, std::complex<float> const *y, da_int incy,
                       std::complex<float> *A, da_int lda) {
    cblas_cher2(layout, uplo, n, &alpha, x, incx, y, incy, A, lda);
}

inline void cblas_her2(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n,
                       std::complex<double> alpha, std::complex<double> const *x,
                       da_int incx, std::complex<double> const *y, da_int incy,
                       std::complex<double> *A, da_int lda) {
    cblas_zher2(layout, uplo, n, &alpha, x, incx, y, incy, A, lda);
}

// -----------------------------------------------------------------------------
inline void cblas_hpr2(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n,
                       std::complex<float> alpha, std::complex<float> const *x,
                       da_int incx, std::complex<float> const *y, da_int incy,
                       std::complex<float> *Ap) {
    cblas_chpr2(layout, uplo, n, &alpha, x, incx, y, incy, Ap);
}

inline void cblas_hpr2(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n,
                       std::complex<double> alpha, std::complex<double> const *x,
                       da_int incx, std::complex<double> const *y, da_int incy,
                       std::complex<double> *Ap) {
    cblas_zhpr2(layout, uplo, n, &alpha, x, incx, y, incy, Ap);
}
// -----------------------------------------------------------------------------
inline void cblas_syr(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, float alpha,
                      float const *x, da_int incx, float *A, da_int lda) {
    cblas_ssyr(layout, uplo, n, alpha, x, incx, A, lda);
}

inline void cblas_syr(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, double alpha,
                      double const *x, da_int incx, double *A, da_int lda) {
    cblas_dsyr(layout, uplo, n, alpha, x, incx, A, lda);
}

// -----------------------------------------------------------------------------
inline void cblas_spr(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, float alpha,
                      float const *x, da_int incx, float *Ap) {
    cblas_sspr(layout, uplo, n, alpha, x, incx, Ap);
}

inline void cblas_spr(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, double alpha,
                      double const *x, da_int incx, double *Ap) {
    cblas_dspr(layout, uplo, n, alpha, x, incx, Ap);
}

// -----------------------------------------------------------------------------
inline void cblas_syr2(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, float alpha,
                       float const *x, da_int incx, float const *y, da_int incy, float *A,
                       da_int lda) {
    cblas_ssyr2(layout, uplo, n, alpha, x, incx, y, incy, A, lda);
}

inline void cblas_syr2(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, double alpha,
                       double const *x, da_int incx, double const *y, da_int incy,
                       double *A, da_int lda) {
    cblas_dsyr2(layout, uplo, n, alpha, x, incx, y, incy, A, lda);
}

// -----------------------------------------------------------------------------
inline void cblas_spr2(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, float alpha,
                       float const *x, da_int incx, float const *y, da_int incy,
                       float *Ap) {
    cblas_sspr2(layout, uplo, n, alpha, x, incx, y, incy, Ap);
}

inline void cblas_spr2(CBLAS_ORDER layout, CBLAS_UPLO uplo, da_int n, double alpha,
                       double const *x, da_int incx, double const *y, da_int incy,
                       double *Ap) {
    cblas_dspr2(layout, uplo, n, alpha, x, incx, y, incy, Ap);
}

// =============================================================================
// Level 3 BLAS

// -----------------------------------------------------------------------------
inline void cblas_gemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                       da_int m, da_int n, da_int k, float alpha, float const *A,
                       da_int lda, float const *B, da_int ldb, float beta, float *C,
                       da_int ldc) {
    cblas_sgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_gemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                       da_int m, da_int n, da_int k, double alpha, double const *A,
                       da_int lda, double const *B, da_int ldb, double beta, double *C,
                       da_int ldc) {
    cblas_dgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_gemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                       da_int m, da_int n, da_int k, std::complex<float> alpha,
                       std::complex<float> const *A, da_int lda,
                       std::complex<float> const *B, da_int ldb, std::complex<float> beta,
                       std::complex<float> *C, da_int ldc) {
    cblas_cgemm(layout, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void cblas_gemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                       da_int m, da_int n, da_int k, std::complex<double> alpha,
                       std::complex<double> const *A, da_int lda,
                       std::complex<double> const *B, da_int ldb,
                       std::complex<double> beta, std::complex<double> *C, da_int ldc) {
    cblas_zgemm(layout, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// -----------------------------------------------------------------------------
inline void cblas_trmm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                       CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, da_int m, da_int n,
                       float alpha, float const *A, da_int lda, float *B, da_int ldb) {
    cblas_strmm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

inline void cblas_trmm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                       CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, da_int m, da_int n,
                       double alpha, double const *A, da_int lda, double *B, da_int ldb) {
    cblas_dtrmm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

inline void cblas_trmm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                       CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, da_int m, da_int n,
                       std::complex<float> alpha, std::complex<float> const *A,
                       da_int lda, std::complex<float> *B, da_int ldb) {
    cblas_ctrmm(layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
}

inline void cblas_trmm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                       CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, da_int m, da_int n,
                       std::complex<double> alpha, std::complex<double> const *A,
                       da_int lda, std::complex<double> *B, da_int ldb) {
    cblas_ztrmm(layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
}

// -----------------------------------------------------------------------------
inline void cblas_trsm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                       CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, da_int m, da_int n,
                       float alpha, float const *A, da_int lda, float *B, da_int ldb) {
    cblas_strsm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

inline void cblas_trsm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                       CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, da_int m, da_int n,
                       double alpha, double const *A, da_int lda, double *B, da_int ldb) {
    cblas_dtrsm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

inline void cblas_trsm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                       CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, da_int m, da_int n,
                       std::complex<float> alpha, std::complex<float> const *A,
                       da_int lda, std::complex<float> *B, da_int ldb) {
    cblas_ctrsm(layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
}

inline void cblas_trsm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                       CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, da_int m, da_int n,
                       std::complex<double> alpha, std::complex<double> const *A,
                       da_int lda, std::complex<double> *B, da_int ldb) {
    cblas_ztrsm(layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
}

// -----------------------------------------------------------------------------
inline void cblas_hemm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo, da_int m,
                       da_int n, float alpha, float const *A, da_int lda, float const *B,
                       da_int ldb, float beta, float *C, da_int ldc) {
    cblas_ssymm(layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_hemm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo, da_int m,
                       da_int n, double alpha, double const *A, da_int lda,
                       double const *B, da_int ldb, double beta, double *C, da_int ldc) {
    cblas_dsymm(layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_hemm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo, da_int m,
                       da_int n, std::complex<float> alpha, std::complex<float> const *A,
                       da_int lda, std::complex<float> const *B, da_int ldb,
                       std::complex<float> beta, std::complex<float> *C, da_int ldc) {
    cblas_chemm(layout, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void cblas_hemm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo, da_int m,
                       da_int n, std::complex<double> alpha,
                       std::complex<double> const *A, da_int lda,
                       std::complex<double> const *B, da_int ldb,
                       std::complex<double> beta, std::complex<double> *C, da_int ldc) {
    cblas_zhemm(layout, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// -----------------------------------------------------------------------------
inline void cblas_symm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo, da_int m,
                       da_int n, float alpha, float const *A, da_int lda, float const *B,
                       da_int ldb, float beta, float *C, da_int ldc) {
    cblas_ssymm(layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_symm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo, da_int m,
                       da_int n, double alpha, double const *A, da_int lda,
                       double const *B, da_int ldb, double beta, double *C, da_int ldc) {
    cblas_dsymm(layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_symm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo, da_int m,
                       da_int n, std::complex<float> alpha, std::complex<float> const *A,
                       da_int lda, std::complex<float> const *B, da_int ldb,
                       std::complex<float> beta, std::complex<float> *C, da_int ldc) {
    cblas_csymm(layout, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void cblas_symm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo, da_int m,
                       da_int n, std::complex<double> alpha,
                       std::complex<double> const *A, da_int lda,
                       std::complex<double> const *B, da_int ldb,
                       std::complex<double> beta, std::complex<double> *C, da_int ldc) {
    cblas_zsymm(layout, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// -----------------------------------------------------------------------------
inline void cblas_syrk(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       da_int n, da_int k, float alpha, float const *A, da_int lda,
                       float beta, float *C, da_int ldc) {
    cblas_ssyrk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

inline void cblas_syrk(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       da_int n, da_int k, double alpha, double const *A, da_int lda,
                       double beta, double *C, da_int ldc) {
    cblas_dsyrk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

inline void cblas_syrk(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       da_int n, da_int k, std::complex<float> alpha,
                       std::complex<float> const *A, da_int lda, std::complex<float> beta,
                       std::complex<float> *C, da_int ldc) {
    cblas_csyrk(layout, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc);
}

inline void cblas_syrk(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       da_int n, da_int k, std::complex<double> alpha,
                       std::complex<double> const *A, da_int lda,
                       std::complex<double> beta, std::complex<double> *C, da_int ldc) {
    cblas_zsyrk(layout, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc);
}

// -----------------------------------------------------------------------------
inline void cblas_herk(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       da_int n, da_int k, float alpha, float const *A, da_int lda,
                       float beta, float *C, da_int ldc) {
    cblas_ssyrk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

inline void cblas_herk(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       da_int n, da_int k, double alpha, double const *A, da_int lda,
                       double beta, double *C, da_int ldc) {
    cblas_dsyrk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

inline void cblas_herk(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       da_int n, da_int k,
                       float alpha, // note: real
                       std::complex<float> const *A, da_int lda,
                       float beta, // note: real
                       std::complex<float> *C, da_int ldc) {
    cblas_cherk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

inline void cblas_herk(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                       da_int n, da_int k,
                       double alpha, // note: real
                       std::complex<double> const *A, da_int lda,
                       double beta, // note: real
                       std::complex<double> *C, da_int ldc) {
    cblas_zherk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

// -----------------------------------------------------------------------------
inline void cblas_syr2k(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                        da_int n, da_int k, float alpha, float const *A, da_int lda,
                        float const *B, da_int ldb, float beta, float *C, da_int ldc) {
    cblas_ssyr2k(layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_syr2k(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                        da_int n, da_int k, double alpha, double const *A, da_int lda,
                        double const *B, da_int ldb, double beta, double *C, da_int ldc) {
    cblas_dsyr2k(layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_syr2k(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                        da_int n, da_int k, std::complex<float> alpha,
                        std::complex<float> const *A, da_int lda,
                        std::complex<float> const *B, da_int ldb,
                        std::complex<float> beta, std::complex<float> *C, da_int ldc) {
    cblas_csyr2k(layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void cblas_syr2k(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                        da_int n, da_int k, std::complex<double> alpha,
                        std::complex<double> const *A, da_int lda,
                        std::complex<double> const *B, da_int ldb,
                        std::complex<double> beta, std::complex<double> *C, da_int ldc) {
    cblas_zsyr2k(layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// -----------------------------------------------------------------------------
inline void cblas_her2k(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                        da_int n, da_int k, float alpha, float const *A, da_int lda,
                        float const *B, da_int ldb, float beta, float *C, da_int ldc) {
    cblas_ssyr2k(layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_her2k(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                        da_int n, da_int k, double alpha, double const *A, da_int lda,
                        double const *B, da_int ldb, double beta, double *C, da_int ldc) {
    cblas_dsyr2k(layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_her2k(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                        da_int n, da_int k, std::complex<float> alpha,
                        std::complex<float> const *A, da_int lda,
                        std::complex<float> const *B, da_int ldb,
                        float beta, // note: real
                        std::complex<float> *C, da_int ldc) {
    cblas_cher2k(layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_her2k(CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                        da_int n, da_int k, std::complex<double> alpha,
                        std::complex<double> const *A, da_int lda,
                        std::complex<double> const *B, da_int ldb,
                        double beta, // note: real
                        std::complex<double> *C, da_int ldc) {
    cblas_zher2k(layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void imatcopy(char trans, da_int m, da_int n, float alpha, float *A, da_int lda_in,
                     da_int lda_out) {
    simatcopy_(&trans, &m, &n, (const float *)&alpha, A, &lda_in, &lda_out);
}

inline void imatcopy(char trans, da_int m, da_int n, double alpha, double *A,
                     da_int lda_in, da_int lda_out) {
    dimatcopy_(&trans, &m, &n, (const double *)&alpha, A, &lda_in, &lda_out);
}

inline void omatcopy(char trans, da_int m, da_int n, float alpha, const float *A,
                     da_int lda_in, float *B, da_int ldb_out) {
    somatcopy_(&trans, &m, &n, (const float *)&alpha, A, &lda_in, B, &ldb_out);
}

inline void omatcopy(char trans, da_int m, da_int n, double alpha, const double *A,
                     da_int lda_in, double *B, da_int ldb_out) {
    domatcopy_(&trans, &m, &n, (const double *)&alpha, A, &lda_in, B, &ldb_out);
}

} // namespace da_blas

#endif //  #ifndef CBLAS_HH
