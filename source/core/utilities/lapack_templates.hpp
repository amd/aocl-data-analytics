#ifndef LAPACK_TEMPLATES_HPP
#define LAPACK_TEMPLATES_HPP
#include "aoclda_types.h"

extern "C" {
void sgesvd_(char *jobu, char *jobv, da_int *m, da_int *n, float *a, da_int *lda,
             float *s, float *u, da_int *ldu, float *vt, da_int *ldvt, float *work,
             da_int *lwork, da_int *info);
void dgesvd_(char *jobu, char *jobv, da_int *m, da_int *n, double *a, da_int *lda,
             double *s, double *u, da_int *ldu, double *vt, da_int *ldvt, double *work,
             da_int *lwork, da_int *info);

void dgeqrf_(da_int *m, da_int *n, double *a, da_int *lda, double *tau, double *work,
             da_int *lwork, da_int *info);
void sgeqrf_(da_int *m, da_int *n, float *a, da_int *lda, float *tau, float *work,
             da_int *lwork, da_int *info);
void dormqr_(char *side, char *trans, da_int *m, da_int *n, da_int *k, double *a,
             da_int *lda, double *tau, double *c, da_int *ldc, double *work,
             da_int *lwork, da_int *info);
void sormqr_(char *side, char *trans, da_int *m, da_int *n, da_int *k, float *a,
             da_int *lda, float *tau, float *c, da_int *ldc, float *work, da_int *lwork,
             da_int *info);
void dtrtrs_(char *uplo, char *trans, char *diag, da_int *n, da_int *nrhs, double *a,
             da_int *lda, double *b, da_int *ldb, da_int *info);
int strtrs_(char *uplo, char *trans, char *diag, da_int *n, da_int *nrhs, float *a,
            da_int *lda, float *b, da_int *ldb, da_int *info);
}

namespace da {
inline void gesvd(char *jobu, char *jobv, da_int *m, da_int *n, float *a, da_int *lda,
                  float *s, float *u, da_int *ldu, float *vt, da_int *ldvt, float *work,
                  da_int *lwork, da_int *info) {
    sgesvd_(jobu, jobv, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
};

inline void gesvd(char *jobu, char *jobv, da_int *m, da_int *n, double *a, da_int *lda,
                  double *s, double *u, da_int *ldu, double *vt, da_int *ldvt,
                  double *work, da_int *lwork, da_int *info) {
    dgesvd_(jobu, jobv, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
};

// --- QR factorization (classic) ---
inline void geqrf(da_int *m, da_int *n, float *a, da_int *lda, float *tau, float *work,
                  da_int *lwork, da_int *info) {
    sgeqrf_(m, n, a, lda, tau, work, lwork, info);
}
inline void geqrf(da_int *m, da_int *n, double *a, da_int *lda, double *tau, double *work,
                  da_int *lwork, da_int *info) {
    dgeqrf_(m, n, a, lda, tau, work, lwork, info);
}
// --- Apply Q or Q' from QR factorization ---
inline void ormqr(char *side, char *trans, da_int *m, da_int *n, da_int *k, float *a,
                  da_int *lda, float *tau, float *c, da_int *ldc, float *work,
                  da_int *lwork, da_int *info) {
    sormqr_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
}
inline void ormqr(char *side, char *trans, da_int *m, da_int *n, da_int *k, double *a,
                  da_int *lda, double *tau, double *c, da_int *ldc, double *work,
                  da_int *lwork, da_int *info) {
    dormqr_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
}
// --- solves a triangular system of the form A * X = B  or  A**T * X = B ---
inline void trtrs(char *uplo, char *trans, char *diag, da_int *n, da_int *nrhs, float *a,
                  da_int *lda, float *b, da_int *ldb, da_int *info) {
    strtrs_(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
}
inline void trtrs(char *uplo, char *trans, char *diag, da_int *n, da_int *nrhs, double *a,
                  da_int *lda, double *b, da_int *ldb, da_int *info) {
    dtrtrs_(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
}
} // namespace da
#endif
