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

#ifndef LAPACK_TEMPLATES_HPP
#define LAPACK_TEMPLATES_HPP
#include "aoclda_types.h"

extern "C" {
void sposv_(char *uplo, da_int *n, da_int *nrhs, float *a, da_int *lda, float *b,
            da_int *ldb, da_int *info);
void dposv_(char *uplo, da_int *n, da_int *nrhs, double *a, da_int *lda, double *b,
            da_int *ldb, da_int *info);

void spotrf_(char *uplo, da_int *n, float *a, da_int *lda, da_int *info);
void dpotrf_(char *uplo, da_int *n, double *a, da_int *lda, da_int *info);

void spotrs_(char *uplo, da_int *n, da_int *nrhs, float *a, da_int *lda, float *b,
             da_int *ldb, da_int *info);
void dpotrs_(char *uplo, da_int *n, da_int *nrhs, double *a, da_int *lda, double *b,
             da_int *ldb, da_int *info);

void spocon_(char *uplo, da_int *n, float *a, da_int *lda, float *anorm, float *rcond,
             float *work, da_int *iwork, da_int *info);
void dpocon_(char *uplo, da_int *n, double *a, da_int *lda, double *anorm, double *rcond,
             double *work, da_int *iwork, da_int *info);
void sgesdd_(char *jobz, da_int *m, da_int *n, float *a, da_int *lda, float *s, float *u,
             da_int *ldu, float *vt, da_int *ldvt, float *work, da_int *lwork,
             da_int *iwork, da_int *info);
void dgesdd_(char *jobz, da_int *m, da_int *n, double *a, da_int *lda, double *s,
             double *u, da_int *ldu, double *vt, da_int *ldvt, double *work,
             da_int *lwork, da_int *iwork, da_int *info);
void sgesvd_(char *jobu, char *jobv, da_int *m, da_int *n, float *a, da_int *lda,
             float *s, float *u, da_int *ldu, float *vt, da_int *ldvt, float *work,
             da_int *lwork, da_int *info);
void dgesvd_(char *jobu, char *jobv, da_int *m, da_int *n, double *a, da_int *lda,
             double *s, double *u, da_int *ldu, double *vt, da_int *ldvt, double *work,
             da_int *lwork, da_int *info);
void sgesvdx_(char *jobu, char *jobv, char *range, da_int *m, da_int *n, float *a,
              da_int *lda, float *vl, float *vu, da_int *il, da_int *iu, da_int *ns,
              float *s, float *u, da_int *ldu, float *vt, da_int *ldvt, float *work,
              da_int *lwork, da_int *iwork, da_int *info);
void dgesvdx_(char *jobu, char *jobv, char *range, da_int *m, da_int *n, double *a,
              da_int *lda, double *vl, double *vu, da_int *il, da_int *iu, da_int *ns,
              double *s, double *u, da_int *ldu, double *vt, da_int *ldvt, double *work,
              da_int *lwork, da_int *iwork, da_int *info);
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
double dlange_(char const *norm, da_int const *m, da_int const *n, double const *A,
               da_int const *lda, double *work);
float slange_(char const *norm, da_int const *m, da_int const *n, float const *A,
              da_int const *lda, float *work);
}

namespace da {
// --- Cholesky factorization ---
inline void posv(char *uplo, da_int *n, da_int *nrhs, float *a, da_int *lda, float *b,
                 da_int *ldb, da_int *info) {
    sposv_(uplo, n, nrhs, a, lda, b, ldb, info);
};
inline void posv(char *uplo, da_int *n, da_int *nrhs, double *a, da_int *lda, double *b,
                 da_int *ldb, da_int *info) {
    dposv_(uplo, n, nrhs, a, lda, b, ldb, info);
};
inline void potrf(char *uplo, da_int *n, float *a, da_int *lda, da_int *info) {
    spotrf_(uplo, n, a, lda, info);
};
inline void potrf(char *uplo, da_int *n, double *a, da_int *lda, da_int *info) {
    dpotrf_(uplo, n, a, lda, info);
};
inline void potrs(char *uplo, da_int *n, da_int *nrhs, float *a, da_int *lda, float *b,
                  da_int *ldb, da_int *info) {
    spotrs_(uplo, n, nrhs, a, lda, b, ldb, info);
};
inline void potrs(char *uplo, da_int *n, da_int *nrhs, double *a, da_int *lda, double *b,
                  da_int *ldb, da_int *info) {
    dpotrs_(uplo, n, nrhs, a, lda, b, ldb, info);
};
inline void pocon(char *uplo, da_int *n, float *a, da_int *lda, float *anorm,
                  float *rcond, float *work, da_int *iwork, da_int *info) {
    spocon_(uplo, n, a, lda, anorm, rcond, work, iwork, info);
}
inline void pocon(char *uplo, da_int *n, double *a, da_int *lda, double *anorm,
                  double *rcond, double *work, da_int *iwork, da_int *info) {
    dpocon_(uplo, n, a, lda, anorm, rcond, work, iwork, info);
}
// --- SVD ---
inline void gesdd(char *jobz, da_int *m, da_int *n, float *a, da_int *lda, float *s,
                  float *u, da_int *ldu, float *vt, da_int *ldvt, float *work,
                  da_int *lwork, da_int *iwork, da_int *info) {
    sgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
};

inline void gesdd(char *jobz, da_int *m, da_int *n, double *a, da_int *lda, double *s,
                  double *u, da_int *ldu, double *vt, da_int *ldvt, double *work,
                  da_int *lwork, da_int *iwork, da_int *info) {
    dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
};

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

inline void gesvdx(char *jobu, char *jobv, char *range, da_int *m, da_int *n, float *a,
                   da_int *lda, float *vl, float *vu, da_int *il, da_int *iu, da_int *ns,
                   float *s, float *u, da_int *ldu, float *vt, da_int *ldvt, float *work,
                   da_int *lwork, da_int *iwork, da_int *info) {
    sgesvdx_(jobu, jobv, range, m, n, a, lda, vl, vu, il, iu, ns, s, u, ldu, vt, ldvt,
             work, lwork, iwork, info);
};

inline void gesvdx(char *jobu, char *jobv, char *range, da_int *m, da_int *n, double *a,
                   da_int *lda, double *vl, double *vu, da_int *il, da_int *iu,
                   da_int *ns, double *s, double *u, da_int *ldu, double *vt,
                   da_int *ldvt, double *work, da_int *lwork, da_int *iwork,
                   da_int *info) {
    dgesvdx_(jobu, jobv, range, m, n, a, lda, vl, vu, il, iu, ns, s, u, ldu, vt, ldvt,
             work, lwork, iwork, info);
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

// --- Compute norm of a real matrix ---
inline float lange(char const *norm, da_int const *m, da_int const *n, float const *A,
                   da_int const *lda, float *work) {
    return slange_(norm, m, n, A, lda, work);
}
inline double lange(char const *norm, da_int const *m, da_int const *n, double const *A,
                    da_int const *lda, double *work) {
    return dlange_(norm, m, n, A, lda, work);
}

} // namespace da
#endif
