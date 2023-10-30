#ifndef LINMOD_QR_REG_HPP
#define LINMOD_QR_REG_HPP

#include "aoclda.h"
#include <vector>

// data for QR factorization used in standard linear least squares
template <typename T> struct qr_data {
    // A needs to be copied as lapack's dgeqr modifies the matrix
    std::vector<T> A, b, tau, work;
    da_int lwork = 0;

    // Constructors
    qr_data(da_int m, da_int n, T *Ai, T *bi, bool intercept, da_int ncoef) {

        // Copy A and b, starting with the first n columns of A
        A.resize(m * ncoef);
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i < m; i++) {
                A[j * m + i] = Ai[j * m + i];
            }
        }
        b.resize(m);
        for (da_int i = 0; i < m; i++)
            b[i] = bi[i];

        // add a column of 1 to A if intercept is required
        if (intercept) {
            for (da_int i = 0; i < m; i++)
                A[n * m + i] = 1.0;
        }

        // work arrays for the LAPACK QR factorization
        tau.resize(std::min(m, ncoef));
        lwork = ncoef;
        work.resize(lwork);
    };
};


#endif