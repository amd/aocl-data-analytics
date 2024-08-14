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

/* Based on code at https://people.sc.fsu.edu/~jburkardt/f77_src/asa136/asa136.html with minimal edits

  AMD CHANGES:
  - templating
  - n_iter added which returns number of iterations used
  - int -> da_iont throughout
  - workspace now passed in
  - deal with lda
  - const a
  - corrected an lda bug
  - fixed compilation warnings
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T>
void kmns(const T a[], da_int m, da_int n, da_int lda, T c[], da_int k, da_int ic1[],
          da_int nc[], da_int iter, T wss[], da_int *ifault, da_int *n_iter, da_int ic2[],
          T an1[], T an2[], da_int ncp[], T d[], da_int itran[], da_int live[]);
template <typename T>
void optra(const T a[], da_int m, da_int n, da_int lda, T c[], da_int k, da_int ic1[],
           da_int ic2[], da_int nc[], T an1[], T an2[], da_int ncp[], T d[],
           da_int itran[], da_int live[], da_int *indx);
template <typename T>
void qtran(const T a[], da_int m, da_int n, da_int lda, T c[], da_int k, da_int ic1[],
           da_int ic2[], da_int nc[], T an1[], T an2[], da_int ncp[], T d[],
           da_int itran[], da_int *indx);
template <typename T> T r8_huge();

/******************************************************************************/

template <typename T>
void kmns(const T a[], da_int m, da_int n, da_int lda, T c[], da_int k, da_int ic1[],
          da_int nc[], da_int iter, T wss[], da_int *ifault, da_int *n_iter, da_int ic2[],
          T an1[], T an2[], da_int ncp[], T d[], da_int itran[], da_int live[])

/******************************************************************************/
/*
  Purpose:

    KMNS carries out the K-means algorithm.

  Discussion:

    This routine attempts to divide M points in N-dimensional space into
    K clusters so that the within cluster sum of squares is minimized.

  Licensing:

    This code is distributed under the MIT license.

  Modified:

    09 November 2010

  Author:

    Original FORTRAN77 version by John Hartigan, Manchek Wong.
    C version by John Burkardt.

  Reference:

    John Hartigan, Manchek Wong,
    Algorithm AS 136:
    A K-Means Clustering Algorithm,
    Applied Statistics,
    Volume 28, Number 1, 1979, pages 100-108.

  Parameters:

    Input, T A(M,N), the points.

    Input, da_int M, the number of points.

    Input, da_int N, the number of spatial dimensions.

    Input/output, T C(K,N), the cluster centers.

    Input, da_int K, the number of clusters.

    Output, da_int IC1(M), the cluster to which each point
    is assigned.

    Output, da_int NC(K), the number of points in each cluster.

    Input, da_int ITER, the maximum number of iterations allowed.

    Output, T WSS(K), the within-cluster sum of squares
    of each cluster.

    Output, da_int *IFAULT, error indicator.
    0, no error was detected.
    1, at least one cluster is empty after the initial assignment.  A better
       set of initial cluster centers is needed.
    2, the allowed maximum number off iterations was exceeded.
    3, K is less than or equal to 1, or greater than or equal to M.
*/
{
    T aa;
    T da;
    T db;
    T dc;
    T dt[2];
    da_int i;
    da_int ii;
    da_int ij;
    da_int il;
    da_int indx;
    da_int j;
    da_int l;
    T temp;

    *ifault = 0;

    if (k <= 1 || m <= k) {
        *ifault = 3;
        return;
    }
    //ic2 = (da_int *)malloc(m * sizeof(int));
    //an1 = (T *)malloc(k * sizeof(T));
    //an2 = (T *)malloc(k * sizeof(T));
    //ncp = (da_int *)malloc(k * sizeof(int));
    //d = (T *)malloc(m * sizeof(T));
    //itran = (da_int *)malloc(k * sizeof(int));
    //live = (da_int *)malloc(k * sizeof(int));
    /*
  For each point I, find its two closest centers, IC1(I) and
  IC2(I).  Assign the point to IC1(I).
*/
    for (i = 1; i <= m; i++) {
        ic1[i - 1] = 1;
        ic2[i - 1] = 2;

        for (il = 1; il <= 2; il++) {
            dt[il - 1] = 0.0;
            for (j = 1; j <= n; j++) {
                da = a[i - 1 + (j - 1) * lda] - c[il - 1 + (j - 1) * k];
                dt[il - 1] = dt[il - 1] + da * da;
            }
        }

        if (dt[1] < dt[0]) {
            ic1[i - 1] = 2;
            ic2[i - 1] = 1;
            temp = dt[0];
            dt[0] = dt[1];
            dt[1] = temp;
        }

        for (l = 3; l <= k; l++) {
            db = 0.0;
            for (j = 1; j <= n; j++) {
                dc = a[i - 1 + (j - 1) * lda] - c[l - 1 + (j - 1) * k];
                db = db + dc * dc;
            }

            if (db < dt[1]) {
                if (dt[0] <= db) {
                    dt[1] = db;
                    ic2[i - 1] = l;
                } else {
                    dt[1] = dt[0];
                    ic2[i - 1] = ic1[i - 1];
                    dt[0] = db;
                    ic1[i - 1] = l;
                }
            }
        }
    }
    /*
  Update cluster centers to be the average of points contained within them.
*/
    for (l = 1; l <= k; l++) {
        nc[l - 1] = 0;
        for (j = 1; j <= n; j++) {
            c[l - 1 + (j - 1) * k] = 0.0;
        }
    }

    for (i = 1; i <= m; i++) {
        l = ic1[i - 1];
        nc[l - 1] = nc[l - 1] + 1;
        for (j = 1; j <= n; j++) {
            c[l - 1 + (j - 1) * k] = c[l - 1 + (j - 1) * k] + a[i - 1 + (j - 1) * lda];
        }
    }
    /*
  Check to see if there is any empty cluster at this stage.
*/
    *ifault = 1;

    for (l = 1; l <= k; l++) {
        if (nc[l - 1] == 0) {
            *ifault = 1;
            return;
        }
    }

    *ifault = 0;

    for (l = 1; l <= k; l++) {
        aa = (T)(nc[l - 1]);

        for (j = 1; j <= n; j++) {
            c[l - 1 + (j - 1) * k] = c[l - 1 + (j - 1) * k] / aa;
        }
        /*
  Initialize AN1, AN2, ITRAN and NCP.

  AN1(L) = NC(L) / (NC(L) - 1)
  AN2(L) = NC(L) / (NC(L) + 1)
  ITRAN(L) = 1 if cluster L is updated in the quick-transfer stage,
           = 0 otherwise

  In the optimal-transfer stage, NCP(L) stores the step at which
  cluster L is last updated.

  In the quick-transfer stage, NCP(L) stores the step at which
  cluster L is last updated plus M.
*/
        an2[l - 1] = aa / (aa + (T)1.0);

        if (1.0 < aa) {
            an1[l - 1] = aa / (aa - (T)1.0);
        } else {
            an1[l - 1] = r8_huge<T>();
        }
        itran[l - 1] = 1;
        ncp[l - 1] = -1;
    }

    indx = 0;
    *ifault = 2;

    for (ij = 1; ij <= iter; ij++) {
        /*
  In this stage, there is only one pass through the data.   Each
  point is re-allocated, if necessary, to the cluster that will
  induce the maximum reduction in within-cluster sum of squares.
*/
        optra(a, m, n, lda, c, k, ic1, ic2, nc, an1, an2, ncp, d, itran, live, &indx);
        /*
  Stop if no transfer took place in the last M optimal transfer steps.
*/
        if (indx == m) {
            *ifault = 0;
            break;
        }
        /*
  Each point is tested in turn to see if it should be re-allocated
  to the cluster to which it is most likely to be transferred,
  IC2(I), from its present cluster, IC1(I).   Loop through the
  data until no further change is to take place.
*/
        qtran(a, m, n, lda, c, k, ic1, ic2, nc, an1, an2, ncp, d, itran, &indx);
        /*
  If there are only two clusters, there is no need to re-enter the
  optimal transfer stage.
*/
        if (k == 2) {
            *ifault = 0;
            break;
        }
        /*
  NCP has to be set to 0 before entering OPTRA.
*/
        for (l = 1; l <= k; l++) {
            ncp[l - 1] = 0;
        }
    }
    /*
  If the maximum number of iterations was taken without convergence,
  IFAULT is 2 now.  This may indicate unforeseen looping.
*/
    *n_iter = ij;
    /*
    if (*ifault == 2) {
        printf("\n");
        printf("KMNS - Warning!\n");
        printf("  Maximum number of iterations reached\n");
        printf("  without convergence.\n");
    }
    */
    /*
  Compute the within-cluster sum of squares for each cluster.
*/
    for (l = 1; l <= k; l++) {
        wss[l - 1] = 0.0;
        for (j = 1; j <= n; j++) {
            c[l - 1 + (j - 1) * k] = 0.0;
        }
    }

    for (i = 1; i <= m; i++) {
        ii = ic1[i - 1];
        for (j = 1; j <= n; j++) {
            c[ii - 1 + (j - 1) * k] = c[ii - 1 + (j - 1) * k] + a[i - 1 + (j - 1) * lda];
        }
    }

    for (j = 1; j <= n; j++) {
        for (l = 1; l <= k; l++) {
            c[l - 1 + (j - 1) * k] = c[l - 1 + (j - 1) * k] / (T)(nc[l - 1]);
        }
        for (i = 1; i <= m; i++) {
            ii = ic1[i - 1];
            da = a[i - 1 + (j - 1) * lda] - c[ii - 1 + (j - 1) * k];
            wss[ii - 1] = wss[ii - 1] + da * da;
        }
    }

    //free(ic2);
    //free(an1);
    //free(an2);
    //free(ncp);
    //free(d);
    //free(itran);
    //free(live);

    return;
}
/******************************************************************************/
template <typename T>
void optra(const T a[], da_int m, da_int n, da_int lda, T c[], da_int k, da_int ic1[],
           da_int ic2[], da_int nc[], T an1[], T an2[], da_int ncp[], T d[],
           da_int itran[], da_int live[], da_int *indx)

/******************************************************************************/
/*
  Purpose:

    OPTRA carries out the optimal transfer stage.

  Discussion:

    This is the optimal transfer stage.

    Each point is re-allocated, if necessary, to the cluster that
    will induce a maximum reduction in the within-cluster sum of
    squares.

  Licensing:

    This code is distributed under the MIT license.

  Modified:

    09 November 2010

  Author:

    Original FORTRAN77 version by John Hartigan, Manchek Wong.
    C version by John Burkardt.

  Reference:

    John Hartigan, Manchek Wong,
    Algorithm AS 136:
    A K-Means Clustering Algorithm,
    Applied Statistics,
    Volume 28, Number 1, 1979, pages 100-108.

  Parameters:

    Input, T A(M,N), the points.

    Input, da_int M, the number of points.

    Input, da_int N, the number of spatial dimensions.

    Input/output, T C(K,N), the cluster centers.

    Input, da_int K, the number of clusters.

    Input/output, da_int IC1(M), the cluster to which each
    point is assigned.

    Input/output, da_int IC2(M), used to store the cluster
    which each point is most likely to be transferred to at each step.

    Input/output, da_int NC(K), the number of points in
    each cluster.

    Input/output, T AN1(K).

    Input/output, T AN2(K).

    Input/output, da_int NCP(K).

    Input/output, T D(M).

    Input/output, da_int ITRAN(K).

    Input/output, da_int LIVE(K).

    Input/output, da_int *INDX, the number of steps since a
    transfer took place.
*/
{
    T al1;
    T al2;
    T alt;
    T alw;
    T da;
    T db;
    T dc;
    T dd;
    T de;
    T df;
    da_int i;
    da_int j;
    da_int l;
    da_int l1;
    da_int l2;
    da_int ll;
    T r2;
    T rr;
    /*
  If cluster L is updated in the last quick-transfer stage, it
  belongs to the live set throughout this stage.   Otherwise, at
  each step, it is not in the live set if it has not been updated
  in the last M optimal transfer steps.
*/
    for (l = 1; l <= k; l++) {
        if (itran[l - 1] == 1) {
            live[l - 1] = m + 1;
        }
    }

    for (i = 1; i <= m; i++) {
        *indx = *indx + 1;
        l1 = ic1[i - 1];
        l2 = ic2[i - 1];
        ll = l2;
        /*
  If point I is the only member of cluster L1, no transfer.
*/
        if (1 < nc[l1 - 1]) {
            /*
  If L1 has not yet been updated in this stage, no need to
  re-compute D(I).
*/
            if (ncp[l1 - 1] != 0) {
                de = 0.0;
                for (j = 1; j <= n; j++) {
                    df = a[i - 1 + (j - 1) * lda] - c[l1 - 1 + (j - 1) * k];
                    de = de + df * df;
                }
                d[i - 1] = de * an1[l1 - 1];
            }
            /*
  Find the cluster with minimum R2.
*/
            da = 0.0;
            for (j = 1; j <= n; j++) {
                db = a[i - 1 + (j - 1) * lda] - c[l2 - 1 + (j - 1) * k];
                da = da + db * db;
            }
            r2 = da * an2[l2 - 1];

            for (l = 1; l <= k; l++) {
                /*
  If LIVE(L1) <= I, then L1 is not in the live set.   If this is
  true, we only need to consider clusters that are in the live set
  for possible transfer of point I.   Otherwise, we need to consider
  all possible clusters.
*/
                if ((i < live[l1 - 1] || i < live[l2 - 1]) && l != l1 && l != ll) {
                    rr = r2 / an2[l - 1];

                    dc = 0.0;
                    for (j = 1; j <= n; j++) {
                        dd = a[i - 1 + (j - 1) * lda] - c[l - 1 + (j - 1) * k];
                        dc = dc + dd * dd;
                    }

                    if (dc < rr) {
                        r2 = dc * an2[l - 1];
                        l2 = l;
                    }
                }
            }
            /*
  If no transfer is necessary, L2 is the new IC2(I).
*/
            if (d[i - 1] <= r2) {
                ic2[i - 1] = l2;
            }
            /*
  Update cluster centers, LIVE, NCP, AN1 and AN2 for clusters L1 and
  L2, and update IC1(I) and IC2(I).
*/
            else {
                *indx = 0;
                live[l1 - 1] = m + i;
                live[l2 - 1] = m + i;
                ncp[l1 - 1] = i;
                ncp[l2 - 1] = i;
                al1 = (T)(nc[l1 - 1]);
                alw = al1 - (T)1.0;
                al2 = (T)(nc[l2 - 1]);
                alt = al2 + (T)1.0;
                for (j = 1; j <= n; j++) {
                    c[l1 - 1 + (j - 1) * k] =
                        (c[l1 - 1 + (j - 1) * k] * al1 - a[i - 1 + (j - 1) * lda]) / alw;
                    c[l2 - 1 + (j - 1) * k] =
                        (c[l2 - 1 + (j - 1) * k] * al2 + a[i - 1 + (j - 1) * lda]) / alt;
                }
                nc[l1 - 1] = nc[l1 - 1] - 1;
                nc[l2 - 1] = nc[l2 - 1] + 1;
                an2[l1 - 1] = alw / al1;
                if ((T)1.0 < alw) {
                    an1[l1 - 1] = alw / (alw - (T)1.0);
                } else {
                    an1[l1 - 1] = r8_huge<T>();
                }
                an1[l2 - 1] = alt / al2;
                an2[l2 - 1] = alt / (alt + (T)1.0);
                ic1[i - 1] = l2;
                ic2[i - 1] = l1;
            }
        }

        if (*indx == m) {
            return;
        }
    }
    /*
  ITRAN(L) = 0 before entering QTRAN.   Also, LIVE(L) has to be
  decreased by M before re-entering OPTRA.
*/
    for (l = 1; l <= k; l++) {
        itran[l - 1] = 0;
        live[l - 1] = live[l - 1] - m;
    }

    return;
}
/******************************************************************************/
template <typename T>
void qtran(const T a[], da_int m, da_int n, da_int lda, T c[], da_int k, da_int ic1[],
           da_int ic2[], da_int nc[], T an1[], T an2[], da_int ncp[], T d[],
           da_int itran[], da_int *indx)

/******************************************************************************/
/*
  Purpose:

    QTRAN carries out the quick transfer stage.

  Discussion:

    This is the quick transfer stage.

    IC1(I) is the cluster which point I belongs to.
    IC2(I) is the cluster which point I is most likely to be
    transferred to.

    For each point I, IC1(I) and IC2(I) are switched, if necessary, to
    reduce within-cluster sum of squares.  The cluster centers are
    updated after each step.

  Licensing:

    This code is distributed under the MIT license.

  Modified:

    09 November 2010

  Author:

    Original FORTRAN77 version by John Hartigan, Manchek Wong.
    C version by John Burkardt.

  Reference:

    John Hartigan, Manchek Wong,
    Algorithm AS 136:
    A K-Means Clustering Algorithm,
    Applied Statistics,
    Volume 28, Number 1, 1979, pages 100-108.

  Parameters:

    Input, T A(M,N), the points.

    Input, da_int M, the number of points.

    Input, da_int N, the number of spatial dimensions.

    Input/output, T C(K,N), the cluster centers.

    Input, da_int K, the number of clusters.

    Input/output, da_int IC1(M), the cluster to which each
    point is assigned.

    Input/output, da_int IC2(M), used to store the cluster
    which each point is most likely to be transferred to at each step.

    Input/output, da_int NC(K), the number of points in
    each cluster.

    Input/output, T AN1(K).

    Input/output, T AN2(K).

    Input/output, da_int NCP(K).

    Input/output, T D(M).

    Input/output, da_int ITRAN(K).

    Input/output, da_int INDX, counts the number of steps
    since the last transfer.
*/
{
    T al1;
    T al2;
    T alt;
    T alw;
    T da;
    T db;
    T dd;
    T de;
    da_int i;
    da_int icoun;
    da_int istep;
    da_int j;
    da_int l1;
    da_int l2;
    T r2;
    /*
  In the optimal transfer stage, NCP(L) indicates the step at which
  cluster L is last updated.   In the quick transfer stage, NCP(L)
  is equal to the step at which cluster L is last updated plus M.
*/
    icoun = 0;
    istep = 0;

    for (;;) {
        for (i = 1; i <= m; i++) {
            icoun = icoun + 1;
            istep = istep + 1;
            l1 = ic1[i - 1];
            l2 = ic2[i - 1];
            /*
  If point I is the only member of cluster L1, no transfer.
*/
            if (1 < nc[l1 - 1]) {
                /*
  If NCP(L1) < ISTEP, no need to re-compute distance from point I to
  cluster L1.   Note that if cluster L1 is last updated exactly M
  steps ago, we still need to compute the distance from point I to
  cluster L1.
*/
                if (istep <= ncp[l1 - 1]) {
                    da = 0.0;
                    for (j = 1; j <= n; j++) {
                        db = a[i - 1 + (j - 1) * lda] - c[l1 - 1 + (j - 1) * k];
                        da = da + db * db;
                    }
                    d[i - 1] = da * an1[l1 - 1];
                }
                /*
  If NCP(L1) <= ISTEP and NCP(L2) <= ISTEP, there will be no transfer of
  point I at this step.
*/
                if (istep < ncp[l1 - 1] || istep < ncp[l2 - 1]) {
                    r2 = d[i - 1] / an2[l2 - 1];

                    dd = 0.0;
                    for (j = 1; j <= n; j++) {
                        de = a[i - 1 + (j - 1) * lda] - c[l2 - 1 + (j - 1) * k];
                        dd = dd + de * de;
                    }
                    /*
  Update cluster centers, NCP, NC, ITRAN, AN1 and AN2 for clusters
  L1 and L2.   Also update IC1(I) and IC2(I).   Note that if any
  updating occurs in this stage, INDX is set back to 0.
*/
                    if (dd < r2) {
                        icoun = 0;
                        *indx = 0;
                        itran[l1 - 1] = 1;
                        itran[l2 - 1] = 1;
                        ncp[l1 - 1] = istep + m;
                        ncp[l2 - 1] = istep + m;
                        al1 = (T)(nc[l1 - 1]);
                        alw = al1 - (T)1.0;
                        al2 = (T)(nc[l2 - 1]);
                        alt = al2 + (T)1.0;
                        for (j = 1; j <= n; j++) {
                            c[l1 - 1 + (j - 1) * k] = (c[l1 - 1 + (j - 1) * k] * al1 -
                                                       a[i - 1 + (j - 1) * lda]) /
                                                      alw;
                            c[l2 - 1 + (j - 1) * k] = (c[l2 - 1 + (j - 1) * k] * al2 +
                                                       a[i - 1 + (j - 1) * lda]) /
                                                      alt;
                        }
                        nc[l1 - 1] = nc[l1 - 1] - 1;
                        nc[l2 - 1] = nc[l2 - 1] + 1;
                        an2[l1 - 1] = alw / al1;
                        if (1.0 < alw) {
                            an1[l1 - 1] = alw / (alw - (T)1.0);
                        } else {
                            an1[l1 - 1] = r8_huge<T>();
                        }
                        an1[l2 - 1] = alt / al2;
                        an2[l2 - 1] = alt / (alt + (T)1.0);
                        ic1[i - 1] = l2;
                        ic2[i - 1] = l1;
                    }
                }
            }
            /*
  If no re-allocation took place in the last M steps, return.
*/
            if (icoun == m) {
                return;
            }
        }
    }
}
/******************************************************************************/
template <typename T>
T r8_huge()

/******************************************************************************/
/*
  Purpose:

    R8_HUGE returns a "huge" R8.

  Discussion:

    The value returned by this function is NOT required to be the
    maximum representable R8.  This value varies from machine to machine,
    from compiler to compiler, and may cause problems when being printed.
    We simply want a "very large" but non-infinite number.

  Licensing:

    This code is distributed under the MIT license.

  Modified:

    06 October 2007

  Author:

    John Burkardt

  Parameters:

    Output, T R8_HUGE, a "huge" R8 value.
*/
{
    T value;

    value = (T)1.0E+30;

    return value;
}
