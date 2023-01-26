#ifndef DRIVERS_HPP
#define DRIVERS_HPP

#include "callbacks.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

extern "C" {
/* C interface to the reverse communication lbfgsb solver */
void setulb_d_(int *n, int *m, double *x, double *l, double *u, int *nbd, double *f,
               double *g, double *factr, double *pgtol, double *wa, int *iwa, int *itask,
               int *iprint, int *icsave, int *lsavei, int *isave, double *dsave);

void setulb_s_(int *n, int *m, float *x, float *l, float *u, int *nbd, float *f, float *g,
               float *factr, float *pgtol, float *wa, int *iwa, int *itask, int *iprint,
               int *icsave, int *lsavei, int *isave, float *dsave);
}

/* Overload the definition of lbfgsb to work with both double and floats */
inline void setulb(int *n, int *m, double *x, double *l, double *u, int *nbd, double *f,
                   double *g, double *factr, double *pgtol, double *wa, int *iwa,
                   int *itask, int *iprint, int *icsave, int *lsavei, int *isave,
                   double *dsave) {
    setulb_d_(n, m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, itask, iprint, icsave,
              lsavei, isave, dsave);
}

inline void setulb(int *n, int *m, float *x, float *l, float *u, int *nbd, float *f,
                   float *g, float *factr, float *pgtol, float *wa, int *iwa, int *itask,
                   int *iprint, int *icsave, int *lsavei, int *isave, float *dsave) {
    setulb_s_(n, m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, itask, iprint, icsave,
              lsavei, isave, dsave);
}

template <typename T> void lbfgs_prec(T &factr, T &pgtol) {
    /* double => pgtol ~ 1.0e-05
     *           factr ~ 1.0e07
     * single => pgtol ~ 6.0e-03
     *        => factr ~ 1.0e03
     */
    pgtol = pow(std::numeric_limits<T>::epsilon(), 0.32);
    factr = pow(10., std::numeric_limits<T>::digits10 / 2);
}

/* Internal memory for lbfgsb */
template <typename T> struct lbfgsb_data {
    int m = 0; // number of vectors of memory used
               // TODO should be an option
    int *nbd = nullptr, *iwa = nullptr;
    T *wa = nullptr;

    // TODO these should be options
    T factr = 1.0e07;
    T pgtol = 1.0e-05;
    int iprint = 0;
};

template <typename T> void free_lbfgsb_data(lbfgsb_data<T> **d) {
    if ((*d)->iwa)
        delete[] (*d)->iwa;
    if ((*d)->wa)
        delete[] (*d)->wa;
    if ((*d)->nbd)
        delete[] (*d)->nbd;

    delete *d;
    *d = nullptr;
}

template <typename T>
int init_lbfgsb_data(lbfgsb_data<T> *d, int n, int m, T bigbnd, std::vector<T> &l,
                     std::vector<T> &u) {

    d->m = m;
    try {
        d->iwa = new int[3 * n];
    } catch (std::bad_alloc &) {
        return 1;
    }
    int nwa = 2 * m * n + 5 * n + 11 * m * m + 8 * m;
    try {
        d->wa = new T[nwa];
    } catch (std::bad_alloc &) {
        return 1;
    }
    try {
        d->nbd = new int[n];
    } catch (std::bad_alloc &) {
        return 1;
    }

    int i;
    if (l.size() == 0 || u.size() == 0) {
        for (i = 0; i < n; i++)
            d->nbd[i] = 0;
    } else {
        for (i = 0; i < n; i++) {
            if (l[i] >= -bigbnd && u[i] <= bigbnd)
                d->nbd[i] = 2;
            else if (l[i] >= -bigbnd)
                d->nbd[i] = 1;
            else if (u[i] <= bigbnd)
                d->nbd[i] = 3;
            else
                d->nbd[i] = 0;
        }
    }
    lbfgs_prec<T>(d->factr, d->pgtol);

    return 0;
}

template <typename T>
int lbfgsb_fc(lbfgsb_data<T> *d, int n, T *x, T *l, T *u, T *f, T *g, objfun_t<T> objfun,
              objgrd_t<T> objgrd, void *usrdata) {

    /* if l nd u are nullptr, d->nbd should be filled with 0 and the bound vector
     * should not be accessed
     */

    int itask = 2;

    int m = d->m;

    /* task = 'START => itask = 2
     * TASK = 'NEW_X' => itask = 1
     */
    bool compute_fg = true;
    int icsave, lsavei[4], isave[44];
    T dsave[29];
    while (itask == 2 || itask == 1 || compute_fg) {
        setulb(&n, &m, &(*x), &(*l), &(*u), &(d->nbd[0]), &f[0], &g[0], &d->factr,
               &d->pgtol, &d->wa[0], &d->iwa[0], &itask, &d->iprint, &icsave, &lsavei[0],
               &isave[0], dsave);
        compute_fg = itask == 4 ||  // 'FG'
                     itask == 21 || // 'FG_START'
                     itask == 20;   // 'FG_LNSRCH
        if (compute_fg) {
            objfun(n, x, f, usrdata);
            objgrd(n, x, g, usrdata);
        }
    }

    return 0;
}

#endif