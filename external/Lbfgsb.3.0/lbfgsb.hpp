#ifndef LBFGSB_HPP
#define LBFGSB_HPP
#include <string>
#include <vector>

// Courtesy export symbol for main entry point
// Add other cases here
#if defined(_WIN32)
  #if defined(__GNUC__) && !defined(__clang)
    #define DLBFGSB_SOLVER dlbfgsb_solver_
    #define SLBFGSB_SOLVER slbfgsb_solver_
  #else
    #define DLBFGSB_SOLVER DLBFGSB_SOLVER
    #define SLBFGSB_SOLVER SLBFGSB_SOLVER
  #endif
#else
  #define DLBFGSB_SOLVER dlbfgsb_solver_
  #define SLBFGSB_SOLVER slbfgsb_solver_
#endif

extern "C" {
/* C interface to the reverse communication lbfgsb solver */
void DLBFGSB_SOLVER(da_int *n, da_int *m, double *x, double *l, double *u, da_int *nbd,
                    double *f, double *g, double *factr, double *pgtol, double *wa,
                    da_int *iwa, da_int *itask, da_int *iprint, da_int *lsavei,
                    da_int *isave, double *dsave);

void SLBFGSB_SOLVER(da_int *n, da_int *m, float *x, float *l, float *u, da_int *nbd,
                    float *f, float *g, float *factr, float *pgtol, float *wa,
                    da_int *iwa, da_int *itask, da_int *iprint, da_int *lsavei,
                    da_int *isave, float *dsave);
}
#endif
