/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LBFGSB_HPP
#define LBFGSB_HPP
#include <string>
#include <vector>

// Courtesy export symbol for main entry point
// Add other cases here
#if defined(WIN_IFORT_NAME_MANGLING)
  #define DLBFGSB_SOLVER DLBFGSB_SOLVER
  #define SLBFGSB_SOLVER SLBFGSB_SOLVER
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
