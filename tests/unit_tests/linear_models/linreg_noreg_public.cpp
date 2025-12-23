/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "linreg_positive.hpp"
#include "gtest/gtest.h"

// clang-format off
/*
 * To keep comments with numbers in sync use reflow_numbers.sh
 *
 * Replicate table for intercept=yes|no
 * Done Solver Regularization Scaling
 * [D]  BFGS      NONE        none
 * [D]  BFGS      NONE        centering
 * [D]  BFGS      NONE        scale only
 * [D]  BFGS      NONE        standardize
 * [D]  Cholesky  NONE        none
 * [D]  Cholesky  NONE        centering
 * [D]  Cholesky  NONE        scale only
 * [D]  Cholesky  NONE        standardize
 * [D]  Coord     NONE        none
 * [D]  Coord     NONE        centering
 * [D]  Coord     NONE        scale only
 * [D]  Coord     NONE        standardization
 * [D]  QR        NONE        none
 * [D]  QR        NONE        scale only
 * [D]  QR        NONE        standardize
 * [D]  Sparse CG NONE        none
 * [D]  Sparse CG NONE        centering
 * [D]  Sparse CG NONE        scale only
 * [D]  Sparse CG NONE        standardization
 */
const linregParam linregParamNoReg[] = {
    // 0
{  "NoReg/trivial/auto/0/a", "trivial", {}, {}, {}, {}},
    // 1
{  "NoReg/trivial/auto/0/z", "trivial", {}, {{"scaling", "standardize"}}, {}, {}},
    // 2
{  "NoReg/trivial/auto/0/s", "trivial", {}, {{"scaling", "scale only"}}, {}, {}},
    // 3
#ifndef NO_FORTRAN
{  "NoReg/trivial/bfgs/0/a", "trivial", {{"print level", 1}}, {{"optim method", "lbfgs"}}, {}, {}},
    // 4
{  "NoReg/trivial/bfgs/0/z", "trivial", {}, {{"optim method", "lbfgs"},{"scaling", "standardize"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 5
{  "NoReg/trivial/bfgs/0/s", "trivial", {}, {{"optim method", "lbfgs"},{"scaling", "scale only"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 6
#endif
{  "NoReg/trivial/auto/1/a", "trivial", {{"intercept", 1}}, {}, {}, {}},
    // 7
{  "NoReg/trivial/auto/1/z", "trivial", {{"intercept", 1}}, {{"scaling", "standardize"}}, {}, {}},
    // 8 QR with intercept and scaling only
{  "NoReg/trivial/auto/1/s", "trivial", {{"intercept", 1}}, {{"scaling", "scale only"}}, {}, {}},
    // 9
#ifndef NO_FORTRAN
{  "NoReg/trivial/bfgs/1/a", "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"}}, {}, {}},
    // 10
{  "NoReg/trivial/bfgs/1/z", "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"},{"scaling", "standardize"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 11
{  "NoReg/trivial/bfgs/1/s", "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"},{"scaling", "scale only"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // Data and solution generated using R (glmnet_trivial.R)
    // 12
#endif
{  "NoReg/trivial/coord/1/c", "trivial", {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}, {"print options", "yes"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim dual gap tol", infs}},
                                     {{"lambda",0.0},{"alpha",0.5}, {"optim convergence tol",1.e-7},{"optim dual gap tol", infd}}
                                     },
    // 13
{  "NoReg/trivial/coord/0/c", "trivial", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-6f},{"optim dual gap tol", infs}},
                                     {{"lambda",0.0},{"alpha",0.5},{"optim convergence tol", 1.0e-6},{"optim dual gap tol", infd}}
                                     },
    // 14
#ifndef NO_FORTRAN
{  "NoReg/trivial/bfgs/0/c", "trivial", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "bfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-6f}},
                                     {{"lambda",0.0},{"alpha",0.5},{"optim convergence tol", 1.0e-6}}
                                     },
    // 15
#endif
{  "NoReg/trivial/coord/0/n", "trivial", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-7f},{"optim dual gap tol", infs}},
                                     {{"lambda",0.0},{"alpha",0.5},{"optim convergence tol", 1.0e-7,},{"optim dual gap tol", infd}}
                                     },
    // [disabled XX: scaling none with intercept assumes data is centered!]
    //{"CoordNoReg+1/n", "trivial",      {{"intercept", 1}, {"print level", 5}, {"optim iteration limit", 1800}},
    //                                 {{"optim method", "coord"}, {"scaling", "none"}},
    //                                 {{"lambda",0.0f},{"alpha",0.5f}},
    //                                 {{"lambda",0.0},{"alpha",0.5}}
    //                                 },
    // 16
{  "NoReg/trivial/coord/1/z", "trivial", {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim dual gap tol", 0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}, {"optim convergence tol",1.e-7}, {"optim dual gap tol", 0.5}}
                                     },
    // 17
{  "NoReg/trivial/coord/0/z", "trivial", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", infs}},
                                     {{"lambda",0.0},{"alpha",0.5}, {"optim convergence tol",1.e-7},{"optim dual gap tol", infd}}
                                     },
    // Data and solution generated using R (glmnet_trivial.R) (STANDARDIZED = FALSE, our scaling = "scale only")
    // 18
{  "NoReg/trivial/coord/1/s", "trivial", {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", 0.5}},
                                     {{"lambda",0.0},{"alpha",0.5}, {"optim convergence tol",1.e-7}, {"optim dual gap tol", 0.5}}
                                     },
    // 19
{  "NoReg/trivial/coord/0/s", "trivial", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 700}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", 1.0}},
                                     {{"lambda",0.0},{"alpha",0.5}, {"optim convergence tol",1.e-7}, {"optim dual gap tol", 1.0}}
                                     },
    // Data and solution generated using R (glmnet_driver.R)
    // 20
{  "NoReg/NormTab/coord/0/z", "glmnet-100x20", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",0.0f},{"alpha",1.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-9},{"lambda",0.0},{"alpha",1.0},{"optim dual gap tol", infd}},
                                     },
    // 21
{  "NoReg/NormTab/coord/1/z", "glmnet-100x20", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",0.0f},{"alpha",1.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-9},{"lambda",0.0},{"alpha",1.0},{"optim dual gap tol", infd}},
                                     },
    // 22 - same set of problems but scaling="scale only" (standardize=FALSE)
{  "NoReg/NormTab/coord/0/s", "glmnet-100x20unscl", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",0.0f},{"alpha",1.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim dual gap tol", infd}},
                                     },
    // 23
{  "NoReg/NormTab/coord/1/s", "glmnet-100x20unscl", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",0.0f},{"alpha",1.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim dual gap tol", infd}},
                                     },
    // 24
#ifndef NO_FORTRAN
{  "NoReg/NormTab/bfgs/0/z", "glmnet-100x20", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-20},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 10.0}},
                                     },
    // 25
{  "NoReg/NormTab/bfgs/1/z", "glmnet-100x20", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 10.0}},
                                     },
    // 26 - same set of problems 12-19 but scaling="scale only" (standardize=FALSE)
{  "NoReg/NormTab/bfgs/0/s", "glmnet-100x20unscl", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 10.0}},
                                     },
    // 27
{  "NoReg/NormTab/bfgs/1/s", "glmnet-100x20unscl", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.0f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     },
#endif
    // same problems solved with QR - selecting only NOREG
    // 28 models y ~ X + 0, y ~ X + 1, no-reg, scaling only OR standardize
{  "NoReg/trivial/QR/1/z", "trivial", {{"intercept", 1}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 29
{  "NoReg/trivial/QR/0/z", "trivial", {{"intercept", 0}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 30
{  "NoReg/trivial/QR/1/s", "trivial", {{"intercept", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 31
{  "NoReg/trivial/QR/0/s", "trivial", {{"intercept", 0}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 32
{  "NoReg/NormTab/QR/0/z", "glmnet-100x20", {{"intercept", 0}},
                                     {{"optim method", "qr"},{"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 33
{  "NoReg/NormTab/QR/1/z", "glmnet-100x20", {{"intercept", 1}},
                                     {{"optim method", "qr"},{"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 34
{  "NoReg/NormTab/QR/0/s", "glmnet-100x20unscl", {{"intercept", 0}},
                                     {{"optim method", "qr"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 35
{  "NoReg/NormTab/QR/1/s", "glmnet-100x20unscl", {{"intercept", 1}},
                                     {{"optim method", "qr"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 36
#ifndef NO_FORTRAN
{  "NoReg/trivial/bfgs/1/c", "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
#endif
    // 37
{  "NoReg/trivial/qr/0/c", "trivial", {{"intercept", 0}}, {{"optim method", "qr"},{"scaling", "centering"}}, {}, {}},
    // 38
{  "NoReg/trivial/qr/1/c", "trivial", {{"intercept", 1}}, {{"optim method", "qr"},{"scaling", "centering"}}, {}, {}},
    // 39 Add some regularization to find minimal norm solution, relax tolerance
    // Compare all scalings of all solvers with sci-kit learn / glmnet (for elasticnet) output
    // =======================================================================================
    // OK - Pass, OK* - Pass with modification to the problem (either add small lambda or relax tolerance)
    // DP - Different problem, NA - Solver not applicable, F - Fail
    //
    // NORMAL
    // SCALING=NONE ===================================
    // NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK    ??   OK
    // tall-thin      OK    OK    OK   OK    OK   OK
    // tall-fat       OK    OK    OK   OK    OK   OK
    //
    // INTERCEPT (solvers unavail because strategy for
    // intercept in under-det. is to center data)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK*   NA    NA   NA    NA   NA
    // tall-thin      OK    NA    OK   OK    NA   NA
    // tall-fat       OK*   NA    NA   NA    NA   NA
    //
    // NORMAL
    // CENTERING =====================================
    // NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK    OK*  OK
    // tall-thin      OK    OK    OK   OK    OK   OK
    // tall-fat       OK    OK    OK   OK    OK   OK
    //
    // INTERCEPT (singular in undetermined situation)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK*  OK    OK*  NA
    // tall-thin      OK    OK    OK   OK    OK*  OK
    // tall-fat       OK    OK    OK*  OK*   OK   NA
    //
    // SCALE ONLY ===================================
    // NORMAL
    // NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK   OK*   OK
    // tall-thin      OK    OK    OK   OK   OK    OK
    // tall-fat       OK    OK    OK   OK   OK    OK
    //
    // INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK*  OK   OK*   NA
    // tall-thin      OK    OK    OK   OK   OK    OK
    // tall-fat       OK    OK    OK*  OK*  OK*   NA
    //
    // STANDARDIZE =================================
    // (HERE DATA PASSED IS PRESCALED TO HAVE VARIANCE=1 AND
    // MEAN=0 IN EACH COLUMN AND OUTPUT IS BEING COMPARED TO GLMNET)
    // QR UNAVAIL BECAUSE PRESCALING UNDERDETERMINED PROBLEM MAKES MATRIX LOW-RANK
    // NORMAL
    // NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK*  OK   OK*   NA
    // tall-thin      OK    OK    OK   OK   OK    OK
    // tall-fat       OK    OK    OK*  OK   OK*   NA
    //
    // INTERCEPT (singular in undetermined situation)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK*  OK   OK*   NA
    // tall-thin      OK    OK    OK   OK   OK    OK
    // tall-fat       OK    OK    OK*  OK*  OK*   NA
    //
{  "NoReg/ShortFat/coord/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.005f}, {"alpha", 0.0f},{"optim dual gap tol", 7e-3}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.005}, {"alpha", 0.0},{"optim dual gap tol", 7e-3}},
                                     true, false, 200.0},
    // 40 Add some regularization to find minimal norm solution, relax tolerance
{  "NoReg/ShortFat/coord/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.005f}, {"alpha", 0.0f},{"optim dual gap tol", 2.8e-4}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.005}, {"alpha", 0.0},{"optim dual gap tol", 2.8e-4}},
                                     true, false, 150.0},
    // 41 Add some regularization to find minimal norm solution, relax tolerance
{  "NoReg/ShortFat/coord/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.005f}, {"alpha", 0.0f},{"optim dual gap tol", 2.8e-4}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.005}, {"alpha", 0.0},{"optim dual gap tol", 2.8e-4}},
                                     true, false, 150.0},
    // 42 NoReg comparison with sklearn results
{  "NoReg/TallFat/coord/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.0f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.0}, {"alpha", 0.0}},
                                     true, false},
    // 43 NoReg comparison with sklearn results
{  "NoReg/TallFat/coord/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.0f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.0}, {"alpha", 0.0}},
                                     true, false},
    // 44 NoReg comparison with sklearn results
    // Add a lot of lambda - also use relaxed tolerance
{  "NoReg/TallFat/coord/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-6f}, {"lambda", 0.1f}, {"alpha", 0.0f},{"optim dual gap tol", 2.0e-4}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.1}, {"alpha", 0.0},{"optim dual gap tol", 2.0e-4}},
                                     true, false, 20.0},
    // 45 NoReg comparison with sklearn results
{  "NoReg/TallThin/coord/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.0f}, {"alpha", 0.0f},{"optim dual gap tol", 7.0e-4}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.0}, {"alpha", 0.0},{"optim dual gap tol", 7.0e-4}},
                                     true, false},
    // 46 NoReg comparison with sklearn results
{  "NoReg/TallThin/coord/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.0f}, {"alpha", 0.0f},{"optim dual gap tol", 7.0e-4}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.0}, {"alpha", 0.0},{"optim dual gap tol", 7.0e-4}},
                                     true, false},
    // 47 NoReg comparison with sklearn results
{  "NoReg/TallThin/coord/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.0f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.0}, {"alpha", 0.0}},
                                     true, false},
    // 48
#ifndef NO_FORTRAN
{  "NoReg/ShortFat/bfgs/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 49
{  "NoReg/ShortFat/svd/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 50
{  "NoReg/ShortFat/chol/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 51
{  "NoReg/ShortFat/cg/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 52
{  "NoReg/ShortFat/qr/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 53
#ifndef NO_FORTRAN
{  "NoReg/TallThin/bfgs/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 54
{  "NoReg/TallThin/svd/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 55
{  "NoReg/TallThin/chol/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 56
{  "NoReg/TallThin/cg/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    // 57
{  "NoReg/TallThin/qr/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 58
#ifndef NO_FORTRAN
{  "NoReg/TallFat/bfgs/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 59
{  "NoReg/TallFat/svd/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 60
{  "NoReg/TallFat/chol/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 61
{  "NoReg/TallFat/cg/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 62
{  "NoReg/TallFat/qr/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // Add small lambda
    // 63
#ifndef NO_FORTRAN
{  "NoReg/ShortFat/bfgs/1/n", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.001f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.00001},{"alpha",0.0}, {"optim progress factor",1.0}},
                                     true, false
                                     },
    // 64
{  "NoReg/TallThin/bfgs/1/n", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 65
{  "NoReg/TallThin/chol/1/n", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 66
{  "NoReg/TallThin/cg/1/n", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    // Add a bit of lambda (a lot for float)
    // 67
#ifndef NO_FORTRAN
{  "NoReg/TallFat/bfgs/1/n", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.1f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.00001},{"alpha",0.0}, {"optim progress factor",1.0}},
                                     true, false
                                     },
    // 68
{  "NoReg/ShortFat/bfgs/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 69
{  "NoReg/ShortFat/svd/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 70
{  "NoReg/ShortFat/chol/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 71
{  "NoReg/ShortFat/cg/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 72
{  "NoReg/ShortFat/qr/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 73
#ifndef NO_FORTRAN
{  "NoReg/TallThin/bfgs/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 74
{  "NoReg/TallThin/svd/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 75
{  "NoReg/TallThin/chol/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 76
{  "NoReg/TallThin/cg/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    // 77
{  "NoReg/TallThin/qr/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 78
#ifndef NO_FORTRAN
{  "NoReg/TallFat/bfgs/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 79
{  "NoReg/TallFat/svd/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 80
{  "NoReg/TallFat/chol/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 81
{  "NoReg/TallFat/cg/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 82
{  "NoReg/TallFat/qr/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    // 83
#ifndef NO_FORTRAN
{  "NoReg/ShortFat/bfgs/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 84
{  "NoReg/ShortFat/svd/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Bump lambda a bit to get around singular matrix
    // 85
{  "NoReg/ShortFat/chol/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.00001f},{"alpha",0.0f}},
                                     {{"lambda",0.00001},{"alpha",0.0}},
                                     true, false
                                     },
    // 86
{  "NoReg/ShortFat/cg/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 87
#ifndef NO_FORTRAN
{  "NoReg/TallThin/bfgs/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 88
{  "NoReg/TallThin/svd/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 89
{  "NoReg/TallThin/chol/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 90
{  "NoReg/TallThin/cg/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    // 91
{  "NoReg/TallThin/qr/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    // 92
#ifndef NO_FORTRAN
{  "NoReg/TallFat/bfgs/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 93
{  "NoReg/TallFat/svd/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda to make it possible to factorise
    // 94
{  "NoReg/TallFat/chol/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 95
{  "NoReg/TallFat/cg/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 96
#ifndef NO_FORTRAN
{  "NoReg/ShortFat/bfgs/0/s", "short_fat", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 100}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 97
{  "NoReg/ShortFat/svd/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 98
{  "NoReg/ShortFat/chol/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 99
{  "NoReg/ShortFat/cg/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Hard to obtain sklearn result due to underdetermined system, need to add 0.0001 lambda and increase tolerance to 0.0028
    // 100 WARM START
{  "NoReg/ShortFat/coord/0/s", "short_fat", {{"intercept", 0}, {"print level", 3},{"optim iteration limit", 100}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",2.e-8f}, {"lambda",0.0001f},{"alpha",0.0f}, {"optim dual gap tol",2.0e-4f}},
                                     {{"optim convergence tol",5.e-9}, {"lambda",0.0001},{"alpha",0.0}, {"optim dual gap tol",2.0e-4}},
                                     true, false, 3.8, {-1, -1}, true
                                     },
    // 101
{  "NoReg/ShortFat/qr/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 102
#ifndef NO_FORTRAN
{  "NoReg/TallThin/bfgs/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 103
{  "NoReg/TallThin/svd/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 104
{  "NoReg/TallThin/chol/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 105
{  "NoReg/TallThin/cg/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    // 106
{  "NoReg/TallThin/coord/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f},{"optim dual gap tol",0.11}},
                                     {{"lambda",0.0},{"alpha",0.0},{"optim dual gap tol",0.11}},
                                     true, false
                                     },
    // 107
{  "NoReg/TallThin/qr/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 108
#ifndef NO_FORTRAN
{  "NoReg/TallFat/bfgs/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-9}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 109
{  "NoReg/TallFat/svd/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 110
{  "NoReg/TallFat/chol/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 111
{  "NoReg/TallFat/cg/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 112
{  "NoReg/TallFat/coord/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 113
{  "NoReg/TallFat/qr/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    /* Tricky situation, calculating solution to undetermined system with
     * intercept in unregularised case leads to dealing with matrix with
     * very high conditional number which makes the solution unstable and
     * difficult to compare between each other
     */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        condition number which makes the solution unstable and difficult to compare between each other */
    // 114
#ifndef NO_FORTRAN
{  "NoReg/ShortFat/bfgs/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 115
{  "NoReg/ShortFat/svd/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 116 Bump lambda a bit to get around singular matrix
{  "NoReg/ShortFat/chol/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.00001f},{"alpha",0.0f}},
                                     {{"lambda",0.00001},{"alpha",0.0}},
                                     true, false
                                     },
    // 117
{  "NoReg/ShortFat/cg/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 118 Add small lambda, WARMSTART and set tolerance to 0.003
{  "NoReg/ShortFat/coord/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f},{"optim dual gap tol", 2.0e-4}},
                                     {{"optim convergence tol",7.e-8f}, {"lambda",0.0001},{"alpha",0.0},{"optim dual gap tol", 2.0e-4}},
                                     true, false, 3, {-1, -1}, true
                                     },
    /* TALL THIN */
    // 119 Fail for single precision
#ifndef NO_FORTRAN
{  "NoReg/TallThin/bfgs/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",0.1f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",0.1}},
                                     true, false, 1.5f
                                     },
#endif
    // 120
{  "NoReg/TallThin/svd/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 121
{  "NoReg/TallThin/chol/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 122
{  "NoReg/TallThin/cg/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    // 123
{  "NoReg/TallThin/coord/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f},{"optim dual gap tol", 5.0e-3}},
                                     {{"lambda",0.0},{"alpha",0.0},{"optim dual gap tol", 5.0e-3}},
                                     true, false
                                     },
    // 124
{  "NoReg/TallThin/qr/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    // 125
#ifndef NO_FORTRAN
{  "NoReg/TallFat/bfgs/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 126
{  "NoReg/TallFat/svd/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda to make it possible to factorise
    // 127
{  "NoReg/TallFat/chol/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 128
{  "NoReg/TallFat/cg/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 129
{  "NoReg/TallFat/coord/1/s", "tall_fat", {{"intercept", 1}, {"print level", 3},{"optim iteration limit", 300}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.01f},{"alpha",0.0f},{"optim dual gap tol", 2.0e-4}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.01},{"alpha",0.0},{"optim dual gap tol", 2.0e-4}},
                                     true, false, 3.5, {-1, -1}, true
                                     },
    /* STANDARDIZE (HERE WE COMPARING TO GLMNET OUTPUT) */
    /* NORMAL TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 130
#ifndef NO_FORTRAN
{  "NoReg/ShortFat/bfgs/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda", 0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 131
{  "NoReg/ShortFat/svd/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 132 Add a bit of lambda
{  "NoReg/ShortFat/chol/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // 133
{  "NoReg/ShortFat/cg/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 134 Add a bit of lambda
{  "NoReg/ShortFat/coord/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 100}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda",0.0001f},{"alpha",0.0f}, {"optim dual gap tol",2.0e-4f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0001},{"alpha",0.0}, {"optim dual gap tol",2.0e-4}},
                                     true, false, 1, {-1, -1}, true
                                     },
    /* TALL THIN */
    // 135
#ifndef NO_FORTRAN
{  "NoReg/TallThin/bfgs/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 136
{  "NoReg/TallThin/svd/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 137
{  "NoReg/TallThin/chol/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 138
{  "NoReg/TallThin/cg/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 139
{  "NoReg/TallThin/coord/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f},{"optim dual gap tol", 0.01}},
                                     {{"lambda",0.0},{"alpha",0.0},{"optim dual gap tol", 0.01}},
                                     true, false
                                     },
    // 140
{  "NoReg/TallThin/qr/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 141
#ifndef NO_FORTRAN
{  "NoReg/TallFat/bfgs/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 142
{  "NoReg/TallFat/svd/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 143
{  "NoReg/TallFat/chol/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // 144
{  "NoReg/TallFat/cg/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 145
{  "NoReg/TallFat/coord/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.001f},{"alpha",0.0f},{"optim dual gap tol", 2.0e-3}},
                                     {{"optim convergence tol",1.e-11f}, {"lambda",0.001},{"alpha",0.0},{"optim dual gap tol", 2.0e-3}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    // 146
#ifndef NO_FORTRAN
{  "NoReg/ShortFat/bfgs/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 147
{  "NoReg/ShortFat/svd/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 148 Add tiny bit of lambda
{  "NoReg/ShortFat/chol/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // 149
{  "NoReg/ShortFat/cg/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda (this problem is too stringent, relax cmp tol?)
    // 150
{  "NoReg/ShortFat/coord/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 100}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}, {"optim dual gap tol",3.0e-8f}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001},{"alpha",0.0}, {"optim dual gap tol",3.0e-8f}},
                                     true, false, 1, {-1, -1}, true
                                     },
    /* TALL THIN */
    // 151
#ifndef NO_FORTRAN
{  "NoReg/TallThin/bfgs/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 152
{  "NoReg/TallThin/svd/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 153
{  "NoReg/TallThin/chol/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 154
{  "NoReg/TallThin/cg/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 155
{  "NoReg/TallThin/coord/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f},{"optim dual gap tol", 5.0e-3}},
                                     {{"lambda",0.0},{"alpha",0.0},{"optim dual gap tol", 5.0e-3}},
                                     true, false
                                     },
    // 156
{  "NoReg/TallThin/qr/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised
     * case leads to dealing with matrix with very high condition number which makes the solution
     * unstable and difficult to compare between each other
     */
    // 157
#ifndef NO_FORTRAN
{  "NoReg/TallFat/bfgs/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 158
{  "NoReg/TallFat/svd/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 159 Add tiny bit of lambda
{  "NoReg/TallFat/chol/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // 160 Add tiny bit of lambda
{  "NoReg/TallFat/cg/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 161 Add tiny bit of lambda
{  "NoReg/TallFat/coord/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 3},{"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",6.e-6f}, {"lambda",0.01f},{"alpha",0.0f},{"optim dual gap tol", 0.02}},
                                     {{"optim convergence tol",5.e-9}, {"lambda",0.0001},{"alpha",0.0},{"optim dual gap tol", 0.02}},
                                     true, false, 1, {-1, -1}, true
                                     },
    // 162 Dual-gap check
{  "NoReg/dualgap/coord/0/a", "dualgap", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 300}},
                                     {{"optim method", "coord"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.0f},{"alpha",0.0f},{"optim dual gap tol", 1.e-7f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0},{"optim dual gap tol", 1.e-7}},
                                     true, false, 1.0f, {1.0e-10f, 1.0e-10f}
                                     },
};
// clang-format on

// Data  Tests
class linregPosD : public testing::TestWithParam<linregParam> {};
class linregPosF : public testing::TestWithParam<linregParam> {};

// Teach GTest how to print the param type
// in this case use only user's unique testname
// It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
void PrintTo(const linregParam &param, ::std::ostream *os) { *os << param.test_name; }

// Positive tests with double type
TEST_P(linregPosD, Double) {
    const linregParam &param = GetParam();
    test_linreg_positive<double>(param.data_name, param.iopts, param.sopts, param.dopts,
                                 param.check_coeff, param.check_predict,
                                 (double)param.check_tol_scale, (double)param.dual_gap[1],
                                 param.initial_guess);
}

// Positive tests with float type
TEST_P(linregPosF, Float) {
    const linregParam &param = GetParam();
    test_linreg_positive<float>(param.data_name, param.iopts, param.sopts, param.fopts,
                                param.check_coeff, param.check_predict,
                                (float)param.check_tol_scale, (float)param.dual_gap[0],
                                param.initial_guess);
}

INSTANTIATE_TEST_SUITE_P(linregPosSuiteD, linregPosD,
                         testing::ValuesIn(linregParamNoReg));
INSTANTIATE_TEST_SUITE_P(linregPosSuiteF, linregPosF,
                         testing::ValuesIn(linregParamNoReg));
