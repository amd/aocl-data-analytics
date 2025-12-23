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
 * [D]  Coord     L1          centering
 * [D]  Coord     L1          none
 * [D]  Coord     L1          scale only
 * [D]  Coord     L1          standardize
 */
const linregParam linregParamLASSO[] = {
    // 0
{  "LASSO/trivial/coord/1/z", "triviall1", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 1
{  "LASSO/trivial/coord/0/z", "triviall1", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 2
{  "LASSO/trivial/coord/1/c", "triviall1unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 1520}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 3
{  "LASSO/trivial/coord/0/c", "triviall1unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 1500}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",2.f},{"alpha",1.0f},{"optim dual gap tol", infs}},
                                     {{"lambda",2.},{"alpha",1.0},{"optim dual gap tol", infd}}
                                     },
    // [Disabled: XX data is assumed to be centered]
    // {"CoordL1Reg+1/n", "triviall1unscl",    {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 1500}},
    //                                 {{"optim method", "coord"}, {"scaling", "none"}},
    //                                 {{"lambda",2.f},{"alpha",1.0f}},
    //                                 {{"lambda",2.},{"alpha",1.0}}
    //                                 },
    // 4
{  "LASSO/trivial/coord/0/n", "triviall1unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 1500}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"lambda",2.f},{"alpha",1.0f},{"optim dual gap tol", infs}},
                                     {{"lambda",2.},{"alpha",1.0},{"optim dual gap tol", infd}}
                                     },
    // 5
{  "LASSO/trivial/coord/1/s", "triviall1unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",2.f},{"alpha",1.0f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", 10.0}},
                                     {{"lambda",2.},{"alpha",1.0},{"optim dual gap tol", 10.0}}
                                     },
    // 6
{  "LASSO/trivial/coord/0/s", "triviall1unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",2.f},{"alpha",1.0f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", 10.0}},
                                     {{"lambda",2.},{"alpha",1.0},{"optim dual gap tol", 10.0}}
                                     },
    // 7
{  "LASSO/NormTab/coord/0/z", "glmnet-100x20l1", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 8
{  "LASSO/NormTab/coord/1/z", "glmnet-100x20l1", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",2.0f},{"alpha",1.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.0},{"alpha",1.0},{"optim dual gap tol", infd}}
                                     },
    // 9
{  "LASSO/NormTab/coord/0/s", "glmnet-100x20l1unscl", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",2.0f},{"alpha",1.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.0},{"alpha",1.0},{"optim dual gap tol", infd}}
                                     },
    // 10
{  "LASSO/NormTab/coord/1/s", "glmnet-100x20l1unscl", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",2.0f},{"alpha",1.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.0},{"alpha",1.0},{"optim dual gap tol", infd}}
                                     },
    // Compare all scalings of all solvers with sci-kit learn / glmnet (for elasticnet) output
    // =======================================================================================
    // OK - Pass, OK* - Pass with modification to the problem (either add small lambda or relax tolerance)
    // DP - Different problem, NA - Solver not applicable, F - Fail
    //
    // NORMAL
    // SCALING=NONE ===================================
    // LASSO - ALL NA except for coord descent  with NO INTERCEPT
    // NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA    OK   NA
    // tall-thin      NA    NA    NA   NA    OK   NA
    // tall-fat       NA    NA    NA   NA    OK   NA
    // CENTERING =====================================
    // LASSO - ALL NA except for coord descent
    // BOTH INTERCEPT AND NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA    OK   NA
    // tall-thin      NA    NA    NA   NA    OK   NA
    // tall-fat       NA    NA    NA   NA    OK   NA
    // SCALE ONLY ===================================
    // LASSO (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA   OK    NA
    // tall-thin      NA    NA    NA   NA   OK    NA
    // tall-fat       NA    NA    NA   NA   OK    NA
    // STANDARDIZE =================================
    // (HERE DATA PASSED IS PRESCALED TO HAVE VARIANCE=1 AND
    // MEAN=0 IN EACH COLUMN AND OUTPUT IS BEING COMPARED TO GLMNET)
    // QR UNAVAIL BECAUSE PRESCALING UNDERDETERMINED PROBLEM MAKES MATRIX LOW-RANK
    // LASSO (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA   OK    NA
    // tall-thin      NA    NA    NA   NA   OK    NA
    // tall-fat       NA    NA    NA   NA   OK    NA
    // =============================================

    // 11
{  "LASSO/ShortFat/coord/0/n", "short_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 12
{  "LASSO/ShortFat/coord/0/c", "short_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 13
{  "LASSO/ShortFat/coord/1/c", "short_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 14 LASSO comparison with sklearn results
{  "LASSO/TallFat/coord/0/n", "tall_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 15 LASSO comparison with sklearn results
{  "LASSO/TallFat/coord/0/c", "tall_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 16 LASSO comparison with sklearn results
{  "LASSO/TallFat/coord/1/c", "tall_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},

    // 17 LASSO comparison with sklearn results
{  "LASSO/TallThin/coord/0/n", "tall_thinl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 18 LASSO comparison with sklearn results
{  "LASSO/TallThin/coord/0/c", "tall_thinl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 19 LASSO comparison with sklearn results
{  "LASSO/TallThin/coord/1/c", "tall_thinl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    /* L1 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 20
{  "LASSO/ShortFat/coord/0/s", "short_fatl1", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 120}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f},{"optim dual gap tol", 1.0}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",1.0},{"optim dual gap tol", 1.0}},
                                     true, false, 1, {-1, -1}, true
                                     },
    /* TALL THIN */
    // 21
{  "LASSO/TallThin/coord/0/s", "tall_thinl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f},{"optim dual gap tol", 200.0}},
                                     {{"lambda",0.3},{"alpha",1.0},{"optim dual gap tol", 200.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 22
{  "LASSO/TallFat/coord/0/s", "tall_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f},{"optim dual gap tol", 8.5e-3}},
                                     {{"lambda",0.3},{"alpha",1.0},{"optim dual gap tol", 8.5e-3}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 23
{  "LASSO/ShortFat/coord/1/s", "short_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f},{"optim dual gap tol", 1.0e-1}, {"optim convergence tol",1.e-7f}},
                                     {{"lambda",0.3},{"alpha",1.0},{"optim dual gap tol", 1.0e-1}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    /* TALL THIN */
    // 24
{  "LASSO/TallThin/coord/1/s", "tall_thinl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f},{"optim dual gap tol", 190.0}},
                                     {{"lambda",0.3},{"alpha",1.0},{"optim dual gap tol", 190.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 25
{  "LASSO/TallFat/coord/1/s", "tall_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f},{"optim dual gap tol", 3.0e-2}},
                                     {{"optim convergence tol",1.e-8}, {"lambda",0.3},{"alpha",1.0},{"optim dual gap tol", 3.0e-2}},
                                     true, false
                                     },
    /* L1 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 26
{  "LASSO/ShortFat/coord/0/z", "scl_short_fatl1", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 100}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-8}, {"lambda",0.3},{"alpha",1.0}},
                                     true, false, 1, {-1, -1}, true
                                     },
    /* TALL THIN */
    // 27
{  "LASSO/TallThin/coord/0/z", "scl_tall_thinl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 28
{  "LASSO/TallFat/coord/0/z", "scl_tall_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 29
{  "LASSO/ShortFat/coord/1/z", "scl_short_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 30
{  "LASSO/TallThin/coord/1/z", "scl_tall_thinl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 31
{  "LASSO/TallFat/coord/1/z", "scl_tall_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    // 32 scikit-learn sparse signal example LASSO GLMnet step to match sklearn)
{  "LASSO/signal/coord/1/s", "signal-scikit", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",0.14f},{"alpha",1.0f},{"optim dual gap tol", 200.0}},
                                     {{"optim convergence tol",1.e-12},{"lambda",0.14},{"alpha",1.0},{"optim dual gap tol", 200.0}},
                                     true, true
                                     },
    // 33 scikit-learn sparse signal example LASSO GLMnet step to match sklearn)
{  "LASSO/signal/coord/0/s", "signal-scikit", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",0.21f},{"alpha",1.0f},{"optim dual gap tol", 200.0}},
                                     {{"optim convergence tol",3.e-7},{"lambda",0.21},{"alpha",1.0},{"optim dual gap tol", 200.0}},
                                     true, false
                                     },
    // 34 scikit-learn sparse signal example LASSO
{  "LASSO/signal/coord/1/c", "signal-scikit", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.14f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.14},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, true
                                     },
    // 35 scikit-learn sparse signal example LASSO
{  "LASSO/signal/coord/0/c", "signal-scikit", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.21f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.21},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 36 scikit-learn sparse signal example LASSO
{  "LASSO/signal/coord/0/n", "signal-scikit", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "none"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.21f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.21},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 37 scikit-learn sparse signal example LASSO to test scaling=AUTO with intercept
{  "LASSO/signal/coord/1/a", "signal-scikit", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "auto"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.14f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.14},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, true
                                     },
    // 38 scikit-learn sparse signal example LASSO to test scaling=AUTO with no intercept
{  "LASSO/signal/coord/0/a", "signal-scikit", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "auto"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.21f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.21},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 39 Dual-gap check
{  "LASSO/dualgap/coord/0/a", "dualgap_lasso", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 300}},
                                     {{"optim method", "coord"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.45f},{"alpha",1.0f},{"optim dual gap tol", 1.e-6f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.45},{"alpha",1.0},{"optim dual gap tol", 1.e-6}},
                                     true, false, 1.0f, {3.0e-6f, 3.0e-9f}
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
                         testing::ValuesIn(linregParamLASSO));
INSTANTIATE_TEST_SUITE_P(linregPosSuiteF, linregPosF,
                         testing::ValuesIn(linregParamLASSO));
