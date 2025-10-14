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
 * [D]  Coord     L1 + L2     centering
 * [D]  Coord     L1 + L2     none
 * [D]  Coord     L1 + L2     scale only
 * [D]  Coord     L1 + L2     standardize
 */
const linregParam linregParamElNet[] = {
    // 0 Code coverage for printing -> print level = 5
{  "ElNet/trivial/coord/1/z", "trivialelnet",{{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",5.0f},{"alpha",0.8f}, {"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",5.0},{"alpha",0.8}, {"optim convergence tol", 1.0e-8}}
                                     },
    // 1
{  "ElNet/trivial/coord/0/z", "trivialelnet",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",6.0f},{"alpha",0.9f}, {"optim convergence tol",1.e-7f}},
                                     {{"lambda",6.0},{"alpha",0.9}, {"optim convergence tol",1.e-7}},
                                     },
    // 2
{  "ElNet/trivial/coord/1/s", "trivialelnetunscl",{{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",5.0f},{"alpha",0.8f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", 20.0}},
                                     {{"lambda",5.0},{"alpha",0.8},{"optim dual gap tol", 20.0}}
                                     },
    // 3
{  "ElNet/trivial/coord/0/s", "trivialelnetunscl",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",6.0f},{"alpha",0.9f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", 20.0}},
                                     {{"lambda",6.0},{"alpha",0.9},{"optim dual gap tol", 20.0}}
                                     },
    // 4
{  "ElNet/trivial/coord/1/c", "trivialelnetunscl",{{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",4.8*10.0f/11.7391f},{"alpha",0.8f},{"optim convergence tol", 1.0e-7f},{"optim progress factor", 100.0}},
                                     {{"lambda",4.8*10.0/11.7391},{"alpha",0.8},{"optim convergence tol", 1.0e-9},{"optim progress factor", 100.0}}
                                     },
    // 5
{  "ElNet/trivial/coord/0/c", "trivialelnetunscl",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",6.0f*10.0f/11.7278f},{"alpha",0.9f},{"optim convergence tol", 1.0e-7f},{"optim progress factor", 100.0}},
                                     {{"lambda",6.0*10.0/11.7278},{"alpha",0.9},{"optim convergence tol", 1.0e-9},{"optim progress factor", 100.0}}
                                     },
    // 6
{  "ElNet/trivial/coord/0/n", "trivialelnetunscl",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"lambda",6.0f*10.0f/11.7278f},{"alpha",0.9f},{"optim convergence tol", 1.0e-7f},{"optim progress factor", 100.0}},
                                     {{"lambda",6.0*10.0/11.7278},{"alpha",0.9},{"optim convergence tol", 1.0e-9},{"optim progress factor", 100.0}}
                                     },
    // 7
{  "ElNet/NormTab/coord/0/z", "glmnet-100x20en", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-5f},{"lambda",2.25f},{"alpha",0.8f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.25},{"alpha",0.8},{"optim dual gap tol", infd}}
                                     },
    // 8
{  "ElNet/NormTab/coord/1/z", "glmnet-100x20en", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",2.25f},{"alpha",0.8f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.25},{"alpha",0.8},{"optim dual gap tol", infd}}
                                     },
    // 9
{  "ElNet/NormTab/coord/0/s", "glmnet-100x20enunscl", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",2.25f},{"alpha",0.8f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.25},{"alpha",0.8},{"optim dual gap tol", infd}}
                                     },
    // 10
{  "ElNet/NormTab/coord/1/s", "glmnet-100x20enunscl", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",2.25f},{"alpha",0.8f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.25},{"alpha",0.8},{"optim dual gap tol", infd}}
                                     },
    // Compare all scalings of all solvers with sci-kit learn / glmnet (for elasticnet) output
    // =======================================================================================
    // OK - Pass, OK* - Pass with modification to the problem (either add small lambda or relax tolerance)
    // DP - Different problem, NA - Solver not applicable, F - Fail
    //
    // SCALING=NONE ===================================
    // ELASTIC NET - ALL NA except for coord descent with NO INTERCEPT
    // Using sklearn reference solution
    // NO INTERCEPT
    // INTERCEPT only for tall-thin DP
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA    OK   NA
    // tall-thin      NA    NA    NA   NA    OK   NA
    // tall-fat       NA    NA    NA   NA    OK   NA
    // CENTERING =====================================
    // ELASTIC NET - ALL NA except for coord descent
    // BOTH INTERCEPT AND NO INTERCEPT
    // Using sklearn solution
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA    OK   NA
    // tall-thin      NA    NA    NA   NA    OK   NA
    // tall-fat       NA    NA    NA   NA    OK   NA
    // SCALE ONLY ===================================
    // ELASTIC NET (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA   NA    NA   OK    NA
    // tall-thin      NA    NA   NA    NA   OK    NA
    // tall-fat       NA    NA   NA    NA   OK    NA
    // STANDARDIZE =================================
    // (HERE DATA PASSED IS PRESCALED TO HAVE VARIANCE=1 AND
    // MEAN=0 IN EACH COLUMN AND OUTPUT IS BEING COMPARED TO GLMNET)
    // QR UNAVAIL BECAUSE PRESCALING UNDERDETERMINED PROBLEM MAKES MATRIX LOW-RANK
    // ELASTIC NET (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA   NA    NA   OK    NA
    // tall-thin      NA    NA   NA    NA   OK    NA
    // tall-fat       NA    NA   NA    NA   OK    NA
    // =============================================

    // 11 Elastic net comparison with sklearn results
{  "ElNet/ShortFat/coord/0/n", "short_fatl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 12 Elastic net comparison with sklearn results
{  "ElNet/ShortFat/coord/0/c", "short_fatl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 13 Elastic net comparison with sklearn results
{  "ElNet/ShortFat/coord/1/c", "short_fatl12_sk", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 14 Elastic net comparison with sklearn results
{  "ElNet/TallFat/coord/0/n", "tall_fatl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 15 Elastic net comparison with sklearn results
{  "ElNet/TallFat/coord/0/c", "tall_fatl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 16 Elastic net comparison with sklearn results
{  "ElNet/TallFat/coord/1/c", "tall_fatl12_sk", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 17 Elastic net comparison with sklearn results
{  "ElNet/TallThin/coord/0/n", "tall_thinl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // DP Elastic net comparison with sklearn results
    // {"TallThin/L12/coord/1/n", "tall_thinl12_sk", {{"intercept", 1}, {"print level", 1}},
    // {{"optim method", "coord"}, {"scaling", "none"}}, {{"optim convergence tol",1.e-7f},{"lambda",0.3f},{"alpha",0.5f}},
    // {{"optim convergence tol",1.e-7},{"lambda",0.3},{"alpha",0.5}}, true, false },
    // 18 Elastic net comparison with sklearn results
{  "ElNet/TallThin/coord/0/c", "tall_thinl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 19 Elastic net comparison with sklearn results
{  "ElNet/TallThin/coord/1/c", "tall_thinl12_sk", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    /* ELASTIC NET TESTS */
    /* OUTPUT HERE IS COMPARED TO GLMNET INSTEAD OF SKLEARN */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 20
{  "ElNet/ShortFat/coord/0/s", "short_fatl12", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f},{"optim dual gap tol", 1.0e-1}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5},{"optim dual gap tol", 1.0e-1}},
                                     true, false
                                     },
    /* TALL THIN */
    // 21
{  "ElNet/TallThin/coord/0/s", "tall_thinl12", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f},{"optim dual gap tol", 200}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5},{"optim dual gap tol", 200}},
                                     true, false
                                     },
    /* TALL FAT */
    // 22
{  "ElNet/TallFat/coord/0/s", "tall_fatl12", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f},{"optim dual gap tol", 5.0e-3}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5},{"optim dual gap tol", 5.0e-3}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 23
{  "ElNet/ShortFat/coord/1/s", "short_fatl12", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f},{"optim dual gap tol", 0.1}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5},{"optim dual gap tol", 0.1}},
                                     true, false
                                     },
    /* TALL THIN */
    // 24
{  "ElNet/TallThin/coord/1/s", "tall_thinl12", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f},{"optim dual gap tol", 150.0}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5},{"optim dual gap tol", 150.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 25
{  "ElNet/TallFat/coord/1/s", "tall_fatl12", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f},{"optim dual gap tol", 1.7e-2}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5},{"optim dual gap tol", 1.7e-2}},
                                     true, false
                                     },
    /* ELASTIC NET TESTS */
    /* NO INTERCEPT */
    // 26 SHORT FAT
{  "ElNet/ShortFat/coord/0/z", "scl_short_fatl12", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    // 27 TALL THIN
{  "ElNet/TallThin/coord/0/z", "scl_tall_thinl12", {{"intercept", 0}, {"print level", 1},},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-6f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    // 28 TALL FAT
{  "ElNet/TallFat/coord/0/z", "scl_tall_fatl12", {{"intercept", 0}, {"print level", 1},},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* INTERCEPT */
    // 29 SHORT FAT
{  "ElNet/ShortFat/coord/1/z", "scl_short_fatl12", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    // 30 TALL THIN
{  "ElNet/TallThin/coord/1/z", "scl_tall_thinl12", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    // 31 TALL FAT
{  "ElNet/TallFat/coord/1/z", "scl_tall_fatl12", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    // DP scikit-learn sparse signal example ELASTIC-NET (GLMNet regularization path does not match with sklearn)
    // {"signal/coord/Enet/+1/s", "signal-scikit-enet", {{"debug", 0},{"intercept", 1},{"print level", 1}},
    //                                  {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
    //                                  {{"optim convergence tol",1.e-8f},{"lambda",0.17f},{"alpha",0.5f},{"optim progress factor", 1.0}},
    //                                  {{"optim convergence tol",1.e-8},{"lambda",0.17},{"alpha",0.5},{"optim progress factor", 1.0}},
    //                                  true, false
    //                                  },
    // DP scikit-learn sparse signal example ELASTIC-NET (GLMNet regularization path does not match with sklearn)
    // {"signal/coord/Enet/+0/s", "signal-scikit-enet", {{"debug", 0},{"intercept", 0},{"print level", 1}},
    //                                  {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
    //                                  {{"optim convergence tol",1.e-8f},{"lambda",0.19f},{"alpha",0.63f},{"optim progress factor", 1.0}},
    //                                  {{"optim convergence tol",1.e-8},{"lambda",0.19},{"alpha",0.63},{"optim progress factor", 1.0}},
    //                                  true, false
    //                                  },
    // 32 scikit-learn sparse signal example ELASTIC-NET
{  "ElNet/signal/coord/1/c", "signal-scikit-enet", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.17f},{"alpha",0.5f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.17},{"alpha",0.5},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 33 scikit-learn sparse signal example ELASTIC-NET
{  "ElNet/signal/coord/0/c", "signal-scikit-enet", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 150}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",5.e-6f},{"lambda",0.19f},{"alpha",0.63f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.19},{"alpha",0.63},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 34 scikit-learn sparse signal example ELASTIC-NET
{  "ElNet/signal/coord/0/n", "signal-scikit-enet", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 150}},
                                     {{"optim method", "coord"},{"scaling", "none"},{"print options", "yes"}},
                                     {{"optim convergence tol",5.e-6f},{"lambda",0.19f},{"alpha",0.63},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.19},{"alpha",0.63},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 35 Dual-gap check
{  "ElNet/dualgap/coord/0/a", "dualgap_elnet", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 300}},
                                     {{"optim method", "coord"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",1.0f},{"alpha",0.5f},{"optim dual gap tol", 1.e-6f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",1.0},{"alpha",0.5},{"optim dual gap tol", 1.e-6}},
                                     true, false, 1.0f, {5.0e-6f, 1.0e-9f}
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
                         testing::ValuesIn(linregParamElNet));
INSTANTIATE_TEST_SUITE_P(linregPosSuiteF, linregPosF,
                         testing::ValuesIn(linregParamElNet));
