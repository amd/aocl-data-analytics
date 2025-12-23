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
 * [D]  BFGS      L2          centering
 * [D]  BFGS      L2          none
 * [D]  BFGS      L2          scale only
 * [D]  BFGS      L2          standardize
 * [D]  Cholesky  L2          centering
 * [D]  Cholesky  L2          none
 * [D]  Cholesky  L2          scale only
 * [D]  Cholesky  L2          standardize
 * [D]  Coord     L2          centering
 * [D]  Coord     L2          none
 * [D]  Coord     L2          scale only
 * [D]  Coord     L2          standardize
 * [D]  SVD       L2          centering
 * [D]  SVD       L2          none
 * [D]  SVD       L2          scale only
 * [D]  SVD       L2          standardize
 * [D]  Sparse CG L2          centering
 * [D]  Sparse CG L2          none
 * [D]  Sparse CG L2          scale only
 * [D]  Sparse CG L2          standardize
 */
const linregParam linregParamRidge[] = {
    // 0
{  "Ridge/trivial/coord/1/z", "triviall2", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f},{"optim dual gap tol", infs}},
                                     {{"lambda",10.0},{"alpha",0.0},{"optim dual gap tol", infd},{"optim convergence tol",1.e-7}}
                                     },
    // 1
{  "Ridge/trivial/coord/0/z", "triviall2", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f},{"optim dual gap tol", 1.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}, {"optim convergence tol",1.e-7}, {"optim dual gap tol", 1.0}}
                                     },
    // 2 matches with Sklearn
{  "Ridge/trivial/coord/1/c", "triviall2unscl", {{"intercept", 1},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",10.0f*6.0f/5.05319f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", 0.3}},
                                     {{"lambda",10.0*6.0/5.05319},{"alpha",0.0},{"optim dual gap tol", 0.3}}
                                     },
    // 3 models y ~ X + 0, y ~ X + 1, Ridge, centering => NEED to scale manually lambda
    // scaling = centering needs to be used as scaling = "scaling only" so _unscl data needs to be used.
    // Also lambda needs to be pre-scaled since sy is set to 1.
    // Model has intercept so lambda is scaled by n * sd(y)*sqrt(n-1)/sqrt(n)
#ifndef NO_FORTRAN
{  "Ridge/trivial/bfgs/1/c", "triviall2unscl", {{"intercept", 1},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",10.0f*6.0f/5.05319f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/5.05319},{"alpha",0.0}}
                                     },
#endif
    // 4 matches with Sklearn. Model has no intercept so we scale lambda by norm2(y)*sqrt(nsamples) and also use _unscl data for the test.
{  "Ridge/trivial/coord/0/c", "triviall2unscl", {{"intercept", 0},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",10.0f*6.0f/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", 2.0e-2}},
                                     {{"lambda",10.0*6.0/11.7278},{"alpha",0.0}, {"optim convergence tol",1.e-7}, {"optim dual gap tol", 2.0e-2}}
                                     },
#ifndef NO_FORTRAN
    // 5 Model has no intercept so we scale lambda by norm2(y)*sqrt(nsamples) and also use _unscl data for the test.
{  "Ridge/trivial/bfgs/0/c", "triviall2unscl", {{"intercept", 0},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",10.0f*6.0f/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/11.7278},{"alpha",0.0}}
                                     },
#endif
                                     // 6
{  "Ridge/trivial/coord/0/n", "triviall2unscl", {{"intercept", 0},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"lambda",10.0f*6.0f/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", 0.02}},
                                     {{"lambda",10.0*6.0/11.7278},{"alpha",0.0}, {"optim convergence tol",1.e-7},{"optim dual gap tol", 0.02}}
                                     },
#ifndef NO_FORTRAN
                                     // 7
{  "Ridge/trivial/bfgs/0/n", "triviall2unscl", {{"intercept", 0},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",10.0f*6/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6/11.7278},{"alpha",0.0}}
                                     },
#endif
                                     // 8
{  "Ridge/trivial/coord/1/s", "triviall2unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",10.0f*6.0f/5.05319f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", 3.0}},
                                     {{"lambda",10.0*6.0/5.05319},{"alpha",0.0},{"optim dual gap tol", 3.0}}
                                     },
    // 9
{  "Ridge/trivial/coord/0/s", "triviall2unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",10.0f*6.0f/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f},{"optim dual gap tol", infs}},
                                     {{"lambda",10.0*6.0/11.7278},{"alpha",0.0}, {"optim convergence tol",1.e-7}, {"optim dual gap tol", infs}}
                                     },
    // 10
{  "Ridge/NormTab/coord/0/z", "glmnet-100x20l2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",22.0f},{"alpha",0.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-9},{"lambda",22.0},{"alpha",0.0},{"optim dual gap tol", infd}}
                                     },
    // 11
{  "Ridge/NormTab/coord/1/z", "glmnet-100x20l2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",22.0f},{"alpha",0.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-9},{"lambda",22.0},{"alpha",0.0},{"optim dual gap tol", infd}}
                                     },
    // 12
{  "Ridge/NormTab/coord/0/s", "glmnet-100x20l2unscl", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",22.0f*100.0f/10.3712f},{"alpha",0.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/10.3712},{"alpha",0.0},{"optim dual gap tol", infd}}
                                     },
    // 13
{  "Ridge/NormTab/coord/1/s", "glmnet-100x20l2unscl", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500},
                                     {"optim coord skip min", 4}, {"optim coord skip max", 25}, {"debug", 1}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",22.0f*100.0f/8.71399f},{"alpha",0.0f},{"optim dual gap tol", infs}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/8.71399},{"alpha",0.0},{"optim dual gap tol", infd}}
                                     },
    // 14
#ifndef NO_FORTRAN
{  "Ridge/trivial/bfgs/1/z", "triviall2", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 15
{  "Ridge/trivial/bfgs/0/z", "triviall2", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 16
{  "Ridge/trivial/bfgs/1/s", "triviall2unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7},{"lambda",10.0f*6.0f/(5.053189312f)},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-20},{"lambda",10.0*6.0/(5.053189312)},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 17
{  "Ridge/trivial/bfgs/0/s", "triviall2unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7},{"lambda",10.0f*6.0f/(11.72781594f)},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-20},{"lambda",10.0*6.0/(11.72781594)},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 18
{  "Ridge/NormTab/bfgs/0/z", "glmnet-100x20l2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 19
{  "Ridge/NormTab/bfgs/1/z", "glmnet-100x20l2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 20 - same set of problems 12-19 but scaling="scale only" (standardize=FALSE)
{  "NoReg/NormTab/bfgs/0/s", "glmnet-100x20unscl", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 10.0}},
                                     },
    // 21
{  "Ridge/NormTab/bfgs/0/s", "glmnet-100x20l2unscl", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f*100.0f/10.3711999994f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/10.3711999994},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 22
{  "Ridge/NormTab/bfgs/1/s", "glmnet-100x20l2unscl", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f*100.0f/8.71398621795f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/8.71398621795},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 23 Model has intercept so lambda is scaled by sd(y)*sqrt(n-1)/sqrt(n)
{  "Ridge/NormTab/bfgs/1/c", "glmnet-100x20l2unscl", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",100*22.0f/8.71398621795f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",100*22.0/8.71398621795},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 24 Model has no intercept so we scale lambda by norm2(y)/sqrt(n) and also use _unscl data for the test.
{  "Ridge/NormTab/bfgs/0/c", "glmnet-100x20l2unscl", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",100*22.0f/10.3711999994f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",100*22.0/10.3711999994},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // Compare with matrix-formulation (solved with normal equations)
    //
    // A \in m x n: m = nsamples
    // BOA = Assumptions of algo not satisfied
    // DP = different problem solved where l2 penalty is scaled
    // NA = Problem can't be solved (e.g. no centering with intercept)
    //
    // [A'A + lambda diag(I,0)] x = A'b <- INTERCEPT NO ASSUMPTIONS ON columns of A
    // ============================================================================
    // scaling type   lbfgs  svd  chol  cg  coord  lambda-fix
    // none             OK   NA    DP   DP   NA    same
    // centering        OK   OK    OK   OK   OK    same
    // scale only       OK   OK    OK   OK   OK    lambda/m * stdev(b)
    // standardize      DP   DP    DP   DP   DP    *  xs[i] /= 1 so different problem solved
    //
    // [A'A + lambda I] x = A'b <- NO INTERCEPT NO ASSUMPTIONS ON columns of A
    // =======================================================================
    // scaling type   lbfgs  svd  chol  cg  coord  lambda-fix
    // none             OK   NA    OK   OK   OK    same
    // centering        OK   OK    OK   OK   OK    same
    // scale only       OK   OK    OK   OK   OK    lambda/m * norm2(b)/sqrt(m)
    // standardize      DP   DP    DP   DP   DP    *  xs[i] /= 1 so different problem solved
    // =======================================================================
    // test only for none/centering and "scale only", standardize would solve a different problem
    // test group works for L-BFGS-B, SVD, CHOL, CG. For COORD ("standardize" is NOT valid - BOA)
    // 25 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/bfgs/0/n", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false,
                                  },
    // 26 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/bfgs/1/n", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 27 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/bfgs/0/c", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 28 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/bfgs/1/c", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 29 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2/bfgs/0/s", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 30 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2/bfgs/1/s", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
#endif
    // 31 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/svd/0/n", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "none"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false,
                                  },
    // [disabled 91] Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // Can't solve with intercept when scaling==none
    // {"NE7x2-l2+1/svd/n", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
    //                               {{"optim method", "svd"},{"scaling", "none"}},
    //                               {{"lambda",1.5f},{"alpha",0.0f}},
    //                               {{"lambda",1.5},{"alpha",0.0}},
    //                               true, false
    //                               },
    // 32 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/svd/0/c", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 33 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/svd/1/c", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 34 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2/svd/0/s", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 35 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2/svd/1/s", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 36 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/chol/0/n", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "none"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false,
                                  },
    // 37 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/chol/1/n", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "none"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 38 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/chol/0/c", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 39 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/chol/1/c", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 40 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2/chol/0/s", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 41 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2/chol/1/s", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 42 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/cg/0/n", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false,
                                  },
    // 43 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/cg/1/n", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 44 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/cg/0/c", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 45 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/cg/1/c", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 46 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2/cg/0/s", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 47 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2/cg/1/s", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 48 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2/coord/0/s", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim dual gap tol", 0.5}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim dual gap tol", 0.5}},
                                     true, false
                                     },
    // 49 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2/coord/1/s", "mtx_7x2",{{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim dual gap tol", 2.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim dual gap tol", 2.0}},
                                     true, false
                                     },
    // 50 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/coord/0/n", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "coord"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim dual gap tol", 0.02}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim dual gap tol", 0.02}},
                                  true, false,
                                  },
    // 51 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/coord/0/c", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "coord"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim dual gap tol", 0.02}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim dual gap tol", 0.02}},
                                  true, false
                                  },
    // 52 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
{  "Ridge/NE7x2/coord/1/c", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "coord"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim dual gap tol", 0.1}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim dual gap tol", 0.1}},
                                  true, false
                                  },
    // Compare with matrix-formulation (solved with normal equations)
    // test ALL none/centering/scale only/standardize
    //
    // A \in m x n: m = nsamples
    // BOA = Assumptions of algo not satisfied
    // DP = different problem solved where l2 penalty is scaled
    // NA = Problem can't be solved (e.g. no centering with intercept)
    //
    // [A'A + lambda diag(I,0)] x = A'b <- INTERCEPT
    // =============================================
    // A is such that for each column, ai, 1/nsamples sum[ai - mean(ai)]^2 = 1
    // scaling type   lbfgs  svd  chol  cg  coord  lambda-fix
    // none             OK   DP    DP   DP   NA    same
    // centering        OK   OK    OK   OK   OK    same
    // scale only       OK   OK    OK   OK   OK    lambda/m * stdev(b)
    // standardize      OK   DP    DP   DP   OK    lambda/m * stdev(b)
    // =============================================
    //
    // [A'A + lambda I] x = A'b <- NO INTERCEPT
    // ========================================
    // A is such that for each column, ai, 1/nsamples sum[ai - mean(ai)]^2 = 1
    // scaling type   lbfgs  svd  chol  cg  coord  lambda-fix
    // none             OK   OK    OK   OK   OK    same
    // centering        OK   OK    OK   OK   OK    same
    // scale only       OK   OK    OK   OK   OK    lambda/m * norm2(b)/sqrt(m)
    // standardize      OK   DP    DP   DP   OK    lambda/m * norm2(b)/sqrt(m)
    //
    // test group works for L-BFGS-B, SVD, CHOL, CG, and COORD
    // data: A is such that for each column, ai, 1/nsamples sum[ai - mean(ai)]^2 = 1
    // 53 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
#ifndef NO_FORTRAN
{  "Ridge/NE7x2P/bfgs/0/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false,
                                     },
    // 54 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/bfgs/1/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 55 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/bfgs/0/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 56 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/bfgs/1/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 57 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2P/bfgs/0/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 58 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2P/bfgs/1/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 59 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/svd/0/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "none"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false,
                                     },
#endif
    // [disabled: 116] Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // Can't solve with intercept when scaling==none
    // {"NE7x2P-l2+1/svd/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
    //                                  {{"optim method", "svd"},{"scaling", "none"}},
    //                                  {{"lambda",1.5f},{"alpha",0.0f}},
    //                                  {{"lambda",1.5},{"alpha",0.0}},
    //                                  true, false
    //                                  },
    // 60 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/svd/0/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 61 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/svd/1/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 62 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2P/svd/0/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 63 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2P/svd/1/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 64 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/chol/0/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "none"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false,
                                     },
    // 65 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/chol/1/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "none"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 66 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/chol/0/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 67 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/chol/1/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 68 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2P/chol/0/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 69 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2P/chol/1/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 70 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/cg/0/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false,
                                     },
    // 71 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/cg/1/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 72 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/cg/0/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 73 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/cg/1/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 74 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2P/cg/0/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 75 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2P/cg/1/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 76 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2P/coord/0/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim dual gap tol", 2.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim dual gap tol", 2.0}},
                                     true, false
                                     },
    // 77 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2P/coord/1/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim dual gap tol", 4.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim dual gap tol", 4.0}},
                                     true, false
                                     },
    // 78 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/coord/0/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim dual gap tol", 0.1}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim dual gap tol", 0.1}},
                                     true, false,
                                     },
    // 79 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/coord/0/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim dual gap tol", 0.1}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim dual gap tol", 0.1}},
                                     true, false
                                     },
    // 80 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
{  "Ridge/NE7x2P/coord/1/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim dual gap tol", 0.15}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim dual gap tol", 0.15}},
                                     true, false
                                     },
    // 81 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
#ifndef NO_FORTRAN
{  "Ridge/NE7x2P/bfgs/0/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-9},{"lambda",1.5/7.0*10.86771},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 82 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2P/bfgs/1/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-9},{"lambda",1.5/7.0*5.76230},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
#endif
    // 83 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2P/svd/0/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*10.86771},{"alpha",0.0}},
                                     true, false
                                     },
    // 84 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2P/svd/1/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "svd"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*5.76230},{"alpha",0.0}},
                                     true, false
                                     },
    // 85 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2P/chol/0/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "cholesky"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*10.86771},{"alpha",0.0}},
                                     true, false
                                     },
    // 86 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2P/chol/1/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "cholesky"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*5.76230},{"alpha",0.0}},
                                     true, false
                                     },
    // 87 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2P/cg/0/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*10.86771},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 88 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2P/cg/1/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*5.76230},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 89 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
{  "Ridge/NE7x2P/coord/0/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f},{"optim dual gap tol", 0.4}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*10.86771},{"alpha",0.0},{"optim dual gap tol", 0.4}},
                                     true, false
                                     },
    // 90 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
{  "Ridge/NE7x2P/coord/1/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f},{"optim dual gap tol", 1.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*5.76230},{"alpha",0.0},{"optim dual gap tol", 1.0}},
                                     true, false
                                     },
    // Compare all scalings of all solvers with sci-kit learn / glmnet (for elasticnet) output
    // =======================================================================================
    // OK - Pass, OK* - Pass with modification to the problem (either add small lambda or relax tolerance)
    // DP - Different problem, NA - Solver not applicable, F - Fail
    //
    // RIDGE
    // NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK    OK   NA
    // tall-thin      OK    OK    OK   OK    OK   NA
    // tall-fat       OK    OK    OK   OK    OK   NA
    //
    // INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    NA    NA   NA    NA   NA
    // tall-thin      OK    NA    OK   OK    NA   NA
    // tall-fat       OK    NA    NA   NA    NA   NA
    //
    // CENTERING =====================================
    // RIDGE (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK    OK   NA
    // tall-thin      OK    OK    OK   OK    OK   NA
    // tall-fat       OK    OK    OK   OK    OK   NA
    //
    // SCALE ONLY ===================================
    // RIDGE (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK   OK    NA
    // tall-thin      OK    OK    OK   OK   OK    NA
    // tall-fat       OK    OK    OK   OK   OK    NA
    //
    // STANDARDIZE =================================
    // (HERE DATA PASSED IS PRESCALED TO HAVE VARIANCE=1 AND
    // MEAN=0 IN EACH COLUMN AND OUTPUT IS BEING COMPARED TO GLMNET)
    // QR UNAVAIL BECAUSE PRESCALING UNDERDETERMINED PROBLEM MAKES MATRIX LOW-RANK
    //
    // RIDGE (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK   OK    NA
    // tall-thin      OK    OK    OK   OK   OK    NA
    // tall-fat       OK    OK    OK   OK   OK    NA
    // =============================================

    // Missing coord test for scaling=none/centering
    // 91
{  "Ridge/ShortFat/coord/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f},{"optim dual gap tol", 0.025}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0},{"optim dual gap tol", 0.025}},
                                     true, false},
    // 92
{  "Ridge/ShortFat/coord/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f},{"optim dual gap tol", 0.025}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0},{"optim dual gap tol", 0.025}},
                                     true, false},
    // 93
{  "Ridge/ShortFat/coord/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f},{"optim dual gap tol", 0.025}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0},{"optim dual gap tol", 0.025}},
                                     true, false},
    // 94 Ridge comparison with sklearn results
{  "Ridge/TallFat/coord/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f},{"optim dual gap tol", 1.0e-3}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0},{"optim dual gap tol", 1.0e-3}},
                                     true, false},
    // 95 Ridge comparison with sklearn results
{  "Ridge/TallFat/coord/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f},{"optim dual gap tol", 1.0e-3}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0},{"optim dual gap tol", 1.0e-3}},
                                     true, false},
    // 96 Ridge comparison with sklearn results
{  "Ridge/TallFat/coord/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f},{"optim dual gap tol", 1.0e-3}},
                                     {{"optim convergence tol", 1.e-8}, {"lambda", 0.5}, {"alpha", 0.0},{"optim dual gap tol", 1.0e-3}},
                                     true, false},
    // 97 Ridge comparison with sklearn results
{  "Ridge/TallThin/coord/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f},{"optim dual gap tol", 2.0e-3}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0},{"optim dual gap tol", 2.0e-3}},
                                     true, false},
    // 98 Ridge comparison with sklearn results
{  "Ridge/TallThin/coord/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f},{"optim dual gap tol", 2.0e-3}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0},{"optim dual gap tol", 2.0e-3}},
                                     true, false},
    // 99 Ridge comparison with sklearn results
{  "Ridge/TallThin/coord/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f},{"optim dual gap tol", 2.0e-3}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0},{"optim dual gap tol", 2.0e-3}},
                                     true, false},
    /* L2 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 100
#ifndef NO_FORTRAN
{  "Ridge/ShortFat/bfgs/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 101
{  "Ridge/ShortFat/svd/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 102
{  "Ridge/ShortFat/chol/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 103
{  "Ridge/ShortFat/cg/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 104
#ifndef NO_FORTRAN
{  "Ridge/TallThin/bfgs/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 105
{  "Ridge/TallThin/svd/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 106
{  "Ridge/TallThin/chol/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 107
{  "Ridge/TallThin/cg/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    /* TALL FAT */
    // 108
#ifndef NO_FORTRAN
{  "Ridge/TallFat/bfgs/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 109
{  "Ridge/TallFat/svd/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 110
{  "Ridge/TallFat/chol/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 111
{  "Ridge/TallFat/cg/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 112
#ifndef NO_FORTRAN
{  "Ridge/ShortFat/bfgs/1/n", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 113
{  "Ridge/TallThin/bfgs/1/n", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 114
{  "Ridge/TallThin/chol/1/n", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 115
{  "Ridge/TallThin/cg/1/n", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    /* TALL FAT */
    // 116
#ifndef NO_FORTRAN
{  "Ridge/TallFat/bfgs/1/n", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",1.0}},
                                     true, false
                                     },

    /* L2 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 117
{  "Ridge/ShortFat/bfgs/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 118
{  "Ridge/ShortFat/svd/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 119
{  "Ridge/ShortFat/chol/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 120
{  "Ridge/ShortFat/cg/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 121
#ifndef NO_FORTRAN
{  "Ridge/TallThin/bfgs/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 122
{  "Ridge/TallThin/svd/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 123
{  "Ridge/TallThin/chol/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 124
{  "Ridge/TallThin/cg/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    /* TALL FAT */
    // 125
#ifndef NO_FORTRAN
{  "Ridge/TallFat/bfgs/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 126
{  "Ridge/TallFat/svd/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 127
{  "Ridge/TallFat/chol/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 128
{  "Ridge/TallFat/cg/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 129
#ifndef NO_FORTRAN
{  "Ridge/ShortFat/bfgs/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 130
{  "Ridge/ShortFat/svd/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 131
{  "Ridge/ShortFat/chol/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 132
{  "Ridge/ShortFat/cg/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 133
#ifndef NO_FORTRAN
{  "Ridge/TallThin/bfgs/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 134
{  "Ridge/TallThin/svd/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 135
{  "Ridge/TallThin/chol/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 136
{  "Ridge/TallThin/cg/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    /* TALL FAT */
    // 137
#ifndef NO_FORTRAN
{  "Ridge/TallFat/bfgs/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 138
{  "Ridge/TallFat/svd/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 139
{  "Ridge/TallFat/chol/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 140
{  "Ridge/TallFat/cg/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* L2 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 141
#ifndef NO_FORTRAN
{  "Ridge/ShortFat/bfgs/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
#endif
    // 142
{  "Ridge/ShortFat/svd/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 143
{  "Ridge/ShortFat/chol/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 144
{  "Ridge/ShortFat/cg/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 145
{  "Ridge/ShortFat/coord/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim dual gap tol",0.5f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}, {"optim dual gap tol",0.5}},
                                     true, false
                                     },
    /* TALL THIN */
    // 146
#ifndef NO_FORTRAN
{  "Ridge/TallThin/bfgs/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 147
{  "Ridge/TallThin/svd/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 148
{  "Ridge/TallThin/chol/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 149
{  "Ridge/TallThin/cg/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    // 150
{  "Ridge/TallThin/coord/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 55.0}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 55.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 151
#ifndef NO_FORTRAN
{  "Ridge/TallFat/bfgs/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 152
{  "Ridge/TallFat/svd/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 153
{  "Ridge/TallFat/chol/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 154
{  "Ridge/TallFat/cg/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 155
{  "Ridge/TallFat/coord/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 6.0e-3}},
                                     {{"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 6.0e-3}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 156
#ifndef NO_FORTRAN
{  "Ridge/ShortFat/bfgs/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 157
{  "Ridge/ShortFat/svd/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 158
{  "Ridge/ShortFat/chol/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 159
{  "Ridge/ShortFat/cg/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 160
{  "Ridge/ShortFat/coord/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 0.5}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 0.5}},
                                     true, false
                                     },
    /* TALL THIN */
    // 161
#ifndef NO_FORTRAN
{  "Ridge/TallThin/bfgs/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",0.1f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false, 1.5f
                                     },
#endif
    // 162
{  "Ridge/TallThin/svd/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 163
{  "Ridge/TallThin/chol/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 164
{  "Ridge/TallThin/cg/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim convergence tol",1.e-7}},
                                     true, false
                                     },
    // 165
{  "Ridge/TallThin/coord/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 200.0}},
                                     {{"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 200.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 166
#ifndef NO_FORTRAN
{  "Ridge/TallFat/bfgs/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 167
{  "Ridge/TallFat/svd/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 168
{  "Ridge/TallFat/chol/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 169
{  "Ridge/TallFat/cg/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 170
{  "Ridge/TallFat/coord/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 6.0e-3}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 6.0e-3}},
                                     true, false
                                     },
    /* L2 TESTS */
    /* NO INTERCEPT */
    // 171 SHORT FAT
#ifndef NO_FORTRAN
{  "Ridge/ShortFat/bfgs/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 172
{  "Ridge/ShortFat/svd/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 173
{  "Ridge/ShortFat/chol/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 174
{  "Ridge/ShortFat/cg/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 175
{  "Ridge/ShortFat/coord/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 4.5e-1}},
                                     {{"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 4.5e-1}},
                                     true, false
                                     },
    // 176 TALL THIN
#ifndef NO_FORTRAN
{  "Ridge/TallThin/bfgs/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 100}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 177
{  "Ridge/TallThin/svd/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 178
{  "Ridge/TallThin/chol/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 179
{  "Ridge/TallThin/cg/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 180
{  "Ridge/TallThin/coord/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 35.0}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 35.0}},
                                     true, false
                                     },
    // 181 TALL FAT
#ifndef NO_FORTRAN
{  "Ridge/TallFat/bfgs/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",0.1}},
                                     true, false
                                     },
#endif
    // 182
{  "Ridge/TallFat/svd/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 183
{  "Ridge/TallFat/chol/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 184
{  "Ridge/TallFat/cg/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 185
{  "Ridge/TallFat/coord/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 0.5}},
                                     {{"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 0.5}},
                                     true, false
                                     },
    /* INTERCEPT */
    // 186 SHORT FAT
#ifndef NO_FORTRAN
{  "Ridge/ShortFat/bfgs/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 187
{  "Ridge/ShortFat/svd/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 188
{  "Ridge/ShortFat/chol/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 189
{  "Ridge/ShortFat/cg/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 190
{  "Ridge/ShortFat/coord/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 4.5e-1}},
                                     {{"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 4.5e-1}},
                                     true, false
                                     },
    // 191 TALL THIN
#ifndef NO_FORTRAN
{  "Ridge/TallThin/bfgs/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 100}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
#endif
    // 192
{  "Ridge/TallThin/svd/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 193
{  "Ridge/TallThin/chol/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 194
{  "Ridge/TallThin/cg/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 195
{  "Ridge/TallThin/coord/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 35.0}},
                                     {{"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 35.0}},
                                     true, false
                                     },
    // 196 TALL FAT
#ifndef NO_FORTRAN
{  "Ridge/TallFat/bfgs/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
#endif
    // 197
{  "Ridge/TallFat/svd/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 198
{  "Ridge/TallFat/chol/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 199
{  "Ridge/TallFat/cg/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 200
{  "Ridge/TallFat/coord/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f},{"optim dual gap tol", 0.5}},
                                     {{"lambda",0.5},{"alpha",0.0},{"optim dual gap tol", 0.5}},
                                     true, false
                                     },
    // 201 scikit-learn sparse signal example Ridge (GLMnet step to match sklearn)
{  "Ridge/signal/coord/1/s", "signal-scikit-l2", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",0.85f},{"alpha",0.0f},{"optim dual gap tol", 6.0}},
                                     {{"optim convergence tol",3.e-7},{"lambda",0.85},{"alpha",0.0},{"optim dual gap tol", 6.0}},
                                     true, false
                                     },
    // 202 scikit-learn sparse signal example Ridge
{  "Ridge/signal/coord/0/s", "signal-scikit-l2", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",0.85f},{"alpha",0.0f},{"optim dual gap tol", 6.0}},
                                     {{"optim convergence tol",3.e-7},{"lambda",0.85},{"alpha",0.0},{"optim dual gap tol", 6.0}},
                                     true, false
                                     },
    // 203 scikit-learn sparse signal example Ridge
{  "Ridge/signal/coord/1/c", "signal-scikit-l2", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",0.85f},{"alpha",0.0f},{"optim dual gap tol", 0.01}},
                                     {{"optim convergence tol",3.e-7},{"lambda",0.85},{"alpha",0.0},{"optim dual gap tol", 0.01}},
                                     true, false
                                     },
    // 204 scikit-learn sparse signal example Ridge
{  "Ridge/signal/coord/0/c", "signal-scikit-l2", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",0.85f},{"alpha",0.0f},{"optim dual gap tol", 0.01}},
                                     {{"optim convergence tol",3.e-7},{"lambda",0.85},{"alpha",0.0},{"optim dual gap tol", 0.01}},
                                     true, false
                                     },
    // 205 scikit-learn sparse signal example Ridge
{  "Ridge/signal/coord/0/n", "signal-scikit-l2", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "none"},{"print options", "yes"}},
                                     {{"optim convergence tol",3.e-7f},{"lambda",0.85f},{"alpha",0.0f},{"optim dual gap tol", 0.01}},
                                     {{"optim convergence tol",3.e-7},{"lambda",0.85},{"alpha",0.0},{"optim dual gap tol", 0.01}},
                                     true, false
                                     },
    // 206 Dual-gap check
{  "Ridge/dualgap/coord/0/a", "dualgap_ridge", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 300}},
                                     {{"optim method", "coord"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",1.e-7f},{"optim dual gap tol", 1.0e-1f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.5},{"alpha",1.e-7},{"optim dual gap tol", 1.e-6}},
                                     true, false, 1.0f, {5.0f, 1.0e-9f}
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
                         testing::ValuesIn(linregParamRidge));
INSTANTIATE_TEST_SUITE_P(linregPosSuiteF, linregPosF,
                         testing::ValuesIn(linregParamRidge));
