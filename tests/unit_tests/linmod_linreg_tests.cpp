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

#include "aoclda.h"
#include "da_handle.hpp"
#include "linmod_linreg.hpp"
#include "options.hpp"
#include "utest_utils.hpp"
#include "gtest/gtest.h"

typedef struct linregParam_t {
    std::string test_name; // name of the ctest test
    std::string data_name; // name of the files to read in
    std::vector<option_t<da_int>> iopts;
    std::vector<option_t<std::string>> sopts;
    std::vector<option_t<float>> fopts;
    std::vector<option_t<double>> dopts;
    // check the solution
    bool check_coeff{true};
    // check the prediction
    bool check_predict{true};
    // scale to pass to expected_precision<T>(T scale=1.0)
    float check_tol_scale{1.0f};
} linregParam;

// clang-format off
/*
 * Replicate table for intercept=yes|no
 * Done Solver Regularization Scaling
 * [D]  QR        NONE        none
 * [D]  QR        NONE        scale only
 * [D]  QR        NONE        standardize
 * [D]  BFGS      NONE        none
 * [D]  BFGS      NONE        scale only
 * [D]  BFGS      NONE        standardize
 * [D]  BFGS      L2          none
 * [D]  BFGS      L2          scale only
 * [D]  BFGS      L2          standardize
 * [ ]  Coord     NONE        none <-------------- NOT TESTED YET
 * [D]  Coord     NONE        standardization
 * [D]  Coord     NONE        scale only
 * [ ]  Coord     L1          none <-------------- NOT TESTED YET
 * [D]  Coord     L1          standardize
 * [D]  Coord     L1          scale only
 * [ ]  Coord     L2          none <-------------- NOT TESTED YET
 * [D]  Coord     L2          standardize
 * [D]  Coord     L2          scale only
 * [ ]  Coord     L1 + L2     none <-------------- NOT TESTED YET
 * [D]  Coord     L1 + L2     standardize
 * [D]  Coord     L1 + L2     scale only
 * [D]  BFGS      L2          centering
 * [D]  BFGS      L2          centering
 * [ ]  SVD      L2          none
 * [ ]  SVD      L2          scale only
 * [ ]  SVD      L2          standardize
 * [ ]  SVD      L2          centering
 * [ ]  Cholesky      L2          none
 * [ ]  Cholesky      L2          scale only
 * [ ]  Cholesky      L2          standardize
 * [ ]  Cholesky      L2          centering
 * [ ]  Cholesky      NONE          none
 * [ ]  Cholesky      NONE          scale only
 * [ ]  Cholesky      NONE          standardize
 * [ ]  Cholesky      NONE          centering
 * [ ]  Sparse CG      L2          none
 * [ ]  Sparse CG      L2          scale only
 * [ ]  Sparse CG      L2          standardize
 * [ ]  Sparse CG      L2          centering
 */
const linregParam linregParamPos[] = {
    // 0
    {"trivialNoint",      "trivial", {}, {}, {}, {}},
    // 1
    {"trivialNoint/z",      "trivial", {}, {{"scaling", "standardize"}}, {}, {}},
    // 2
    {"trivialNoint/s",      "trivial", {}, {{"scaling", "scale only"}}, {}, {}},
    // 3
    {"trivialNointLbfgs", "trivial", {{"print level", 5}}, {{"optim method", "lbfgs"}}, {}, {}},
    // 4
    {"trivialNointLbfgs", "trivial", {}, {{"optim method", "lbfgs"},{"scaling", "standardize"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 5
    {"trivialNointLbfgs", "trivial", {}, {{"optim method", "lbfgs"},{"scaling", "scale only"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 6
    {"trivialIntercept",  "trivial", {{"intercept", 1}}, {}, {}, {}},
    // 7
    {"trivialIntercept/z",  "trivial", {{"intercept", 1}}, {{"scaling", "standardize"}}, {}, {}},
    // 8 QR with intercept and scaling only
    {"trivialIntercept/s",  "trivial", {{"intercept", 1}}, {{"scaling", "scale only"}}, {}, {}},
    // 9
    {"trivialILbfgs",     "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"}}, {}, {}},
    // 10
    {"trivialILbfgs/z",     "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"},{"scaling", "standardize"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 11
    {"trivialILbfgs/s",     "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"},{"scaling", "scale only"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // Data and solution generated using R (glmnet_trivial.R)
    // 12
    {"CoordNoReg+1/z", "trivial",      {{"intercept", 1}, {"print level", 5}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 13
    {"CoordNoReg+0/z", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 14
    {"CoordL1Reg+1/z", "triviall1",    {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 15
    {"CoordL1Reg+0/z", "triviall1",    {{"intercept", 0},{"print level", 4}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 16
    {"CoordL2Reg+1/z", "triviall2", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 17
    {"CoordL2Reg+0/z", "triviall2", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 18 Code coverage for printing -> print level = 5
    {"CoordElastic+1/z", "trivialelnet",{{"intercept", 1},{"print level", 5}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",5.0f},{"alpha",0.8f}},
                                     {{"lambda",5.0},{"alpha",0.8}}
                                     },
    // 19
    {"CoordElastic+0/z", "trivialelnet",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",6.0f},{"alpha",0.9f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",6.0},{"alpha",0.9}},
                                     },
    // Data and solution generated using R (glmnet_trivial.R) (STANDARDIZED = FALSE, our scaling = "scale only")
    // 20
    {"CoordNoReg+1/s", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 21
    {"CoordNoReg+0/s", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 22
    {"CoordL1Reg+1/s", "triviall1unscl",    {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",2.f},{"alpha",1.0f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 23
    {"CoordL1Reg+0/s", "triviall1unscl",    {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",2.f},{"alpha",1.0f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 24
    {"CoordL2Reg+1/s", "triviall2unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",10.0f*6.0f/5.05319f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/5.05319},{"alpha",0.0}}
                                     },
    // 25
    {"CoordL2Reg+0/s", "triviall2unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",10.0f*6.0f/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/11.7278},{"alpha",0.0}}
//                                   {{"lambda",10.0f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
//                                   {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 26
    {"CoordElastic+1/s", "trivialelnetunscl",{{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",5.0f},{"alpha",0.8f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",5.0},{"alpha",0.8}}
                                     },
    // 27
    {"CoordElastic+0/s", "trivialelnetunscl",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",6.0f},{"alpha",0.9f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",6.0},{"alpha",0.9}}
                                     },
    // Data and solution generated using R (glmnet_driver.R)
    // 28
    {"NormTab+0/z", "glmnet-100x20",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 29
    {"NormTab+1/z", "glmnet-100x20",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 30
    {"NormTab-LASSO+0/z", "glmnet-100x20l1",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 31
    {"NormTab-LASSO+1/z", "glmnet-100x20l1",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 32
    {"NormTab-Ridge+0/z", "glmnet-100x20l2",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 33
    {"NormTab-Ridge+1/z", "glmnet-100x20l2",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 34
    {"NormTab-ElNet+0/z", "glmnet-100x20en",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-5f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 35
    {"NormTab-ElNet+1/z", "glmnet-100x20en",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 36 - same set of problems 12-19 but scaling="scale only" (standardize=FALSE)
    {"NormTab+0/s", "glmnet-100x20unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 37
    {"NormTab+1/s", "glmnet-100x20unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 38
    {"NormTab-LASSO+0/s", "glmnet-100x20l1unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 39
    {"NormTab-LASSO+1/s", "glmnet-100x20l1unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 40
    {"NormTab-Ridge+0/s", "glmnet-100x20l2unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f*100.0f/10.3712f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/10.3712},{"alpha",0.0}}
                                     },
    // 41
    {"NormTab-Ridge+1/s", "glmnet-100x20l2unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500},
                                     {"optim coord skip min", 4}, {"optim coord skip max", 25}, {"debug", 1}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f*100.0f/8.71399f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/8.71399},{"alpha",0.0}}
                                     },
    // 42
    {"NormTab-ElNet+0/s", "glmnet-100x20enunscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 43
    {"NormTab-ElNet+1/s", "glmnet-100x20enunscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 44 models y ~ X + 0, y ~ X + 1, no-reg OR Ridge, scaling only OR standardize
    {"LbfgsStdNoReg+1", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 45
    {"LbfgsStdNoReg+0", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 46
    {"LbfgsStdL2Reg+1", "triviall2", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 47
    {"LbfgsStdL2Reg+0", "triviall2", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 48
    {"LbfgsSclNoReg+1", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 49
    {"LbfgsSclNoReg+0", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 10000}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 50
    {"LbfgsSclL2Reg+1", "triviall2unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7},{"lambda",10.0f*6.0f/(5.053189312f)},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-20},{"lambda",10.0*6.0/(5.053189312)},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 51
    {"LbfgsSclL2Reg+0", "triviall2unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7},{"lambda",10.0f*6.0f/(11.72781594f)},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-20},{"lambda",10.0*6.0/(11.72781594)},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 52
    {"LbfgsStdNormTab+0", "glmnet-100x20",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-20},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 10.0}},
                                     },
    // 53
    {"LbfgsStdNormTab+1", "glmnet-100x20",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 10.0}},
                                     },
    // 54
    {"LbfgsStdNormTab-Ridge+0", "glmnet-100x20l2",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 55
    {"LbfgsStdNormTab-Ridge+1", "glmnet-100x20l2",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 56 - same set of problems 12-19 but scaling="scale only" (standardize=FALSE)
    {"LbfgsSclNormTab+0", "glmnet-100x20unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 10.0}},
                                     },
    // 57
    {"LbfgsSclNormTab+1", "glmnet-100x20unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.0f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     },
    // 58
    {"LbfgsSclNormTab-Ridge+0", "glmnet-100x20l2unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f*100.0f/10.3711999994f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/10.3711999994},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 59
    {"LbfgsSclNormTab-Ridge+1", "glmnet-100x20l2unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f*100.0f/8.71398621795f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/8.71398621795},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // same problems 44-59 solved with QR - selectinh only NOREG
    // 60 models y ~ X + 0, y ~ X + 1, no-reg, scaling only OR standardize
    {"QRStdNoReg+1", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 61
    {"QRStdNoReg+0", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 62
    {"QRSclNoReg+1", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 63
    {"QRSclNoReg+0", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 10000}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 64
    {"QRStdNormTab+0", "glmnet-100x20",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "qr"},{"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 65
    {"QRStdNormTab+1", "glmnet-100x20",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "qr"},{"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 66
    {"QRSclNormTab+0", "glmnet-100x20unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "qr"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 67
    {"QRSclNormTab+1", "glmnet-100x20unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "qr"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 68
    {"trivialNointLbfgsCent", "trivial", {{"intercept", 0}}, {{"optim method", "lbfgs"},{"scaling", "centering"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 69
    {"trivialIntLbfgsCent", "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"},{"scaling", "centering"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 70
    {"trivialNointQRCent", "trivial", {{"intercept", 0}}, {{"optim method", "qr"},{"scaling", "centering"}}, {}, {}},
    // 71
    {"trivialIntQRCent", "trivial", {{"intercept", 1}}, {{"optim method", "qr"},{"scaling", "centering"}}, {}, {}},
    // 72 models y ~ X + 0, y ~ X + 1, Ridge, centering => NEED to scale manually lambda
    // scaling = centering needs to be used as scaling = "scaling only" so _unscl data needs to be used.
    // Also lambda needs to be pre-scaled since sy is set to 1.
    // Model has intercept so lambda is scaled by n * sd(y)*sqrt(n-1)/sqrt(n)
    {"LbfgsCenL2Reg+1", "triviall2unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",6*10.0f/(5.053189312f)},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 10.0}},
                                     {{"lambda",6*10.0/(5.053189312)},{"alpha",0.0},{"optim convergence tol", 1.0e-9f},{"optim progress factor", 10.0}}
                                     },
    // 73 Model has no intercept so we scale lambda by norm2(y)*sqrt(nsamples) and also use _unscl data for the test.
    {"LbfgsCenL2Reg+0", "triviall2unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",6*10.0f/(11.72781594f)},{"alpha",0.0f}},
                                     {{"lambda",6*10.0/(11.72781594)},{"alpha",0.0}}
                                     },
    // 74 Model has intercept so lambda is scaled by sd(y)*sqrt(n-1)/sqrt(n)
    {"LbfgsCenNormTab-Ridge+1", "glmnet-100x20l2unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",100*22.0f/8.71398621795f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",100*22.0/8.71398621795},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 75 Model has no intercept so we scale lambda by norm2(y)/sqrt(n) and also use _unscl data for the test.
    {"LbfgsCenNormTab-Ridge+0", "glmnet-100x20l2unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",100*22.0f/10.3711999994f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",100*22.0/10.3711999994},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // Compare with matrix-formulation (solved with normal equations)
    //
    // [A'A + lambda diag(I,0)] x = A'b <- INTERCEPT NO ASSUMPTIONS ON columns of A
    // ============================================================================
    // scaling type   lbfgs  svd  chol  cg  coord  lambda-fix
    // none             OK   X     DP   DP   BOA   same
    // centering        OK   OK    OK   OK   BOA   same
    // scale only       OK   OK    OK   OK   OK    lambda/m * stdev(b)
    // standardize      DP   DP    DP   DP   DP    *  xs[i] /= 1 so different problem solved
    //
    // [A'A + lambda I] x = A'b <- NO INTERCEPT NO ASSUMPTIONS ON columns of A
    // =======================================================================
    // scaling type   lbfgs  svd  chol  cg  coord  lambda-fix
    // none             OK   X     OK   OK   BOA   same
    // centering        OK   OK    OK   OK   BOA   same
    // scale only       OK   OK    OK   OK   OK    lambda/m * norm2(b)/sqrt(m)
    // standardize      DP   DP    DP   DP   DP    *  xs[i] /= 1 so different problem solved
    // =======================================================================
    // test only for none/centering and "scale only", standardize would solve a different problem
    // test group works for L-BFGS-B, SVD, CHOL, CG. For COORD (only "scale only" is valid, otherwise
    // assumptions not met, so not testing)
    // 76 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/L/n", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false,
                                  },
    // 77 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/L/n", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 78 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/L/c", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 79 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/L/c", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 80 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/L/s", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 81 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/L/s", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 76 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/svd/n", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "none"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false,
                                  },
    // 77 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // Can't solve with intercept when scaling==none
    // {"NE7x2-l2+1/svd/n", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
    //                               {{"optim method", "svd"},{"scaling", "none"}},
    //                               {{"lambda",1.5f},{"alpha",0.0f}},
    //                               {{"lambda",1.5},{"alpha",0.0}},
    //                               true, false
    //                               },
    // 77 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/svd/c", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 78 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/svd/c", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 79 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/svd/s", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 80 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/svd/s", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 76 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/chol/n", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "none"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false,
                                  },
    // 77 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/chol/n", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "none"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 77 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/chol/c", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 78 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/chol/c", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 79 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/chol/s", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 80 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/chol/s", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 76 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/cg/n", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false,
                                  },
    // 77 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/cg/n", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 77 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/cg/c", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 78 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/cg/c", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 79 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/cg/s", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 80 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/cg/s", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 81 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/Coord/s", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 83 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/Coord/s", "mtx_7x2",{{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // Compare with matrix-formulation (solved with normal equations)
    // test ALL none/centering/scale only/standardize
    //
    // A \in m x n: m = nsamples
    // BOA = Assumptions of algo not satisfied
    // DP = different problem solved where l2 penalty is scaled
    // [A'A + lambda I] x = A'b <- NO INTERCEPT
    // ========================================
    // A is such that for each column, ai, 1/nsamples sum[ai - mean(ai)]^2 = 1
    // scaling type   lbfgs  svd  chol  cg  coord  lambda-fix
    // none             OK   OK    OK   OK   BOA   same
    // centering        OK   OK    OK   OK   BOA   same
    // scale only       OK   OK    OK   OK   OK    lambda/m * norm2(b)/sqrt(m)
    // standardize      OK   DP    DP   DP   OK    lambda/m * norm2(b)/sqrt(m)
    //
    // [A'A + lambda diag(I,0)] x = A'b <- INTERCEPT
    // =============================================
    // A is such that for each column, ai, 1/nsamples sum[ai - mean(ai)]^2 = 1
    // scaling type   lbfgs  svd  chol  cg  coord  lambda-fix
    // none             OK   DP    DP   DP   BOA   same
    // centering        OK   OK    OK   OK   BOA   same
    // scale only       OK   OK    OK   OK   OK    lambda/m * stdev(b)
    // standardize      OK   DP    DP   DP   OK    lambda/m * stdev(b)
    // =============================================
    // test group works for L-BFGS-B, SVD, CHOL, CG, and COORD
    // data: A is such that for each column, ai, 1/nsamples sum[ai - mean(ai)]^2 = 1
    // 84 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2P-l2+0/L/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false,
                                     },
    // 85 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2P-l2+1/L/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 86 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2P-l2+0/L/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 87 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2P-l2+1/L/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 88 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2P-l2+0/L/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 89 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2P-l2+1/L/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 83 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/svd/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "none"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false,
                                     },
    // 84 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // Can't solve with intercept when scaling==none
    // {"NE7x2-l2+1/svd/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
    //                                  {{"optim method", "svd"},{"scaling", "none"}},
    //                                  {{"lambda",1.5f},{"alpha",0.0f}},
    //                                  {{"lambda",1.5},{"alpha",0.0}},
    //                                  true, false
    //                                  },
    // 85 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/svd/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 86 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2-l2+1/svd/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 87 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/svd/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 88 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/svd/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 83 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/chol/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "none"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false,
                                     },
    // 84 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2-l2+1/chol/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "none"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 85 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/chol/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 86 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2-l2+1/chol/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 87 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/chol/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 88 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/chol/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 83 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/cg/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false,
                                     },
    // 84 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2-l2+1/cg/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 85 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/cg/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 86 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2-l2+1/cg/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 87 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/cg/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 88 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/cg/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 89 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2P-l2+0/Coord/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 91 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2P-l2+1/Coord/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 92 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2P-l2+0/L/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-9},{"lambda",1.5/7.0*10.86771},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 93 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2P-l2+1/L/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-9},{"lambda",1.5/7.0*5.76230},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 91 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/svd/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*10.86771},{"alpha",0.0}},
                                     true, false
                                     },
    // 92 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/svd/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "svd"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*5.76230},{"alpha",0.0}},
                                     true, false
                                     },
    // 91 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/chol/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "cholesky"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*10.86771},{"alpha",0.0}},
                                     true, false
                                     },
    // 92 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/chol/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "cholesky"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*5.76230},{"alpha",0.0}},
                                     true, false
                                     },
    // 91 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/cg/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*10.86771},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 92 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/cg/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*5.76230},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 93 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2P-l2+0/Coord/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*10.86771},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 95 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2P-l2+1/Coord/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*5.76230},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // Compare all scalings of all solvers with sci-kit learn / glmnet (for elasticnet) output
    // ========================================
    // OK - Pass, OK* - Pass with modification to the problem (either add small lambda or relax tolerance)
    // DP - Different problem, NA - Solver not applicable, F - Fail
    //
    // NONE SCALING (coord incompatible)
    // NORMAL
    // NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK   NA    OK
    // tall-thin      OK    OK    OK   OK   NA    OK
    // tall-fat       OK    OK    OK   OK   NA    OK
    // INTERCEPT (solvers unavail because strategy for intercept in under-det. is to center data)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK*   NA    NA   NA   NA    NA
    // tall-thin      OK    NA    OK   OK   NA    NA
    // tall-fat       OK*   NA    NA   NA   NA    NA
    //
    // LASSO - ALL NA (BOTH INTERCEPT AND NO INTERCEPT)
    //
    // RIDGE
    // NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK   NA    NA
    // tall-thin      OK    OK    OK   OK   NA    NA
    // tall-fat       OK    OK    OK   OK   NA    NA
    // INTERCEPT (same as normal regression)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    NA    NA   NA   NA    NA
    // tall-thin      OK    NA    OK   OK   NA    NA
    // tall-fat       OK    NA    NA   NA   NA    NA
    //
    // ELASTIC NET - ALL NA (BOTH INTERCEPT AND NO INTERCEPT)
    //
    // CENTERING (coord incompatible)
    // NORMAL
    // NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK   NA    OK
    // tall-thin      OK    OK    OK   OK   NA    OK
    // tall-fat       OK    OK    OK   OK   NA    OK
    //
    // INTERCEPT (singular in undetermined situation)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK*  OK   NA    NA
    // tall-thin      OK    OK    OK   OK   NA    OK
    // tall-fat       OK    OK    OK*  OK*  NA    NA
    //
    // LASSO - ALL NA (BOTH INTERCEPT AND NO INTERCEPT)
    //
    // RIDGE (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK   NA    NA
    // tall-thin      OK    OK    OK   OK   NA    NA
    // tall-fat       OK    OK    OK   OK   NA    NA
    //
    // ELASTIC NET - ALL NA (BOTH INTERCEPT AND NO INTERCEPT)
    //
    // SCALE ONLY
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
    // LASSO (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA   OK    NA
    // tall-thin      NA    NA    NA   NA   OK    NA
    // tall-fat       NA    NA    NA   NA   OK    NA
    //
    // RIDGE (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK   OK    NA
    // tall-thin      OK    OK    OK   OK   OK    NA
    // tall-fat       OK    OK    OK   OK   OK    NA
    //
    // ELASTIC NET (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA   NA    NA   OK    NA
    // tall-thin      NA    NA   NA    NA   OK    NA
    // tall-fat       NA    NA   NA    NA   OK    NA
    //
    // STANDARDISE (HERE DATA PASSED IS PRESCALED TO HAVE VARIANCE=1 AND MEAN=0 IN EACH COLUMN AND OUTPUT IS BEING COMPARED TO GLMNET)
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
    // LASSO (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA   OK    NA
    // tall-thin      NA    NA    NA   NA   OK    NA
    // tall-fat       NA    NA    NA   NA   OK    NA
    //
    // RIDGE (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK   OK    NA
    // tall-thin      OK    OK    OK   OK   OK    NA
    // tall-fat       OK    OK    OK   OK   OK    NA
    //
    // ELASTIC NET (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA   NA    NA   OK    NA
    // tall-thin      NA    NA   NA    NA   OK    NA
    // tall-fat       NA    NA   NA    NA   OK    NA
    // =============================================

    /* NONE SCALING */
    /* NORMAL TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/norm/lbfgs/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/svd/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/chol/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/cg/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/qr/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/norm/lbfgs/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/svd/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/chol/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/cg/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/qr/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/norm/lbfgs/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/norm/svd/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/chol/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/cg/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/qr/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // Add small lambda
    {"ShortFat/norm/lbfgs/1/n", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.001f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.00001},{"alpha",0.0}, {"optim progress factor",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/norm/lbfgs/1/n", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallThin/norm/chol/1/n", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/cg/1/n", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // Add a bit of lambda (a lot for float)
    {"TallFat/norm/lbfgs/1/n", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.1f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.00001},{"alpha",0.0}, {"optim progress factor",1.0}},
                                     true, false
                                     },

    /* L2 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L2/lbfgs/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"ShortFat/L2/svd/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/chol/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/cg/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L2/lbfgs/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/svd/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/chol/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/cg/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L2/lbfgs/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/L2/svd/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/chol/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/cg/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L2/lbfgs/1/n", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L2/lbfgs/1/n", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/chol/1/n", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/cg/1/n", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L2/lbfgs/1/n", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",1.0}},
                                     true, false
                                     },

    /* CENTERING */
    /* NORMAL TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/norm/lbfgs/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/svd/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/chol/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/cg/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/qr/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/norm/lbfgs/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/svd/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/chol/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/cg/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/qr/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/norm/lbfgs/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/norm/svd/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/chol/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/cg/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/qr/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    {"ShortFat/norm/lbfgs/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"ShortFat/norm/svd/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Bump lambda a bit to get around singular matrix
    {"ShortFat/norm/chol/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.00001f},{"alpha",0.0f}},
                                     {{"lambda",0.00001},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/cg/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/norm/lbfgs/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallThin/norm/svd/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/chol/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/cg/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/qr/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    {"TallFat/norm/lbfgs/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/norm/svd/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda to make it possible to factorise
    {"TallFat/norm/chol/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    {"TallFat/norm/cg/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* L2 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L2/lbfgs/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"ShortFat/L2/svd/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/chol/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/cg/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L2/lbfgs/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/svd/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/chol/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/cg/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L2/lbfgs/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/L2/svd/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/chol/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/cg/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L2/lbfgs/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"ShortFat/L2/svd/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/chol/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/cg/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L2/lbfgs/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/svd/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/chol/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/cg/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L2/lbfgs/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/L2/svd/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/chol/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/cg/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },

    /* SCALE ONLY */
    /* NORMAL TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/norm/lbfgs/0/s", "short_fat", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 600000}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/svd/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/chol/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/cg/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Hard to obtain sklearn result due to underdetermined system, need to add 0.0001 lambda and increase tolerance to 0.0021
    {"ShortFat/norm/coord/0/s", "short_fat", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 600000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda",0.0001f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-11}, {"lambda",0.0001},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false, 2.1
                                     },
    {"ShortFat/norm/qr/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/norm/lbfgs/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallThin/norm/svd/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/chol/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/cg/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/coord/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/qr/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/norm/lbfgs/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-9}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/norm/svd/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/chol/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/cg/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/coord/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/qr/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
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
        conditional number which makes the solution unstable and difficult to compare between each other */
    {"ShortFat/norm/lbfgs/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"ShortFat/norm/svd/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Bump lambda a bit to get around singular matrix
    {"ShortFat/norm/chol/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.00001f},{"alpha",0.0f}},
                                     {{"lambda",0.00001},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/cg/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add small lambda, bump max iter to 1,000,000 and set tolerance to 0.003
    {"ShortFat/norm/coord/1/s", "short_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 1000000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15f}, {"lambda",0.0001},{"alpha",0.0}},
                                     true, false, 3
                                     },
    /* TALL THIN */
    // Fail for single precision
    {"TallThin/norm/lbfgs/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",0.1f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",0.1}},
                                     true, false, 1.5f
                                     },
    {"TallThin/norm/svd/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/chol/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/cg/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/coord/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/qr/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    {"TallFat/norm/lbfgs/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/norm/svd/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda to make it possible to factorise
    {"TallFat/norm/chol/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    {"TallFat/norm/cg/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    {"TallFat/norm/coord/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 300000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.01f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.01},{"alpha",0.0}},
                                     true, false
                                     },
    /* L1 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L1/coord/0/s", "short_fatl1", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 100000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L1/coord/0/s", "tall_thinl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L1/coord/0/s", "tall_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L1/coord/1/s", "short_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L1/coord/1/s", "tall_thinl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L1/coord/1/s", "tall_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* L2 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L2/lbfgs/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"ShortFat/L2/svd/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/chol/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/cg/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/coord/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L2/lbfgs/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallThin/L2/svd/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/chol/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/cg/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/coord/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L2/lbfgs/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/L2/svd/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/chol/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/cg/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/coord/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L2/lbfgs/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"ShortFat/L2/svd/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/chol/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/cg/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/coord/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L2/lbfgs/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",0.1f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false, 1.5f
                                     },
    {"TallThin/L2/svd/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/chol/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/cg/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/coord/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L2/lbfgs/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/L2/svd/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/chol/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/cg/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/coord/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },

    /* ELASTIC NET TESTS */
    /* OUTPUT HERE IS COMPARED TO GLMNET INSTEAD OF SKLEARN */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L12/coord/0/s", "short_fatl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L12/coord/0/s", "tall_thinl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L12/coord/0/s", "tall_fatl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 100000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L12/coord/1/s", "short_fatl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L12/coord/1/s", "tall_thinl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L12/coord/1/s", "tall_fatl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },

    /* STANDARDISE (HERE WE COMPARING TO GLMNET OUTPUT) */
    /* NORMAL TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/norm/lbfgs/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda", 0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/svd/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add a bit of lambda
    {"ShortFat/norm/chol/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/cg/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add a bit of lambda
    {"ShortFat/norm/coord/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 600000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda",0.0001f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-11}, {"lambda",0.0001},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/norm/lbfgs/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/svd/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/chol/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/cg/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/coord/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/qr/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/norm/lbfgs/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/norm/svd/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    {"TallFat/norm/chol/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/norm/cg/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    {"TallFat/norm/coord/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-14f}, {"lambda",0.001},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    {"ShortFat/norm/lbfgs/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"ShortFat/norm/svd/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    {"ShortFat/norm/chol/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/norm/cg/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    {"ShortFat/norm/coord/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 300000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15f}, {"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/norm/lbfgs/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallThin/norm/svd/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/chol/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/cg/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/coord/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/norm/qr/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    {"TallFat/norm/lbfgs/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 300000}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/norm/svd/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    {"TallFat/norm/chol/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    {"TallFat/norm/cg/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    {"TallFat/norm/coord/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 300000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.01f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    /* L1 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L1/coord/0/z", "scl_short_fatl1", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 100000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L1/coord/0/z", "scl_tall_thinl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L1/coord/0/z", "scl_tall_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L1/coord/1/z", "scl_short_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L1/coord/1/z", "scl_tall_thinl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L1/coord/1/z", "scl_tall_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* L2 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L2/lbfgs/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"ShortFat/L2/svd/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/chol/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/cg/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/coord/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L2/lbfgs/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 100000}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/svd/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/chol/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/cg/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/coord/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L2/lbfgs/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",0.1}},
                                     true, false
                                     },
    {"TallFat/L2/svd/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/chol/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/cg/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/coord/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L2/lbfgs/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"ShortFat/L2/svd/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/chol/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/cg/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"ShortFat/L2/coord/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L2/lbfgs/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 100000}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/svd/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/chol/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/cg/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallThin/L2/coord/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L2/lbfgs/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    {"TallFat/L2/svd/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/chol/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/cg/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    {"TallFat/L2/coord/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* ELASTIC NET TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L12/coord/0/z", "scl_short_fatl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L12/coord/0/z", "scl_tall_thinl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L12/coord/0/z", "scl_tall_fatl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 100000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    {"ShortFat/L12/coord/1/z", "scl_short_fatl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL THIN */
    {"TallThin/L12/coord/1/z", "scl_tall_thinl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL FAT */
    {"TallFat/L12/coord/1/z", "scl_tall_fatl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },

    // 96 scikit-learn sparse signal example LASSO
    {"signal-l1+1/Coord/s", "signal-scikit", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.14f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.14},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, false
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
                                 (double)param.check_tol_scale);
}

// Positive tests with float type
TEST_P(linregPosF, Float) {
    const linregParam &param = GetParam();
    test_linreg_positive<float>(param.data_name, param.iopts, param.sopts, param.fopts,
                                param.check_coeff, param.check_predict,
                                (double)param.check_tol_scale);
}

// Test public option registry printing
// this is done in doc_tests.cpp

INSTANTIATE_TEST_SUITE_P(linregPosSuiteD, linregPosD, testing::ValuesIn(linregParamPos));
INSTANTIATE_TEST_SUITE_P(linregPosSuiteF, linregPosF, testing::ValuesIn(linregParamPos));
