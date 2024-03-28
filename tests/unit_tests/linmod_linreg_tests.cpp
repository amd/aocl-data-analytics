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
 * [W]  BFGS      L2          centering
 * [W]  BFGS      L2          centering
 */
const linregParam linregParamPos[] = {
    // 0
    {"trivialNoint",      "trivial", {}, {}, {}, {}},
    // 1
    {"trivialNoint",      "trivial", {}, {{"scaling", "standardize"}}, {}, {}},
    // 2
    {"trivialNoint",      "trivial", {}, {{"scaling", "scale only"}}, {}, {}},
    // 3
    {"trivialNointLbfgs", "trivial", {}, {{"optim method", "lbfgs"}}, {}, {}},
    // 4
    {"trivialNointLbfgs", "trivial", {}, {{"optim method", "lbfgs"},{"scaling", "standardize"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 5
    {"trivialNointLbfgs", "trivial", {}, {{"optim method", "lbfgs"},{"scaling", "scale only"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 6
    {"trivialIntercept",  "trivial", {{"intercept", 1}}, {}, {}, {}},
    // 7
    {"trivialIntercept",  "trivial", {{"intercept", 1}}, {{"scaling", "standardize"}}, {}, {}},
    // 8 QR with intercept and scaling only
    {"trivialIntercept",  "trivial", {{"intercept", 1}}, {{"scaling", "scale only"}}, {}, {}},
    // 9
    {"trivialILbfgs",     "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"}}, {}, {}},
    // 10
    {"trivialILbfgs",     "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"},{"scaling", "standardize"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 11
    {"trivialILbfgs",     "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"},{"scaling", "scale only"}}, {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // Data and solution generated using R (glmnet_trivial.R)
    // 12
    {"CoordNoReg+1", "trivial",      {{"intercept", 1}, {"print level", 5}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 13
    {"CoordNoReg+0", "trivial",      {{"intercept", 0}, {"print level", 5}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 14
    {"CoordL1Reg+1", "triviall1",    {{"intercept", 1},{"print level", 5}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 15
    {"CoordL1Reg+0", "triviall1",    {{"intercept", 0},{"print level", 5}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 16
    {"CoordL2Reg+1", "triviall2", {{"intercept", 1},{"print level", 5}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 17
    {"CoordL2Reg+0", "triviall2", {{"intercept", 0},{"print level", 5}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 18
    {"CoordElastic+1", "trivialelnet",{{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}},
                                     {{"lambda",5.0f},{"alpha",0.8f}},
                                     {{"lambda",5.0},{"alpha",0.8}}
                                     },
    // 19
    {"CoordElastic+0", "trivialelnet",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}},
                                     {{"lambda",6.0f},{"alpha",0.9f}},
                                     {{"lambda",6.0},{"alpha",0.9}},
                                     },
    // Data and solution generated using R (glmnet_trivial.R) (STANDARDIZED = FALSE, our scaling = "scale only")
    // 20
    {"CoordNoReg+1", "trivial",      {{"intercept", 1}, {"print level", 5}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 21
    {"CoordNoReg+0", "trivial",      {{"intercept", 0}, {"print level", 5}, {"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 22
    {"CoordL1Reg+1", "triviall1unscl",    {{"intercept", 1},{"print level", 5}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 23
    {"CoordL1Reg+0", "triviall1unscl",    {{"intercept", 0},{"print level", 5}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",2.f},{"alpha",1.0f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 24
    {"CoordL2Reg+1", "triviall2unscl", {{"intercept", 1},{"print level", 5}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 25
    {"CoordL2Reg+0", "triviall2unscl", {{"intercept", 0},{"print level", 5}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 26
    {"CoordElastic+1", "trivialelnetunscl",{{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",5.0f},{"alpha",0.8f}},
                                     {{"lambda",5.0},{"alpha",0.8}}
                                     },
    // 27
    {"CoordElastic+0", "trivialelnetunscl",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",6.0f},{"alpha",0.9f}},
                                     {{"lambda",6.0},{"alpha",0.9}}
                                     },
    // Data and solution generated using R (glmnet_driver.R)
    // 28
    {"NormTab+0", "glmnet-100x20",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 29
    {"NormTab+1", "glmnet-100x20",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 30
    {"NormTab-LASSO+0", "glmnet-100x20l1",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 31
    {"NormTab-LASSO+1", "glmnet-100x20l1",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 32
    {"NormTab-Ridge+0", "glmnet-100x20l2",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "auto"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 33
    {"NormTab-Ridge+1", "glmnet-100x20l2",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "auto"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 34
    {"NormTab-ElNet+0", "glmnet-100x20en",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "auto"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 35
    {"NormTab-ElNet+1", "glmnet-100x20en",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "auto"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 36 - same set of problems 12-19 but scaling="scale only" (standardize=FALSE)
    {"NormTab+0", "glmnet-100x20unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 37
    {"NormTab+1", "glmnet-100x20unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 38
    {"NormTab-LASSO+0", "glmnet-100x20l1unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 39
    {"NormTab-LASSO+1", "glmnet-100x20l1unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 40
    {"NormTab-Ridge+0", "glmnet-100x20l2unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 41
    {"NormTab-Ridge+1", "glmnet-100x20l2unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 42
    {"NormTab-ElNet+0", "glmnet-100x20enunscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 43
    {"NormTab-ElNet+1", "glmnet-100x20enunscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
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
    {"LbfgsStdL2Reg+1", "triviall2", {{"intercept", 1},{"print level", 5}, {"optim iteration limit", 500}},
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
                                     {{"optim convergence tol",1.e-7},{"lambda",10.0f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-20},{"lambda",10.0},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 51
    {"LbfgsSclL2Reg+0", "triviall2unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7},{"lambda",10.0f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-20},{"lambda",10.0},{"alpha",0.0},{"optim progress factor", 10.0}}
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
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 59
    {"LbfgsSclNormTab-Ridge+1", "glmnet-100x20l2unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0},{"optim progress factor", 10.0}}
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
    // Model has intercept so lambda is scaled by sd(y)*sqrt(n-1)/sqrt(n)
    {"LbfgsCenL2Reg+1", "triviall2unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",10.0f/5.053189312f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 10.0}},
                                     {{"lambda",10.0/5.053189312},{"alpha",0.0},{"optim convergence tol", 1.0e-9f},{"optim progress factor", 10.0}}
                                     },
    // 73 Model has no intercept so we scale lambda by norm2(y)*sqrt(nsamples) and also use _unscl data for the test.
    {"LbfgsCenL2Reg+0", "triviall2unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",10.0f/11.72781594f},{"alpha",0.0f}},
                                     {{"lambda",10.0/11.72781594},{"alpha",0.0}}
                                     },
    // 74 Model has intercept so lambda is scaled by sd(y)*sqrt(n-1)/sqrt(n)
    {"LbfgsCenNormTab-Ridge+1", "glmnet-100x20l2unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f/8.71398621795f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0/8.71398621795},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 75 Model has no intercept so we scale lambda by norm2(y)/sqrt(n) and also use _unscl data for the test.
    {"LbfgsCenNormTab-Ridge+0", "glmnet-100x20l2unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f/10.3711999994f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0/10.3711999994},{"alpha",0.0},{"optim progress factor", 10.0}}
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
