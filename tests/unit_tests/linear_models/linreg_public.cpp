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

#include "aoclda.h"
#include "linreg_positive.hpp"
// #include "options.hpp"
#include "../utest_utils.hpp"
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
 * To keep comments with numbers in sync use reflow_numbers.sh
 *
 * Replicate table for intercept=yes|no
 * Done Solver Regularization Scaling
 * [D]  BFGS      L2          centering
 * [D]  BFGS      L2          none
 * [D]  BFGS      L2          scale only
 * [D]  BFGS      L2          standardize
 * [D]  BFGS      NONE        centering
 * [D]  BFGS      NONE        none
 * [D]  BFGS      NONE        scale only
 * [D]  BFGS      NONE        standardize
 * [D]  Cholesky  L2          centering
 * [D]  Cholesky  L2          none
 * [D]  Cholesky  L2          scale only
 * [D]  Cholesky  L2          standardize
 * [D]  Cholesky  NONE        centering
 * [D]  Cholesky  NONE        none
 * [D]  Cholesky  NONE        scale only
 * [D]  Cholesky  NONE        standardize
 * [D]  Coord     L1          centering
 * [D]  Coord     L1          none
 * [D]  Coord     L1          scale only
 * [D]  Coord     L1          standardize
 * [D]  Coord     L1 + L2     centering
 * [D]  Coord     L1 + L2     none
 * [D]  Coord     L1 + L2     scale only
 * [D]  Coord     L1 + L2     standardize
 * [D]  Coord     L2          centering
 * [D]  Coord     L2          none
 * [D]  Coord     L2          scale only
 * [D]  Coord     L2          standardize
 * [D]  Coord     NONE        centering
 * [D]  Coord     NONE        none
 * [D]  Coord     NONE        scale only
 * [D]  Coord     NONE        standardization
 * [D]  QR        NONE        none
 * [D]  QR        NONE        scale only
 * [D]  QR        NONE        standardize
 * [D]  SVD       L2          centering
 * [D]  SVD       L2          none
 * [D]  SVD       L2          scale only
 * [D]  SVD       L2          standardize
 * [D]  Sparse CG L2          centering
 * [D]  Sparse CG L2          none
 * [D]  Sparse CG L2          scale only
 * [D]  Sparse CG L2          standardize
 */
const linregParam linregParamPos[] = {
    // 0
    {"trivialNoint",      "trivial", {}, {}, {}, {}},
    // 1
    {"trivialNoint/z",      "trivial", {}, {{"scaling", "standardize"}}, {}, {}},
    // 2
    {"trivialNoint/s",      "trivial", {}, {{"scaling", "scale only"}}, {}, {}},
    // 3
    {"trivialNointLbfgs", "trivial", {{"print level", 1}}, {{"optim method", "lbfgs"}}, {}, {}},
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
    {"CoordNoReg+1/c", "trivial", {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 13
    {"CoordNoReg+0/c", "trivial", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-6f}},
                                     {{"lambda",0.0},{"alpha",0.5},{"optim convergence tol", 1.0e-6}}
                                     },
    // 14
    {"LbfgsbNoReg+0/c", "trivial", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "bfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-6f}},
                                     {{"lambda",0.0},{"alpha",0.5},{"optim convergence tol", 1.0e-6}}
                                     },
    // 15
    {"CoordNoReg+0/n", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-7f}},
                                     {{"lambda",0.0},{"alpha",0.5},{"optim convergence tol", 1.0e-7}}
                                     },
    // [disabled XX: scaling none with intercept assumes data is centered!]
    //{"CoordNoReg+1/n", "trivial",      {{"intercept", 1}, {"print level", 5}, {"optim iteration limit", 1800}},
    //                                 {{"optim method", "coord"}, {"scaling", "none"}},
    //                                 {{"lambda",0.0f},{"alpha",0.5f}},
    //                                 {{"lambda",0.0},{"alpha",0.5}}
    //                                 },
    // 16
    {"CoordNoReg+1/z", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 17
    {"CoordNoReg+0/z", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 18
    {"CoordL1Reg+1/z", "triviall1",    {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 19
    {"CoordL1Reg+0/z", "triviall1",    {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 20
    {"CoordL2Reg+1/z", "triviall2", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 21
    {"CoordL2Reg+0/z", "triviall2", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 22 Code coverage for printing -> print level = 5
    {"CoordElastic+1/z", "trivialelnet",{{"intercept", 1},{"print level", 5}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",5.0f},{"alpha",0.8f}},
                                     {{"lambda",5.0},{"alpha",0.8}}
                                     },
    // 23
    {"CoordElastic+0/z", "trivialelnet",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "standardize"}},
                                     {{"lambda",6.0f},{"alpha",0.9f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",6.0},{"alpha",0.9}},
                                     },
    // Data and solution generated using R (glmnet_trivial.R) (STANDARDIZED = FALSE, our scaling = "scale only")
    // 24
    {"CoordNoReg+1/s", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 25
    {"CoordNoReg+0/s", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 26
    {"CoordL1Reg+1/c", "triviall1unscl",    {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 1520}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 27
    {"CoordL1Reg+0/c", "triviall1unscl",    {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 1500}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // [Disabled: XX data is assumed to be centered]
    // {"CoordL1Reg+1/n", "triviall1unscl",    {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 1500}},
    //                                 {{"optim method", "coord"}, {"scaling", "none"}},
    //                                 {{"lambda",2.f},{"alpha",1.0f}},
    //                                 {{"lambda",2.},{"alpha",1.0}}
    //                                 },
    // 28
    {"CoordL1Reg+0/n", "triviall1unscl",    {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 1500}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"lambda",2.f},{"alpha",1.0f}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 29
    {"CoordL1Reg+1/s", "triviall1unscl",    {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 150}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",2.f},{"alpha",1.0f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 30
    {"CoordL1Reg+0/s", "triviall1unscl",    {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",2.f},{"alpha",1.0f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",2.},{"alpha",1.0}}
                                     },
    // 31 matches with Sklearn
    {"CoordL2Reg+1/c", "triviall2unscl", {{"intercept", 1},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",10.0f*6.0f/5.05319f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/5.05319},{"alpha",0.0}}
                                     },
    // 32
    {"LbfgsL2Reg+1/c", "triviall2unscl", {{"intercept", 1},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",10.0f*6.0f/5.05319f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/5.05319},{"alpha",0.0}}
                                     },
    // 33 matches with Sklearn
    {"CoordL2Reg+0/c", "triviall2unscl", {{"intercept", 0},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",10.0f*6.0f/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/11.7278},{"alpha",0.0}}
                                     },
    // 34
    {"LbfgsL2Reg+0/c", "triviall2unscl", {{"intercept", 0},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",10.0f*6.0f/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/11.7278},{"alpha",0.0}}
                                     },
    // 35
    {"CoordL2Reg+0/n", "triviall2unscl", {{"intercept", 0},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"lambda",10.0f*6.0f/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/11.7278},{"alpha",0.0}}
                                     },
    // 36
    {"LbfgsL2Reg+0/n", "triviall2unscl", {{"intercept", 0},{"print level", 3}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",10.0f*6/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6/11.7278},{"alpha",0.0}}
                                     },
    // 37
    {"CoordL2Reg+1/s", "triviall2unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",10.0f*6.0f/5.05319f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/5.05319},{"alpha",0.0}}
                                     },
    // 38
    {"CoordL2Reg+0/s", "triviall2unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",10.0f*6.0f/11.7278f},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f}},
                                     {{"lambda",10.0*6.0/11.7278},{"alpha",0.0}}
                                     },
    // 39
    {"CoordElastic+1/s", "trivialelnetunscl",{{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",5.0f},{"alpha",0.8f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",5.0},{"alpha",0.8}}
                                     },
    // 40
    {"CoordElastic+0/s", "trivialelnetunscl",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",6.0f},{"alpha",0.9f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",6.0},{"alpha",0.9}}
                                     },
    // 41
    {"CoordElastic+1/c", "trivialelnetunscl",{{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",4.8*10.0f/11.7391f},{"alpha",0.8f},{"optim convergence tol", 1.0e-7f},{"optim progress factor", 100.0}},
                                     {{"lambda",4.8*10.0/11.7391},{"alpha",0.8},{"optim convergence tol", 1.0e-9},{"optim progress factor", 100.0}}
                                     },
    // 42
    {"CoordElastic+0/c", "trivialelnetunscl",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"lambda",6.0f*10.0f/11.7278f},{"alpha",0.9f},{"optim convergence tol", 1.0e-7f},{"optim progress factor", 100.0}},
                                     {{"lambda",6.0*10.0/11.7278},{"alpha",0.9},{"optim convergence tol", 1.0e-9},{"optim progress factor", 100.0}}
                                     },
    // 43
    {"CoordElastic+0/n", "trivialelnetunscl",{{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"lambda",6.0f*10.0f/11.7278f},{"alpha",0.9f},{"optim convergence tol", 1.0e-7f},{"optim progress factor", 100.0}},
                                     {{"lambda",6.0*10.0/11.7278},{"alpha",0.9},{"optim convergence tol", 1.0e-9},{"optim progress factor", 100.0}}
                                     },
    // Data and solution generated using R (glmnet_driver.R)
    // 44
    {"NormTab+0/z", "glmnet-100x20",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 45
    {"NormTab+1/z", "glmnet-100x20",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 46
    {"NormTab-LASSO+0/z", "glmnet-100x20l1",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 47
    {"NormTab-LASSO+1/z", "glmnet-100x20l1",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 48
    {"NormTab-Ridge+0/z", "glmnet-100x20l2",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 49
    {"NormTab-Ridge+1/z", "glmnet-100x20l2",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 50
    {"NormTab-ElNet+0/z", "glmnet-100x20en",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-5f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 51
    {"NormTab-ElNet+1/z", "glmnet-100x20en",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-9},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 52 - same set of problems 12-19 but scaling="scale only" (standardize=FALSE)
    {"NormTab+0/s", "glmnet-100x20unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 53
    {"NormTab+1/s", "glmnet-100x20unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 54
    {"NormTab-LASSO+0/s", "glmnet-100x20l1unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 55
    {"NormTab-LASSO+1/s", "glmnet-100x20l1unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.0},{"alpha",1.0}}
                                     },
    // 56
    {"NormTab-Ridge+0/s", "glmnet-100x20l2unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f*100.0f/10.3712f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/10.3712},{"alpha",0.0}}
                                     },
    // 57
    {"NormTab-Ridge+1/s", "glmnet-100x20l2unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500},
                                     {"optim coord skip min", 4}, {"optim coord skip max", 25}, {"debug", 1}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f*100.0f/8.71399f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/8.71399},{"alpha",0.0}}
                                     },
    // 58
    {"NormTab-ElNet+0/s", "glmnet-100x20enunscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 59
    {"NormTab-ElNet+1/s", "glmnet-100x20enunscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",2.25f},{"alpha",0.8f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",2.25},{"alpha",0.8}}
                                     },
    // 60 models y ~ X + 0, y ~ X + 1, no-reg OR Ridge, scaling only OR standardize
    {"LbfgsStdNoReg+1", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 61
    {"LbfgsStdNoReg+0", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardize"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 62
    {"LbfgsStdL2Reg+1", "triviall2", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 63
    {"LbfgsStdL2Reg+0", "triviall2", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardize"}},
                                     {{"lambda",10.0f},{"alpha",0.0f}},
                                     {{"lambda",10.0},{"alpha",0.0}}
                                     },
    // 64
    {"LbfgsSclNoReg+1", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 65
    {"LbfgsSclNoReg+0", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 10000}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 66
    {"LbfgsSclL2Reg+1", "triviall2unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7},{"lambda",10.0f*6.0f/(5.053189312f)},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-20},{"lambda",10.0*6.0/(5.053189312)},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 67
    {"LbfgsSclL2Reg+0", "triviall2unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7},{"lambda",10.0f*6.0f/(11.72781594f)},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-20},{"lambda",10.0*6.0/(11.72781594)},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 68
    {"LbfgsStdNormTab+0", "glmnet-100x20",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-20},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 10.0}},
                                     },
    // 69
    {"LbfgsStdNormTab+1", "glmnet-100x20",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 10.0}},
                                     },
    // 70
    {"LbfgsStdNormTab-Ridge+0", "glmnet-100x20l2",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 71
    {"LbfgsStdNormTab-Ridge+1", "glmnet-100x20l2",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0},{"alpha",0.0}}
                                     },
    // 72 - same set of problems 12-19 but scaling="scale only" (standardize=FALSE)
    {"LbfgsSclNormTab+0", "glmnet-100x20unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 10.0}},
                                     },
    // 73
    {"LbfgsSclNormTab+1", "glmnet-100x20unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.0f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     },
    // 74
    {"LbfgsSclNormTab-Ridge+0", "glmnet-100x20l2unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f*100.0f/10.3711999994f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/10.3711999994},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 75
    {"LbfgsSclNormTab-Ridge+1", "glmnet-100x20l2unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",22.0f*100.0f/8.71398621795f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",22.0*100.0/8.71398621795},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // same problems solved with QR - selecting only NOREG
    // 76 models y ~ X + 0, y ~ X + 1, no-reg, scaling only OR standardize
    {"QRStdNoReg+1", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 77
    {"QRStdNoReg+0", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 78
    {"QRSclNoReg+1", "trivial",      {{"intercept", 1}, {"print level", 1}, {"optim iteration limit", 1800}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 79
    {"QRSclNoReg+0", "trivial",      {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 10000}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.5f}},
                                     {{"lambda",0.0},{"alpha",0.5}}
                                     },
    // 80
    {"QRStdNormTab+0", "glmnet-100x20",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "qr"},{"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 81
    {"QRStdNormTab+1", "glmnet-100x20",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "qr"},{"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 82
    {"QRSclNormTab+0", "glmnet-100x20unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "qr"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 83
    {"QRSclNormTab+1", "glmnet-100x20unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "qr"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",0.0f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15},{"lambda",0.0},{"alpha",1.0}},
                                     },
    // 84
    {"trivialNointLbfgsCent", "trivial", {{"intercept", 0}}, {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 85
    {"trivialIntLbfgsCent", "trivial", {{"intercept", 1}}, {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol", 1.0e-5f},{"optim progress factor", 100.0}}, {}},
    // 86
    {"trivialNointQRCent", "trivial", {{"intercept", 0}}, {{"optim method", "qr"},{"scaling", "centering"}}, {}, {}},
    // 87
    {"trivialIntQRCent", "trivial", {{"intercept", 1}}, {{"optim method", "qr"},{"scaling", "centering"}}, {}, {}},
    // 88 models y ~ X + 0, y ~ X + 1, Ridge, centering => NEED to scale manually lambda
    // scaling = centering needs to be used as scaling = "scaling only" so _unscl data needs to be used.
    // Also lambda needs to be pre-scaled since sy is set to 1.
    // Model has intercept so lambda is scaled by n * sd(y)*sqrt(n-1)/sqrt(n)
    {"LbfgsCenL2Reg+1", "triviall2unscl", {{"intercept", 1},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",6*10.0f/(5.053189312f)},{"alpha",0.0f},{"optim convergence tol", 1.0e-5f},{"optim progress factor", 10.0}},
                                     {{"lambda",6*10.0/(5.053189312)},{"alpha",0.0},{"optim convergence tol", 1.0e-9f},{"optim progress factor", 10.0}}
                                     },
    // 89 Model has no intercept so we scale lambda by norm2(y)*sqrt(nsamples) and also use _unscl data for the test.
    {"LbfgsCenL2Reg+0", "triviall2unscl", {{"intercept", 0},{"print level", 1}, {"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",6*10.0f/(11.72781594f)},{"alpha",0.0f}},
                                     {{"lambda",6*10.0/(11.72781594)},{"alpha",0.0}}
                                     },
    // 90 Model has intercept so lambda is scaled by sd(y)*sqrt(n-1)/sqrt(n)
    {"LbfgsCenNormTab-Ridge+1", "glmnet-100x20l2unscl",   {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",100*22.0f/8.71398621795f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",100*22.0/8.71398621795},{"alpha",0.0},{"optim progress factor", 10.0}}
                                     },
    // 91 Model has no intercept so we scale lambda by norm2(y)/sqrt(n) and also use _unscl data for the test.
    {"LbfgsCenNormTab-Ridge+0", "glmnet-100x20l2unscl",   {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
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
    // 92 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/lbfgsb/n", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false,
                                  },
    // 93 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/lbfgsb/n", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 94 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/lbfgsb/c", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 95 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/lbfgsb/c", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 96 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/lbfgsb/s", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 97 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/lbfgsb/s", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 98 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/svd/n", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
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
    // 99 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/svd/c", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 100 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/svd/c", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 101 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/svd/s", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 102 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/svd/s", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "svd"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 103 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/chol/n", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "none"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false,
                                  },
    // 104 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/chol/n", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "none"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 105 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/chol/c", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 106 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/chol/c", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "centering"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 107 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/chol/s", "mtx_7x2", {{"intercept", 0},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 108 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/chol/s", "mtx_7x2", {{"intercept", 1},{"print level", 1}},
                                  {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                  {{"lambda",1.5f},{"alpha",0.0f}},
                                  {{"lambda",1.5},{"alpha",0.0}},
                                  true, false
                                  },
    // 109 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/cg/n", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false,
                                  },
    // 110 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/cg/n", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 111 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/cg/c", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 112 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/cg/c", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 113 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/cg/s", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 114 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/cg/s", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 115 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/Coord/s", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 116 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/Coord/s", "mtx_7x2",{{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 117 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/Coord/n", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "coord"},{"scaling", "none"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false,
                                  },
    // 118 Solve x [A'*A + lambda*eye(n)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+0/Coord/c", "mtx_7x2", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "coord"},{"scaling", "centering"}},
                                  {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                  {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                  true, false
                                  },
    // 119 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [no prescaling of data]
    {"NE7x2-l2+1/Coord/c", "mtx_7x2", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                  {{"optim method", "coord"},{"scaling", "centering"}},
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
    // 120 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2P-l2+0/lbfgsb/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false,
                                     },
    // 121 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2P-l2+1/lbfgsb/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 122 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2P-l2+0/lbfgsb/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 123 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2P-l2+1/lbfgsb/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 124 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2P-l2+0/lbfgsb/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 125 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2P-l2+1/lbfgsb/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 126 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/svd/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "none"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false,
                                     },
    // [disabled: 116] Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // Can't solve with intercept when scaling==none
    // {"NE7x2-l2+1/svd/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
    //                                  {{"optim method", "svd"},{"scaling", "none"}},
    //                                  {{"lambda",1.5f},{"alpha",0.0f}},
    //                                  {{"lambda",1.5},{"alpha",0.0}},
    //                                  true, false
    //                                  },
    // 127 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/svd/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 128 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2-l2+1/svd/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 129 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/svd/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 130 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/svd/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 131 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/chol/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "none"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false,
                                     },
    // 132 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2-l2+1/chol/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "none"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 133 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/chol/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 134 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2-l2+1/chol/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "centering"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 135 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/chol/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 136 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/chol/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1}},
                                     {{"optim method", "cholesky"},{"scaling", "scale only"}},
                                     {{"lambda",1.5f},{"alpha",0.0f}},
                                     {{"lambda",1.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 137 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/cg/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false,
                                     },
    // 138 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2-l2+1/cg/n", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 139 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2-l2+0/cg/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 140 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2-l2+1/cg/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 141 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/cg/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 142 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/cg/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 143 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2P-l2+0/Coord/s", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 144 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2P-l2+1/Coord/s", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 145 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2P-l2+0/Coord/n", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false,
                                     },
    // 146 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    {"NE7x2P-l2+0/Coord/c", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 147 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    {"NE7x2P-l2+1/Coord/c", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 148 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2P-l2+0/lbfgsb/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-9},{"lambda",1.5/7.0*10.86771},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 149 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2P-l2+1/lbfgsb/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "lbfgs"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-6f},{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-9},{"lambda",1.5/7.0*5.76230},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 150 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/svd/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1}},
                                     {{"optim method", "svd"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*10.86771},{"alpha",0.0}},
                                     true, false
                                     },
    // 151 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/svd/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "svd"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*5.76230},{"alpha",0.0}},
                                     true, false
                                     },
    // 152 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/chol/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "cholesky"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*10.86771},{"alpha",0.0}},
                                     true, false
                                     },
    // 153 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/chol/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "cholesky"},{"scaling", "standardize"}},
                                     {{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f}},
                                     {{"lambda",1.5/7.0*5.76230},{"alpha",0.0}},
                                     true, false
                                     },
    // 154 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2-l2+0/cg/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*10.86771},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 155 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2-l2+1/cg/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "sparse_cg"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*5.76230},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 156 Solve x [A'*A + lambda*eye(n)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * norm2(y)/sqrt(m)
    {"NE7x2P-l2+0/Coord/z", "mtx_7x2_sd", {{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*10.86771f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*10.86771},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
    // 157 Solve x [A'*A + lambda*diag(1,1,0)] \ A'*b [data prescaled]
    // lambda is inflated to lambda/m * stdev(y)/sqrt(m)
    {"NE7x2P-l2+1/Coord/z", "mtx_7x2_sd", {{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "standardize"}},
                                     {{"optim convergence tol",1.e-7f},{"lambda",1.5f/7.0f*5.76230f},{"alpha",0.0f},{"optim progress factor", 10.0}},
                                     {{"optim convergence tol",1.e-10},{"lambda",1.5/7.0*5.76230},{"alpha",0.0},{"optim progress factor", 10.0}},
                                     true, false
                                     },
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
    // LASSO - ALL NA except for coord descent  with NO INTERCEPT
    // NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA    OK   NA
    // tall-thin      NA    NA    NA   NA    OK   NA
    // tall-fat       NA    NA    NA   NA    OK   NA
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
    // ELASTIC NET - ALL NA except for coord descent with NO INTERCEPT
    // Using sklearn refernece solution
    // NO INTERCEPT
    // INTERCEPT only for tall-thin DP
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA    OK   NA
    // tall-thin      NA    NA    NA   NA    OK   NA
    // tall-fat       NA    NA    NA   NA    OK   NA
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
    // LASSO - ALL NA except for coord descent
    // BOTH INTERCEPT AND NO INTERCEPT
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA    OK   NA
    // tall-thin      NA    NA    NA   NA    OK   NA
    // tall-fat       NA    NA    NA   NA    OK   NA
    //
    // RIDGE (BOTH INTERCEPT AND NO INTERCEPT)
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      OK    OK    OK   OK    OK   NA
    // tall-thin      OK    OK    OK   OK    OK   NA
    // tall-fat       OK    OK    OK   OK    OK   NA
    //
    // ELASTIC NET - ALL NA except for coord descent
    // BOTH INTERCEPT AND NO INTERCEPT
    // Using sklearn solution
    // matrix size   lbfgs  svd  chol  cg  coord  qr
    // short-fat      NA    NA    NA   NA    OK   NA
    // tall-thin      NA    NA    NA   NA    OK   NA
    // tall-fat       NA    NA    NA   NA    OK   NA
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

    // Missing coord test for scaling=none/centering
    // 158 Add some regularization to find minimal norm solution, relax tolerance
    {"ShortFat/norm/coord/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.005f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.005}, {"alpha", 0.0}},
                                     true, false, 200.0},
    // 159 Add some regularization to find minimal norm solution, relax tolerance
    {"ShortFat/norm/coord/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.005f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.005}, {"alpha", 0.0}},
                                     true, false, 150.0},
    // 160 Add some regularization to find minimal norm solution, relax tolerance
    {"ShortFat/norm/coord/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.005f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.005}, {"alpha", 0.0}},
                                     true, false, 150.0},
    // 161
    {"ShortFat/L1/coord/0/n", "short_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 162
    {"ShortFat/L1/coord/0/c", "short_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 163
    {"ShortFat/L1/coord/1/c", "short_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 164
    {"ShortFat/L2/coord/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0}},
                                     true, false},
    // 165
    {"ShortFat/L2/coord/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0}},
                                     true, false},
    // 166
    {"ShortFat/L2/coord/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0}},
                                     true, false},
    // 167 Elastic net comparison with sklearn results
    {"ShortFat/L12/coord/0/n", "short_fatl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 168 Elastic net comparison with sklearn results
    {"ShortFat/L12/coord/0/c", "short_fatl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 169 Elastic net comparison with sklearn results
    {"ShortFat/L12/coord/1/c", "short_fatl12_sk", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},

    // 170 NoReg comparison with sklearn results
    {"TallFat/norm/coord/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.0f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.0}, {"alpha", 0.0}},
                                     true, false},
    // 171 NoReg comparison with sklearn results
    {"TallFat/norm/coord/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.0f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.0}, {"alpha", 0.0}},
                                     true, false},
    // 172 NoReg comparison with sklearn results
    // Add a lot of lambda - also use relaxed tolerance
    {"TallFat/norm/coord/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-6f}, {"lambda", 0.1f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-6}, {"lambda", 0.1}, {"alpha", 0.0}},
                                     true, false, 20.0},

    // 173 LASSO comparison with sklearn results
    {"TallFat/L1/coord/0/n", "tall_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 174 LASSO comparison with sklearn results
    {"TallFat/L1/coord/0/c", "tall_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 175 LASSO comparison with sklearn results
    {"TallFat/L1/coord/1/c", "tall_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},

    // 176 Ridge comparison with sklearn results
    {"TallFat/L2/coord/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0}},
                                     true, false},
    // 177 Ridge comparison with sklearn results
    {"TallFat/L2/coord/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0}},
                                     true, false},
    // 178 Ridge comparison with sklearn results
    {"TallFat/L2/coord/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0}},
                                     true, false},

    // 179 Elastic net comparison with sklearn results
    {"TallFat/L12/coord/0/n", "tall_fatl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 180 Elastic net comparison with sklearn results
    {"TallFat/L12/coord/0/c", "tall_fatl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 181 Elastic net comparison with sklearn results
    {"TallFat/L12/coord/1/c", "tall_fatl12_sk", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},

    // 182 NoReg comparison with sklearn results
    {"TallThin/norm/coord/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.0f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.0}, {"alpha", 0.0}},
                                     true, false},
    // 183 NoReg comparison with sklearn results
    {"TallThin/norm/coord/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.0f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.0}, {"alpha", 0.0}},
                                     true, false},
    // 184 NoReg comparison with sklearn results
    {"TallThin/norm/coord/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.0f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.0}, {"alpha", 0.0}},
                                     true, false},

    // 185 LASSO comparison with sklearn results
    {"TallThin/L1/coord/0/n", "tall_thinl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 186 LASSO comparison with sklearn results
    {"TallThin/L1/coord/0/c", "tall_thinl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},
    // 187 LASSO comparison with sklearn results
    {"TallThin/L1/coord/1/c", "tall_thinl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 1.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 1.0}},
                                     true, false},

    // 188 Ridge comparison with sklearn results
    {"TallThin/L2/coord/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0}},
                                     true, false},
    // 189 Ridge comparison with sklearn results
    {"TallThin/L2/coord/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0}},
                                     true, false},
    // 190 Ridge comparison with sklearn results
    {"TallThin/L2/coord/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.5f}, {"alpha", 0.0f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.5}, {"alpha", 0.0}},
                                     true, false},

    // 191 Elastic net comparison with sklearn results
    {"TallThin/L12/coord/0/n", "tall_thinl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "none"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // DP Elastic net comparison with sklearn results
    // {"TallThin/L12/coord/1/n", "tall_thinl12_sk", {{"intercept", 1}, {"print level", 1}},
    // {{"optim method", "coord"}, {"scaling", "none"}}, {{"optim convergence tol",1.e-7f},{"lambda",0.3f},{"alpha",0.5f}},
    // {{"optim convergence tol",1.e-7},{"lambda",0.3},{"alpha",0.5}}, true, false },
    // 192 Elastic net comparison with sklearn results
    {"TallThin/L12/coord/0/c", "tall_thinl12_sk", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},
    // 193 Elastic net comparison with sklearn results
    {"TallThin/L12/coord/1/c", "tall_thinl12_sk", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "centering"}},
                                     {{"optim convergence tol", 1.e-7f}, {"lambda", 0.3f}, {"alpha", 0.5f}},
                                     {{"optim convergence tol", 1.e-7}, {"lambda", 0.3}, {"alpha", 0.5}},
                                     true, false},

    /* NONE SCALING */
    /* NORMAL TESTS = NoReg = No regularization*/
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 194
    {"ShortFat/norm/lbfgs/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 195
    {"ShortFat/norm/svd/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 196
    {"ShortFat/norm/chol/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 197
    {"ShortFat/norm/cg/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 198
    {"ShortFat/norm/qr/0/n", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 199
    {"TallThin/norm/lbfgs/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 200
    {"TallThin/norm/svd/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 201
    {"TallThin/norm/chol/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 202
    {"TallThin/norm/cg/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 203
    {"TallThin/norm/qr/0/n", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 204
    {"TallFat/norm/lbfgs/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 205
    {"TallFat/norm/svd/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 206
    {"TallFat/norm/chol/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 207
    {"TallFat/norm/cg/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 208
    {"TallFat/norm/qr/0/n", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // Add small lambda
    // 209
    {"ShortFat/norm/lbfgs/1/n", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.001f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.00001},{"alpha",0.0}, {"optim progress factor",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 210
    {"TallThin/norm/lbfgs/1/n", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 211
    {"TallThin/norm/chol/1/n", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 212
    {"TallThin/norm/cg/1/n", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // Add a bit of lambda (a lot for float)
    // 213
    {"TallFat/norm/lbfgs/1/n", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.1f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.00001},{"alpha",0.0}, {"optim progress factor",1.0}},
                                     true, false
                                     },

    /* L2 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 214
    {"ShortFat/L2/lbfgs/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 215
    {"ShortFat/L2/svd/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 216
    {"ShortFat/L2/chol/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 217
    {"ShortFat/L2/cg/0/n", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 218
    {"TallThin/L2/lbfgs/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 219
    {"TallThin/L2/svd/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 220
    {"TallThin/L2/chol/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 221
    {"TallThin/L2/cg/0/n", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 222
    {"TallFat/L2/lbfgs/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 223
    {"TallFat/L2/svd/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 224
    {"TallFat/L2/chol/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 225
    {"TallFat/L2/cg/0/n", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 226
    {"ShortFat/L2/lbfgs/1/n", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 227
    {"TallThin/L2/lbfgs/1/n", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 228
    {"TallThin/L2/chol/1/n", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 229
    {"TallThin/L2/cg/1/n", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "none"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 230
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
    // 231
    {"ShortFat/norm/lbfgs/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 232
    {"ShortFat/norm/svd/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 233
    {"ShortFat/norm/chol/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 234
    {"ShortFat/norm/cg/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 235
    {"ShortFat/norm/qr/0/c", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 236
    {"TallThin/norm/lbfgs/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 237
    {"TallThin/norm/svd/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 238
    {"TallThin/norm/chol/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 239
    {"TallThin/norm/cg/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 240
    {"TallThin/norm/qr/0/c", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 241
    {"TallFat/norm/lbfgs/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 242
    {"TallFat/norm/svd/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 243
    {"TallFat/norm/chol/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 244
    {"TallFat/norm/cg/0/c", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 245
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
    // 246
    {"ShortFat/norm/lbfgs/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 247
    {"ShortFat/norm/svd/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Bump lambda a bit to get around singular matrix
    // 248
    {"ShortFat/norm/chol/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.00001f},{"alpha",0.0f}},
                                     {{"lambda",0.00001},{"alpha",0.0}},
                                     true, false
                                     },
    // 249
    {"ShortFat/norm/cg/1/c", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 250
    {"TallThin/norm/lbfgs/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 251
    {"TallThin/norm/svd/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 252
    {"TallThin/norm/chol/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 253
    {"TallThin/norm/cg/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 254
    {"TallThin/norm/qr/1/c", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    // 255
    {"TallFat/norm/lbfgs/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 256
    {"TallFat/norm/svd/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda to make it possible to factorise
    // 257
    {"TallFat/norm/chol/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 258
    {"TallFat/norm/cg/1/c", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* L2 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 259
    {"ShortFat/L2/lbfgs/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 260
    {"ShortFat/L2/svd/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 261
    {"ShortFat/L2/chol/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 262
    {"ShortFat/L2/cg/0/c", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 263
    {"TallThin/L2/lbfgs/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 264
    {"TallThin/L2/svd/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 265
    {"TallThin/L2/chol/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 266
    {"TallThin/L2/cg/0/c", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 267
    {"TallFat/L2/lbfgs/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 268
    {"TallFat/L2/svd/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 269
    {"TallFat/L2/chol/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 270
    {"TallFat/L2/cg/0/c", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 271
    {"ShortFat/L2/lbfgs/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 272
    {"ShortFat/L2/svd/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 273
    {"ShortFat/L2/chol/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 274
    {"ShortFat/L2/cg/1/c", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 275
    {"TallThin/L2/lbfgs/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 276
    {"TallThin/L2/svd/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 277
    {"TallThin/L2/chol/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 278
    {"TallThin/L2/cg/1/c", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 279
    {"TallFat/L2/lbfgs/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "centering"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 280
    {"TallFat/L2/svd/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 281
    {"TallFat/L2/chol/1/c", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "centering"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 282
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
    // 283
    {"ShortFat/norm/lbfgs/0/s", "short_fat", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 600000}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 284
    {"ShortFat/norm/svd/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 285
    {"ShortFat/norm/chol/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 286
    {"ShortFat/norm/cg/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Hard to obtain sklearn result due to underdetermined system, need to add 0.0001 lambda and increase tolerance to 0.0021
    // 287
    {"ShortFat/norm/coord/0/s", "short_fat", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 600000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda",0.0001f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-11}, {"lambda",0.0001},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false, 2.1
                                     },
    // 288
    {"ShortFat/norm/qr/0/s", "short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 289
    {"TallThin/norm/lbfgs/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 290
    {"TallThin/norm/svd/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 291
    {"TallThin/norm/chol/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 292
    {"TallThin/norm/cg/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 293
    {"TallThin/norm/coord/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 294
    {"TallThin/norm/qr/0/s", "tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 295
    {"TallFat/norm/lbfgs/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-9}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 296
    {"TallFat/norm/svd/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 297
    {"TallFat/norm/chol/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 298
    {"TallFat/norm/cg/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 299
    {"TallFat/norm/coord/0/s", "tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 300
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
        condition number which makes the solution unstable and difficult to compare between each other */
    // 301
    {"ShortFat/norm/lbfgs/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 302
    {"ShortFat/norm/svd/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 303 Bump lambda a bit to get around singular matrix
    {"ShortFat/norm/chol/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.00001f},{"alpha",0.0f}},
                                     {{"lambda",0.00001},{"alpha",0.0}},
                                     true, false
                                     },
    // 304
    {"ShortFat/norm/cg/1/s", "short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 305 Add small lambda, bump max iter to 1,000,000 and set tolerance to 0.003
    {"ShortFat/norm/coord/1/s", "short_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 1000000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15f}, {"lambda",0.0001},{"alpha",0.0}},
                                     true, false, 3
                                     },
    /* TALL THIN */
    // 306 Fail for single precision
    {"TallThin/norm/lbfgs/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",0.1f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",0.1}},
                                     true, false, 1.5f
                                     },
    // 307
    {"TallThin/norm/svd/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 308
    {"TallThin/norm/chol/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 309
    {"TallThin/norm/cg/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 310
    {"TallThin/norm/coord/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 311
    {"TallThin/norm/qr/1/s", "tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    /* Tricky situation, calculating solution to undetermined system with intercept in unregularised case leads to dealing with matrix with very high
        conditional number which makes the solution unstable and difficult to compare between each other */
    // 312
    {"TallFat/norm/lbfgs/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 313
    {"TallFat/norm/svd/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda to make it possible to factorise
    // 314
    {"TallFat/norm/chol/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 315
    {"TallFat/norm/cg/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 316
    {"TallFat/norm/coord/1/s", "tall_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 300000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.01f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.01},{"alpha",0.0}},
                                     true, false
                                     },
    /* L1 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 317
    {"ShortFat/L1/coord/0/s", "short_fatl1", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 100000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 318
    {"TallThin/L1/coord/0/s", "tall_thinl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 319
    {"TallFat/L1/coord/0/s", "tall_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 320
    {"ShortFat/L1/coord/1/s", "short_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 321
    {"TallThin/L1/coord/1/s", "tall_thinl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 322
    {"TallFat/L1/coord/1/s", "tall_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* L2 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 323
    {"ShortFat/L2/lbfgs/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 324
    {"ShortFat/L2/svd/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 325
    {"ShortFat/L2/chol/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 326
    {"ShortFat/L2/cg/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 327
    {"ShortFat/L2/coord/0/s", "short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 328
    {"TallThin/L2/lbfgs/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 329
    {"TallThin/L2/svd/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 330
    {"TallThin/L2/chol/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 331
    {"TallThin/L2/cg/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 332
    {"TallThin/L2/coord/0/s", "tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 333
    {"TallFat/L2/lbfgs/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 334
    {"TallFat/L2/svd/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 335
    {"TallFat/L2/chol/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 336
    {"TallFat/L2/cg/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 337
    {"TallFat/L2/coord/0/s", "tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 338
    {"ShortFat/L2/lbfgs/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 339
    {"ShortFat/L2/svd/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 340
    {"ShortFat/L2/chol/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 341
    {"ShortFat/L2/cg/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 342
    {"ShortFat/L2/coord/1/s", "short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 343
    {"TallThin/L2/lbfgs/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",0.1f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false, 1.5f
                                     },
    // 344
    {"TallThin/L2/svd/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 345
    {"TallThin/L2/chol/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 346
    {"TallThin/L2/cg/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 347
    {"TallThin/L2/coord/1/s", "tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 348
    {"TallFat/L2/lbfgs/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 349
    {"TallFat/L2/svd/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 350
    {"TallFat/L2/chol/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 351
    {"TallFat/L2/cg/1/s", "tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "scale only"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 352
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
    // 353
    {"ShortFat/L12/coord/0/s", "short_fatl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL THIN */
    // 354
    {"TallThin/L12/coord/0/s", "tall_thinl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL FAT */
    // 355
    {"TallFat/L12/coord/0/s", "tall_fatl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 100000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 356
    {"ShortFat/L12/coord/1/s", "short_fatl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL THIN */
    // 357
    {"TallThin/L12/coord/1/s", "tall_thinl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* TALL FAT */
    // 358
    {"TallFat/L12/coord/1/s", "tall_fatl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "scale only"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },

    /* STANDARDIZE (HERE WE COMPARING TO GLMNET OUTPUT) */
    /* NORMAL TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 359
    {"ShortFat/norm/lbfgs/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda", 0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 360
    {"ShortFat/norm/svd/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 361 Add a bit of lambda
    {"ShortFat/norm/chol/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // 362
    {"ShortFat/norm/cg/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 363 Add a bit of lambda
    {"ShortFat/norm/coord/0/z", "scl_short_fat", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 600000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-8f}, {"lambda",0.0001f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-11}, {"lambda",0.0001},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 364
    {"TallThin/norm/lbfgs/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-14}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 365
    {"TallThin/norm/svd/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 366
    {"TallThin/norm/chol/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 367
    {"TallThin/norm/cg/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 368
    {"TallThin/norm/coord/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 369
    {"TallThin/norm/qr/0/z", "scl_tall_thin", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "qr"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 370
    {"TallFat/norm/lbfgs/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 371
    {"TallFat/norm/svd/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 372
    {"TallFat/norm/chol/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // 373
    {"TallFat/norm/cg/0/z", "scl_tall_fat", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 374
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
    // 375
    {"ShortFat/norm/lbfgs/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 376
    {"ShortFat/norm/svd/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 377 Add tiny bit of lambda
    {"ShortFat/norm/chol/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // 378
    {"ShortFat/norm/cg/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // Add tiny bit of lambda
    // 379
    {"ShortFat/norm/coord/1/z", "scl_short_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 300000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15f}, {"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 380
    {"TallThin/norm/lbfgs/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 381
    {"TallThin/norm/svd/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 382
    {"TallThin/norm/chol/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 383
    {"TallThin/norm/cg/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 384
    {"TallThin/norm/coord/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 385
    {"TallThin/norm/qr/1/z", "scl_tall_thin", {{"intercept", 1}, {"print level", 1}},
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
    // 386
    {"TallFat/norm/lbfgs/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 300000}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.0f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-13}, {"lambda",0.0},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 387
    {"TallFat/norm/svd/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0f},{"alpha",0.0f}},
                                     {{"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 388 Add tiny bit of lambda
    {"TallFat/norm/chol/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    // 389 Add tiny bit of lambda
    {"TallFat/norm/cg/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.0001f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0},{"alpha",0.0}},
                                     true, false
                                     },
    // 390 Add tiny bit of lambda
    {"TallFat/norm/coord/1/z", "scl_tall_fat", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 300000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-9f}, {"lambda",0.01f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-10}, {"lambda",0.0001},{"alpha",0.0}},
                                     true, false
                                     },
    /* L1 TESTS */
    /* NO INTERCEPT */
    /* SHORT FAT */
    // 391
    {"ShortFat/L1/coord/0/z", "scl_short_fatl1", {{"intercept", 0}, {"print level", 1}, {"optim iteration limit", 100000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 392
    {"TallThin/L1/coord/0/z", "scl_tall_thinl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 393
    {"TallFat/L1/coord/0/z", "scl_tall_fatl1", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    /* SHORT FAT */
    // 394
    {"ShortFat/L1/coord/1/z", "scl_short_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL THIN */
    // 395
    {"TallThin/L1/coord/1/z", "scl_tall_thinl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* TALL FAT */
    // 396
    {"TallFat/L1/coord/1/z", "scl_tall_fatl1", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",1.0f}},
                                     {{"lambda",0.3},{"alpha",1.0}},
                                     true, false
                                     },
    /* L2 TESTS */
    /* NO INTERCEPT */
    // 397 SHORT FAT
    {"ShortFat/L2/lbfgs/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 398
    {"ShortFat/L2/svd/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 399
    {"ShortFat/L2/chol/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 400
    {"ShortFat/L2/cg/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 401
    {"ShortFat/L2/coord/0/z", "scl_short_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 402 TALL THIN
    {"TallThin/L2/lbfgs/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 100000}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 403
    {"TallThin/L2/svd/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 404
    {"TallThin/L2/chol/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 405
    {"TallThin/L2/cg/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 406
    {"TallThin/L2/coord/0/z", "scl_tall_thinl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 407 TALL FAT
    {"TallFat/L2/lbfgs/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",1.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",0.1}},
                                     true, false
                                     },
    // 408
    {"TallFat/L2/svd/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 409
    {"TallFat/L2/chol/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 410
    {"TallFat/L2/cg/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 411
    {"TallFat/L2/coord/0/z", "scl_tall_fatl2", {{"intercept", 0}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* INTERCEPT */
    // 412 SHORT FAT
    {"ShortFat/L2/lbfgs/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 413
    {"ShortFat/L2/svd/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 414
    {"ShortFat/L2/chol/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 415
    {"ShortFat/L2/cg/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 416
    {"ShortFat/L2/coord/1/z", "scl_short_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 417 TALL THIN
    {"TallThin/L2/lbfgs/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 100000}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 418
    {"TallThin/L2/svd/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 419
    {"TallThin/L2/chol/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 420
    {"TallThin/L2/cg/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 421
    {"TallThin/L2/coord/1/z", "scl_tall_thinl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 422 TALL FAT
    {"TallFat/L2/lbfgs/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "lbfgs"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}, {"optim progress factor",10.0f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.5},{"alpha",0.0}, {"optim progress factor",10.0}},
                                     true, false
                                     },
    // 423
    {"TallFat/L2/svd/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "svd"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 424
    {"TallFat/L2/chol/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "cholesky"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 425
    {"TallFat/L2/cg/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "sparse_cg"}, {"scaling", "standardise"}},
                                     {{"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    // 426
    {"TallFat/L2/coord/1/z", "scl_tall_fatl2", {{"intercept", 1}, {"print level", 1}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.5f},{"alpha",0.0f}},
                                     {{"lambda",0.5},{"alpha",0.0}},
                                     true, false
                                     },
    /* ELASTIC NET TESTS */
    /* NO INTERCEPT */
    // 427 SHORT FAT
    {"ShortFat/L12/coord/0/z", "scl_short_fatl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    // 428 TALL THIN
    {"TallThin/L12/coord/0/z", "scl_tall_thinl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    // 429 TALL FAT
    {"TallFat/L12/coord/0/z", "scl_tall_fatl12", {{"intercept", 0}, {"print level", 1},{"optim iteration limit", 100000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    /* INTERCEPT */
    // 430 SHORT FAT
    {"ShortFat/L12/coord/1/z", "scl_short_fatl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-15}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    // 431 TALL THIN
    {"TallThin/L12/coord/1/z", "scl_tall_thinl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    // 432 TALL FAT
    {"TallFat/L12/coord/1/z", "scl_tall_fatl12", {{"intercept", 1}, {"print level", 1},{"optim iteration limit", 10000}},
                                     {{"optim method", "coord"}, {"scaling", "standardise"}},
                                     {{"optim convergence tol",1.e-7f}, {"lambda",0.3f},{"alpha",0.5f}},
                                     {{"optim convergence tol",1.e-7}, {"lambda",0.3},{"alpha",0.5}},
                                     true, false
                                     },
    // 433 scikit-learn sparse signal example LASSO GLMnet step to match sklearn)
    {"signal/coord/l1/+1/s", "signal-scikit", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.14f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.14},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, true
                                     },
    // 434 scikit-learn sparse signal example LASSO GLMnet step to match sklearn)
    {"signal/coord/l1/+0/s", "signal-scikit", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.21f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.21},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 435 scikit-learn sparse signal example LASSO
    {"signal/coord/l1/+1/c", "signal-scikit", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.14f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.14},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, true
                                     },
    // 436 scikit-learn sparse signal example LASSO
    {"signal/coord/l1/+0/c", "signal-scikit", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.21f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.21},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 437 scikit-learn sparse signal example LASSO
    {"signal/coord/l1/+0/n", "signal-scikit", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "none"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.21f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.21},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 438 scikit-learn sparse signal example Ridge (GLMnet step to match sklearn)
    {"signal/coord/l2/+1/s", "signal-scikit-l2", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.85f},{"alpha",0.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.85},{"alpha",0.0},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 439 scikit-learn sparse signal example Ridge - timeout on Windows
    //{"signal/coord/l2/+0/s", "signal-scikit-l2", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 1000}},
    //                                 {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
    //                                 {{"optim convergence tol",1.e-8f},{"lambda",0.85f},{"alpha",0.0f},{"optim progress factor", 1.0}},
    //                                 {{"optim convergence tol",1.e-8},{"lambda",0.85},{"alpha",0.0},{"optim progress factor", 1.0}},
    //                                 true, false
    //                                 },

    // 440 scikit-learn sparse signal example Ridge
    {"signal/coord/l2/+1/c", "signal-scikit-l2", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.85f},{"alpha",0.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.85},{"alpha",0.0},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 441 scikit-learn sparse signal example Ridge
    {"signal/coord/l2/+0/c", "signal-scikit-l2", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.85f},{"alpha",0.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.85},{"alpha",0.0},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 442 scikit-learn sparse signal example Ridge
    {"signal/coord/l2/+0/n", "signal-scikit-l2", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "none"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.85f},{"alpha",0.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.85},{"alpha",0.0},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // DP scikit-learn sparse signal example ELASTIC-NET (GLMNet regularization path does not match with sklearn)
    // {"signal/coord/Enet/+1/s", "signal-scikit-enet", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 1000}},
    //                                  {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
    //                                  {{"optim convergence tol",1.e-8f},{"lambda",0.17f},{"alpha",0.5f},{"optim progress factor", 1.0}},
    //                                  {{"optim convergence tol",1.e-8},{"lambda",0.17},{"alpha",0.5},{"optim progress factor", 1.0}},
    //                                  true, false
    //                                  },
    // DP scikit-learn sparse signal example ELASTIC-NET (GLMNet regularization path does not match with sklearn)
    // {"signal/coord/Enet/+0/s", "signal-scikit-enet", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 1000}},
    //                                  {{"optim method", "coord"},{"scaling", "scale only"},{"print options", "yes"}},
    //                                  {{"optim convergence tol",1.e-8f},{"lambda",0.19f},{"alpha",0.63f},{"optim progress factor", 1.0}},
    //                                  {{"optim convergence tol",1.e-8},{"lambda",0.19},{"alpha",0.63},{"optim progress factor", 1.0}},
    //                                  true, false
    //                                  },
    // 443 scikit-learn sparse signal example ELASTIC-NET
    {"signal/coord/Enet/+1/c", "signal-scikit-enet", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.17f},{"alpha",0.5f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.17},{"alpha",0.5},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 444 scikit-learn sparse signal example ELASTIC-NET
    {"signal/coord/Enet/+0/c", "signal-scikit-enet", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "centering"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.19f},{"alpha",0.63f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.19},{"alpha",0.63},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 445 scikit-learn sparse signal example ELASTIC-NET
    {"signal/coord/Enet/+0/n", "signal-scikit-enet", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 1000}},
                                     {{"optim method", "coord"},{"scaling", "none"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.19f},{"alpha",0.63},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.19},{"alpha",0.63},{"optim progress factor", 1.0}},
                                     true, false
                                     },
    // 446 scikit-learn sparse signal example LASSO to test scaling=AUTO with intercept
    {"signal/coord/l1/+1/c", "signal-scikit", {{"debug", 0},{"intercept", 1},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "auto"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.14f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.14},{"alpha",1.0},{"optim progress factor", 1.0}},
                                     true, true
                                     },
    // 447 scikit-learn sparse signal example LASSO to test scaling=AUTO with no intercept
    {"signal/coord/l1/+0/c", "signal-scikit", {{"debug", 0},{"intercept", 0},{"print level", 1},{"optim iteration limit", 500}},
                                     {{"optim method", "coord"},{"scaling", "auto"},{"print options", "yes"}},
                                     {{"optim convergence tol",1.e-8f},{"lambda",0.21f},{"alpha",1.0f},{"optim progress factor", 1.0}},
                                     {{"optim convergence tol",1.e-8},{"lambda",0.21},{"alpha",1.0},{"optim progress factor", 1.0}},
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
                                (float)param.check_tol_scale);
}

// Test public option registry printing
// this is done in doc_tests.cpp

INSTANTIATE_TEST_SUITE_P(linregPosSuiteD, linregPosD, testing::ValuesIn(linregParamPos));
INSTANTIATE_TEST_SUITE_P(linregPosSuiteF, linregPosF, testing::ValuesIn(linregParamPos));
