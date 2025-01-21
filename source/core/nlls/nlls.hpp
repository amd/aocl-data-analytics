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
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "macros.h"
#include "optimization.hpp"
#include "options.hpp"
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

/* Nonlinear Least Square Model and solver
 *
 * Solve the problem   minimize   F(x) = 1/2 \sum_{i=0}^{n_res-1} ri(x)^2_W + sigma/p ||x||_2^p
 *                   x \in R^n_coef
 * where
 *  * ri() are the model residuals
 *  * sigma > 0, p=2,3 are the regularization hyperparams
 *
 */

namespace da_optim {
template <typename T> class da_optimization;
}

namespace ARCH {

namespace da_nlls {

using namespace da_optim;

template <typename T> class nlls : public da_optimization<T> {
  private:
  public:
    // Constructor
    nlls(da_status &status, da_errors::da_error_t &err);

    // da_status define_residuals(da_int n_coef, da_int n_res);
    da_status define_callbacks(resfun_t<T> resfun, resgrd_t<T> resgrd, reshes_t<T> reshes,
                               reshp_t<T> reshp);
    da_status fit(da_int n_coef, T *coef, void *udata);
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
};

} // namespace da_nlls

} // namespace ARCH
