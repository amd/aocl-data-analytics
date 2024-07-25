/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LINMOD_TYPES_HPP
#define LINMOD_TYPES_HPP

#include "optim_types.hpp"

namespace da_linmod {
enum linmod_method {
    undefined = da_optim::solvers::solver_undefined,
    lbfgsb = da_optim::solvers::solver_lbfgsb,
    coord = da_optim::solvers::solver_coord,
    svd = 30,
    cholesky = 31,
    cg = 32,
    qr = 33
};
// static struct see if a method is iterative
struct linmod_method_type {
    static bool is_iterative(linmod_method mid) {
        return mid == linmod_method::lbfgsb || mid == linmod_method::coord || mid == cg;
    }
};

/* type of scaling to perform */
enum scaling_t : da_int {
    none = 0 /* must be zero */,
    automatic,
    scale_only,
    standardize,
    centering,
};

/* Affects only multinomial logistic regression. Type of constraint put on coefficients. 
   This will affect number of coefficients returned. RSC - means we choose a 
   reference catergory whose coefficients will be set to all 0. This results 
   in K-1 class coefficients for K class problem. SSC - sum of coefficients
   class-wise for each feature is 0. It will result in K class coefficients
   for K class problem.
   https://epub.ub.uni-muenchen.de/11001/1/tr067.pdf
   */
enum logistic_constraint {
    no = 0,
    rsc = 1, // Reference category constraint
    ssc = 2, // Symmetric side constraint
};
} // namespace da_linmod

#endif //LINMOD_TYPES_HPP
