/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include "aoclda.h"
#include "da_cblas.hh"
#include "linear_model.hpp"
#include "macros.h"
#include "sparse_overloads.hpp"
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace ARCH {

namespace da_linmod {

using namespace da_linmod_types;

template <typename T>
cg_data<T>::cg_data(da_int nsamples, da_int ncoef, T tol, da_int maxit)
    : tol(tol), nsamples(nsamples), ncoef(ncoef), maxit(maxit) {
    min_order = std::min(nsamples, ncoef);
    coef.resize(min_order, 0);       // Initialize starting point to be vector of 0s
    A.resize(min_order * min_order); // Initialize array for X'X or XX'
    b.resize(min_order);             // Initialize array for X'y
    // Create handle
    handle = nullptr;
    if (aoclsparse_itsol_init<T>(&handle) != aoclsparse_status_success) {
        throw std::bad_alloc();
    }
    // Set handle options
    // Stream formatting avoids std::to_string() truncating small tolerances to 0.
    std::ostringstream tol_stream;
    tol_stream << std::scientific << std::setprecision(2) << tol;
    const std::string tol_str = tol_stream.str();
    const std::string maxit_str = std::to_string(maxit);
    if (aoclsparse_itsol_option_set(handle, "CG abs tolerance", tol_str.c_str()) !=
            aoclsparse_status_success ||
        aoclsparse_itsol_option_set(handle, "CG rel tolerance", tol_str.c_str()) !=
            aoclsparse_status_success ||
        aoclsparse_itsol_option_set(handle, "CG preconditioner", "none") !=
            aoclsparse_status_success ||
        aoclsparse_itsol_option_set(handle, "CG iteration limit", maxit_str.c_str()) !=
            aoclsparse_status_success)
        throw std::runtime_error("Internal error with CG solver");
};

template <typename T> cg_data<T>::~cg_data() { aoclsparse_itsol_destroy(&handle); }

template <typename T> da_status cg_data<T>::compute_cg() {
    aoclsparse_status status;
    status = aoclsparse_itsol_rci_input(handle, min_order, b.data());
    if (status != aoclsparse_status_success) {
        if (status == aoclsparse_status_memory_error) {
            return da_status_memory_error;
        } else {
            return da_status_internal_error;
        }
    }
    // Call CG solver
    ircomm = aoclsparse_rci_start;
    u = nullptr;
    v = nullptr;
    while (ircomm != aoclsparse_rci_stop) {
        status = aoclsparse_itsol_rci_solve(handle, &ircomm, &u, &v, coef.data(), rinfo);
        if (status != aoclsparse_status_success)
            break;
        switch (ircomm) {
        case aoclsparse_rci_mv:
            // Compute v = Au
            // There is an alternative to explicitly computing A. reverse communication CG doesn't actually require A
            // and only asks for v = (X'X +lambda I)u which can be done on the fly with pointers to X and lambda.
            // It would be more expensive per iteration to compute (2 gemv calls instead of one) but could save a lot
            // of memory space. I could see it being a useful alternative when X and X'X are huge and copying X is prohibitive.
            da_blas::cblas_symv(CblasColMajor, CblasUpper, min_order, alpha, A.data(),
                                min_order, u, 1, beta, v, 1);
            break;

        default:
            break;
        }
    }
    da_status exit_status;
    switch (status) {
    case aoclsparse_status_success:
        exit_status = da_status_success;
        break;
    case aoclsparse_status_numerical_error:
        exit_status = da_status_numerical_difficulties;
        break;
    case aoclsparse_status_maxit:
        exit_status = da_status_maxit;
        break;
    default:
        exit_status = da_status_internal_error;
        break;
    }
    return exit_status;
}

template <typename T>
da_status cg_data<T>::get_info([[maybe_unused]] da_int linfo, T *info) {
    // Save information about the norm of the gradient of the loss function
    info[da_linmod_info_t::linmod_info_grad_norm] = static_cast<T>(rinfo[0] * rinfo[1]);
    // Save information about the number of iterations
    info[da_linmod_info_t::linmod_info_iter] = static_cast<T>(rinfo[30]);
    return da_status_success;
}

template struct cg_data<float>;
template struct cg_data<double>;

} // namespace da_linmod

} // namespace ARCH