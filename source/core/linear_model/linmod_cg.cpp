/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "aoclda.h"
#include "convert_num.hpp"
#include "da_cblas.hh"
#include "linear_model.hpp"
#include "macros.h"
#include "sparse_overloads.hpp"
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
    // The following workaround with sprintf is due to the fact that std::to_string()
    // truncates small numbers to 0.
    char tol_str[16];
    char maxit_str[16];
    convert_num_to_char<T, 16>(tol, tol_str);
    convert_num_to_char<da_int, 16>(maxit, maxit_str);
    if (aoclsparse_itsol_option_set(handle, "CG abs tolerance", tol_str) !=
            aoclsparse_status_success ||
        aoclsparse_itsol_option_set(handle, "CG rel tolerance", tol_str) !=
            aoclsparse_status_success ||
        aoclsparse_itsol_option_set(handle, "CG preconditioner", "none") !=
            aoclsparse_status_success ||
        aoclsparse_itsol_option_set(handle, "CG iteration limit", maxit_str) !=
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