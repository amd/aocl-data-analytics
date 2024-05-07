/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#ifndef LINMOD_SPARSECG_REG_HPP
#define LINMOD_SPARSECG_REG_HPP

#include "aoclda.h"
#include "da_cblas.hh"
#include "sparse_overloads.hpp"
#include <vector>

namespace da_linmod {
template <typename T> struct cg_data {
    // Declare objects needed for conjugate gradient solver
    aoclsparse_itsol_handle handle;
    aoclsparse_itsol_rci_job ircomm;
    T *u, *v, rinfo[100], tol;
    T beta = T(0.0), alpha = T(1.0);
    da_int nsamples, ncoef, min_order, maxit;
    std::vector<T> coef, A, b;

    // Constructors
    cg_data(da_int nsamples, da_int ncoef, T tol, da_int maxit)
        : nsamples(nsamples), ncoef(ncoef), tol(tol), maxit(maxit) {
        min_order = std::min(nsamples, ncoef);
        coef.resize(min_order, 0);       // Initilise starting point to be vector of 0s
        A.resize(min_order * min_order); // Initilise array for X'X or XX'
        b.resize(min_order);             // Initilise array for X'y
        // Create handle
        handle = nullptr;
        if (aoclsparse_itsol_init<T>(&handle) != aoclsparse_status_success) {
            throw std::bad_alloc();
        }
        // Set handle options
        // TODO: Handle absolute tolerance as a function of relative tolerance
        // The following workaround with sprintf is due to the fact that std::to_string()
        // truncates small numbers to 0.
        char tolerance[16];
        sprintf(tolerance, "%9.2e", tol);
        char max_iteration[16];
        sprintf(max_iteration, "%d", maxit);
        if (aoclsparse_itsol_option_set(handle, "CG abs tolerance", tolerance) !=
                aoclsparse_status_success ||
            aoclsparse_itsol_option_set(handle, "CG rel tolerance", tolerance) !=
                aoclsparse_status_success ||
            aoclsparse_itsol_option_set(handle, "CG preconditioner", "none") !=
                aoclsparse_status_success ||
            aoclsparse_itsol_option_set(handle, "CG iteration limit", max_iteration) !=
                aoclsparse_status_success)
            throw std::runtime_error("Internal error with CG solver");
    };
    ~cg_data() { aoclsparse_itsol_destroy(&handle); }

    da_status compute_cg() {
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
            status =
                aoclsparse_itsol_rci_solve(handle, &ircomm, &u, &v, coef.data(), rinfo);
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

                // TODO: Preconditioner
                // case aoclsparse_rci_precond:

                // case aoclsparse_rci_stopping_criterion:
                //     break;

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
            exit_status = da_status_optimization_num_difficult;
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
};
} // namespace da_linmod

#endif