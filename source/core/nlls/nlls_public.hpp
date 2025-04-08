/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"
#include <type_traits>

namespace nlls_public {

template <typename nlls_class, typename resfun_t, typename resgrd_t, typename reshes_t,
          typename reshp_t, typename T>
da_status nlls_define_residuals(da_handle handle, da_int n_coef, da_int n_res,
                                resfun_t *resfun, resgrd_t *resgrd, reshes_t *reshes,
                                reshp_t *reshp) {
    nlls_class *nlls = dynamic_cast<nlls_class *>(handle->get_alg_handle<T>());
    if (nlls == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    nlls->refresh();
    da_status status;
    status = nlls->add_vars(n_coef);
    if (status != da_status_success)
        return status; // Error message already loaded
    status = nlls->add_res(n_res);
    if (status != da_status_success)
        return status; // Error message already loaded
    return nlls->define_callbacks(resfun, resgrd, reshes, reshp);
}

template <typename nlls_class, typename T>
da_status nlls_define_bounds(da_handle handle, da_int n_coef, T *lower, T *upper) {
    nlls_class *nlls = dynamic_cast<nlls_class *>(handle->get_alg_handle<T>());
    if (nlls == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    nlls->refresh();
    return nlls->add_bound_cons(n_coef, lower, upper);
}

template <typename nlls_class, typename T>
da_status nlls_define_weights(da_handle handle, da_int n_coef, T *weights) {
    nlls_class *nlls = dynamic_cast<nlls_class *>(handle->get_alg_handle<T>());
    if (nlls == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    nlls->refresh();
    return nlls->add_weights(n_coef, weights);
}

template <typename nlls_class, typename T>
da_status nlls_fit(da_handle handle, da_int n_coef, T *coef, void *udata) {
    nlls_class *nlls = dynamic_cast<nlls_class *>(handle->get_alg_handle<T>());
    if (nlls == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    return nlls->fit(n_coef, coef, udata);
}

} // namespace nlls_public