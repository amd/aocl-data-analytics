/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

namespace tsne_public {

template <typename tsne_class, typename T>
da_status tsne_set_data(da_handle handle, da_int n_samples, da_int n_features, const T *X,
                        da_int ldx) {
    tsne_class *tsne = dynamic_cast<tsne_class *>(handle->get_alg_handle<T>());
    if (tsne == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_tsne or "
                        "handle is invalid.");
    return tsne->set_data(n_samples, n_features, X, ldx);
}

template <typename tsne_class, typename T>
da_status tsne_set_init_embedding(da_handle handle, const T *Y, da_int ldy) {
    tsne_class *tsne = dynamic_cast<tsne_class *>(handle->get_alg_handle<T>());
    if (tsne == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_tsne or "
                        "handle is invalid.");
    return tsne->set_init_embedding(Y, ldy);
}

template <typename tsne_class, typename T> da_status tsne_compute(da_handle handle) {
    tsne_class *tsne = dynamic_cast<tsne_class *>(handle->get_alg_handle<T>());
    if (tsne == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_tsne or "
                        "handle is invalid.");
    return tsne->compute();
}

} // namespace tsne_public
