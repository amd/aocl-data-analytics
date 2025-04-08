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

#ifndef SVM_TYPES_HPP
#define SVM_TYPES_HPP
#include <functional>

#define SVM_MAX_KERNEL_SIZE da_int(1024)
#define SVM_MAX_BLOCK_SIZE da_int(2048)

namespace da_svm_types {

enum svm_kernel { rbf = 0, linear, polynomial, sigmoid };

template <typename T> struct meta_kernel_f {
    using type =
        std::function<void(da_order order, da_int m, da_int n, da_int k, const T *X,
                           T *x_norm, da_int ldx, const T *Y, T *y_norm, da_int ldy, T *D,
                           da_int ldd, T gamma, da_int degree, T coef0, bool X_is_Y)>;
};
template <typename T> using kernel_f_type = typename meta_kernel_f<T>::type;

} // namespace da_svm_types

#endif //SVM_TYPES_HPP
