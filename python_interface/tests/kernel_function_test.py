# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# pylint: disable = missing-module-docstring
"""
Kernel functions Python test script
"""

import numpy as np
import importlib
import pytest
from aoclda.kernel_functions import rbf_kernel, linear_kernel, polynomial_kernel, sigmoid_kernel


@pytest.mark.parametrize("numpy_precision", [
    np.float64, np.float32
])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_rbf_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the rbf kernel function
    """
    tol = np.finfo(numpy_precision).eps

    X = np.array([[1., 2.],
                  [3., 4.],
                  [5., 6.]], dtype=numpy_precision, order=numpy_order)

    Y = np.array([[7., 8.],
                  [9., 10.]], dtype=numpy_precision, order=numpy_order)

    expected_XY = np.array([[0.23692775868212176, 0.07730474044329974], [
                           0.5272924240430485, 0.23692775868212176], [0.8521437889662113, 0.5272924240430485]])
    expected_XX = np.array([[1.0, 0.8521437889662113, 0.5272924240430485], [
                           0.8521437889662113, 1.0, 0.8521437889662113], [0.5272924240430485, 0.8521437889662113, 1.0]])

    kernel_XY = rbf_kernel(X, Y, gamma=0.02)
    assert X.dtype == kernel_XY.dtype
    assert kernel_XY.shape == expected_XY.shape
    assert np.allclose(kernel_XY, expected_XY, rtol=tol)

    kernel_XX = rbf_kernel(X, gamma=0.02)
    assert X.dtype == kernel_XX.dtype
    assert kernel_XX.shape == expected_XX.shape
    assert np.allclose(kernel_XX, expected_XX, rtol=tol)


@pytest.mark.parametrize("numpy_precision", [
    np.float64, np.float32
])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_linear_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the linear kernel function
    """
    tol = np.finfo(numpy_precision).eps

    X = np.array([[1., 2.],
                  [3., 4.],
                  [5., 6.]], dtype=numpy_precision, order=numpy_order)

    Y = np.array([[7., 8.],
                  [9., 10.]], dtype=numpy_precision, order=numpy_order)

    expected_XY = np.array([[23.0, 29.0], [53.0, 67.0], [83.0, 105.0]])
    expected_XX = np.array(
        [[5.0, 11.0, 17.0], [11.0, 25.0, 39.0], [17.0, 39.0, 61.0]])

    kernel_XY = linear_kernel(X, Y)
    assert X.dtype == kernel_XY.dtype
    assert kernel_XY.shape == expected_XY.shape
    assert np.allclose(kernel_XY, expected_XY, rtol=tol)

    kernel_XX = linear_kernel(X)
    assert X.dtype == kernel_XX.dtype
    assert kernel_XX.shape == expected_XX.shape
    assert np.allclose(kernel_XX, expected_XX, rtol=tol)


@pytest.mark.parametrize("numpy_precision", [
    np.float64, np.float32
])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_sigmoid_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the sigmoid kernel function
    """
    tol = np.finfo(numpy_precision).eps

    X = np.array([[1., 2.],
                  [3., 4.],
                  [5., 6.]], dtype=numpy_precision, order=numpy_order)

    Y = np.array([[7., 8.],
                  [9., 10.]], dtype=numpy_precision, order=numpy_order)

    expected_XY = np.array([[0.23549574953849797, 0.2913126124515909], [
                           0.49298796667532435, 0.5915193954318164], [0.6858090622290945, 0.7856638590269437]])
    expected_XX = np.array([[0.0599281035291435, 0.11942729853438588, 0.1780808681173302], [
                           0.11942729853438588, 0.25429553262639115, 0.3799489622552249], [0.1780808681173302, 0.3799489622552249, 0.5511280285381469]])

    kernel_XY = sigmoid_kernel(
        X, Y, gamma=0.01, coef0=0.01)
    assert X.dtype == kernel_XY.dtype
    assert kernel_XY.shape == expected_XY.shape
    assert np.allclose(kernel_XY, expected_XY, rtol=tol)

    kernel_XX = sigmoid_kernel(X, gamma=0.01, coef0=0.01)
    assert X.dtype == kernel_XX.dtype
    assert kernel_XX.shape == expected_XX.shape
    assert np.allclose(kernel_XX, expected_XX, rtol=tol)


@pytest.mark.parametrize("numpy_precision", [
    np.float64, np.float32
])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_polynomial_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the polynomial kernel function
    """
    tol = np.finfo(numpy_precision).eps

    X = np.array([[1., 2.],
                  [3., 4.],
                  [5., 6.]], dtype=numpy_precision, order=numpy_order)

    Y = np.array([[7., 8.],
                  [9., 10.]], dtype=numpy_precision, order=numpy_order)

    expected_XY = np.array([[34.2225, 54.0225], [
                           178.2225, 283.9225], [434.7225, 694.3225]])
    expected_XX = np.array([[1.8225, 8.1225, 18.9225], [
                           8.1225, 40.3225, 97.0225], [18.9225, 97.0225, 235.6225]])

    kernel_XY = polynomial_kernel(
        X, Y, gamma=0.25, degree=2, coef0=0.1)
    assert X.dtype == kernel_XY.dtype
    assert kernel_XY.shape == expected_XY.shape
    assert np.allclose(kernel_XY, expected_XY, rtol=tol)

    kernel_XX = polynomial_kernel(
        X, gamma=0.25, degree=2, coef0=0.1)
    assert X.dtype == kernel_XX.dtype
    assert kernel_XX.shape == expected_XX.shape
    assert np.allclose(kernel_XX, expected_XX, rtol=tol)


@pytest.mark.parametrize("kernel_function", [
    "rbf_kernel", "linear_kernel", "polynomial_kernel", "sigmoid_kernel"
])
@pytest.mark.parametrize("numpy_precision", [
    np.float64, np.float32
])
def test_kernel_function_error_exits(kernel_function, numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision)
    Y = np.array([[1, 1], [2, 2]], dtype=numpy_precision)

    module = importlib.import_module('aoclda.kernel_functions')
    function = getattr(module, kernel_function)

    if kernel_function is not "linear_kernel":
        with pytest.raises(ValueError):
            kernel_XX = function(X, gamma=-1)
        with pytest.raises(ValueError):
            kernel_XY = function(X, Y, gamma=-1)

    if kernel_function is "polynomial_kernel":
        with pytest.raises(ValueError):
            kernel_XX = function(X, degree=-1)
        with pytest.raises(ValueError):
            kernel_XY = function(X, Y, degree=-1)
