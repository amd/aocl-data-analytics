# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# pylint: disable = import-error, anomalous-backslash-in-string, invalid-name, too-many-arguments
"""
aoclda.kernel_functions module
"""
import numpy as np

from ._aoclda.kernel_functions import (
    pybind_rbf_kernel, pybind_linear_kernel,
    pybind_polynomial_kernel, pybind_sigmoid_kernel)


def rbf_kernel(X, Y=None, gamma=1.0):
    """
    Compute the RBF (Radial Basis Function) kernel matrix.

    .. math::
        K(x, y) = \\exp(-\\gamma ||x - y||^2)

    If Y is not provided, the RBF kernel is computed between the rows of X,
    resulting in a matrix of shape (n_samples_X, n_samples_X). Otherwise, it computes
    the RBF kernel between X and Y, with shape (n_samples_X, n_samples_Y).

    Args:
        X (numpy.ndarray): The feature matrix of shape (n_samples_X, n_features).
        Y (numpy.ndarray, optional): The optional matrix of shape (n_samples_Y, n_features).
        gamma (float, optional): The kernel coefficient. Has to be greater or equal to 0.

    Returns:
        numpy.ndarray: The RBF kernel matrix of shape (n_samples_X, n_samples_X) if Y is None,
        or (n_samples_X, n_samples_Y) otherwise.
    """
    if X.dtype == "float32":
        gamma = np.float32(gamma)
    else:
        gamma = np.float64(gamma)
    return pybind_rbf_kernel(X, Y, gamma)


def linear_kernel(X, Y=None):
    """
    Compute the linear kernel matrix.

    .. math::
        K(x, y) = x * y

    If Y is not provided, the linear kernel is computed between the rows of X,
    resulting in a matrix of shape (n_samples_X, n_samples_X). Otherwise, it computes
    the linear kernel between X and Y, with shape (n_samples_X, n_samples_Y).

    Args:
        X (numpy.ndarray): The feature matrix of shape (n_samples_X, n_features).
        Y (numpy.ndarray, optional): The optional matrix of shape (n_samples_Y, n_features).

    Returns:
        numpy.ndarray: The linear kernel matrix of shape (n_samples_X, n_samples_X) if Y is None,
        or (n_samples_X, n_samples_Y) otherwise.
    """
    return pybind_linear_kernel(X, Y)


def polynomial_kernel(X, Y=None, degree=3, gamma=1.0, coef0=1.0):
    """
    Compute the polynomial kernel matrix.

    .. math::
        K(x, y) = (\\gamma x * y + c)^{d}

    If Y is not provided, the polynomial kernel is computed between the rows of X,
    resulting in a matrix of shape (n_samples_X, n_samples_X). Otherwise, it computes
    the polynomial kernel between X and Y, with shape (n_samples_X, n_samples_Y).

    Args:
        X (numpy.ndarray): The feature matrix of shape (n_samples_X, n_features).
        Y (numpy.ndarray, optional): The optional matrix of shape (n_samples_Y, n_features).
        degree (int, optional): The degree of the polynomial.
        gamma (float, optional): The kernel coefficient. Has to be greater or equal to 0.
        coef0 (float, optional): The independent term in the polynomial kernel.

    Returns:
        numpy.ndarray: The polynomial kernel matrix of shape (n_samples_X, n_samples_X)
        if Y is None, or (n_samples_X, n_samples_Y) otherwise.
    """
    if X.dtype == "float32":
        gamma = np.float32(gamma)
        coef0 = np.float32(coef0)
    else:
        gamma = np.float64(gamma)
        coef0 = np.float64(coef0)
    return pybind_polynomial_kernel(X, Y, degree, gamma, coef0)


def sigmoid_kernel(X, Y=None, gamma=1.0, coef0=1.0):
    """
    Compute the sigmoid kernel matrix.

    .. math::
        K(x, y) = \\tanh(\\gamma x * y + c)

    If Y is not provided, the sigmoid kernel is computed between the rows of X,
    resulting in a matrix of shape (n_samples_X, n_samples_X). Otherwise, it computes
    the sigmoid kernel between X and Y, with shape (n_samples_X, n_samples_Y).

    Args:
        X (numpy.ndarray): The feature matrix of shape (n_samples_X, n_features).
        Y (numpy.ndarray, optional): The optional matrix of shape (n_samples_Y, n_features).
        gamma (float, optional): The kernel coefficient. Has to be greater or equal to 0.
        coef0 (float, optional): The constant factor for the sigmoid kernel.

    Returns:
        numpy.ndarray: The sigmoid kernel matrix of shape (n_samples_X, n_samples_X) if Y is None,
        or (n_samples_X, n_samples_Y) otherwise.
    """
    if X.dtype == "float32":
        gamma = np.float32(gamma)
        coef0 = np.float32(coef0)
    else:
        gamma = np.float64(gamma)
        coef0 = np.float64(coef0)
    return pybind_sigmoid_kernel(X, Y, gamma, coef0)
