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

"""
Kernel functions example Python script
"""

import sys
import numpy as np
from aoclda.kernel_functions import rbf_kernel, linear_kernel, polynomial_kernel, sigmoid_kernel

def kernel_functions_examples():
    """
    Demonstrate usage of the kernel functions on small datasets.
    """

    X = np.array([
        [1.0,  2.0],
        [-1.0, 0.0],
        [2.0, -2.0]
    ])
    Y = np.array([
        [0.5, -0.5],
        [1.0,  3.0]
    ])

    print("\n--- Kernel Functions Examples ---\n")

    # Linear kernel
    print("Linear kernel (X, Y):")
    try:
        lin = linear_kernel(X, Y)
    except RuntimeError:
        sys.exit(1)
    print(lin)

    # RBF kernel
    print("\nRBF kernel (X, Y), gamma=0.5:")
    try:
        rbf = rbf_kernel(X, Y, gamma=0.5)
    except RuntimeError:
        sys.exit(1)
    print(rbf)

    # Polynomial kernel
    print("\nPolynomial kernel (X, X), degree=2, gamma=1.0, coef0=1.0:")
    try:
        poly = polynomial_kernel(X, degree=2, gamma=1.0, coef0=1.0)
    except RuntimeError:
        sys.exit(1)
    print(poly)

    # Sigmoid kernel
    print("\nSigmoid kernel (X, Y), gamma=0.2, coef0=0.0:")
    try:
        sig = sigmoid_kernel(X, gamma=0.2, coef0=0.0)
    except RuntimeError:
        sys.exit(1)
    print(sig)


if __name__ == "__main__":
    kernel_functions_examples()