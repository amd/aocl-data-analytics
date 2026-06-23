# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
Kernel principal component analysis example Python script
"""

import sys
import numpy as np
from aoclda.factorization import KernelPCA


def kernel_pca_example():
    """
    Kernel principal component analysis example
    """

    # Define data arrays
    A = np.array([[1.0, 2.0, -1.0],
                  [3.0, -1.0, 0.5],
                  [0.5, 1.5, 2.0],
                  [-2.0, 0.0, 1.0],
                  [1.5, -0.5, -1.5],
                  [-1.0, 1.0, 0.0]])

    X = np.array([[0.5, -1.0, 1.5],
                  [-0.5, 2.0, -1.0],
                  [1.0, 0.0, 0.5]])

    print("\nKernel PCA with RBF kernel for a 6x3 data matrix\n")
    try:
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.5)
        kpca.fit(A)
        X_transform = kpca.transform(X)
    except RuntimeError:
        sys.exit(1)

    # Print results
    print("\nEigenvalues:\n")
    print(kpca.eigenvalues)
    print("\nX_transform:\n")
    print(X_transform)

    # Check against expected results (first 2 components from RBF kernel PCA)
    expected_eigenvalues = np.array([
        1.13889228957286082e+00, 1.00597570048067242e+00])

    expected_X_transform = np.array([
        [-3.23178684524370874e-02, 1.99334687074597637e-02],
        [6.82903208331758355e-02, 5.21438300158077322e-02],
        [-8.32522890797626203e-02, 1.86186052944305354e-02]])

    norm_eigenvalues = np.linalg.norm(
        kpca.eigenvalues - expected_eigenvalues)
    norm_X_transform = np.linalg.norm(
        np.abs(X_transform) - np.abs(expected_X_transform))

    tol = 1.0e-10

    if norm_eigenvalues > tol or norm_X_transform > tol:
        print("\nSolution is not within expected tolerance\n")
        sys.exit(1)

    print("\nKernel PCA successfully computed\n")


if __name__ == "__main__":
    kernel_pca_example()
