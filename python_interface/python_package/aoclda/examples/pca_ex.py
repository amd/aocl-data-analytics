# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
Principal component analysis example Python script
"""

import sys
import numpy as np
from aoclda.factorization import PCA

def pca_example():
    """
    Principal component analysis example
    """

    # Define data arrays
    a = np.array([[2., 2., 3., 2., 4.],
                  [2., 5., 4., 8., 3.],
                  [3., 2., 4., 4., 1.],
                  [4., 8., 3., 6., 4.],
                  [4., 3., 2., 9., 2.],
                  [3., 2., 1., 5., 2.]])

    x = np.array([[7., 4., 2., 9., 3.],
                  [3., 2., 5., 6., 4.],
                  [3., 3., 2., 4., 1.]])

    print("\nPrincipal component analysis for a 6x5 data matrix\n")
    try:
        pca = PCA(n_components=3)
        pca.fit(a)
        x_transform = pca.transform(x)
    except RuntimeError:
        sys.exit(1)

    # Print results
    print("\nPrincipal components of a:\n")
    print(pca.principal_components)
    print("\nx_transform:\n")
    print(x_transform)

    # Check against expected results
    expected_components = np.array([[-0.14907884486130418, -0.6612054163818867,
                                     -0.031706610956396264, -0.7289116905829763,
                                     -0.09091387966203135],
                                    [-0.07220367025708045,  0.623738867070505,
                                     0.20952521660694667, -0.6138062400926413,
                                     0.4302063910917139],
                                    [-0.38718653977350936, -0.06907631947413592,
                                     0.8854125206703791,  0.1296593407398653,
                                     -0.21106437194645863]])

    expected_x_transform = np.array([[-3.250305270939447, -2.1581247424555086, -1.9477723543266676],
                                     [0.6691223004872521, -0.21658703437771865,
                                         1.7953216115607247],
                                     [1.833601737126601, -0.2844305102179128, -0.5561178355649032]])

    norm_components = np.linalg.norm(pca.principal_components - expected_components)
    norm_x_transform = np.linalg.norm(x_transform - expected_x_transform)

    tol = 1.0e-12

    if norm_components > tol or norm_x_transform > tol:
        print("\nSolution is not within expected tolerance\n")
        sys.exit(1)

    print("\nPCA successfully computed\n")

if __name__ == "__main__":
    pca_example()
