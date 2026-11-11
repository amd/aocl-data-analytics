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
Nonnegative matrix factorization (NMF) example Python script
"""

import sys
import numpy as np
from aoclda.factorization import NMF


def nmf_example():
    """
    Nonnegative matrix factorization example
    """

    # Define data array - 5 samples x 4 features, nonnegative integers
    a = np.array([[1., 2., 3., 4.],
                  [2., 4., 6., 8.],
                  [3., 6., 9., 12.],
                  [4., 1., 5., 2.],
                  [5., 3., 8., 4.]])

    n_components = 2

    print("\nNonnegative matrix factorization of a 5x4 data matrix\n")

    # Use NNDSVDa initialization
    try:
        nmf = NMF(n_components=n_components, solver='mu', beta_loss='frobenius',
                  max_iter=200, tol=1.0e-6, init='nndsvda')
        w = nmf.fit_transform(a)
    except RuntimeError:
        sys.exit(1)

    h = nmf.h

    print("W:\n", w)
    print("\nH:\n", h)
    print("\nReconstruction error:", nmf.reconstruction_err)
    print("Iterations performed:", nmf.n_iter)

    # Transform two new samples into the component space, then reconstruct
    a_new = np.array([[2., 3., 5., 4.],
                      [1., 1., 2., 1.]])

    try:
        w_new = nmf.transform(a_new)
        a_rec = nmf.inverse_transform(w_new)
    except RuntimeError:
        sys.exit(1)

    print("\nW_new = transform(A_new):\n", w_new)
    print("\ninverse_transform(W_new) ~= A_new:\n", a_rec)

    # Check against expected results
    expected_w = np.array([[0.25349316836646635, 0.6243580088015558],
                           [0.5069863368726557, 1.2487160174957959],
                           [0.7604795054091785, 1.873074026166739],
                           [0.9984616720806825, 0.02497846388800884],
                           [1.3634545777991705, 0.3453326815415483]])

    expected_h = np.array([[3.783381296576913, 1.3014499288537753, 5.085367089249122,
                            1.621905609470656],
                           [0.05579148017166895, 2.6919101871859, 2.746939596185101,
                            5.7369326140531305]])

    expected_w_new = np.array([[0.6158527985328428, 0.5921255830249437],
                               [0.30681084256346264, 0.1194642135450567]])

    expected_a_rec = np.array([[2.3630415221381815, 2.395450469873203,
                                4.758370763406295, 4.3958396774196435],
                               [1.1674474886428317, 0.7208858826718753,
                                1.8884067399110551, 1.182976369498942]])

    expected_recon_err = 0.6588699104776147

    tol = 1.0e-4

    if (np.linalg.norm(w - expected_w) > tol
            or np.linalg.norm(h - expected_h) > tol
            or np.linalg.norm(w_new - expected_w_new) > tol
            or np.linalg.norm(a_rec - expected_a_rec) > tol
            or abs(nmf.reconstruction_err - expected_recon_err) > tol):
        print("\nSolution is not within expected tolerance\n")
        sys.exit(1)

    print("\nNMF example completed successfully\n")


if __name__ == "__main__":
    nmf_example()
