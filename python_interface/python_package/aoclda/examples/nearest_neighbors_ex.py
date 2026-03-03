# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
Nearest neighbors example Python script
"""

import sys
import numpy as np
from aoclda.neighbors import nearest_neighbors


def nearest_neighbors_example():
    """
    Nearest Neighbors Example
    """

    # Define data arrays
    x_train = np.array([[-1, -1, 2],
                        [-2, -1, 3],
                        [-3, -2, -1],
                        [1, 3, 1],
                        [2, 5, 1],
                        [3, -1, 2]], dtype=np.float64)

    x_test = np.array([[-2, 2, 3],
                       [-1, -2, -1],
                       [2, 1, -3]], dtype=np.float64)

    # Check against expected results
    expected_k_ind = np.array([[1, 0, 3],
                               [2, 0, 1],
                               [3, 5, 4]])
    expected_k_dist = np.array([[3., 3.31662479, 3.74165739],
                                [2., 3.16227766, 4.24264069],
                                [4.58257569, 5.47722558, 5.65685425]])
    expected_r_dist = [np.array([]), np.array([2.]), np.array([])]
    expected_r_ind = [np.array([]), np.array([2]), np.array([])]
    tol = 1.0e-8

    print("\nNearest Neighbors for a small data matrix\n")
    nn = nearest_neighbors()
    nn.fit(x_train)

    # Compute k-nearest neighbors
    k_dist, k_ind = nn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    # Print results
    print("\nIndices of the nearest neighbors:\n")
    print(k_ind)
    print("\nCorresponding distances of the nearest neighbors:\n")
    print(k_dist)

    # Test k-nearest neighbors
    norm_dist = np.linalg.norm(k_dist - expected_k_dist)
    incorrect_indices = np.any(k_ind - expected_k_ind)

    if norm_dist > tol or incorrect_indices:
        print("\nSolution is not within expected tolerance\n")
        print("norm_dist = ", norm_dist)
        print("incorrect_indices = ", incorrect_indices)
        sys.exit(1)

    print("\nk-nearest neighbors successfully computed\n")

    # Compute radius neighbors
    r_dist, r_ind = nn.radius_neighbors(
        x_test, radius=2.0, return_distance=True, sort_results=False)

    print("\nIndices of the radius neighbors:\n")
    print(r_ind)
    print("\nCorresponding distances of the radius neighbors:\n")
    print(r_dist)

    # Test radius neighbors
    radius_errors = False

    # Check radius neighbors results with simplified validation
    try:
        # Check distances - will raise if shapes don't match
        dist_diff = np.linalg.norm(
            np.concatenate(r_dist) -
            np.concatenate(expected_r_dist))
        if dist_diff > tol:
            print("\nRadius neighbors: distances not within tolerance\n")
            print("Norm difference: ", dist_diff)
            radius_errors = True

        # Check indices - will raise if shapes don't match
        for i, expected_indices in enumerate(expected_r_ind):
            if not np.array_equal(r_ind[i], expected_indices):
                print("\nRadius neighbors query ", i, ": incorrect indices")
                print("Got: ", r_ind[i])
                print("Expected: ", expected_indices)
                radius_errors = True
    except (ValueError, IndexError) as e:
        print("\nRadius neighbors: incorrect structure or size mismatch")
        print("Got ", len(r_dist), " distance arrays, expected ", len(expected_r_dist))
        print("Got ", len(r_ind), " index arrays, expected ", len(expected_r_ind))
        print("Error: ", e)
        radius_errors = True

    if radius_errors:
        print("\nRadius neighbors solution is not within expected tolerance\n")
        sys.exit(1)

    print("\nRadius neighbors successfully computed\n")


if __name__ == "__main__":
    try:
        nearest_neighbors_example()
    except RuntimeError:
        sys.exit(1)
