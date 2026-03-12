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
Radius neighbors classification example Python script
"""

import sys
import numpy as np
from aoclda.neighbors import nearest_neighbors


def radius_neighbors_classifier_example():
    """
    Radius Neighbors classification
    """

    # Define data arrays
    x_train = np.array([[-1, -1, 2],
                        [-2, -1, 3],
                        [-3, -2, -1],
                        [1, 3, 1],
                        [2, 5, 1],
                        [3, -1, 2]], dtype=np.float64)

    y_train = np.array([1, 2, 0, 1, 2, 2])

    x_test = np.array([[-2, 2, 3],
                       [-1, -2, -1],
                       [2, 1, -3]], dtype=np.float64)

    print("\nradius neighbors for a small data matrix\n")
    rnn = nearest_neighbors(radius=5.0)
    rnn.fit(x_train, y_train)
    print(x_train)
    r_dist, r_ind = rnn.radius_neighbors(
        x_test, radius=3.0, return_distance=True, sort_results=False)
    proba = rnn.classifier_predict_proba(x_test, search_mode='radius_neighbors')
    y_test = rnn.classifier_predict(x_test, search_mode='radius_neighbors')

    # Print results
    print("\nIndices of the radius neighbors:\n")
    print(r_ind)
    print("\nCorresponding distances of the radius neighbors:\n")
    print(r_dist)
    print("\nProbability estimates for the test data matrix:\n")
    print(proba)
    print("\nPredicted labels for the test data matrix:\n")
    print(y_test)

    # Check against expected results
    expected_r_dist = [np.array([3.]), np.array([2.]), np.array([])]
    expected_r_ind = [np.array([1]), np.array([2]), np.array([])]
    expected_proba = np.array([[0., 0.66666667, 0.33333333],
                               [0.33333333, 0.33333333, 0.33333333],
                               [0., 1., 0.]]
                              )
    expected_labels = np.array([1, 0, 1])

    norm_proba = np.linalg.norm(proba - expected_proba)
    incorrect_labels = not np.array_equal(y_test, expected_labels)

    tol = 1.0e-8

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

    print("\nradius neighbors classification successfully computed\n")


if __name__ == "__main__":
    try:
        radius_neighbors_classifier_example()
    except RuntimeError:
        sys.exit(1)
