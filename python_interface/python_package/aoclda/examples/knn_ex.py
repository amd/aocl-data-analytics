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
k-nearest neighbors classification example Python script
"""

import sys
import numpy as np
from aoclda.nearest_neighbors import knn_classifier


def knn_classifier_example():
    """
    k-Nearest Neighbors classification
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

    print("\nk-nearest neighbors for a small data matrix\n")
    knn = knn_classifier()
    knn.fit(x_train, y_train)
    print(x_train)
    k_dist, k_ind = knn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    proba = knn.predict_proba(x_test)
    y_test = knn.predict(x_test)

    # Print results
    print("\nIndices of the nearest neighbors:\n")
    print(k_ind)
    print("\nCorresponding distances of the nearest neighbors:\n")
    print(k_dist)
    print("\nProbability estimates for the test data matrix:\n")
    print(proba)
    print("\nPredicted labels for the test data matrix:\n")
    print(y_test)

    # Check against expected results
    expected_ind = np.array([[1, 0, 3],
                             [2, 0, 1],
                             [3, 5, 4]])
    expected_dist = np.array([[3., 3.31662479, 3.74165739],
                              [2., 3.16227766, 4.24264069],
                              [4.58257569, 5.47722558, 5.65685425]])
    expected_proba = np.array([[0.2, 0.4, 0.4],
                               [0.2, 0.4, 0.4],
                               [0.2, 0.4, 0.4]])
    expected_labels = np.array([[1, 1, 1]])

    norm_dist = np.linalg.norm(k_dist - expected_dist)
    norm_proba = np.linalg.norm(proba - expected_proba)
    incorrect_indices = np.any(k_ind - expected_ind)
    incorrect_labels = np.any(y_test - expected_labels)

    tol = 1.0e-8

    if norm_dist > tol or norm_proba > tol or incorrect_indices or incorrect_labels:
        print("\nSolution is not within expected tolerance\n")
        print("norm_dist = ", norm_dist)
        print("norm_proba = ", norm_proba)
        print("incorrect_indices = ", incorrect_indices)
        print("incorrect_labels = ", incorrect_labels)
        sys.exit(1)

    print("\nk-nearest neighbors successfully computed\n")


if __name__ == "__main__":
    try:
        knn_classifier_example()
    except RuntimeError:
        sys.exit(1)
