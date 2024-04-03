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
k-means clustering example Python script
"""

import sys
import numpy as np
from aoclda.clustering import kmeans


def kmeans_example():
    """
    k-means clustering example
    """

    # Define data arrays
    a = np.array([[2., 1.],
                  [-1., -2.],
                  [3., 2.],
                  [2., 3.],
                  [-3., -2.],
                  [-2., -1.],
                  [-2., -3.],
                  [1., 2.]])

    c = np.array([[1., 1.],
                  [-3., -3.]])

    x = np.array([[0., 1.],
                  [0., -1.]])

    print("\nk-means clustering for a small data matrix\n")
    try:
        km = kmeans(n_clusters=2)
        km.fit(a, c)
        x_transform = km.transform(x)
        x_labels = km.predict(x)
    except RuntimeError:
        sys.exit(1)

    # Print results
    print("\nComputed cluster centres:\n")
    print(km.cluster_centres)
    print("\nLabels for original data matrix:\n")
    print(km.labels)
    print("\nx_transform:\n")
    print(x_transform)
    print("\nx labels:\n")
    print(x_labels)

    # Check against expected results

    expected_centres = np.array([[2., 2.], [-2., -2.]])
    expected_labels = np.array([0, 1, 0, 0, 1, 1, 1, 0])

    expected_x_transform = np.array([[2.23606797749979, 3.605551275463989],
                                     [3.605551275463989, 2.23606797749979]])
    expected_x_labels = np.array([0, 1])

    norm_components = np.linalg.norm(
        km.cluster_centres - expected_centres)
    norm_x_transform = np.linalg.norm(x_transform - expected_x_transform)

    incorrect_labels = np.any(
        expected_labels - km.labels) or np.any(expected_x_labels - x_labels)

    tol = 1.0e-12

    if norm_components > tol or norm_x_transform > tol or incorrect_labels:
        print("\nSolution is not within expected tolerance\n")
        sys.exit(1)

    print("\nk-means clusters successfully computed\n")


if __name__ == "__main__":
    kmeans_example()
