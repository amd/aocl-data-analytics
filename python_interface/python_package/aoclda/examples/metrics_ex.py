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
Metrics example python script
"""

import sys
import numpy as np
from aoclda.metrics import pairwise_distances


def metrics_example():
    """
    Metrics examples
    """
    X = np.array([[2, 3], [3, 5], [5, 8]],
                 dtype=np.float32,
                 order='F')
    print("Sample X:")
    print(X)

    Y = np.array([[1, 0], [2, 1]],
                 dtype=np.float32,
                 order='F')

    print("\nSample Y:")
    print(Y)

    euclidean_distance_xy = pairwise_distances(X, Y)
    print(f"\nEuclidean pairwise distances of X and Y:\n {euclidean_distance_xy}")

    euclidean_distance_xx = pairwise_distances(X)
    print(f"\nEuclidean pairwise distances of rows of X:\n {euclidean_distance_xx}")

    squared_euclidean_distance_xy = pairwise_distances(X, Y, metric='sqeuclidean')
    print(f"\nSquared euclidean pairwise distances of X and Y:\n {squared_euclidean_distance_xy}")

    squared_euclidean_distance_xx = pairwise_distances(X, metric='sqeuclidean')
    print(f"\nSquared euclidean pairwise distances of rows of X:\n {squared_euclidean_distance_xx}")

    print("\nMetrics calculations successful")
    print("---------------------------")


if __name__ == "__main__":
    try:
        metrics_example()
    except RuntimeError:
        sys.exit(1)
