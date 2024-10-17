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
DBSCAN clustering example Python script
"""

import sys
import numpy as np
from aoclda.clustering import DBSCAN


def dbscan_example():
    """
    DBSCAN clustering example
    """

    # Define data arrays
    a = np.array([[2., 1.],
                  [-1., -2.],
                  [3., 2.],
                  [2., 3.],
                  [-3., -2.],
                  [-2., -1.],
                  [-2., -3.],
                  [1., 2.],
                  [2., 2.],
                  [-2., -2.]])

    print("\nDBSCAN clustering for a small data matrix\n")
    try:
        db = DBSCAN(eps=1.1, min_samples=4)
        db.fit(a)
    except RuntimeError:
        sys.exit(1)

    # Print results
    print("\nLabels:\n")
    print(db.labels)
    print("\nCore sample indices:\n")
    print(db.core_sample_indices)

    # Check against expected results

    expected_labels = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
    expected_core_sample_indices = np.array([8, 9])

    incorrect_results = np.any(expected_labels - db.labels) or np.any(
        expected_core_sample_indices - db.core_sample_indices)

    tol = 1.0e-12

    if incorrect_results:
        print("\nThe expected solution was not obtained\n")
        sys.exit(1)

    print("\nDBSCAN clusters successfully computed\n")


if __name__ == "__main__":
    dbscan_example()
