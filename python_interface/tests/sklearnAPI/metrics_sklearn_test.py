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
Metrics tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch

@pytest.mark.parametrize("precision", [np.float64,  np.float32])
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
def test_metrics(precision, metric):
    """
    Basic euclidean distance computation
    """
    X = np.array([[1., 2.],
                  [3., 4.],
                  [5., 6.]], dtype=precision)

    Y = np.array([[7., 8. ],
                  [9., 10.],
                  [11., 12.],
                  [13., 14.]], dtype=precision)

    tol = np.finfo(precision).eps

    #patch and import scikit-learn
    skpatch()
    from sklearn.metrics.pairwise import pairwise_distances
    da_euclidean_distance_XY = pairwise_distances(X, Y, metric=metric)
    da_euclidean_distance_XX = pairwise_distances(X, metric=metric)

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    from sklearn.metrics.pairwise import pairwise_distances
    sk_euclidean_distance_XY = pairwise_distances(X, Y, metric=metric)
    sk_euclidean_distance_XX = pairwise_distances(X, metric=metric)

    # Check results
    assert da_euclidean_distance_XY == pytest.approx(sk_euclidean_distance_XY, tol)
    assert da_euclidean_distance_XX == pytest.approx(sk_euclidean_distance_XX, tol)

    # print the results if pytest is invoked with the -rA option
    print("\nEuclidean pairwise distances of X and Y")
    print("     aocl: \n", da_euclidean_distance_XY)
    print("\n  sklearn: \n", sk_euclidean_distance_XY)
    print("\nEuclidean pairwise distances of rows of X")
    print("     aocl: \n", da_euclidean_distance_XX)
    print("\n  sklearn: \n", sk_euclidean_distance_XX)

def test_metrics_errors():
    '''
    Check we can catch errors in the sklearn metrics patch
    '''
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    Y = np.array([[1, 1], [2, 2]])

    #patch and import scikit-learn
    skpatch()
    from sklearn.metrics.pairwise import pairwise_distances
    with pytest.raises(ValueError):
        euclidean_distance_XX = pairwise_distances(X, Y=None, metric="nonexistent")

    with pytest.raises(ValueError):
        euclidean_distance_XX = pairwise_distances(X, Y=None, force_all_finite="nonexistent")

    with pytest.raises(ValueError):
        euclidean_distance_XX = pairwise_distances(X, Y)

    undo_skpatch()
