# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# pylint: disable = import-outside-toplevel, reimported, no-member, invalid-name

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch

@pytest.mark.parametrize("precision", [np.float64,  np.float32])
@pytest.mark.parametrize("metric", ['euclidean', 'l2', 'sqeuclidean', 'manhattan',
                         'l1', 'cityblock', 'cosine', 'minkowski'])
@pytest.mark.parametrize("numpy_order", ['C', 'F'])
def test_metrics(precision, metric, numpy_order):
    """
    Basic euclidean distance computation
    """
    X = np.array([[-1, -1, 2],
                  [-2, -1, 3],
                  [-3, -2, -1],
                  [1, 3, 1],
                  [2, 5, 1],
                  [3, -1, 2]], dtype=precision, order=numpy_order)

    Y = np.array([[-2, 2, 3],
                  [-1, -2, -1],
                  [2, 1, -3]], dtype=precision, order=numpy_order)

    if metric=='cosine':
        tol = 80*np.finfo(precision).eps
    else:
        tol = np.finfo(precision).eps

    #patch and import scikit-learn
    skpatch()
    from sklearn.metrics.pairwise import pairwise_distances
    with pytest.warns(RuntimeWarning):
        if metric=='minkowski':
            p = 5.5
            da_distance_XY = pairwise_distances(X, Y, metric=metric, p=p)
            da_distance_XX = pairwise_distances(X, metric=metric, p=p)
        else:
            da_distance_XY = pairwise_distances(X, Y, metric=metric)
            da_distance_XX = pairwise_distances(X, metric=metric)
    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    from sklearn.metrics.pairwise import pairwise_distances
    if metric=='minkowski':
        p = 5.5
        sk_distance_XY = pairwise_distances(X, Y, metric=metric, p=p)
        sk_distance_XX = pairwise_distances(X, metric=metric, p=p)
    else:
        sk_distance_XY = pairwise_distances(X, Y, metric=metric)
        sk_distance_XX = pairwise_distances(X, metric=metric)

    # print the results if pytest is invoked with the -rA option
    print("\n" + metric + " pairwise distances of X and Y")
    print("     aocl: \n", da_distance_XY)
    print("\n  sklearn: \n", sk_distance_XY)
    print("\n" + metric + " pairwise distances of rows of X")
    print("     aocl: \n", da_distance_XX)
    print("\n  sklearn: \n", sk_distance_XX)

    # Check results
    assert da_distance_XY == pytest.approx(sk_distance_XY, tol)
    assert da_distance_XX == pytest.approx(sk_distance_XX, tol)



def test_metrics_errors():
    '''
    Check we can catch errors in the sklearn metrics patch
    '''
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    Y = np.array([[1, 1], [2, 2]])

    #patch and import scikit-learn
    skpatch()
    from sklearn.metrics.pairwise import pairwise_distances

    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError):
            euclidean_distance_XX = pairwise_distances(X, Y=None, metric="nonexistent")

        with pytest.raises(ValueError):
            euclidean_distance_XX = pairwise_distances(X, Y)

    undo_skpatch()
