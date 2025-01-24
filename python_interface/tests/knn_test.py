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
kNN Python test script
"""

import numpy as np
import pytest
from aoclda.nearest_neighbors import knn_classifier

@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_knn_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the Python wrapper
    """
    x_train = np.array([[-1, -1, 2],
                        [-2, -1, 3],
                        [-3, -2, -1],
                        [1, 3, 1],
                        [2, 5, 1],
                        [3, -1, 2]],
                       dtype=numpy_precision, order=numpy_order)

    y_train = np.array([1, 2, 0, 1, 2, 2],
                       dtype=numpy_precision, order=numpy_order)

    x_test = np.array([[-2 , 2, 3],
                       [-1, -2, -1],
                       [2, 1, -3]],
                       dtype=numpy_precision, order=numpy_order)

    knn = knn_classifier()
    knn.fit(x_train, y_train)
    k_dist, k_ind = knn.kneighbors(x_test, n_neighbors=3, return_distance=True)

    assert k_dist.flags.f_contiguous == x_test.flags.f_contiguous
    assert k_ind.flags.f_contiguous == x_test.flags.f_contiguous

    proba = knn.predict_proba(x_test)
    assert proba.flags.f_contiguous == x_test.flags.f_contiguous

    y_test = knn.predict(x_test)

    expected_ind = np.array([[1, 0, 3],
                             [2, 0, 1],
                             [3, 5, 4]])
    expected_dist = np.array([[3.        , 3.31662479, 3.74165739 ],
                              [2.        , 3.16227766, 4.24264069 ],
                              [4.58257569, 5.47722558, 5.65685425]])
    expected_proba = np.array([[0.2, 0.4, 0.4],
                               [0.2, 0.4, 0.4],
                               [0.2, 0.4, 0.4]])
    expected_labels = np.array([[1, 1, 1,]])

    # Check we have the right answer
    tol = np.sqrt(np.finfo(numpy_precision).eps)

    assert k_dist == pytest.approx(expected_dist, tol)
    assert not np.any(k_ind - expected_ind)
    assert proba == pytest.approx(expected_proba, tol)
    assert not np.any(y_test - expected_labels)

@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
def test_knn_error_exits(numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        knn = knn_classifier(n_neighbors = -1)
    with pytest.raises(RuntimeError):
        knn = knn_classifier(weights = "ones")
    with pytest.raises(RuntimeError):
        knn = knn_classifier(metric = "nonexistent")
    with pytest.raises(RuntimeError):
        knn = knn_classifier(algorithm = "kdtree")
    y_train = np.array([[1,2,3]], dtype=numpy_precision)
    knn = knn_classifier()
    knn.fit(x_train, y_train)
    x_test = np.array([[1, 1], [2, 2], [3, 3]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        knn.kneighbors(X=x_test)
    with pytest.raises(RuntimeError):
        knn.predict(X=x_test)
    with pytest.raises(RuntimeError):
        knn.predict_proba(X=x_test)

    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision, order="F")
    y_train = np.array([[1, 2, 3]], dtype=numpy_precision)
    knn = knn_classifier()
    knn.fit(x_train, y_train)
    x_test = np.array([[1, 1, 3], [2, 2, 3]], dtype=numpy_precision, order="C")
    with pytest.raises(RuntimeError):
        knn.kneighbors(X=x_test, n_neighbors=2)
    with pytest.raises(RuntimeError):
        knn.predict(X=x_test)
    with pytest.raises(RuntimeError):
        knn.predict_proba(X=x_test)

    knn = knn_classifier(check_data=True)
    x_train = np.array([[1, 1, np.nan], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision, order="F")
    with pytest.raises(RuntimeError):
        knn.fit(x_train, y_train)
