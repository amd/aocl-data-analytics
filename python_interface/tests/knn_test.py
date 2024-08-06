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
kNN Python test script
"""

import numpy as np
import pytest
from aoclda.nearest_neighbors import knn_classifier

@pytest.mark.parametrize("da_precision, numpy_precision", [
    ("double", np.float64), ("single", np.float32),
])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_knn_functionality(da_precision, numpy_precision, numpy_order):
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

    knn = knn_classifier(precision=da_precision)
    knn.fit(x_train, y_train)
    k_dist, k_ind = knn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    proba = knn.predict_proba(x_test)
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

@pytest.mark.parametrize("da_precision, numpy_precision", [
    ("double", np.float64), ("single", np.float32),
])
def test_knn_error_exits(da_precision, numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        knn = knn_classifier(n_neighbors = -1, precision=da_precision)
    with pytest.raises(RuntimeError):
        knn = knn_classifier(weights = "ones", precision=da_precision)
    with pytest.raises(RuntimeError):
        knn = knn_classifier(metric = "manhattan", precision=da_precision)
    with pytest.raises(RuntimeError):
        knn = knn_classifier(algorithm = "kdtree", precision=da_precision)
    y_train = np.array([[1,2,3]], dtype=numpy_precision)
    knn = knn_classifier(precision=da_precision)
    knn.fit(x_train, y_train)
    x_test = np.array([[1, 1], [2, 2], [3, 3]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        knn.kneighbors(X=x_test)
    with pytest.raises(RuntimeError):
        knn.predict(X=x_test)
    with pytest.raises(RuntimeError):
        knn.predict_proba(X=x_test)
