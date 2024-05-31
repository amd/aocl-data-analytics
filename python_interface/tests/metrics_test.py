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
# pylint: disable = missing-module-docstring
"""
Metric Python test script
"""

import math
import numpy as np
import pytest
import aoclda.metrics as da_metrics

@pytest.mark.parametrize("numpy_precision", [
    np.float64, np.float32
])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])

def test_metrics_functionality(numpy_precision, numpy_order, metric):
    """
    Test the functionality of the Python wrapper
    """

    X = np.array([[1., 2.],
                  [3., 4.],
                  [5., 6.]], dtype=numpy_precision, order=numpy_order)

    Y = np.array([[7., 8. ],
                  [9., 10.]], dtype=numpy_precision, order=numpy_order)

    euclidean_distance_XY = da_metrics.pairwise_distances(X, Y, metric=metric)

    sqrt2 = math.sqrt(2)
    if metric == "euclidean":
        expected_XY = np.array([[6.*sqrt2, 8.*sqrt2],
                                [4.*sqrt2, 6.*sqrt2],
                                [2.*sqrt2, 4.*sqrt2]])
        expected_XX = np.array([[0.,       2.*sqrt2, 4.*sqrt2],
                                [2.*sqrt2, 0.,       2.*sqrt2],
                                [4.*sqrt2, 2.*sqrt2, 0.]])
    elif metric == "sqeuclidean":
        expected_XY = np.array([[72., 128.],
                                [32.,  72.],
                                [8.,   32.]])
        expected_XX = np.array([[0.,  8., 32.],
                                [8.,  0.,  8.],
                                [32., 8.,  0.]])
    # Check that distance matrix had the expected shape
    assert euclidean_distance_XY.shape == expected_XY.shape
    # Compare matrices element-wise
    tol = np.finfo(numpy_precision).eps
    assert np.allclose(euclidean_distance_XY, expected_XY, rtol=tol)
    if metric == "euclidean":
        euclidean_distance_XX = da_metrics.pairwise_distances(X)
    else:
        euclidean_distance_XX = da_metrics.pairwise_distances(X, metric=metric)

    # Check that distance matrix had the expected shape
    assert euclidean_distance_XX.shape == expected_XX.shape
    # Compare matrices element-wise
    assert np.allclose(euclidean_distance_XX, expected_XX, rtol=tol)

@pytest.mark.parametrize("numpy_precision", [
    np.float64, np.float32
])
def test_metrics_error_exits(numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision)
    Y = np.array([[1, 1], [2, 2]], dtype=numpy_precision)

    with pytest.raises(ValueError):
        euclidean_distance_XX = da_metrics.pairwise_distances(X, Y=None, metric="nonexistent")

    with pytest.raises(ValueError):
        euclidean_distance_XX = da_metrics.pairwise_distances(X, Y=None,
            force_all_finite="nonexistent")

    with pytest.raises(ValueError):
        euclidean_distance_XX = da_metrics.pairwise_distances(X, Y)
