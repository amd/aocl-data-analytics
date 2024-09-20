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
# pylint: disable = import-error

"""
k-means clustering Python test script
"""

import numpy as np
import pytest
from aoclda.clustering import kmeans


@pytest.mark.parametrize("da_precision, numpy_precision", [
    ("double", np.float64), ("single", np.float32),
])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_kmeans_functionality(da_precision, numpy_precision, numpy_order):
    """
    Test the functionality of the Python wrapper
    """

    a = np.array([[2., 1.],
                  [-1., -2.],
                  [3., 2.],
                  [2., 3.],
                  [-3., -2.],
                  [-2., -1.],
                  [-2., -3.],
                  [1., 2.]], dtype=numpy_precision, order=numpy_order)

    c = np.array([[1., 1.],
                  [-3., -3.]], dtype=numpy_precision, order=numpy_order)

    x = np.array([[0., 1.],
                  [0., -1.]], dtype=numpy_precision, order=numpy_order)

    km = kmeans(n_clusters=2, precision=da_precision)
    km.fit(a, c, 1.0e-4)
    x_transform = km.transform(x)
    x_labels = km.predict(x)

    expected_centres = np.array([[2., 2.], [-2., -2.]])
    expected_labels = np.array([0, 1, 0, 0, 1, 1, 1, 0])

    expected_x_transform = np.array([[2.23606797749979, 3.605551275463989],
                                     [3.605551275463989, 2.23606797749979]])
    expected_x_labels = np.array([0, 1])

    assert x.flags.f_contiguous == x_transform.flags.f_contiguous

    # Check we have the right answer
    tol = np.finfo(numpy_precision).eps * 1000

    norm = np.linalg.norm(km.cluster_centres - expected_centres)
    assert norm < tol

    norm = np.linalg.norm(x_transform - expected_x_transform)
    assert norm < tol

    assert not np.any(km.labels - expected_labels)

    assert not np.any(x_labels - expected_x_labels)

    assert km.n_samples == 8

    assert km.n_features == 2

    assert km.n_clusters == 2

    assert km.n_iter == 1


@pytest.mark.parametrize("da_precision, numpy_precision", [
    ("double", np.float64), ("single", np.float32),
])
def test_kmeans_error_exits(da_precision, numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision)

    with pytest.raises(RuntimeError):
        km = kmeans(n_clusters=2, precision=da_precision, algorithm="floyd")

    km = kmeans(n_clusters=10, precision=da_precision)
    with pytest.warns(RuntimeWarning):
        km.fit(a)

    b = np.array([1], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        km.transform(b)

    a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision, order="F")
    b = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision, order="C")
    km = kmeans(n_clusters=10, precision=da_precision)
    with pytest.warns(RuntimeWarning):
        km.fit(a)
    with pytest.raises(RuntimeError):
        km.transform(b)
