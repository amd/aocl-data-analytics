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
# pylint: disable = import-error

"""
DBSCAN clustering Python test script
"""

import numpy as np
import pytest
from aoclda.clustering import DBSCAN, kmeans

@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_dbscan_functionality(numpy_precision, numpy_order):
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

    db = DBSCAN(eps=2.0, min_samples=2)
    db.fit(a)

    expected_n_clusters = 2
    expected_n_core_samples = 8
    expected_labels = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    expected_core_sample_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    assert db.n_clusters == expected_n_clusters

    assert db.n_core_samples == expected_n_core_samples

    assert db.n_samples == a.shape[0]

    assert db.n_features == a.shape[1]

    assert not np.any(db.labels - expected_labels)

    assert not np.any(db.core_sample_indices - expected_core_sample_indices)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
def test_dbscan_error_exits(numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision)

    with pytest.raises(RuntimeError):
        db = DBSCAN(algorithm="bruce")

    with pytest.raises(RuntimeError):
        db = DBSCAN(metric="sqeuclidead")

    with pytest.raises(RuntimeError):
        db = DBSCAN(min_samples=-45)

    a = np.array([[2., 1.],
                  [-1., -2.],
                  [np.nan, 2.],
                  [2., 3.],
                  [-3., -2.],
                  [-2., -1.],
                  [-2., -3.],
                  [1., 2.]], dtype=numpy_precision)

    db = DBSCAN(check_data=True)
    with pytest.raises(RuntimeError):
        db.fit(a)
