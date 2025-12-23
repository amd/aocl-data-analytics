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
k-means clustering Python test script
"""

import numpy as np
import pytest
from aoclda.clustering import kmeans
from aoclda._internal_utils import debug as dbg


@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, 'object'])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_kmeans_all_dtypes(numpy_precision, numpy_order):
    """
    Test it runs when supported/unsupported C-interface type is provided.
    """

    a = np.array([[2.5, 1.4],
                  [-1.0, -2.3],
                  [3.8, 2.6],
                  [2.4, 3.6],
                  [-3.0, -2.4],
                  [-2.2, -1.5],
                  [-2.3, -3.1],
                  [1.5, 2.1]], dtype=numpy_precision, order=numpy_order)

    c = np.array([[1.6, 1.6],
                  [-3.2, -3.4]], dtype=numpy_precision, order=numpy_order)

    x = np.array([[0.1, 1.6],
                  [0.3, -1.5]], dtype=numpy_precision, order=numpy_order)

    km = kmeans(n_clusters=2, C=c, tol=1.0e-4, seed=23)
    km.fit(a)

    x_transform = km.transform(x)
    x_labels = km.predict(x)


@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders",
                         [("C", "F"), ("F", "C")])
def test_kmeans_multiple_orders(numpy_precision, numpy_orders):
    """
    Test it runs when arrays of multiple orders are provided.
    """

    a = np.array([[2.5, 1.4],
                  [-1.0, -2.3],
                  [3.8, 2.6],
                  [2.4, 3.6],
                  [-3.0, -2.4],
                  [-2.2, -1.5],
                  [-2.3, -3.1],
                  [1.5, 2.1]], dtype=numpy_precision, order=numpy_orders[0])

    x = np.array([[0.1, 1.6],
                  [0.3, -1.5]], dtype=numpy_precision, order=numpy_orders[0])

    c = np.array([[1.6, 1.6],
                  [-3.2, -3.4]], dtype=numpy_precision, order=numpy_orders[1])

    km = kmeans(n_clusters=2, C=c, tol=1.0e-4, seed=23)

    with pytest.warns(UserWarning):
        km.fit(a)

    x_transform = km.transform(x)
    x_labels = km.predict(x)

    # Change order of a to be different to x
    a = np.array(a, order=numpy_orders[1])

    km = kmeans(n_clusters=2, tol=1.0e-4, seed=23)
    km.fit(a)

    with pytest.warns(UserWarning):
        x_transform = km.transform(x)
    with pytest.warns(UserWarning):
        x_labels = km.predict(x)

    a = np.array(a, order=numpy_orders[0])
    with pytest.warns(UserWarning):
        km.fit(a)


@pytest.mark.parametrize(
    "numpy_precisions", [('float32', 'float64'),
                         ('float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_kmeans_multiple_dtypes(numpy_precisions, numpy_order):
    """
    Test it runs when arrays of multiple dtypes are provided.
    """

    a = np.array([[2.5, 1.4],
                  [-1.0, -2.3],
                  [3.8, 2.6],
                  [2.4, 3.6],
                  [-3.0, -2.4],
                  [-2.2, -1.5],
                  [-2.3, -3.1],
                  [1.5, 2.1]], dtype=numpy_precisions[0], order=numpy_order)

    x = np.array([[0.1, 1.6],
                  [0.3, -1.5]], dtype=numpy_precisions[0], order=numpy_order)

    c = np.array([[1.6, 1.6],
                  [-3.2, -3.4]], dtype=numpy_precisions[1], order=numpy_order)

    km = kmeans(n_clusters=2, C=c, tol=1.0e-4, seed=23)

    km.fit(a)

    x_transform = km.transform(x)
    x_labels = km.predict(x)

    # Change order of a to be different to x
    a = np.array(a, dtype=numpy_precisions[1], order=numpy_order)

    km = kmeans(n_clusters=2, tol=1.0e-4, seed=23)

    km.fit(a)

    x_transform = km.transform(x)
    x_labels = km.predict(x)

    a = np.array(a, dtype=numpy_precisions[0])
    km.fit(a)


@pytest.mark.parametrize("isa", ["scalar", "avx"])
def test_context_getsetters_kmeans(isa):
    """
    Check kmeans setup registry, request a specific kernel, expect setup to
    record it
    """
    isae = {"scalar": "0", "avx": "2"}  # matches with enum in kmeans_type.hpp
    dbg.set({"kmeans.isa": isa})
    _ = test_kmeans_functionality(np.float32, "C")
    d = dbg.get("kmeans.setup")
    tokens = (d["kmeans.setup"]).split(",")
    ok = False
    for t in tokens:
        pair = t.split("=")
        if (pair[0] == "kernel.type") and (pair[1] == isae[isa]):
            ok = True
            break
    assert ok


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_kmeans_functionality(numpy_precision, numpy_order):
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

    km = kmeans(n_clusters=2, C=c, tol=1.0e-4, seed=23)
    km.fit(a)
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


@pytest.mark.parametrize("_da_precision, numpy_precision", [
    ("double", np.float64), ("single", np.float32),
])
def test_kmeans_error_exits(_da_precision, numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision)

    with pytest.raises(RuntimeError):
        km = kmeans(n_clusters=2, algorithm="floyd")

    km = kmeans(n_clusters=10)
    with pytest.warns(RuntimeWarning):
        km.fit(a)

    b = np.array([1], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        km.transform(b)

    a = np.array([[2., 1.],
                  [-1., -2.],
                  [np.nan, 2.],
                  [2., 3.],
                  [-3., -2.],
                  [-2., -1.],
                  [-2., -3.],
                  [1., 2.]], dtype=numpy_precision)

    c = np.array([[1., 1.],
                  [-3., -3.]], dtype=numpy_precision)

    km = kmeans(n_clusters=2, C=c, tol=1.0e-4, check_data=True)
    with pytest.raises(RuntimeError):
        km.fit(a)
