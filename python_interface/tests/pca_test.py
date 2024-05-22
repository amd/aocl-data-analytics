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
Principal component analysis Python test script
"""

import numpy as np
import pytest
from aoclda.factorization import PCA


@pytest.mark.parametrize("da_precision, numpy_precision", [
    ("double", np.float64), ("single", np.float32),
])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_pca_functionality(da_precision, numpy_precision, numpy_order):
    """
    Test the functionality of the Python wrapper
    """

    a = np.array([[1, 2, 3], [2, 5, 4], [3, 6, 1]],
                 dtype=numpy_precision, order=numpy_order)
    x = np.array([[1, 1, 4], [3, 2, 3], [0, 2, 3], [1, 0, 1]],
                 dtype=numpy_precision, order=numpy_order)

    pca = PCA(n_components=2, precision=da_precision)
    pca.fit(a)

    expected_components = np.array([[0.4082482904638631,  0.816496580927726, -0.408248290463863],
                                   [0.,  0.447213595499958,  0.8944271909999159]])

    expected_explained_variance = np.array(
        [5.999999999999999, 1.6666666666666665])
    expected_mean = np.array([2., 4.333333333333333, 2.6666666666666665])
    expected_singular_values = np.array(
        [3.4641016151377544, 1.8257418583505536])
    expected_u = np.array([[-7.0710678118654724e-01, -4.0824829046386296e-01],
                           [3.7336028004349413e-17, 8.1649658092772626e-01],
                           [7.0710678118654746e-01, -4.0824829046386302e-01]])
    expected_scores = np.array([[-2.4494897427831770e+00, -7.4535599249992979e-01],
                                [1.2933579491269523e-16,  1.4907119849998600e+00],
                                [2.4494897427831779e+00, -7.4535599249992990e-01]])
    expected_transform = np.array([[0.5, 1.2000000000000002, 3.8999999999999995],
                                   [1.3333333333333335, 2.6666666666666665,
                                    2.6666666666666665],
                                   [0.8333333333333333, 1.6666666666666665,
                                    3.1666666666666665],
                                   [0.6666666666666665, 0.13333333333333286,
                                    0.9333333333333333]])

    # Check we have the right answer, note use of abs to allow for different normalization
    tol = np.finfo(numpy_precision).eps * 1000

    norm = np.linalg.norm(
        np.abs(pca.principal_components) - np.abs(expected_components))
    assert norm < tol

    norm = np.linalg.norm(pca.column_means - expected_mean)
    assert norm < tol

    norm = np.linalg.norm(pca.sigma - expected_singular_values)
    assert norm < tol

    norm = np.linalg.norm(pca.variance - expected_explained_variance)
    assert norm < tol

    # Final singular value is 10^-33 so we are OK ignoring it for this test
    assert np.abs(np.sum(expected_explained_variance) -
                  pca.total_variance[0]) < tol

    norm = np.linalg.norm(np.abs(pca.vt) - np.abs(expected_components))
    assert norm < tol

    norm = np.linalg.norm(np.abs(pca.u) - np.abs(expected_u))
    assert norm < tol

    norm = np.linalg.norm(np.abs(pca.scores) - np.abs(expected_scores))
    assert norm < tol

    norm = np.linalg.norm(pca.inverse_transform(
        pca.transform(x)) - expected_transform)
    assert norm < tol

    assert pca.n_samples == 3

    assert pca.n_features == 3

    assert pca.n_components == 2

    # Simple test to check we can also get column_sdevs
    a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                 dtype=numpy_precision, order=numpy_order)

    pca = PCA(n_components=3, precision=da_precision, method="correlation")
    pca.fit(a)
    norm = np.linalg.norm(pca.column_sdevs - np.array([1., 1., 1.]))
    assert norm < tol


@pytest.mark.parametrize("da_precision, numpy_precision", [
    ("double", np.float64), ("single", np.float32),
])
def test_pca_error_exits(da_precision, numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision)

    with pytest.raises(RuntimeError):
        pca = PCA(n_components=3, precision=da_precision, method="corelation")

    pca = PCA(n_components=10, precision=da_precision, method="correlation")
    with pytest.warns(RuntimeWarning):
        pca.fit(a)

    b = np.array([1])
    with pytest.raises(RuntimeError):
        pca.transform(b)

    with pytest.raises(RuntimeError):
        pca.inverse_transform(b)
