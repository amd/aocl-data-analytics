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
Miscellaneous Python tests
"""

import numpy as np
import pytest
from aoclda.basic_stats import harmonic_mean, mean, variance, quantile, covariance_matrix, standardize
from aoclda.factorization import PCA

@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_array_slicing(numpy_precision, numpy_order):
    """
    Use basic statistics and PCA APIs to test we are correctly handling array slicing for input data
    """
    a_tmp = np.array([[1.1, 2.213, 3.3, 1.2], [4, 5.013, 6, 4.3], [1.5, 2.833, 4.3, 4.2]],
                 dtype=numpy_precision, order=numpy_order)

    a = np.array([[1.1, 2.213, 3.3], [4, 5.013, 6]],
                 dtype=numpy_precision, order=numpy_order)

    a_slice = a_tmp[0:2,0:3]

    # Compute some statistics using a
    means = mean(a, axis="row")

    harmonic_means = harmonic_mean(a, axis="col")

    var = variance(a, axis="all")

    medians = quantile(a, 0.5, axis="row")

    covar = covariance_matrix(a)

    standardized = standardize(a)

    # Compute the same statistics using a_slice
    means_slice = mean(a_slice, axis="row")

    harmonic_means_slice = harmonic_mean(a_slice, axis="col")

    var_slice = variance(a_slice, axis="all")

    medians_slice = quantile(a_slice, 0.5, axis="row")

    covar_slice = covariance_matrix(a_slice)

    standardized_slice = standardize(a_slice)

    # Check we get the same results
    tol = np.finfo(numpy_precision).eps * 100

    assert means_slice == pytest.approx(means, tol)
    assert harmonic_means_slice == pytest.approx(harmonic_means, tol)
    assert var_slice == pytest.approx(var, tol)
    assert medians_slice == pytest.approx(medians, tol)
    assert covar_slice == pytest.approx(covar, tol)
    assert standardized_slice == pytest.approx(standardized, tol)

    # Check PCA
    pca = PCA(n_components=2)
    pca.fit(a)

    pca_slice = PCA(n_components=2)
    pca_slice.fit(a_slice)

    assert pca.principal_components == pytest.approx(pca_slice.principal_components, tol)

