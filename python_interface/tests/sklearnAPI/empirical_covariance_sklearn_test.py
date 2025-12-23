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
# pylint: disable = import-outside-toplevel, reimported, no-member
"""
EmpiricalCovariance, empirical_covariance Python tests, check output of skpatch versus sklearn
"""

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch


@pytest.mark.parametrize("np_precision", [np.float64, np.float32])
@pytest.mark.parametrize("order", ['F', 'C'])
@pytest.mark.parametrize("assume_centered", [True, False])
def test_EmpiricalCovariance_works(np_precision, order, assume_centered):
    """
    Basic test to see that the skpatch for EmpiricalCovariance works.
    """

    X = np.random.rand(30, 30)
    X = np.array(X, dtype=np_precision, order=order)

    tol = np.sqrt(np.finfo(np_precision).eps)

    skpatch()
    from sklearn.covariance import EmpiricalCovariance

    emp_cov = EmpiricalCovariance(assume_centered=assume_centered, store_precision=False)

    covariance_da = emp_cov.fit(X).covariance_

    location_da = emp_cov.location_

    assert emp_cov.aocl is True

    undo_skpatch()

    from sklearn.covariance import EmpiricalCovariance

    covariance_sk = EmpiricalCovariance(
        assume_centered=assume_centered, store_precision=False
    ).fit(X).covariance_

    error_cov = np.max(np.abs(covariance_da - covariance_sk))

    assert error_cov < tol

    location_sk = emp_cov.location_

    error_loc = np.max(np.abs(location_da - location_sk))

    assert error_loc < tol


@pytest.mark.parametrize("np_precision", [np.float64, np.float32])
@pytest.mark.parametrize("order", ['F', 'C'])
@pytest.mark.parametrize("assume_centered", [True, False])
def test_empirical_covariance_works(np_precision, order, assume_centered):
    """
    Basic test to see that the skpatch for empirical_covariance works.
    """

    X = np.random.rand(30, 30).astype(np_precision)
    X = np.array(X, dtype=np_precision, order=order)

    tol = np.sqrt(np.finfo(np_precision).eps)

    skpatch()
    from sklearn.covariance import empirical_covariance

    covariance_da = empirical_covariance(X, assume_centered=assume_centered)

    undo_skpatch()

    covariance_sk = empirical_covariance(X, assume_centered=assume_centered)

    error = np.max(np.abs(covariance_da - covariance_sk))

    assert error < tol


def test_EmpiricalCovariance_errors():
    """
    Basic test to see that the skpatch for empirical_covariance give appropraite errors.
    """

    X = np.random.rand(30, 30)
    X = np.array(X, dtype='float32', order='C')

    skpatch()
    from sklearn.covariance import EmpiricalCovariance

    with pytest.warns(UserWarning):
        emp_cov = EmpiricalCovariance(assume_centered=False, store_precision=True)

    assert emp_cov.aocl is True

    covariance = emp_cov.fit(X)

    with pytest.raises(RuntimeError):
        covariance.get_precision()

    with pytest.raises(RuntimeError):
        covariance.score(X)

    with pytest.raises(RuntimeError):
        covariance.error_norm(covariance.covariance_)

    with pytest.raises(RuntimeError):
        covariance.mahalanobis(X)

    assert covariance.precision_ is None

    undo_skpatch()
