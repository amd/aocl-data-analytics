# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
Basic statistics tests, check output of sppatch versus scipy.stats
"""

# pylint: disable = import-outside-toplevel, reimported, no-member, invalid-name

import numpy as np
import pytest
from scipy import stats as scipy_stats
from aoclda.scipy import sppatch, undo_sppatch


SCIPY_HAS_QUANTILE = hasattr(scipy_stats, "quantile")
requires_scipy_quantile = pytest.mark.skipif(
    not SCIPY_HAS_QUANTILE,
    reason="scipy.stats.quantile is unavailable in this SciPy version",
)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ['C', 'F'])
@pytest.mark.parametrize("axis", [0, 1, None])
def test_gmean_hmean(precision, numpy_order, axis):
    """
    Check geometric and harmonic means against scipy
    """
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [2.0, 4.0, 6.0]], dtype=precision, order=numpy_order)

    tol = 100 * np.finfo(precision).eps

    # patch and import scipy
    sppatch(["gmean", "hmean"])
    from scipy import stats
    da_gmean = stats.gmean(X, axis=axis)
    da_hmean = stats.hmean(X, axis=axis)

    # unpatch and solve the same problem with scipy
    undo_sppatch()
    from scipy import stats
    sk_gmean = stats.gmean(X, axis=axis)
    sk_hmean = stats.hmean(X, axis=axis)

    assert da_gmean == pytest.approx(sk_gmean, rel=tol, abs=tol)
    assert da_hmean == pytest.approx(sk_hmean, rel=tol, abs=tol)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("fisher", [True, False])
def test_skew_kurtosis(precision, axis, bias, fisher):
    """
    Check skewness and kurtosis against scipy for the bias and fisher options
    """
    rng = np.random.default_rng(42)
    X = rng.uniform(1.0, 10.0, size=(20, 5)).astype(precision)

    tol = 100 * np.finfo(precision).eps

    sppatch()
    from scipy import stats
    da_skew = stats.skew(X, axis=axis, bias=bias)
    da_kurt = stats.kurtosis(X, axis=axis, fisher=fisher, bias=bias)

    undo_sppatch()
    from scipy import stats
    sk_skew = stats.skew(X, axis=axis, bias=bias)
    sk_kurt = stats.kurtosis(X, axis=axis, fisher=fisher, bias=bias)

    assert da_skew == pytest.approx(sk_skew, rel=tol, abs=tol)
    assert da_kurt == pytest.approx(sk_kurt, rel=tol, abs=tol)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("axis", [0, 1, None])
def test_moment(precision, order, axis):
    """
    Check central moments against scipy
    """
    rng = np.random.default_rng(7)
    X = rng.uniform(0.0, 5.0, size=(30, 4)).astype(precision)

    tol = 100 * np.finfo(precision).eps

    sppatch()
    from scipy import stats
    da_moment = stats.moment(X, order=order, axis=axis)

    undo_sppatch()
    from scipy import stats
    sk_moment = stats.moment(X, order=order, axis=axis)

    assert da_moment == pytest.approx(sk_moment, rel=tol, abs=tol)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("axis", [0, 1, None])
def test_moment_array_order(precision, axis):
    """
    Check central moments with a vector of orders against scipy
    """
    rng = np.random.default_rng(7)
    X = rng.uniform(0.0, 5.0, size=(30, 4)).astype(precision)
    order = np.array([2, 3, 4])

    tol = 100 * np.finfo(precision).eps

    sppatch()
    from scipy import stats
    da_moment = stats.moment(X, order=order, axis=axis)

    undo_sppatch()
    from scipy import stats
    sk_moment = stats.moment(X, order=order, axis=axis)

    assert np.shape(da_moment) == np.shape(sk_moment)
    assert da_moment == pytest.approx(sk_moment, rel=tol, abs=tol)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("q", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("keepdims", [None, True, False])
@requires_scipy_quantile
def test_quantile(precision, q, axis, keepdims):
    """
    Check quantiles against scipy
    """
    rng = np.random.default_rng(13)
    X = rng.uniform(0.0, 100.0, size=(50, 3)).astype(precision)

    tol = 100 * np.finfo(precision).eps

    sppatch()
    from scipy import stats
    da_quant = stats.quantile(X, q, axis=axis, keepdims=keepdims)

    undo_sppatch()
    from scipy import stats
    sk_quant = stats.quantile(X, q, axis=axis, keepdims=keepdims)

    assert np.shape(da_quant) == np.shape(sk_quant)
    assert da_quant == pytest.approx(sk_quant, rel=tol, abs=tol)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("keepdims", [None, True])
@requires_scipy_quantile
def test_quantile_array_p(precision, axis, keepdims):
    """
    Check quantiles with a vector of probabilities against scipy. scipy
    positions p within the coordinate system of the data, so the resulting
    shapes differ per axis; the patch must reproduce them exactly.
    """
    rng = np.random.default_rng(13)
    X = rng.uniform(0.0, 100.0, size=(50, 3)).astype(precision)
    if axis is None:
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    else:
        p = np.array([0.2, 0.5, 0.9])

    tol = 100 * np.finfo(precision).eps

    sppatch()
    from scipy import stats
    da_quant = stats.quantile(X, p, axis=axis, keepdims=keepdims)

    undo_sppatch()
    from scipy import stats
    sk_quant = stats.quantile(X, p, axis=axis, keepdims=keepdims)

    assert np.shape(da_quant) == np.shape(sk_quant)
    assert da_quant == pytest.approx(sk_quant, rel=tol, abs=tol)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("keepdims", [None, True])
@requires_scipy_quantile
def test_quantile_p_grid(precision, axis, keepdims):
    """
    Check that a 2D p produces the full quantile grid as in scipy. A column
    vector is used for axis=0 and a row vector for axis=1 so the probabilities
    lie along the reduction axis.
    """
    rng = np.random.default_rng(21)
    X = rng.uniform(0.0, 100.0, size=(50, 3)).astype(precision)
    p = np.array([[0.1], [0.5], [0.9]]) if axis == 0 \
        else np.array([[0.1, 0.5, 0.9]])

    tol = 100 * np.finfo(precision).eps

    sppatch()
    from scipy import stats
    da_quant = stats.quantile(X, p, axis=axis, keepdims=keepdims)

    undo_sppatch()
    from scipy import stats
    sk_quant = stats.quantile(X, p, axis=axis, keepdims=keepdims)

    assert np.shape(da_quant) == np.shape(sk_quant)
    assert da_quant == pytest.approx(sk_quant, rel=tol, abs=tol)


@requires_scipy_quantile
def test_quantile_unsupported_parameters():
    """
    Check that quantiles run successfully even when
    weights and nan_policy are different from defaults.
    """

    X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    p = [0.5]
    nan_policy = 'omit'
    weights = [100]

    sppatch()
    from scipy.stats import quantile

    with pytest.warns(RuntimeWarning):
        da_quants_nan = quantile(X, p, nan_policy=nan_policy)
    with pytest.warns(RuntimeWarning):
        da_quants_wei = quantile(X, p, weights=weights)
    with pytest.warns(RuntimeWarning):
        da_quants_both = quantile(X, p, nan_policy=nan_policy, weights=weights)

    undo_sppatch()

    # Verify they work for scipy
    sp_quants_both = quantile(X, p, nan_policy=nan_policy, weights=weights)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("axis", [0, 1, None])
def test_describe(precision, axis):
    """
    Check the describe result fields against scipy
    """
    rng = np.random.default_rng(99)
    X = rng.uniform(1.0, 10.0, size=(40, 4)).astype(precision)

    tol = 100 * np.finfo(precision).eps

    sppatch()
    from scipy import stats
    da_res = stats.describe(X, axis=axis)

    undo_sppatch()
    from scipy import stats
    sk_res = stats.describe(X, axis=axis)

    assert da_res.nobs == sk_res.nobs
    assert da_res.minmax[0] == pytest.approx(
        sk_res.minmax[0], rel=tol, abs=tol)
    assert da_res.minmax[1] == pytest.approx(
        sk_res.minmax[1], rel=tol, abs=tol)
    assert da_res.mean == pytest.approx(sk_res.mean, rel=tol, abs=tol)
    assert da_res.variance == pytest.approx(sk_res.variance, rel=tol, abs=tol)
    assert da_res.skewness == pytest.approx(sk_res.skewness, rel=tol, abs=tol)
    assert da_res.kurtosis == pytest.approx(sk_res.kurtosis, rel=tol, abs=tol)


def test_patching():
    """
    Test we can patch and unpatch using string and list arguments to sppatch
    """

    sppatch("gmean")
    from scipy import stats
    assert stats.gmean.__module__.startswith("aoclda")

    undo_sppatch("gmean")
    from scipy import stats
    assert not stats.gmean.__module__.startswith("aoclda")

    sppatch(["skew", "kurtosis"])
    from scipy import stats
    assert stats.skew.__module__.startswith("aoclda")
    assert stats.kurtosis.__module__.startswith("aoclda")

    undo_sppatch(["skew", "kurtosis"])
    from scipy import stats
    assert not stats.skew.__module__.startswith("aoclda")
    assert not stats.kurtosis.__module__.startswith("aoclda")

    with pytest.raises(ValueError, match="Wibble"):
        sppatch(["skew", "Wibble"])

    with pytest.raises(ValueError, match="Wobble"):
        undo_sppatch(["skew", "Wobble"])

    with pytest.raises(TypeError):
        sppatch(1)

    with pytest.raises(TypeError):
        undo_sppatch(1)


if __name__ == "__main__":
    test_patching()
