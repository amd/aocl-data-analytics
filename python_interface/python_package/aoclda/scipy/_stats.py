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
Patching scipy.stats basic statistics:
describe, gmean, hmean, kurtosis, moment, skew, quantile
"""
# pylint: disable = too-many-positional-arguments, invalid-name, too-many-arguments, too-many-locals

import numpy as np

from aoclda.basic_stats import (
    mean as mean_da,
    variance as variance_da,
    geometric_mean as geometric_mean_da,
    harmonic_mean as harmonic_mean_da,
    skewness as skewness_da,
    kurtosis as kurtosis_da,
    moment as moment_da,
    quantile as quantile_da,
    five_point_summary as five_point_summary_da,
)
from ._utils import _warn_if

try:
    from scipy.stats._stats_py import DescribeResult
except ImportError:
    from collections import namedtuple
    DescribeResult = namedtuple(
        "DescribeResult",
        ("nobs", "minmax", "mean", "variance", "skewness", "kurtosis"))


def _apply(da_func, a, axis, **kwargs):
    """
    Map a scipy-style (a, axis) reduction onto an AOCL-DA basic_stats routine.

    scipy reduces along an integer ``axis`` (or all elements if ``axis`` is None)
    whereas AOCL-DA uses the string axes 'col' (axis=0), 'row' (axis=1) and 'all'.
    A 1D input is treated as a single column and a scalar is returned, matching
    scipy's behaviour.

    The input dtype is preserved (float32 stays float32); the AOCL-DA routines
    handle the array conversion via ``check_convert_data``.
    """
    ndim = np.ndim(a)
    if ndim == 1:
        return da_func(np.reshape(a, (-1, 1)), axis="col", **kwargs)[0]
    if ndim != 2:
        raise ValueError("Only 1D and 2D arrays are supported")
    if axis is None:
        return da_func(a, axis="all", **kwargs)[0]
    if axis == 0:
        return da_func(a, axis="col", **kwargs)
    if axis == 1:
        return da_func(a, axis="row", **kwargs)
    raise ValueError(f"Unsupported axis {axis}")


def _nobs(a, axis):
    """
    Number of observations scipy reduces over for the given ``axis``.

    Matches the sample count AOCL-DA scales its central moments by: the number
    of rows for ``axis=0``, the number of columns for ``axis=1``, every element
    for ``axis=None`` and the full length of a 1D input.
    """
    shape = np.shape(a)
    if np.ndim(a) == 1:
        return shape[0]
    elif axis is None:
        return shape[0] * shape[1]
    elif axis == 0:
        return shape[0]
    elif axis == 1:
        return shape[1]
    else:
        raise ValueError(f"Unsupported axis {axis}")


def gmean(a, axis=0, dtype=None, weights=None, *, nan_policy='propagate',
          keepdims=False):
    """
    Overwrite scipy.stats.gmean to call AOCL-DA library
    """
    _warn_if(dtype is not None, "dtype")
    _warn_if(weights is not None, "weights")
    _warn_if(nan_policy != 'propagate', "nan_policy")
    _warn_if(keepdims, "keepdims")
    return _apply(geometric_mean_da, a, axis)


def hmean(a, axis=0, dtype=None, *, weights=None, nan_policy='propagate',
          keepdims=False):
    """
    Overwrite scipy.stats.hmean to call AOCL-DA library
    """
    _warn_if(dtype is not None, "dtype")
    _warn_if(weights is not None, "weights")
    _warn_if(nan_policy != 'propagate', "nan_policy")
    _warn_if(keepdims, "keepdims")
    return _apply(harmonic_mean_da, a, axis)


def skew(a, axis=0, bias=True, nan_policy='propagate', *, keepdims=False):
    """
    Overwrite scipy.stats.skew to call AOCL-DA library
    """
    _warn_if(nan_policy != 'propagate', "nan_policy")
    _warn_if(keepdims, "keepdims")
    # AOCL-DA returns the biased (Fisher-Pearson) skewness g1.
    g1 = _apply(skewness_da, a, axis)
    if not bias:
        # Correct to the adjusted Fisher-Pearson estimator used by scipy.
        n = _nobs(a, axis)
        g1 = g1 * np.sqrt(n * (n - 1)) / (n - 2)
    return g1


def kurtosis(a, axis=0, fisher=True, bias=True, nan_policy='propagate', *,
             keepdims=False):
    """
    Overwrite scipy.stats.kurtosis to call AOCL-DA library
    """
    _warn_if(nan_policy != 'propagate', "nan_policy")
    _warn_if(keepdims, "keepdims")
    # AOCL-DA returns the biased Fisher (excess) kurtosis g2 (i.e. with 3
    # already subtracted).
    g2 = _apply(kurtosis_da, a, axis)
    if not bias:
        # Correct to the unbiased excess kurtosis estimator used by scipy.
        n = _nobs(a, axis)
        g2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
    if not fisher:
        # Pearson's definition adds 3 back to the excess kurtosis.
        g2 = g2 + 3
    return g2


def moment(a, order=1, axis=0, nan_policy='propagate', *, center=None,
           keepdims=False):
    """
    Overwrite scipy.stats.moment to call AOCL-DA library
    """
    _warn_if(nan_policy != 'propagate', "nan_policy")
    _warn_if(keepdims, "keepdims")

    def _center_for(n):
        if center is None:
            return None
        if np.ndim(center) == 0:
            return np.full(n, center)
        return center

    def _single(order_k):
        ndim = np.ndim(a)
        if ndim == 1:
            return moment_da(np.reshape(a, (-1, 1)), order_k,
                             mean=_center_for(1), axis="col")[0]
        if ndim != 2:
            raise ValueError("Only 1D and 2D arrays are supported")
        shape = np.shape(a)
        if axis is None:
            return moment_da(a, order_k, mean=_center_for(1), axis="all")[0]
        if axis == 0:
            return moment_da(a, order_k, mean=_center_for(shape[1]), axis="col")
        if axis == 1:
            return moment_da(a, order_k, mean=_center_for(shape[0]), axis="row")
        raise ValueError(f"Unsupported axis {axis}")

    # AOCL-DA's moment only accepts a single order, so a vector of orders is
    # handled with one call per order and stacked along a new leading axis to
    # match scipy's behaviour.
    if np.ndim(order) > 0:
        return np.array([_single(int(k)) for k in order])
    return _single(order)


def _quantile_grid(x, probs, method, da_axis):
    """
    Compute AOCL-DA quantiles for the unique probabilities in ``probs`` and
    return a ``(len(unique), n_slices)`` lookup grid together with the inverse
    index that maps every entry of ``probs`` back into that grid.
    """
    p_unique, inv = np.unique(np.ravel(probs), return_inverse=True)
    grid = np.asarray(
        quantile_da(x, p_unique, method=method, axis=da_axis)
    ).reshape(p_unique.size, -1)
    return grid, inv.reshape(-1)


def quantile(x, p, *, method='linear', axis=0, nan_policy='propagate',
             keepdims=None):
    """
    Overwrite scipy.stats.quantile to call AOCL-DA library.

    scipy positions ``p`` inside the coordinate system of ``x``: values along
    the reduction ``axis`` are the requested quantiles, whereas values along the
    remaining axes broadcast against the corresponding (per-column/per-row)
    statistics. AOCL-DA instead returns the full ``(len(p), n_stats)`` grid, so
    the grid is indexed here to reproduce scipy's broadcast result.
    """
    _warn_if(nan_policy != 'propagate', "nan_policy")

    x_arr = np.asarray(x)
    p_arr = np.asarray(p)
    ndim = x_arr.ndim
    if ndim not in (1, 2):
        raise ValueError("Only 1D and 2D arrays are supported")

    n_dim = max(ndim, p_arr.ndim)

    # --- 1D data or axis=None: reduce over every element, p selects quantiles --
    if ndim == 1 or axis is None:
        mat = x_arr.reshape(-1, 1)
        p_flat = p_arr.reshape(-1)
        grid, inv = _quantile_grid(mat, p_flat, method, "col")
        res = grid[inv, 0]
        k = p_flat.size
        keepdims_eff = (k != 1) if keepdims is None else keepdims
        if ndim == 1:
            # Result follows the shape of p (scalar in, scalar out).
            res = res.reshape(p_arr.shape)
            return res[()] if res.ndim == 0 else res
        # 2D data with axis=None: scipy restores the input dimensionality.
        if keepdims_eff:
            return res.reshape((1,) * (n_dim - 1) + (k,))
        if k != 1:
            raise ValueError("`keepdims` may be False only if the length of "
                             "`p` along `axis` is 1.")
        return res[0]

    if axis not in (0, 1, -1, -2):
        raise ValueError(f"Unsupported axis {axis}")
    red = axis % 2
    other = 1 - red
    da_axis = "col" if red == 0 else "row"

    # Pad p with leading singleton axes so it lines up with the 2D data.
    p2 = p_arr.reshape((1,) * (2 - p_arr.ndim) + p_arr.shape)
    k = p2.shape[red]
    n_slices = x_arr.shape[other]
    if p2.shape[other] not in (1, n_slices):
        raise ValueError("Array shapes are incompatible for broadcasting.")
    batch = max(p2.shape[other], n_slices)

    # Pre-squeeze output shape: quantiles along the reduction axis, statistics
    # along the other axis.
    out_shape = [0, 0]
    out_shape[red] = k
    out_shape[other] = batch
    out_shape = tuple(out_shape)

    p_b = np.broadcast_to(p2, out_shape)
    slice_shape = [1, 1]
    slice_shape[other] = batch
    slice_idx = np.broadcast_to(
        np.arange(batch).reshape(slice_shape) % n_slices, out_shape)

    grid, inv = _quantile_grid(x_arr, p_b, method, da_axis)
    vals = grid[inv, slice_idx.ravel()].reshape(out_shape)

    keepdims_eff = (k != 1) if keepdims is None else keepdims
    if not keepdims_eff:
        vals = np.squeeze(vals, axis=red)
    return vals


def describe(a, axis=0, ddof=1, bias=True, nan_policy='propagate'):
    """
    Overwrite scipy.stats.describe to call AOCL-DA library
    """
    _warn_if(not bias, "bias=False")
    _warn_if(nan_policy != 'propagate', "nan_policy")

    ndim = np.ndim(a)
    shape = np.shape(a)

    if ndim == 1:
        mat, da_axis, scalar, nobs = np.reshape(
            a, (-1, 1)), "col", True, shape[0]
    elif ndim == 2:
        if axis is None:
            mat, da_axis, scalar, nobs = a, "all", True, shape[0] * shape[1]
        elif axis == 0:
            mat, da_axis, scalar, nobs = a, "col", False, shape[0]
        elif axis == 1:
            mat, da_axis, scalar, nobs = a, "row", False, shape[1]
        else:
            raise ValueError(f"Unsupported axis {axis}")
    else:
        raise ValueError("Only 1D and 2D arrays are supported")

    # scipy divides the variance by (nobs - ddof); AOCL-DA divides by the
    # supplied number of degrees of freedom directly.
    dof = nobs - ddof

    mean_v = mean_da(mat, axis=da_axis)
    var_v = variance_da(mat, dof=dof, axis=da_axis)
    skew_v = skewness_da(mat, axis=da_axis)
    kurt_v = kurtosis_da(mat, axis=da_axis)
    fps = five_point_summary_da(mat, axis=da_axis)
    minimum, maximum = fps[0], fps[4]

    if scalar:
        mean_v, var_v = mean_v[0], var_v[0]
        skew_v, kurt_v = skew_v[0], kurt_v[0]
        minimum, maximum = minimum[0], maximum[0]

    return DescribeResult(nobs, (minimum, maximum), mean_v, var_v, skew_v, kurt_v)
