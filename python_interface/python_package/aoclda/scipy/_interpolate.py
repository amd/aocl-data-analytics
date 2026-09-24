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
Patching scipy.interpolate cubic splines: CubicSpline and CubicHermiteSpline
"""
# pylint: disable = invalid-name, too-many-arguments, too-many-positional-arguments

import numpy as np

from aoclda.interpolation import cubic_spline as cubic_spline_da
from ._utils import _warn_if


def _prepare_values(y, axis):
    """
    Reshape scipy-style ``y`` into the (n, dim) layout expected by AOCL-DA.

    Returns the reshaped values, the dimensionality ``dim`` and the trailing
    shape of the original array (the shape of ``y`` excluding the interpolation
    axis) so that outputs can be reshaped back afterwards.
    """
    y = np.asarray(y)
    if axis != 0:
        y = np.moveaxis(y, axis, 0)

    if y.ndim == 1:
        return y, 1, ()

    trailing = y.shape[1:]
    dim = int(np.prod(trailing))
    return y.reshape(y.shape[0], dim), dim, trailing


def _map_bc_type(bc_type):
    """
    Translate a scipy ``bc_type`` into an AOCL-DA spline type and boundary
    conditions.

    Returns a ``(spline_type, boundary_conditions)`` pair. ``boundary_conditions``
    is ``None`` unless a custom derivative specification was supplied.
    Unsupported boundary types (``'not-a-knot'`` and ``'periodic'``) fall back to
    a natural spline with a warning.
    """
    string_map = {'natural': 'natural', 'clamped': 'clamped zero'}

    if isinstance(bc_type, str):
        _warn_if(bc_type in ('not-a-knot', 'periodic'), "bc_type")
        return string_map.get(bc_type, 'natural'), None

    # bc_type is a 2-tuple (left, right). Each side is either a string or a
    # (deriv_order, deriv_value) pair.
    left, right = bc_type
    if isinstance(left, str) or isinstance(right, str):
        # Mixed string boundary specifications are not supported individually.
        _warn_if(True, "bc_type")
        return 'natural', None

    left_order, left_val = left
    right_order, right_val = right
    boundary_conditions = ((int(left_order), np.atleast_1d(left_val)),
                           (int(right_order), np.atleast_1d(right_val)))
    return 'custom', boundary_conditions


class _PPolyWrapper:
    """
    Base class providing a minimal ``scipy.interpolate.PPoly``-like interface on
    top of an AOCL-DA cubic spline. It exposes the breakpoints (``x``), the
    piecewise polynomial coefficients (``c``) and evaluation via ``__call__``.
    """

    def __init__(self, spline, x, dim, trailing):
        self._spline = spline
        self.x = np.asarray(x)
        self._dim = dim
        self._trailing = trailing

    @property
    def c(self):
        r"""
        Piecewise polynomial coefficients in the same convention as
        ``scipy.interpolate.PPoly``: descending power order along the leading
        axis, with shape ``(4, n_cells)`` for one-dimensional data or
        ``(4, n_cells) + trailing`` otherwise.
        """
        coeffs = self._spline.coefficients
        if self._dim == 1:
            # (n_cells, 4) ascending power -> (4, n_cells) descending power
            c = coeffs[:, ::-1].T
        else:
            # (dim, n_cells, 4) ascending power -> (4, n_cells, dim) descending
            c = np.transpose(coeffs[:, :, ::-1], (2, 1, 0))
            if self._trailing:
                c = c.reshape((4, c.shape[1]) + self._trailing)
        return c

    def __call__(self, x, nu=0, extrapolate=None):
        r"""
        Evaluate the spline (or its ``nu``-th derivative) at the points ``x``.
        """
        _warn_if(extrapolate is not None, "extrapolate")

        scalar = np.isscalar(x)
        x_eval = np.atleast_1d(np.asarray(x))

        values = self._spline.evaluate(x_eval, orders=int(nu))

        if self._dim == 1:
            result = values
        else:
            # AOCL-DA returns (dim, n_eval); scipy expects (n_eval,) + trailing.
            result = np.moveaxis(values, 0, -1)
            if self._trailing:
                result = result.reshape((x_eval.shape[0],) + self._trailing)

        if scalar:
            result = result[0] if self._dim == 1 else result[0, ...]
        return result


class CubicSpline(_PPolyWrapper):
    """
    Drop-in replacement for ``scipy.interpolate.CubicSpline`` backed by AOCL-DA.

    Interpolate data with a piecewise cubic polynomial that is twice
    continuously differentiable.

    Args:
        x (array-like): 1D array of strictly increasing interpolation sites.

        y (array-like): Array of values to interpolate, with ``y.shape[axis]``
            equal to ``len(x)``.

        axis (int, optional): Axis of ``y`` along which to interpolate.
            Default = 0.

        bc_type (str or tuple, optional): Boundary condition type. Supported
            values are ``'natural'`` and ``'clamped'``, or a 2-tuple
            ``((order_left, value_left), (order_right, value_right))`` specifying
            a derivative order (1 or 2) and value at each boundary. The scipy
            defaults ``'not-a-knot'`` and ``'periodic'`` are not supported and
            fall back to a natural spline with a warning. Default = 'not-a-knot'.

        extrapolate: Not supported; a warning is emitted if a value other than
            ``None`` is supplied. AOCL-DA always extrapolates using the nearest
            boundary cell.
    """

    def __init__(self, x, y, axis=0, bc_type='not-a-knot', extrapolate=None):
        _warn_if(extrapolate is not None, "extrapolate")
        spline_type, boundary_conditions = _map_bc_type(bc_type)

        y_values, dim, trailing = _prepare_values(y, axis)

        spline = cubic_spline_da(spline_type=spline_type)
        spline.fit(x, y_values, boundary_conditions=boundary_conditions)
        super().__init__(spline, x, dim, trailing)


class CubicHermiteSpline(_PPolyWrapper):
    """
    Drop-in replacement for ``scipy.interpolate.CubicHermiteSpline`` backed by
    AOCL-DA.

    Construct a piecewise cubic interpolant matching both values and first
    derivatives at the interpolation sites.

    Args:
        x (array-like): 1D array of strictly increasing interpolation sites.

        y (array-like): Array of values to interpolate, with ``y.shape[axis]``
            equal to ``len(x)``.

        dydx (array-like): Array of first derivatives at the sites, the same
            shape as ``y``.

        axis (int, optional): Axis of ``y`` along which to interpolate.
            Default = 0.

        extrapolate: Not supported; a warning is emitted if a value other than
            ``None`` is supplied. AOCL-DA always extrapolates using the nearest
            boundary cell.
    """

    def __init__(self, x, y, dydx, axis=0, extrapolate=None):
        _warn_if(extrapolate is not None, "extrapolate")

        y_values, dim, trailing = _prepare_values(y, axis)
        dydx_values, _, _ = _prepare_values(dydx, axis)

        spline = cubic_spline_da(spline_type='hermite')
        spline.fit(x, y_values, y_derivative=dydx_values)
        super().__init__(spline, x, dim, trailing)
