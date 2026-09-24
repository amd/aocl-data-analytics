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
Cubic spline interpolation tests, check output of sppatch versus scipy.interpolate
"""

# pylint: disable = import-outside-toplevel, reimported, no-member, invalid-name

import numpy as np
import pytest
from aoclda.scipy import sppatch, undo_sppatch

_X = np.linspace(0.0, 2.0 * np.pi, 11)
_Y = np.sin(_X)
_X_EVAL = np.linspace(0.0, 2.0 * np.pi, 51)


def _tol(precision):
    return 1.0e-4 if precision == np.float32 else 1.0e-8


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("bc_type", ["natural", "clamped"])
def test_cubic_spline_values(precision, bc_type):
    """
    Compare the patched CubicSpline against scipy for the supported bc_types.
    """
    tol = _tol(precision)
    x = _X.astype(precision)
    y = _Y.astype(precision)
    x_eval = _X_EVAL.astype(precision)

    sppatch()
    from scipy import interpolate
    da_spline = interpolate.CubicSpline(x, y, bc_type=bc_type)
    da_values = da_spline(x_eval)
    da_coeffs = da_spline.c

    undo_sppatch()
    from scipy import interpolate
    sk_spline = interpolate.CubicSpline(_X, _Y, bc_type=bc_type)
    sk_values = sk_spline(_X_EVAL)
    sk_coeffs = sk_spline.c

    assert da_values == pytest.approx(sk_values, abs=tol)
    assert da_coeffs == pytest.approx(sk_coeffs, abs=tol)
    assert da_spline.x == pytest.approx(sk_spline.x, abs=tol)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_cubic_spline_derivative(precision):
    """
    Compare evaluated derivatives of the patched CubicSpline against scipy.
    """
    tol = _tol(precision)
    x = _X.astype(precision)
    y = _Y.astype(precision)
    x_eval = _X_EVAL.astype(precision)

    sppatch()
    from scipy import interpolate
    da_spline = interpolate.CubicSpline(x, y, bc_type='natural')
    da_d1 = da_spline(x_eval, 1)
    da_d2 = da_spline(x_eval, 2)

    undo_sppatch()
    from scipy import interpolate
    sk_spline = interpolate.CubicSpline(_X, _Y, bc_type='natural')
    sk_d1 = sk_spline(_X_EVAL, 1)
    sk_d2 = sk_spline(_X_EVAL, 2)

    assert da_d1 == pytest.approx(sk_d1, abs=tol)
    assert da_d2 == pytest.approx(sk_d2, abs=tol)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_cubic_spline_custom_bc(precision):
    """
    Compare the patched CubicSpline against scipy for custom boundary conditions.
    """
    tol = _tol(precision)
    bc_type = ((1, 1.0), (2, 0.0))
    x = _X.astype(precision)
    y = _Y.astype(precision)
    x_eval = _X_EVAL.astype(precision)

    sppatch()
    from scipy import interpolate
    da_spline = interpolate.CubicSpline(x, y, bc_type=bc_type)
    da_values = da_spline(x_eval)

    undo_sppatch()
    from scipy import interpolate
    sk_spline = interpolate.CubicSpline(_X, _Y, bc_type=bc_type)
    sk_values = sk_spline(_X_EVAL)

    assert da_values == pytest.approx(sk_values, abs=tol)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_cubic_hermite_spline(precision):
    """
    Compare the patched CubicHermiteSpline against scipy.
    """
    tol = _tol(precision)
    x = _X.astype(precision)
    y = _Y.astype(precision)
    dydx = np.cos(_X).astype(precision)
    x_eval = _X_EVAL.astype(precision)

    sppatch()
    from scipy import interpolate
    da_spline = interpolate.CubicHermiteSpline(x, y, dydx)
    da_values = da_spline(x_eval)
    da_coeffs = da_spline.c

    undo_sppatch()
    from scipy import interpolate
    sk_spline = interpolate.CubicHermiteSpline(_X, _Y, np.cos(_X))
    sk_values = sk_spline(_X_EVAL)
    sk_coeffs = sk_spline.c

    assert da_values == pytest.approx(sk_values, abs=tol)
    assert da_coeffs == pytest.approx(sk_coeffs, abs=tol)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("bc_type", ["natural", "clamped"])
def test_cubic_spline_multidim(precision, numpy_order, bc_type):
    """
    Compare the patched CubicSpline against scipy for multi-dimensional ``y``
    supplied in both C and Fortran memory layouts.
    """
    tol = _tol(precision)
    y_multi = np.column_stack((np.sin(_X), np.cos(_X), _X ** 2))

    x = _X.astype(precision)
    y = np.array(y_multi, dtype=precision, order=numpy_order)
    x_eval = _X_EVAL.astype(precision)

    sppatch()
    from scipy import interpolate
    da_spline = interpolate.CubicSpline(x, y, bc_type=bc_type)
    da_values = da_spline(x_eval)
    da_coeffs = da_spline.c

    undo_sppatch()
    from scipy import interpolate
    sk_spline = interpolate.CubicSpline(_X, y_multi, bc_type=bc_type)
    sk_values = sk_spline(_X_EVAL)
    sk_coeffs = sk_spline.c

    assert da_values.shape == sk_values.shape
    assert da_coeffs.shape == sk_coeffs.shape
    assert da_values == pytest.approx(sk_values, abs=tol)
    assert da_coeffs == pytest.approx(sk_coeffs, abs=tol)


def test_not_a_knot_warns():
    """
    The default 'not-a-knot' boundary condition is unsupported and warns.
    """
    sppatch()
    from scipy import interpolate
    with pytest.warns(RuntimeWarning):
        interpolate.CubicSpline(_X, _Y)
    undo_sppatch()
