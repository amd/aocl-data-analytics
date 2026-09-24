# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
Cubic spline interpolation Python test script.
"""

import numpy as np
import pytest
from aoclda.interpolation import cubic_spline


def _tol(precision):
    return 1.0e-4 if precision == np.float32 else 1.0e-10


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("spline_type", ["natural", "clamped zero"])
def test_interpolates_at_sites(precision, numpy_order, spline_type):
    """A spline must reproduce the data values exactly at the interpolation sites."""
    x = np.linspace(0.0, 2.0 * np.pi, 11).astype(precision)
    y = np.sin(x).astype(precision)
    y = np.asarray(y, order=numpy_order)

    spline = cubic_spline(spline_type=spline_type)
    spline.fit(x, y)

    values = spline.evaluate(x)
    assert values.shape == (x.shape[0],)
    assert np.allclose(values, y, atol=_tol(precision))


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_evaluate_single_order_shapes(precision):
    """Single-order evaluation drops the order axis (and dim axis for 1D data)."""
    x = np.linspace(0.0, 1.0, 6).astype(precision)
    y = (x ** 2).astype(precision)

    spline = cubic_spline()
    spline.fit(x, y)

    x_eval = np.linspace(0.0, 1.0, 20).astype(precision)
    values = spline.evaluate(x_eval, orders=0)
    assert values.shape == (20,)

    first_deriv = spline.evaluate(x_eval, orders=1)
    assert first_deriv.shape == (20,)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_evaluate_multiple_orders(precision):
    """Requesting several orders retains the leading axis and respects input order."""
    x = np.linspace(0.0, 1.0, 6).astype(precision)
    y = (x ** 3).astype(precision)

    spline = cubic_spline()
    spline.fit(x, y)

    x_eval = np.linspace(0.0, 1.0, 7).astype(precision)
    result = spline.evaluate(x_eval, orders=[0, 1, 2])
    assert result.shape == (3, 7)

    # Order of the returned rows must match the requested orders.
    reordered = spline.evaluate(x_eval, orders=[2, 0])
    assert reordered.shape == (2, 7)
    assert np.allclose(reordered[1], result[0], atol=_tol(precision))
    assert np.allclose(reordered[0], result[2], atol=_tol(precision))


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_multidimensional_values(precision, numpy_order):
    """Multiple dimensions can be interpolated with a single set of sites."""
    x = np.linspace(0.0, 1.0, 8).astype(precision)
    y = np.empty((x.shape[0], 3), dtype=precision, order=numpy_order)
    y[:, 0] = np.sin(x)
    y[:, 1] = np.cos(x)
    y[:, 2] = x ** 2

    spline = cubic_spline()
    spline.fit(x, y)
    assert spline.n_dim == 3

    values = spline.evaluate(x)
    assert values.shape == (3, x.shape[0])
    assert np.allclose(values.T, y, atol=_tol(precision))


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_coefficients_shape(precision):
    """Coefficients have shape (n_cells, 4) for 1D data and reproduce the values."""
    x = np.linspace(0.0, 1.0, 5).astype(precision)
    y = np.sin(x).astype(precision)

    spline = cubic_spline()
    spline.fit(x, y)

    coeffs = spline.coefficients
    n_cells = x.shape[0] - 1
    assert coeffs.shape == (n_cells, 4)

    # The constant coefficient of each cell equals the value at its left site.
    assert np.allclose(coeffs[:, 0], y[:-1], atol=_tol(precision))


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_hermite_spline(precision):
    """Hermite splines accept user-supplied first derivatives at the sites."""
    x = np.linspace(0.0, 2.0 * np.pi, 9).astype(precision)
    y = np.sin(x).astype(precision)
    dydx = np.cos(x).astype(precision)

    spline = cubic_spline(spline_type="hermite")
    spline.fit(x, y, y_derivative=dydx)

    values = spline.evaluate(x)
    derivs = spline.evaluate(x, orders=1)
    assert np.allclose(values, y, atol=_tol(precision))
    assert np.allclose(derivs, dydx, atol=_tol(precision))


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_custom_boundary_conditions(precision):
    """Custom boundary conditions impose the requested derivative at the ends."""
    x = np.linspace(0.0, 1.0, 6).astype(precision)
    y = (x ** 2).astype(precision)

    # Clamp the first derivative to 0 at the left and 2 at the right.
    bcs = ((1, np.array([0.0], dtype=precision)),
           (1, np.array([2.0], dtype=precision)))

    spline = cubic_spline(spline_type="custom")
    spline.fit(x, y, boundary_conditions=bcs)

    deriv = spline.evaluate(np.array([0.0, 1.0], dtype=precision), orders=1)
    assert np.allclose(deriv, np.array([0.0, 2.0]), atol=_tol(precision))


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_incomplete_boundary_conditions(precision):
    """Supplying only one of the left/right boundary values raises an error."""
    x = np.linspace(0.0, 1.0, 6).astype(precision)
    y = (x ** 2).astype(precision)

    # Only the left boundary value is supplied; the right is missing.
    bcs = ((1, np.array([0.0], dtype=precision)), (1, None))

    spline = cubic_spline(spline_type="custom")
    with pytest.raises((TypeError, ValueError)):
        spline.fit(x, y, boundary_conditions=bcs)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_search_cells(precision):
    """search_cells returns the index of the containing cell for each query point."""
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=precision)
    y = np.array([0.0, 1.0, 4.0, 9.0], dtype=precision)

    spline = cubic_spline()
    spline.fit(x, y)

    cells = spline.search_cells(np.array([0.5, 1.5, 2.5], dtype=precision))
    assert np.array_equal(cells, np.array([0, 1, 2]))

    # Out-of-bounds points are clamped to the first and last cells.
    clamped = spline.search_cells(np.array([-1.0, 10.0], dtype=precision))
    assert np.array_equal(clamped, np.array([0, 2]))


@pytest.mark.parametrize("precisions", [('float32', 'float64'),
                                        ('float64', 'float32')])
def test_dtype_dispatch(precisions):
    """The wrapper dispatches to the single or double precision binding by dtype."""
    x = np.linspace(0.0, 1.0, 6).astype(precisions[0])
    y = np.sin(x).astype(precisions[0])

    spline = cubic_spline()
    spline.fit(x, y)
    values = spline.evaluate(x)
    assert values.dtype == np.dtype(precisions[0])
