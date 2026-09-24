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
aoclda.interpolation module
"""

from ._aoclda.interpolation import pybind_cubic_spline
from ._internal_utils import check_convert_data
import numpy as np


class cubic_spline():
    r"""
    Cubic spline interpolation.

    Construct a piecewise cubic polynomial that passes through a set of data
    points :math:`(x_i, y_i)`. The interpolant is twice continuously
    differentiable. One or more dimensions of data can be interpolated
    simultaneously using a single set of interpolation sites.

    Args:
        spline_type (str, optional): The type of cubic spline to compute, which
            fixes the boundary conditions imposed on the spline. Default = 'natural'.

            - If ``spline_type = 'natural'`` the second derivative of the spline is
              set to zero at both end points.

            - If ``spline_type = 'clamped zero'`` the first derivative of the spline
              is set to zero at both end points.

            - If ``spline_type = 'custom'`` the boundary conditions are supplied by
              the user through the ``boundary_conditions`` argument of
              :func:`fit`.

            - If ``spline_type = 'hermite'`` the first derivatives of the data are
              supplied by the user through the ``y_derivative`` argument of
              :func:`fit`.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(self, spline_type='natural', check_data=False):
        self.spline_double = pybind_cubic_spline(
            spline_type, 'double', check_data)
        self.spline_single = pybind_cubic_spline(
            spline_type, 'single', check_data)
        self.spline = self.spline_double
        self.order = 'A'
        self.dtype = 'float'
        self.dim = 0
        self._x = None

    def _match_order(self, arr):
        """Return arr with a memory layout matching the input data order."""
        if self.order == 'C':
            return np.ascontiguousarray(arr)
        if self.order == 'F':
            return np.asfortranarray(arr)
        return arr

    @property
    def x(self):
        r"""numpy.ndarray of shape (:nref:`n_sites`, ): The interpolation sites used
            to compute the spline."""
        return self._x

    @property
    def n_sites(self):
        """int: The number of interpolation sites used to compute the spline."""
        return self.spline.get_n_sites()

    @property
    def n_dim(self):
        """int: The dimensionality of the values used to compute the spline."""
        return self.spline.get_dim()

    @property
    def coefficients(self):
        r"""numpy.ndarray: The piecewise polynomial coefficients of the fitted spline.

            The array has shape (:nref:`n_dim`, :nref:`n_sites` - 1, 4) (the leading
            dimension is dropped when the data is one-dimensional). For cell ``i`` and
            dimension ``d`` the spline is

            .. math::
                S_{d,i}(x) = c_0 + c_1 t + c_2 t^2 + c_3 t^3, \quad t = x - x_i,

            where the four coefficients ``[c_0, c_1, c_2, c_3]`` are stored along the
            trailing axis in ascending power order."""
        if self.dim == 0:
            raise RuntimeError("The spline must be fitted before extracting "
                               "coefficients.")
        coeffs = self.spline.pybind_get_coefficients()
        if self.dim == 1:
            coeffs = coeffs[0]
        return self._match_order(coeffs)

    def fit(self, x, y, y_derivative=None, boundary_conditions=None):
        r"""
        Compute the cubic spline interpolant of the supplied data.

        Args:
            x (array-like): The interpolation sites, a one-dimensional array of shape
                (:nref:`n_sites`, ). The sites must be sorted in strictly increasing
                order with no repeated values. Constraint: :nref:`n_sites` :math:`\ge` 2.

            y (array-like): The values to interpolate at the sites. It has shape
                (:nref:`n_sites`, ) for one-dimensional data, or
                (:nref:`n_sites`, :nref:`n_dim`) to interpolate ``n_dim`` dimensions
                simultaneously.

            y_derivative (array-like, optional): The first derivatives of the data at
                the interpolation sites, used only when ``spline_type = 'hermite'``. It
                has the same shape as ``y``. Default = None.

            boundary_conditions (tuple, optional): The custom boundary conditions, used
                only when ``spline_type = 'custom'``. It is a pair
                ``((left_order, left_values), (right_order, right_values))`` where
                ``left_order`` and ``right_order`` are the derivative orders (1 for the
                first derivative or 2 for the second derivative) imposed at the left and
                right boundaries, and ``left_values`` and ``right_values`` are arrays of
                shape (:nref:`n_dim`, ) holding the corresponding derivative values for
                each dimension. Default = None.

        Returns:
            self (object): Returns the instance itself.
        """
        y, self.order, self.dtype = check_convert_data(
            y, order=self.order, dtype=self.dtype, force_dtype=True
        )
        x, _, _ = check_convert_data(
            x, order=self.order, dtype=self.dtype, force_dtype=True
        )

        if self.dtype == "float32":
            self.spline = self.spline_single
            self.spline_double = None
        else:
            self.spline = self.spline_double
            self.spline_single = None

        y_deriv = None
        if y_derivative is not None:
            y_deriv, _, _ = check_convert_data(
                y_derivative, order=self.order, dtype=self.dtype, force_dtype=True
            )

        left_order, left_values, right_order, right_values = 2, None, 2, None
        if boundary_conditions is not None:
            (left_order, left_values), (right_order,
                                        right_values) = boundary_conditions
            left_values, _, _ = check_convert_data(
                left_values, order=self.order, dtype=self.dtype, force_dtype=True
            )
            right_values, _, _ = check_convert_data(
                right_values, order=self.order, dtype=self.dtype, force_dtype=True
            )

        self.spline.pybind_fit(x, y, y_deriv, int(left_order), left_values,
                               int(right_order), right_values)
        self.dim = self.spline.get_dim()
        self._x = x
        return self

    def evaluate(self, x_eval, orders=0):
        r"""
        Evaluate the fitted spline (or one of its derivatives) at a set of query points.

        Args:
            x_eval (array-like): The query points, a one-dimensional array of shape
                (n_eval, ). Points lying outside the range of the interpolation sites are
                extrapolated using the nearest boundary cell.

            orders (int or array-like, optional): The derivative order, or orders, to
                evaluate. ``0`` returns the spline value, ``1`` the first derivative,
                ``2`` the second derivative and ``3`` the third derivative. An array of
                orders may be supplied to evaluate several at once. Default = 0.

        Returns:
            numpy.ndarray: The evaluated values.

            - When a single order is requested for one-dimensional data the result has
              shape (n_eval, ).

            - When a single order is requested for ``n_dim``-dimensional data the result
              has shape (:nref:`n_dim`, n_eval).

            - When multiple orders are requested the leading axis of length
              ``len(orders)`` is retained, giving shape (len(orders), n_eval) for
              one-dimensional data or (len(orders), :nref:`n_dim`, n_eval) otherwise. The
              rows are ordered to match the supplied ``orders``.
        """
        x_eval, _, _ = check_convert_data(
            x_eval, order=self.order, dtype=self.dtype, force_dtype=True
        )

        scalar_order = np.isscalar(orders)
        requested = np.atleast_1d(np.asarray(orders, dtype=int)).tolist()

        # The underlying library evaluates each distinct order once, in ascending
        # order, so deduplicate the request and map the result back afterwards.
        unique_sorted = sorted(set(requested))
        res = self.spline.pybind_evaluate(x_eval, unique_sorted)

        index_map = {o: i for i, o in enumerate(unique_sorted)}
        rows = [index_map[o] for o in requested]
        res = res[rows]

        squeeze_axes = []
        if scalar_order:
            squeeze_axes.append(0)
        if self.dim == 1:
            squeeze_axes.append(1)
        if squeeze_axes:
            res = np.squeeze(res, axis=tuple(squeeze_axes))

        return self._match_order(res)

    def search_cells(self, x_eval):
        r"""
        Find, for each query point, the index of the cell that contains it.

        Args:
            x_eval (array-like): The query points, a one-dimensional array of shape
                (n_eval, ).

        Returns:
            numpy.ndarray of shape (n_eval, ): For each query point ``x_eval[i]`` the
            index ``j`` such that ``x_eval[i]`` lies in the interval
            ``[x[j], x[j+1]]``. Points outside the range of the interpolation sites are
            clamped to the first or last cell.
        """
        x_eval, _, _ = check_convert_data(
            x_eval, order=self.order, dtype=self.dtype, force_dtype=True
        )
        return self.spline.pybind_search_cells(x_eval)
