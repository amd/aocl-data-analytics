..
    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.


.. _chapter_interpolation:

Interpolation
*******************************

Interpolation is the process of estimating unknown values that fall between known data points. Given a discrete set of data points,
interpolation methods construct a continuous function that passes through these points, enabling the computation of intermediate values.

This chapter describes the interpolation capabilities available in the library. Currently, only cubic spline interpolation is supported for one-dimensional data.

Cubic splines
=================

Cubic spline interpolation constructs a smooth piecewise cubic polynomial that passes through a given set of data points.
Given a set of :math:`n` interpolation sites :math:`(x_i, y_i)` where :math:`i = 0, 1, \ldots, n-1`, the cubic spline
produces a function :math:`S(x)` that is continuous and has continuous first and second derivatives across the entire domain.

Between each pair of consecutive points :math:`[x_i, x_{i+1}]`, the spline is a cubic polynomial of the form:

.. math::
   S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3 \quad x \in [x_i, x_{i+1}]

The coefficients are determined by ensuring continuity of the function and its first two derivatives at each interpolation point :math:`x_i`,
along with boundary conditions at the endpoints. Common boundary conditions include natural splines (second derivative equals zero
at the boundaries), clamped splines (first derivative specified at the boundaries). Hermite splines are a variant where not only
the function values but also the derivative values are specified at the interpolation sites, providing additional control over
the shape of the interpolating curve.

Cubic splines provide a smooth approximation with minimal oscillations and are widely used in computer graphics, data fitting,
and numerical analysis applications.

Typical workflow
------------------------------------------------

The standard way of using cubic spline interpolation in AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      Python interface not yet available.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_interpolation``.
      2. Select the interpolation model using :ref:`da_interpolation_select_model_? <da_interpolation_select_model>`.
      3. Define the interpolation sites using either :ref:`da_interpolation_set_sites_? <da_interpolation_set_sites>` (for custom x-coordinates) or :ref:`da_interpolation_set_sites_uniform_? <da_interpolation_set_sites_uniform>` (for uniformly spaced points).
      4. Set the function values (and optionally first derivatives for Hermite splines) using :ref:`da_interpolation_set_values_? <da_interpolation_set_values>`.
      5. Set the spline type and other options using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <interpolation_options>`).
      6. Optionally set custom boundary conditions using :ref:`da_interpolation_set_boundary_conditions_? <da_interpolation_set_boundary_conditions>` (required when "cubic spline type" is set to "custom").
      7. Compute the spline coefficients using :ref:`da_interpolation_interpolate_? <da_interpolation_interpolate>`.
      8. Evaluate the spline at query points using :ref:`da_interpolation_evaluate_? <da_interpolation_evaluate>`.
      9. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`.


.. _interpolation_options:

Supported Spline Types
----------------------

The library supports several spline types, controlled by the "cubic spline type" option:

- **natural**: Natural cubic splines with zero second derivatives at both endpoints. This is the default option and provides a smooth curve with minimal curvature at the boundaries.

- **clamped zero**: Clamped cubic splines with zero first derivatives at both endpoints. The curve enters and exits horizontally at the boundary points.

- **custom**: Custom boundary conditions where the user specifies the first or second derivative values at the endpoints. This type requires calling :ref:`da_interpolation_set_boundary_conditions <da_interpolation_set_boundary_conditions>` before computing the spline to define the boundary conditions.

- **hermite**: Piecewise cubic Hermite interpolation where both function values and first derivative values are specified at each interpolation site. This type requires providing the first derivative values through a call to :ref:`da_interpolation_set_values <da_interpolation_set_values>` with the ``order`` parameter set to 1, in addition to the function values (``order`` = 0).


Multi-dimensional data
------------------------

The interpolation functions support multi-dimensional data through the :ref:`da_interpolation_set_values_? <da_interpolation_set_values>` function.
When working with multi-dimensional data, the function expects an :math:`n \times \text{dim}` matrix where :math:`n` is the number of interpolation
sites and :math:`\text{dim}` is the dimensionality of the data.
The storage order (row-major or column-major) is controlled by the "storage order" option (column-major is the default).

Evaluating the spline
-----------------------

The spline can be evaluated at query points using the :ref:`da_interpolation_evaluate_? <da_interpolation_evaluate>` function. Multiple derivative orders can be requested at once.
The requested values are returned in a flat array :math:`y_{\text{data}}` of size :math:`n \times \text{dim} \times n_{\text{order}}`, always giving in contiguous memory all the values of a given dimension and order.
The ordering can be visualized as follows:

.. math::

   \begin{aligned}
   & S^0(x_0), \, S^0(x_1), \, \ldots, \, S^0(x_{n-1}) \curvearrowright && \text{(dimension 0, order 0)} \\
   & S^1(x_0), \, S^1(x_1), \, \ldots, \, S^1(x_{n-1}) \curvearrowright && \text{(dimension 1, order 0)} \\
   & \qquad \vdots && \vdots \\
   & S^{\text{dim}}(x_0), \, S^{\text{dim}}(x_1), \, \ldots, \, S^{\text{dim}}(x_{n-1}) \curvearrowright && \text{(dimension dim, order 0)} \\
   & \qquad \vdots && \vdots \\
   & {S^{\text{dim}}}'(x_0), \, {S^{\text{dim}}}'(x_1), \, \ldots, \, {S^{\text{dim}}}'(x_{n-1}) \curvearrowright && \text{(dimension dim, order 1)} \\
   & \qquad \vdots && \vdots \\
   & {S^{\text{dim}}}''(x_0), \, {S^{\text{dim}}}''(x_1), \, \ldots, \, {S^{\text{dim}}}''(x_{n-1}) \curvearrowright && \text{(dimension dim, order 2)} \\
   & \qquad \vdots && \vdots
   \end{aligned}


Extracting the polynomial coefficients
--------------------------------------

After computing the spline using :ref:`da_interpolation_interpolate_? <da_interpolation_interpolate>`, the polynomial coefficients can be extracted
using the :ref:`da_handle_get_result_? <da_handle_get_result>` function with the ``da_cubic_spline_coefficients`` query.
For polynomials of the form

.. math::
   S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3

the coefficients of the k dimensions are returned as a flat array in the following order:

.. math::

    \begin{aligned}
    & a_{0,0}, b_{0,0}, c_{0,0}, d_{0,0}, a_{1,0}, b_{1,0}, c_{1,0}, d_{1,0}, \ldots, a_{n-2,0}, b_{n-2,0}, c_{n-2,0}, d_{n-2,0}, \\
    &a_{0,1}, b_{0,1}, c_{0,1}, d_{0,1}, a_{1,1}, b_{1,1}, c_{1,1}, d_{1,1}, \ldots, a_{n-2,1}, b_{n-2,1}, c_{n-2,1}, d_{n-2,1}, \\
    &\vdots \\
    &a_{0,k-1}, b_{0,k-1}, c_{0,k-1}, d_{0,k-1}, a_{1,k-1}, b_{1,k-1}, c_{1,k-1}, d_{1,k-1}, \ldots, a_{n-2,k-1}, b_{n-2,k-1}, c_{n-2,k-1}, d_{n-2,k-1} \\
    \end{aligned}

where :math:`n` is the number of interpolation sites and :math:`k` is the dimensionality of the data.




Options
----------

.. update options using table _opts_interpolation

.. csv-table:: Interpolation options
   :header: "Option name", "Type", "Default", "Description", "Constraints"

   "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
   "cubic spline type", "string", ":math:`s=` `natural`", "Type of cubic spline to construct. Options: 'natural' (zero second derivatives at endpoints), 'clamped zero' (zero first derivatives at endpoints), 'custom' (user-specified first or second derivatives at endpoints), 'Hermite' (piecewise cubic Hermite interpolation).", ":math:`s=` `clamped zero`, `custom`, `hermite`, or `natural`."
   "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."



Interpolation APIs
=========================

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      Python interfaces not yet available.

   .. tab-item:: C
      :sync: C

        .. _da_interpolation_select_model:

        .. doxygenfunction:: da_interpolation_select_model_s
            :project: da
            :outline:
        .. doxygenfunction:: da_interpolation_select_model_d
            :project: da

        .. _da_interpolation_set_sites:

        .. doxygenfunction:: da_interpolation_set_sites_s
            :project: da
            :outline:
        .. doxygenfunction:: da_interpolation_set_sites_d
            :project: da

        .. _da_interpolation_set_sites_uniform:

        .. doxygenfunction:: da_interpolation_set_sites_uniform_s
            :project: da
            :outline:
        .. doxygenfunction:: da_interpolation_set_sites_uniform_d
            :project: da

        .. _da_interpolation_set_values:

        .. doxygenfunction:: da_interpolation_set_values_s
            :project: da
            :outline:
        .. doxygenfunction:: da_interpolation_set_values_d
            :project: da

        .. _da_interpolation_search_cells:

        .. doxygenfunction:: da_interpolation_search_cells_s
            :project: da
            :outline:
        .. doxygenfunction:: da_interpolation_search_cells_d
            :project: da

        .. _da_interpolation_set_boundary_conditions:

        .. doxygenfunction:: da_interpolation_set_boundary_conditions_s
            :project: da
            :outline:
        .. doxygenfunction:: da_interpolation_set_boundary_conditions_d
            :project: da

        .. _da_interpolation_interpolate:

        .. doxygenfunction:: da_interpolation_interpolate_s
            :project: da
            :outline:
        .. doxygenfunction:: da_interpolation_interpolate_d
            :project: da

        .. _da_interpolation_evaluate:

        .. doxygenfunction:: da_interpolation_evaluate_s
            :project: da
            :outline:
        .. doxygenfunction:: da_interpolation_evaluate_d
            :project: da

        .. _da_interpolation_model:

        .. doxygentypedef:: da_interpolation_model
            :project: da
        .. doxygenenum:: da_interpolation_model_
            :project: da



