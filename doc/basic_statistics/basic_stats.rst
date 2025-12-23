..
    Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.

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


Basic Statistics
****************


This chapter contains functions to compute basic statistical quantities such as the mean,
variance or quantiles of a data matrix. Utility routines for standardizing the data are
also available, and correlation and covariance matrices can also be computed. For more information on basic statistical quantities, see cite:t:`da_rice`, cite:t:`da_kozw2000`.
These functions operate on a ``n_rows`` :math:`\times`  ``n_cols`` matrix stored in either column- or row-major order.

Choosing an axis
------------------

Most statistical quantities can be computed by column, by row or for the data matrix overall.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      This is specified using the :py:attr:`axis` function parameter.

      The ``axis`` parameter can take the following values:

      - ``'col'`` - statistical quantities will be computed for each column of the data matrix

      - ``'row'`` - statistical quantities will be computed for each row of the data matrix

      - ``'all'`` - statistical quantities will be computed for the whole data matrix

      For example, if the function :py:func:`mean` is called with the :py:attr:`axis` argument set
      to ``col``, then ``n_cols`` means will be computed, one for each column. If the
      :py:attr:`axis` argument is set to ``all``, then a single mean will be computed.

      Note that the functions in this chapter do not check for the presence of NaNs in your input data.

   .. tab-item:: C
      :sync: C

      This is specified using the :cpp:type:`da_axis` enum.

      The :cpp:type:`da_axis` enum can take the following values:

      - ``da_axis_col`` - statistical quantities will be computed for each column of the data matrix

      - ``da_axis_row`` - statistical quantities will be computed for each row of the data matrix

      - ``da_axis_all`` - statistical quantities will be computed for the whole data matrix

      For example, if the routine :cpp:func:`da_mean_s` is called with the :cpp:type:`da_axis` argument set
      to :cpp:enumerator:`da_axis_col`, then ``n_cols`` means will be computed, one for each column. If the
      :cpp:type:`da_axis` argument is set to ``da_axis_all``, then a single mean will be computed.

      The functions in this chapter do not check for the presence of NaNs in your input data.
      You can use the :cpp:func:`da_check_data_s` function to check for NaNs in your data.

Examples
--------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. collapse:: Basic Statistics Example

          .. literalinclude:: ../../python_interface/python_package/aoclda/examples/basic_stats_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The code below can be found in ``basic_statistics.cpp`` in the ``examples`` folder of your installation.

      .. collapse:: Basic Statistics Example Code

         .. literalinclude:: ../../tests/examples/basic_statistics.cpp
            :language: C++
            :linenos:



Basic Statistics APIs
---------------------

.. tab-set::

   .. tab-item:: Python

      .. autofunction:: aoclda.basic_stats.mean(X, axis="col")
      .. autofunction:: aoclda.basic_stats.harmonic_mean(X, axis="col")
      .. autofunction:: aoclda.basic_stats.geometric_mean(X, axis="col")
      .. autofunction:: aoclda.basic_stats.variance(X, dof=0, axis="col")
      .. autofunction:: aoclda.basic_stats.skewness(X, axis="col")
      .. autofunction:: aoclda.basic_stats.kurtosis(X, axis="col")
      .. autofunction:: aoclda.basic_stats.moment(X, k, mean=None, axis="col")
      .. autofunction:: aoclda.basic_stats.quantile(X, q, method="linear", axis="col")
      .. autofunction:: aoclda.basic_stats.five_point_summary(X, axis="col")
      .. autofunction:: aoclda.basic_stats.standardize(X, shift=None, scale=None, dof=0, reverse=False, inplace=False, axis="col")
      .. autofunction:: aoclda.basic_stats.covariance_matrix(X, dof=0)
      .. autofunction:: aoclda.basic_stats.correlation_matrix(X)

   .. tab-item:: C

      .. _da_mean:

      .. doxygenfunction:: da_mean_s
         :project: da
         :outline:
      .. doxygenfunction:: da_mean_d
         :project: da

      .. _da_geometric_mean:

      .. doxygenfunction:: da_geometric_mean_s
         :project: da
         :outline:
      .. doxygenfunction:: da_geometric_mean_d
         :project: da

      .. _da_harmonic_mean:

      .. doxygenfunction:: da_harmonic_mean_s
         :project: da
         :outline:
      .. doxygenfunction:: da_harmonic_mean_d
         :project: da

      .. _da_variance:

      .. doxygenfunction:: da_variance_s
         :project: da
         :outline:
      .. doxygenfunction:: da_variance_d
         :project: da

      .. _da_skewness:

      .. doxygenfunction:: da_skewness_s
         :project: da
         :outline:
      .. doxygenfunction:: da_skewness_d
         :project: da

      .. _da_kurtosis:

      .. doxygenfunction:: da_kurtosis_s
         :project: da
         :outline:
      .. doxygenfunction:: da_kurtosis_d
         :project: da

      .. _da_moment:

      .. doxygenfunction:: da_moment_s
         :project: da
         :outline:
      .. doxygenfunction:: da_moment_d
         :project: da

      .. _da_quantile:

      .. doxygenfunction:: da_quantile_s
         :project: da
         :outline:
      .. doxygenfunction:: da_quantile_d
         :project: da

      .. _da_five_point_summary:

      .. doxygenfunction:: da_five_point_summary_s
         :project: da
         :outline:
      .. doxygenfunction:: da_five_point_summary_d
         :project: da

      .. _da_standardize:

      .. doxygenfunction:: da_standardize_s
         :project: da
         :outline:
      .. doxygenfunction:: da_standardize_d
         :project: da

      .. _da_covariance_matrix:

      .. doxygenfunction:: da_covariance_matrix_s
         :project: da
         :outline:
      .. doxygenfunction:: da_covariance_matrix_d
         :project: da

      .. _da_correlation:

      .. doxygenfunction:: da_correlation_matrix_s
         :project: da
         :outline:
      .. doxygenfunction:: da_correlation_matrix_d
         :project: da

      .. doxygentypedef:: da_axis
         :project: da
      .. doxygenenum:: da_axis_
         :project: da
      .. doxygentypedef:: da_quantile_type
         :project: da
      .. doxygenenum:: da_quantile_type_
         :project: da
