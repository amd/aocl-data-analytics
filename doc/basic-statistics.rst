
Basic statistics
================


This chapter contains functions to compute basic statistical quantities such as the mean, variance or quantiles of a data matrix. Utility routines for standardizing the data are also available, and correlation and covariance matrices can also be computed.
These functions operate on an :math:`n \times p` data array (:math:`n` observations and :math:`p` variables) stored in column major format.

Choosing an axis
-------------------

Most statistical quantities can be computed by column, by row or for the data matrix overall. This is specified using the :cpp:enum:`da_axis_` enum.

The :cpp:enum:`da_axis_` enum can take the following values:

- :cpp:enumerator:`da_axis_col` - statistical quantities will be computed for each column of the data matrix

- :cpp:enumerator:`da_axis_row` - statistical quantities will be computed for each row of the data matrix

- :cpp:enumerator:`da_axis_all` - statistical quantities will be computed for the whole data matrix

For example, if the routine :cpp:func:`da_mean_s` is called with the :cpp:enum:`da_axis_` argument set to :cpp:enumerator:`da_axis_col`, then :math:`p` means will be computed, one for each column. If the :cpp:enum:`da_axis_` argument is set to :cpp:enumerator:`da_axis_all`, then a single mean will be computed.

Examples
--------

See :math:`\texttt{basic_statistics.cpp}` in the examples folder of your installation for examples of how to use these functions.

API Reference
-------------

.. doxygenenum:: da_axis_

.. doxygenenum:: da_quantile_type_

.. doxygenfunction:: da_mean_d

.. doxygenfunction:: da_mean_s

.. doxygenfunction:: da_geometric_mean_d

.. doxygenfunction:: da_geometric_mean_s

.. doxygenfunction:: da_harmonic_mean_d

.. doxygenfunction:: da_harmonic_mean_s

.. doxygenfunction:: da_variance_d

.. doxygenfunction:: da_variance_s

.. doxygenfunction:: da_skewness_d

.. doxygenfunction:: da_skewness_s

.. doxygenfunction:: da_kurtosis_d

.. doxygenfunction:: da_kurtosis_s

.. doxygenfunction:: da_moment_d

.. doxygenfunction:: da_moment_s

.. doxygenfunction:: da_quantile_d

.. doxygenfunction:: da_quantile_s

.. doxygenfunction:: da_five_point_summary_d

.. doxygenfunction:: da_five_point_summary_s

.. doxygenfunction:: da_standardize_d

.. doxygenfunction:: da_standardize_s

.. doxygenfunction:: da_covariance_matrix_d

.. doxygenfunction:: da_covariance_matrix_s

.. doxygenfunction:: da_correlation_matrix_d

.. doxygenfunction:: da_correlation_matrix_s