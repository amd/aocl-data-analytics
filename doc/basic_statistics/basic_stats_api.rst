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



Basic Statistics APIs
*********************

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
