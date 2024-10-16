..
    Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

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




Matrix Factorizations
*********************

This chapter contains functions for decomposing a data matrix into the product of two or more matrices.
Matrix factorizations are commonly used for dimensionality reduction and feature extraction.

.. _pca_intro:

Principal component analysis
============================

In a principal component analysis (PCA) a set of possibly correlated feature vectors (the columns of the data matrix) is transformed linearly into a new, uncorrelated coordinate system.
The new coordinates (which are known as the principal components) are chosen such that the first coordinate accounts for the greatest variance in the data, the second coordinate accounts for the second greatest variance, etc.
By using only the first few such coordinates, the data matrix can be reduced in dimension.

Prior to computing the PCA the data matrix is typically standardized by shifting each column so that it has a mean of zero.
It can then be shown that the principal components are the eigenvalues of the *covariance matrix* corresponding to the mean-centered data matrix.

If the features of the data matrix vary greatly in magnitude, then in addition to mean-centering it can be useful to normalize each column by its standard deviation.
In this case the principal components are the eigenvalues of the *correlation matrix* corresponding to the mean-centered data matrix.

The PCA is closely related to a matrix factorization known as the *singular value decomposition* (or SVD),

.. math::
   A = U\Sigma V^T,

where :math:`A` is a (standardized) data matrix of size :math:`n_{\mathrm{samples}} \times n_{\mathrm{features}}`, :math:`\Sigma` is a non-negative diagonal matrix of size :math:`n_{\mathrm{samples}} \times n_{\mathrm{features}}` and :math:`U` and :math:`V` are orthogonal matrices of size :math:`n_{\mathrm{samples}} \times n_{\mathrm{samples}}` and  :math:`n_{\mathrm{features}} \times n_{\mathrm{features}}` respectively.
The nonzero entries of :math:`\Sigma` are known as the *singular values* of :math:`A`.

AOCL-DA can compute the PCA via the SVD (with :math:`\Sigma` and :math:`V` truncated according to the number of principal components requested) or using an eigenvalue decomposition of the covariance/correlation matrix (which is faster for tall, thin matrices, but can be numerically unstable for ill-conditioned problems).

Outputs from the PCA
---------------------
After a PCA computation the following results are stored:

- **principal components** - the "new coordinates" expressed in terms of the old coordinates. These are sorted in order of decreasing variance, and are given by the rows of :math:`V^T`.
- **scores** - the data matrix expressed in terms of the new coordinates. This is given by :math:`U\Sigma`.
- **variance** - the amount of variance explained by each of the principal components. Note that :math:`n\_samples -1` degrees of freedom are used when computing variances.
- **total variance** - the total variance across the whole dataset.
- **the SVD matrices** - :math:`U`, :math:`V^T` and :math:`\Sigma` together with the column means and standard deviations.

After the PCA has been computed, two post-processing operations may be of interest:

- **transform** - given a data matrix :math:`X` in the same coordinates as the original data matrix :math:`A`, express :math:`X` in terms of the new coordinates (the principal components of :math:`A`). This is computed by applying any standardization used on :math:`A` to :math:`X` and post-multiplying by :math:`V`.
- **inverse transform** - given a data matrix :math:`Y` in the new coordinate system, express :math:`Y` in terms of the original coordinates. This is computed by post-multiplying by :math:`V^T` and inverting the standardization used on :math:`A`.


Typical workflow for PCA
------------------------
The standard way of computing the principal component analysis using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.factorization.PCA` object with options set in the class constructor.
      2. Compute the PCA for your data matrix using :func:`aoclda.factorization.PCA.fit`.
      3. Perform further transformations in necessary using :func:`aoclda.factorization.PCA.transform` or :func:`aoclda.factorization.PCA.inverse_transform`.
      4. Extract results from the :func:`aoclda.factorization.PCA` object via its class attributes.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_pca``.
      2. Pass data to the handle using :ref:`da_pca_set_data_? <da_pca_set_data>`.
      3. Set the number of principal components required and the type of PCA using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <pca_options>`).
      4. Compute the PCA using :ref:`da_pca_compute_? <da_pca_compute>`.
      5. Perform further transformations as required, using :ref:`da_pca_transform_? <da_pca_transform>` or :ref:`da_pca_inverse_transform_? <da_pca_inverse_transform>`.
      6. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`.

.. _pca_options:

Options
-------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.factorization.PCA` class constructor.

   .. tab-item:: C
      :sync: C

      The following options can be set using :ref:`da_options_set_? <da_options_set>`:

      .. update options using table _opts_principalcomponentanalysis

      .. csv-table:: PCA options
         :header: "Option Name", "Type", "Default", "Description", "Constraints"

         "pca method", "string", ":math:`s=` `covariance`", "Compute PCA based on the covariance or correlation matrix.", ":math:`s=` `correlation`, `covariance`, or `svd`."
         "degrees of freedom", "string", ":math:`s=` `unbiased`", "Whether to use biased or unbiased estimators for standard deviations and variances.", ":math:`s=` `biased`, or `unbiased`."
         "n_components", "integer", ":math:`i=1`", "Number of principal components to compute. If 0, then all components will be kept.", ":math:`0 \le i`"
         "svd solver", "string", ":math:`s=` `auto`", "Which LAPACK routine to use for the underlying singular value decomposition.", ":math:`s=` `auto`, `gesdd`, `gesvd`, `gesvdx`, or `syevd`."
         "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
         "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."

      If the `pca method` option is set to `svd` then no standardization is performed. This option should be used if the input data is already standardized or if an explicit singular value decomposition is required.
      Note, however, that if the columns of the data matrix are not mean-centered, then the computed **variance** and **total_variance** will be meaningless.

      If a full decomposition is required (so that all principal components are found) then `svd solver` should be set to `gesdd`. The LAPACK routines DGESDD or SGESDD (for double and single precision data respectively) will then be used. This choice offers the best performance, while maintaining high accuracy.
      Note that if internal heuristics determine that it is useful, a QR decomposition may be performed prior to the SVD.

      If `svd solver` is set to `syevd` then the SVD will be found by explicitly forming the covariance or correlation matrix and using LAPACK routines DSYEVD or SSYEVD to perform an eigendecomposition. This is very fast for tall, thin data matrices but for wider matrices it requires a lot of memory.
      The method is also more susceptible to ill-conditioning so must be used with care. It is incompatible with the `store U` option.

      `svd solver` should only be set to `gesvd` (so that the LAPACK routines DGESVD or SGESVD are used) if there is insufficient memory for the workspace requirements of `gesdd`, or if `gesdd` encounters convergence issues.
      If only one or two principal components are required then, depending on your data matrix, `gesvdx` may be faster (so that the LAPACK routines DGESVDX or SGESVDX are used).

      If `svd solver` is set to `auto`, then DGESDD or SGESDD will be used unless internal heuristics determine that the eigendecomposition may be used.

      If `store U` is set to 1, then the matrix :math:`U` from the SVD will be stored and used to ensure deterministic results in the signs of the principal components. Note that there may be a small performance penalty in setting this option and it cannot be used if `svd solver` is set to `syevd`.

Examples
========

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. collapse:: PCA Example

          .. literalinclude:: ../../python_interface/python_package/aoclda/examples/pca_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The code below can be found in ``pca.cpp`` in the ``examples`` folder of your installation.

      .. collapse:: PCA Example

          .. literalinclude:: ../../tests/examples/pca.cpp
              :language: C++
              :linenos:

.. toctree::
    :maxdepth: 1
    :hidden:

    factorization_api
