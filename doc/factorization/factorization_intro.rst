..
    Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
    
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

where :math:`A` is a (standardized) data matrix of size :math:`n\_samples \times n\_features`, :math:`\Sigma` is a non-negative diagonal matrix of size :math:`n\_samples \times n\_features` and :math:`U` and :math:`V` are orthogonal matrices of size :math:`n\_samples \times n\_samples` and :math:`n\_features \times n\_features` respectively.
The nonzero entries of :math:`\Sigma` are known as the *singular values* of :math:`A`. 

Internally, AOCL-DA computes the PCA via the SVD rather than by eigenvalue decomposition of the covariance/correlation matrix.

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
The standard way of computing the principal component analysis using AOCL-DA  is as follows.

1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` :cpp:enum:`da_handle_pca`.
2. Pass data to the handle using either :cpp:func:`da_pca_set_data_s` or :cpp:func:`da_pca_set_data_d`.
3. Set the number of principal components required and the type of PCA using :cpp:func:`da_options_set_int` and :cpp:func:`da_options_set_string` (see :ref:`below <pca_options>`).
4. Compute the PCA using :cpp:func:`da_pca_compute_s` or :cpp:func:`da_pca_compute_d`.
5. Perform further transformations as required, using :cpp:func:`da_pca_transform_s`, :cpp:func:`da_pca_transform_d`, :cpp:func:`da_pca_inverse_transform_s`, or :cpp:func:`da_pca_inverse_transform_d`.
6. Extract results using :cpp:func:`da_handle_get_results_d` or :cpp:func:`da_handle_get_results_s`.


.. _pca_options:

Options
-------

The following option is set using  :cpp:func:`da_options_set_string`:

- ``PCA method`` - the type of PCA to compute (and, equivalently, the type of standardization applied to :math:`A`). This option can take the following values:

   - ``covariance`` - the data matrix is mean-centered. The computed PCA then corresponds to the eigendecomposition of the covariance matrix.

   - ``correlation`` - the data matrix is mean-centered and normalized so that each column has unit standard deviation. The computed PCA then corresponds to the eigendecomposition of the correlation matrix. This option should be chosen if the columns vary significantly in magnitude.
  
   - ``svd`` - no standardization is applied to the data matrix. This option should be chosen if the data matrix has already been normalized, or if a plain SVD is required.

The following option is set using  :cpp:func:`da_options_set_int`:

- ``n_components`` - the number of principal components (or singular values) to compute. This option must lie between :math:`1` and :math:`\min(n\_samples, n\_features)`.


Examples
========

See ``basic_pca.cpp`` in the examples folder of your installation for examples of how to use these functions.

.. toctree::
    :maxdepth: 1
    :hidden:
    
    factorization_api
