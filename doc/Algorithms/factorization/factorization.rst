..
    Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.

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
AOCL-DA can also compute an approximate low-rank SVD using a randomized algorithm :cite:t:`da_hmt11`, which can be significantly faster than the exact solvers when only a small number of principal components is required relative to the matrix dimensions.

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

         "power normalization", "string", ":math:`s=` `qr`", "Normalization method used in the randomized solver power iteration.", ":math:`s=` `lu`, `none`, or `qr`."
         "power iterations", "integer", ":math:`i=-1`", "Number of power iterations used in the randomized solver.", ":math:`-1 \le i`"
         "degrees of freedom", "string", ":math:`s=` `unbiased`", "Whether to use biased or unbiased estimators for standard deviations and variances.", ":math:`s=` `biased`, or `unbiased`."
         "pca method", "string", ":math:`s=` `covariance`", "Compute PCA based on the covariance or correlation matrix.", ":math:`s=` `correlation`, `covariance`, or `svd`."
         "whiten", "integer", ":math:`i=0`", "Whether or not we whiten when transforming the data.", ":math:`0 \le i \le 1`"
         "seed", "integer", ":math:`i=0`", "Seed for random number generation; set to -1 for non-deterministic results. This option is only used in the randomized solver.", ":math:`-1 \le i`"
         "store u", "integer", ":math:`i=0`", "Whether or not to store the matrix U from the SVD.", ":math:`0 \le i \le 1`"
         "n_components", "integer", ":math:`i=1`", "Number of principal components to compute. If 0, then all components will be kept.", ":math:`0 \le i`"
         "svd solver", "string", ":math:`s=` `auto`", "Which LAPACK routine to use for the underlying singular value decomposition.", ":math:`s=` `auto`, `gesdd`, `gesvd`, `gesvdx`, `randomized`, or `syevd`."
         "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
         "n_oversamples", "integer", ":math:`i=10`", "Extra columns added to the random sample to reduce approximation error. This option is only used in the randomized solver.", ":math:`0 \le i`"
         "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."

      If the `pca method` option is set to `svd` then no standardization is performed. This option should be used if the input data is already standardized or if an explicit singular value decomposition is required.
      Note, however, that if the columns of the data matrix are not mean-centered, then the computed **variance** and **total_variance** will be meaningless.

      If a full decomposition is required (so that all principal components are found) then `svd solver` should be set to `gesdd`. The LAPACK routines DGESDD or SGESDD (for double and single precision data respectively) will then be used. This choice offers the best performance, while maintaining high accuracy.
      Note that if internal heuristics determine that it is useful, a QR decomposition may be performed prior to the SVD.

      If `svd solver` is set to `syevd` then the SVD will be found by explicitly forming the covariance or correlation matrix and using LAPACK routines DSYEVD or SSYEVD to perform an eigendecomposition. This is very fast for tall, thin data matrices but for wider matrices it requires a lot of memory.
      The method is also more susceptible to ill-conditioning so must be used with care. It is incompatible with the `store U` option.

      `svd solver` should only be set to `gesvd` (so that the LAPACK routines DGESVD or SGESVD are used) if there is insufficient memory for the workspace requirements of `gesdd`, or if `gesdd` encounters convergence issues.
      If only one or two principal components are required then, depending on your data matrix, `gesvdx` may be faster (so that the LAPACK routines DGESVDX or SGESVDX are used).

      If `svd solver` is set to `auto`, then internal heuristics will be used to choose from one of `gesdd`, `syevd` or `randomized`.

      If `store U` is set to 1, then the matrix :math:`U` from the SVD will be stored and used to ensure deterministic results in the signs of the principal components. Note that there may be a small performance penalty in setting this option and it cannot be used if `svd solver` is set to `syevd`.

      If `whiten` is set to 1, then the data is whitened upon transformation. This divides each principal component by its corresponding singular value and multiplies the component by a dimensional factor so the transformed data, specifically that data used to fit the PCA, has a unit diagonal covariance matrix.

      If `svd solver` is set to `randomized`, a randomized low-rank SVD :cite:t:`da_hmt11` is
      computed. Only the leading `n_components` singular values and their associated left and 
      right singular vectors are approximated, making this solver efficient when `n_components` 
      is small relative to both the number of samples and features.
      Increasing `n_oversamples` adds extra random columns beyond `n_components` to
      reduce approximation error; the default of 10 is suitable for most problems. Increasing
      `power iterations` improves accuracy for matrices whose singular values decay slowly;
      setting it to -1 lets AOCL-DA choose automatically based on the ratio of `n_components`
      to the matrix dimensions. The `power normalization` option selects the normalization
      scheme applied at each power iteration step: `qr` (default) uses a QR factorization and
      is more numerically stable, `lu` uses an LU factorization and is faster, and `none`
      skips normalization entirely (fastest, least stable). Setting `power normalization` to 
      `none` is typically not recommended as it can amplify numerical errors.

.. _kernel_pca_intro:

Kernel principal component analysis
====================================

Kernel PCA :cite:t:`da_ssm97` extends standard PCA to capture non-linear structure in data.
Given an :math:`n_{\mathrm{samples}} \times n_{\mathrm{features}}` data matrix :math:`X` whose rows
are data points :math:`x_1, \ldots, x_{n_{\mathrm{samples}}}`, kernel PCA diagonalizes a *kernel matrix*
:math:`K` where :math:`K_{ij} = k(x_i, x_j)` for a user-selected kernel function :math:`k`,
rather than diagonalizing the covariance matrix of the data directly.
This implicitly performs PCA in a (potentially infinite-dimensional) feature space without ever computing
the feature map explicitly.

AOCL-DA supports five kernel choices:

1. **Linear Kernel**:

   .. math::
       k(x, y) = x \cdot y.

2. **Polynomial Kernel**:

   .. math::
       k(x, y) = (\gamma x \cdot y + c)^d.

3. **RBF (Radial Basis Function) Kernel**:

   .. math::
       k(x, y) = \exp(-\gamma \|x - y\|^2).

4. **Sigmoid Kernel**:

   .. math::
       k(x, y) = \tanh(\gamma x \cdot y + c).

5. **Precomputed**: the user supplies the :math:`n_{\mathrm{samples}} \times n_{\mathrm{samples}}` kernel matrix directly via :ref:`da_kernel_pca_set_data_? <da_kernel_pca_set_data>`.

.. note::
   A kernel matrix must be positive semi-definite (PSD) to represent an inner product in a feature space.
   Sigmoid kernels are not guaranteed to be PSD. This can produce negative
   eigenvalues, which are clamped to zero. If large negative eigenvalues are detected,
   ``da_status_numerical_difficulties`` is returned.

.. note::
   When ``kernel='precomputed'``, the supplied :math:`n_{\mathrm{samples}} \times n_{\mathrm{samples}}` matrix
   is assumed to be symmetric; symmetry is not verified. The matrix is also assumed to be positive semi-definite.
   If the eigendecomposition contains large negative eigenvalues, ``da_status_numerical_difficulties`` is returned.
   Inverse transform is not supported with a precomputed kernel.

Outputs from the kernel PCA
-----------------------------

After a kernel PCA computation the following results are stored:

- **eigenvalues** - the eigenvalues :math:`\lambda_1 \ge \cdots \ge \lambda_{n_{\mathrm{components}}}` of the centered kernel matrix :math:`\tilde{K}`.
- **eigenvectors** - corresponding eigenvectors :math:`V`, shape :math:`n_{\mathrm{samples}} \times n_{\mathrm{components}}`.
- **scores** - training data projected into the kernel PCA space: :math:`Z = V \cdot \sqrt{\Lambda}`, shape :math:`n_{\mathrm{samples}} \times n_{\mathrm{components}}`, where :math:`\Lambda` is the diagonal matrix containing the eigenvalues.
- **dual coefficients** - coefficients :math:`W` (shape :math:`n_{\mathrm{samples}} \times n_{\mathrm{features}}`) from kernel ridge regression; populated only when ``fit inverse transform`` is set to ``"yes"``.
- **gamma** - the resolved gamma value (useful when ``gamma`` = :math:`-1.0` auto-resolves to :math:`1/n_{\mathrm{features}}`).
- **n_components** - the number of principal components computed.
- **n_samples, n_features** - the number of training samples and features used to fit the model.

After the kernel PCA has been computed, two post-processing operations may be of interest:

- **transform** - given a data matrix :math:`X`, project it into the learned kernel PCA space.
- **inverse transform** - given a matrix :math:`Y` in the reduced space, obtain an approximate reconstruction in the original feature space. Kernel PCA does not generally provide an exact mapping back to the original feature space. When inverse transforms are needed, AOCL-DA can learn an approximate inverse transform during compute using kernel ridge regression :cite:t:`da_bws04`. Incompatible with `kernel='precomputed'`.

Typical workflow for kernel PCA
---------------------------------

The standard way of computing the kernel PCA using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.factorization.KernelPCA` object with options set in the class constructor.
      2. Compute the kernel PCA for your data matrix using :func:`aoclda.factorization.KernelPCA.fit`.
      3. Perform further transformations as necessary using :func:`aoclda.factorization.KernelPCA.transform` or :func:`aoclda.factorization.KernelPCA.inverse_transform`.
      4. Extract results from the :func:`aoclda.factorization.KernelPCA` object via its class attributes.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_kernel_pca``.
      2. Pass data to the handle using :ref:`da_kernel_pca_set_data_? <da_kernel_pca_set_data>`.
      3. Set the kernel type, number of components, and other parameters using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <kernel_pca_options>`).
      4. Compute the kernel PCA using :ref:`da_kernel_pca_compute_? <da_kernel_pca_compute>`.
      5. Perform further transformations as necessary using :ref:`da_kernel_pca_transform_? <da_kernel_pca_transform>` or :ref:`da_kernel_pca_inverse_transform_? <da_kernel_pca_inverse_transform>`.
      6. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`.

.. _kernel_pca_options:

Options
-------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.factorization.KernelPCA` class constructor.

   .. tab-item:: C
      :sync: C

      The following options can be set using :ref:`da_options_set_? <da_options_set>`:

      .. update options using table _opts_kernelprincipalcomponentanalysis

      .. csv-table:: Kernel PCA options
         :header: "Option Name", "Type", "Default", "Description", "Constraints"

         "coef0", "real", ":math:`r=1`", "Independent term for polynomial and sigmoid kernels.", "There are no constraints on :math:`r`."
         "kernel", "string", ":math:`s=` `linear`", "Kernel function to use.", ":math:`s=` `linear`, `poly`, `precomputed`, `rbf`, or `sigmoid`."
         "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."
         "n_oversamples", "integer", ":math:`i=10`", "Extra columns added to the random sample to reduce approximation error. This option is only used in the randomized solver.", ":math:`0 \le i`"
         "n_components", "integer", ":math:`i=0`", "Number of kernel principal components to compute.", ":math:`0 \le i`"
         "degree", "integer", ":math:`i=3`", "Degree for the polynomial kernel.", ":math:`1 \le i`"
         "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
         "fit inverse transform", "string", ":math:`s=` `no`", "Whether to fit the inverse transform.", ":math:`s=` `no`, or `yes`."
         "gamma", "real", ":math:`r=-1`", "Kernel coefficient for rbf, poly, and sigmoid kernels.", "There are no constraints on :math:`r`."
         "remove zero eig", "string", ":math:`s=` `no`", "Whether to remove components whose eigenvalue is zero.", ":math:`s=` `no`, or `yes`."
         "power normalization", "string", ":math:`s=` `qr`", "Normalization method used in the randomized solver power iteration.", ":math:`s=` `lu`, `none`, or `qr`."
         "alpha", "real", ":math:`r=1`", "Ridge regularization parameter for the inverse transform linear solve.", ":math:`0 < r`"
         "copy data", "string", ":math:`s=` `yes`", "Whether or not to store a copy of the training data.", ":math:`s=` `no`, or `yes`."
         "eigensolver", "string", ":math:`s=` `auto`", "Which method to use for computing the eigendecomposition of the kernel matrix", ":math:`s=` `auto`, `randomized`, or `syevd`."
         "power iterations", "integer", ":math:`i=-1`", "Number of power iterations used in the randomized solver.", ":math:`-1 \le i`"
         "seed", "integer", ":math:`i=0`", "Seed for random number generation; set to -1 for non-deterministic results. This option is only used in the randomized solver.", ":math:`-1 \le i`"

      If ``n_components`` is set to 0, all principal components with a non-zero eigenvalue are retained.

      When ``fit inverse transform`` is set to ``"yes"``, an approximate inverse transform is fitted during ``compute`` using kernel ridge regression.
      This is performed at ``compute`` time rather than on the first call to ``inverse_transform``, so it carries additional memory and compute overhead.
      This option should only be set when ``inverse_transform`` will be used. It is incompatible with ``kernel='precomputed'``.

      When ``remove zero eig`` is enabled, components corresponding to near-zero eigenvalues are discarded after ``compute``.
      The resulting ``n_components`` may be smaller than the value requested; query ``da_handle_get_result_int`` with ``da_kernel_pca_n_components`` to obtain the number of components computed.

      When ``copy data`` is set to ``"no"``, the user's data pointer is retained rather than copied and must remain valid through all subsequent calls.

      If ``eigensolver`` is set to ``auto``, then internal heuristics are used to choose between ``syevd`` and ``randomized``. 

      If ``eigensolver`` is set to ``randomized``, a randomized symmetric eigendecomposition
      :cite:t:`da_hmt11` is used to compute the leading eigenpairs of the centered kernel
      matrix. This is efficient when ``n_components`` is small relative to ``n_samples``.
      The ``n_oversamples``, ``power iterations``, and ``power normalization`` options have the
      same meaning as for the :ref:`PCA randomized solver <pca_options>`.

      For the ``rbf``, ``poly``, and ``sigmoid`` kernels, when ``gamma`` is not set (or set to a value less than :math:`0`) it defaults to :math:`1/n_{\mathrm{features}}` and is resolved at compute time.
      The resolved value can be retrieved via ``da_handle_get_result_?`` with ``da_kernel_pca_gamma``.

Examples
========

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. dropdown:: PCA Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/pca_ex.py
              :language: Python
              :linenos:

      .. dropdown:: Kernel PCA Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/kernel_pca_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The code below can be found in the ``examples`` folder of your installation.

      .. dropdown:: PCA Example

          .. literalinclude:: ../../../tests/examples/pca.cpp
              :language: C++
              :linenos:

      .. dropdown:: Kernel PCA Example

          .. literalinclude:: ../../../tests/examples/kernel_pca.cpp
              :language: C++
              :linenos:


Factorization APIs
=========================

Principal component analysis and the SVD
-----------------------------------------
.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.factorization.PCA(n_components=1, bias='unbiased', method='covariance', solver='auto', store_U=False, whiten=False, n_oversamples=10, power_iterations=-1, power_normalization='QR', check_data=False)
         :members:

   .. tab-item:: C

      .. _da_pca_set_data:

      .. doxygenfunction:: da_pca_set_data_s
         :project: da
         :outline:
      .. doxygenfunction:: da_pca_set_data_d
         :project: da

      .. _da_pca_compute:

      .. doxygenfunction:: da_pca_compute_s
         :project: da
         :outline:
      .. doxygenfunction:: da_pca_compute_d
         :project: da

      .. _da_pca_transform:

      .. doxygenfunction:: da_pca_transform_s
         :project: da
         :outline:
      .. doxygenfunction:: da_pca_transform_d
         :project: da

      .. _da_pca_inverse_transform:

      .. doxygenfunction:: da_pca_inverse_transform_s
         :project: da
         :outline:
      .. doxygenfunction:: da_pca_inverse_transform_d
         :project: da

Kernel principal component analysis
-------------------------------------
.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.factorization.KernelPCA(n_components=0, kernel='linear', eigensolver='syevd', gamma=-1.0, degree=3, coef0=1.0, fit_inverse_transform=False, alpha=1.0, remove_zero_eig=False, copy_X=True, n_oversamples=10, power_iterations=-1, power_normalization='QR', check_data=False)
         :members:

   .. tab-item:: C

      .. _da_kernel_pca_set_data:

      .. doxygenfunction:: da_kernel_pca_set_data_s
         :project: da
         :outline:
      .. doxygenfunction:: da_kernel_pca_set_data_d
         :project: da

      .. _da_kernel_pca_compute:

      .. doxygenfunction:: da_kernel_pca_compute_s
         :project: da
         :outline:
      .. doxygenfunction:: da_kernel_pca_compute_d
         :project: da

      .. _da_kernel_pca_transform:

      .. doxygenfunction:: da_kernel_pca_transform_s
         :project: da
         :outline:
      .. doxygenfunction:: da_kernel_pca_transform_d
         :project: da

      .. _da_kernel_pca_inverse_transform:

      .. doxygenfunction:: da_kernel_pca_inverse_transform_s
         :project: da
         :outline:
      .. doxygenfunction:: da_kernel_pca_inverse_transform_d
         :project: da
