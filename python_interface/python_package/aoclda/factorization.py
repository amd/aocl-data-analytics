# Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
aoclda.factorization module
"""

import pickle
from ._aoclda.factorization import pybind_PCA, pybind_KernelPCA, pybind_NMF
from ._internal_utils import check_convert_data
import numpy as np


class PCA():
    """
    Principal component analysis (PCA).

    Find all or some of the principal components of a data matrix.

    Args:
        n_components (int, optional): Number of components to keep. Default=1.

        bias (str, optional): Whether to use unbiased or biased estimators for standard deviations
            and variances. It can take the values 'unbiased' or 'biased'. Default: 'unbiased'.

        method (str, optional): The method used to compute the PCA. Default = 'covariance'.

            - If ``method = 'covariance'`` then the columns are mean-centered so the PCA is based on
              the covariance matrix.

            - If ``method = 'correlation'`` then the columns are mean-centered and scaled by the
              standard deviation so the PCA is based on the correlation matrix.

            - If ``method = 'svd'`` then no normalization occurs so the PCA reduces to the singular
              value decomposition.

        solver (str, optional): Which LAPACK solver to use to compute the underlying singular value
            decomposition, allowed values: 'auto', 'gesdd', 'gesvd', 'gesvdx', 'syevd',
            'randomized'.
            If ``solver = 'syevd'`` then then the SVD will be found by explicitly forming the
            covariance or correlation matrix and performing an eigendecomposition. This is very fast
            for tall, thin data matrices but for wider matrices it requires a lot of memory. The
            method is also more susceptible to ill-conditioning so must be used with care. It is
            incompatible with the 'store_U' option.
            If ``solver = 'auto'`` then internal heuristics will be used to choose from one of
            `gesdd`, `syevd` or `randomized`.
            If ``solver = 'randomized'`` then a randomized SVD algorithm is used. This can be
            significantly faster for large matrices when only a small number of components is
            required. See ``n_oversamples``, ``power_iterations``, and
            ``power_normalization`` to control accuracy.
            Default='auto'.

        store_U (bool, optional): Controls whether to store the matrix ``U`` from the singular
            value decomposition. This allows deterministic results regarding the sign of the
            principal components, at the expense of some extra computation. This option cannot be
            used if ``solver = 'syevd'``. Default = False.

        whiten (bool, optional): Whether to whiten the data upon transformation.
            This divides each principal component by its corresponding singular value and
            multiplies the component by a dimensional factor so the transformed data, specifically
            that data used to fit the PCA, has a unit diagonal covariance matrix. Default = False.

        n_oversamples (int, optional): Number of extra columns used in the randomized SVD to
            improve numerical accuracy. Higher values give better results at the cost of more
            computation. Only used when ``solver='randomized'``. Default = 10.

        power_iterations (int, optional): Number of power iterations performed by the randomized
            SVD. Higher values improve accuracy for matrices whose singular values decay slowly.
            ``-1`` selects an automatic heuristic. Only used when ``solver='randomized'``.
            Default = -1.

        power_normalization (str, optional): Normalization method used during power
            iterations in the randomized SVD. ``'QR'`` uses QR decomposition (numerically
            stable); ``'LU'`` uses LU decomposition (faster, slightly less stable);
            ``'none'`` skips normalization entirely (fastest, least stable). Only used
            when ``solver='randomized'``. Default = 'QR'.

        seed (int, optional): Seed for random number generation. Set to ``-1`` for
            non-deterministic results. Only used when ``solver='randomized'``.
            Default = -1.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(
            self,
            n_components=1,
            bias='unbiased',
            method='covariance',
            solver='auto',
            store_U=False,
            whiten=False,
            n_oversamples=10,
            power_iterations=-1,
            power_normalization='QR',
            seed=-1,
            check_data=False):
        self.pca_double = pybind_PCA(
            n_components,
            bias,
            method,
            solver,
            store_U,
            'double',
            whiten,
            n_oversamples,
            power_iterations,
            power_normalization.lower(),
            seed,
            check_data)
        self.pca_single = pybind_PCA(
            n_components,
            bias,
            method,
            solver,
            store_U,
            'single',
            whiten,
            n_oversamples,
            power_iterations,
            power_normalization.lower(),
            seed,
            check_data)
        self.pca = self.pca_double
        self.order = 'A'
        self.dtype = 'float'

    @property
    def principal_components(self):
        r"""numpy.ndarray of shape (:nref:`n_components`, :nref:`n_features`): Principal axes in
            feature space, representing the directions of maximum variance in the data.
            Equivalently, the right singular vectors of the normalized input data, parallel to its
            eigenvectors. The components are sorted by decreasing ``variance``."""
        return self.pca.get_principal_components()

    @property
    def scores(self):
        r"""numpy.ndarray of shape (:nref:`n_samples`, :nref:`n_components`): The principal
            component scores, :math:`U\Sigma`."""
        return self.pca.get_scores()

    @property
    def variance(self):
        r"""numpy.ndarray of shape (:nref:`n_components`, ): The amount of variance explained by
            each of the selected components."""
        return self.pca.get_variance()

    @property
    def total_variance(self):
        """numpy.ndarray of shape (1, ): The total amount of variance within the dataset."""
        return self.pca.get_total_variance()

    @property
    def u(self):
        r"""numpy.ndarray of shape (:nref:`n_samples`, :nref:`n_samples`): The matrix :math:`U` from
            the SVD."""
        return self.pca.get_u()

    @property
    def sigma(self):
        r"""numpy.ndarray of shape (:nref:`n_components`,): The diagonal values of :math:`\Sigma`
            from the SVD."""
        return self.pca.get_sigma()

    @property
    def vt(self):
        r"""numpy.ndarray of shape (:nref:`n_components`, :nref:`n_features`): The matrix
            :math:`V^T` from the SVD."""
        return self.pca.get_vt()

    @property
    def column_means(self):
        r"""numpy.ndarray of shape (:nref:`n_features`, ): The column means of the data matrix.
            These are computed if ``method = 'covariance'`` or ``method = 'correlation'``."""
        return self.pca.get_column_means()

    @property
    def column_sdevs(self):
        r"""numpy.ndarray of shape (:nref:`n_features`, ): The column standard deviations of the
            data matrix. These are only computed if ``method = 'correlation'``"""
        return self.pca.get_column_sdevs()

    @property
    def n_samples(self):
        """int: The number of samples in the data matrix used to compute the PCA. """
        return self.pca.get_n_samples()

    @property
    def n_features(self):
        """int: The number of features in the data matrix used to compute the PCA. """
        return self.pca.get_n_features()

    @property
    def n_components(self):
        """int: The number of components found in the PCA. """
        return self.pca.get_n_components()

    def fit(self, A):
        r"""
        Computes the principal component analysis on the supplied data matrix.

        Args:
            A (array-like): The data matrix with which to compute the PCA. It has shape
              (:nref:`n_samples`, :nref:`n_features`).

        Returns:
            self (object): Returns the instance itself.
        """
        A, self.order, self.dtype = check_convert_data(
            A, order=self.order, dtype=self.dtype, force_dtype=True
        )

        if self.dtype == "float32":
            self.pca = self.pca_single
            self.pca_double = None

        self.pca.pybind_fit(A)
        return self

    def transform(self, X):
        r"""
        Transform a data matrix into new feature space.

        Transforms a data matrix ``X`` from the original coordinate system into the new coordinates
        previously computed by ``pca.fit``.
        The transformation is computed by applying any standardization used on the original data
        matrix to ``X``, then projecting ``X`` into the previously computed principal components.

        Args:
            X (array-like): The data matrix to be transformed. It has shape
              (m_samples, m_features). Note that m\_features must match :nref:`n_features`,
              the number of features in the data matrix originally supplied to ``pca.fit``.

        Returns:
            numpy.ndarray of shape (m_samples, :nref:`n_components`): The transformed matrix.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        return self.pca.pybind_transform(X)

    def inverse_transform(self, Y):
        r"""
        Transform a data matrix into the original coordinate space.

        Transforms a data matrix ``Y`` in the new feature space back into the original coordinate
        space used by the matrix which was supplied to ``pca.fit``. The transformation is computed
        by projecting ``Y`` into the original coordinate space, then inverting any standardization
        used on the original data matrix.

        Args:
            Y (array-like): The data matrix to be transformed. It has shape
              (k_samples, k_features). Note that k\_features must match n\_components,
              the number of principal components computed by ``pca.fit``.

        Returns:
            numpy.ndarray of shape (k_samples, :nref:`n_features`): The transformed matrix.
        """
        Y, _, _ = check_convert_data(
            Y, order=self.order, dtype=self.dtype, force_dtype=True
        )
        return self.pca.pybind_inverse_transform(Y)

    def __getstate__(self):
        """Support for pickle serialization."""
        return {
            'pybind_state': pickle.dumps(self.pca),
            'order': self.order,
            'dtype': self.dtype,
        }

    def __setstate__(self, state):
        """Support for pickle deserialization."""
        self.pca = pickle.loads(state['pybind_state'])
        self.order = state['order']
        self.dtype = state['dtype']

        if self.dtype == 'float64':
            self.pca_double = self.pca
            self.pca_single = None
        elif self.dtype == 'float32':
            self.pca_double = None
            self.pca_single = self.pca
        else:
            raise ValueError(
                f"Invalid dtype '{self.dtype}' when loading " +
                "model. Expected 'float32' or 'float64'."
            )
        return

    def print_model_versions(self):
        return self.pca.pybind_print_model_versions()


class KernelPCA():
    """
    Kernel principal component analysis (kernel PCA).

    Performs nonlinear dimensionality reduction using kernels.

    Args:
        n_components (int, optional): Number of components to keep.
            If 0, all non-zero components are kept. Default = 0.

        kernel (str, optional): Kernel function to use. Possible values:
            ``'linear'``, ``'poly'``, ``'rbf'``, ``'sigmoid'``, ``'precomputed'``.
            Default = ``'linear'``.

        eigensolver (str, optional): Method to use for eigendecomposition of the kernel matrix.
            Possible values: ``'auto'``, ``'syevd'``, ``'randomized'``.
            If ``eigensolver = 'auto'``, then internal heuristics are used to choose between
            ``syevd`` and ``randomized``.
            If ``eigensolver = 'randomized'`` then a randomized eigendecomposition algorithm is
            used. This can be significantly faster for large kernel matrices when only a small
            number of components is required. See ``n_oversamples``, ``power_iterations``, and
            ``power_normalization`` to control accuracy. Default = ``'auto'``.

        gamma (float, optional): Kernel coefficient for ``'rbf'``, ``'poly'``, and
            ``'sigmoid'`` kernels. If ``-1.0``, it is set to ``1 / n_features``.
            Default = ``-1.0``.

        degree (int, optional): Degree for the polynomial kernel. Default = 3.

        coef0 (float, optional): Independent term for ``'poly'`` and ``'sigmoid'``
            kernels. Default = ``1.0``.

        fit_inverse_transform (bool, optional): Whether to learn the approximate
            inverse transform during ``fit``, enabling ``inverse_transform``.
            Cannot be used with a precomputed kernel. Default = ``False``.

        alpha (float, optional): Ridge regularization parameter for the inverse
            transform. Must be positive. Default = ``1.0``.

        remove_zero_eig (bool, optional): Whether to remove components whose
            eigenvalue is zero. Default = ``False``.

        copy_X (bool, optional): Whether to store a copy of the user data.
            Default = ``True``.

        n_oversamples (int, optional): Number of extra columns used in the randomized
            eigendecomposition to improve numerical accuracy. Higher values give better results
            at the cost of more computation. Only used when ``eigensolver='randomized'``.
            Default = 10.

        power_iterations (int, optional): Number of power iterations performed by the randomized
            eigendecomposition. Higher values improve accuracy for kernel matrices whose
            eigenvalues decay slowly. ``-1`` selects an automatic heuristic. Only used when
            ``eigensolver='randomized'``. Default = -1.

        power_normalization (str, optional): Normalization method used during power
            iterations in the randomized eigendecomposition. ``'QR'`` uses QR decomposition
            (numerically stable); ``'LU'`` uses LU decomposition (faster, slightly less stable);
            ``'none'`` skips normalization entirely (fastest, least stable).
            Only used when ``eigensolver='randomized'``. Default = 'QR'.

        seed (int, optional): Seed for random number generation. Set to ``-1`` for
            non-deterministic results. Only used when ``eigensolver='randomized'``.
            Default = -1.

        check_data (bool, optional): Whether to check the data for NaNs prior to
            computation. Default = ``False``.
    """

    def __init__(
            self,
            n_components=0,
            kernel='linear',
            eigensolver='syevd',
            gamma=-1.0,
            degree=3,
            coef0=1.0,
            fit_inverse_transform=False,
            alpha=1.0,
            remove_zero_eig=False,
            copy_X=True,
            n_oversamples=10,
            power_iterations=-1,
            power_normalization='QR',
            seed=-1,
            check_data=False):
        self.kpca_double = pybind_KernelPCA(
            n_components,
            kernel,
            eigensolver,
            degree,
            fit_inverse_transform,
            remove_zero_eig,
            'double',
            copy_X,
            n_oversamples,
            power_iterations,
            power_normalization.lower(),
            seed,
            check_data)
        self.kpca_single = pybind_KernelPCA(
            n_components,
            kernel,
            eigensolver,
            degree,
            fit_inverse_transform,
            remove_zero_eig,
            'single',
            copy_X,
            n_oversamples,
            power_iterations,
            power_normalization.lower(),
            seed,
            check_data)
        self.kpca = self.kpca_double
        self.gamma = gamma
        self.coef0 = coef0
        self.alpha = alpha
        self.order = 'A'
        self.dtype = 'float'
        self.copy_X = copy_X

    @property
    def eigenvalues(self):
        r"""numpy.ndarray of shape (:nref:`n_components`, ): Eigenvalues
            :math:`\lambda_1 \ge \cdots \ge \lambda_{n_\mathrm{components}}` of the centered
            kernel matrix."""
        return self.kpca.get_eigenvalues()

    @property
    def eigenvectors(self):
        r"""numpy.ndarray of shape (:nref:`n_samples`, :nref:`n_components`): Eigenvectors
            :math:`V` of the centered kernel matrix."""
        return self.kpca.get_eigenvectors()

    @property
    def scores(self):
        r"""numpy.ndarray of shape (:nref:`n_samples`, :nref:`n_components`): The kernel
            PCA scores, :math:`Z = V \cdot \mathrm{diag}(\sqrt{\lambda_i})`"""
        return self.kpca.get_scores()

    @property
    def dual_coef(self):
        r"""numpy.ndarray of shape (:nref:`n_samples`, :nref:`n_features`): Dual
            coefficients from the kernel ridge regression inverse transform. Only
            available when ``fit_inverse_transform=True``."""
        return self.kpca.get_dual_coef()

    @property
    def X_fit_(self):
        r"""numpy.ndarray of shape (:nref:`n_samples`, :nref:`n_features`): The training
            data used to fit the kernel PCA, tight-packed in the fitted storage order."""
        return self.kpca.get_X_fit()

    @property
    def gamma_(self):
        """float: The resolved gamma parameter used in the computation.
            When gamma is set to a value less than 0, this returns 1 / n_features."""
        return self.kpca.get_gamma().item()

    @property
    def n_samples(self):
        """int: The number of samples in the data matrix used to compute the kernel PCA."""
        return self.kpca.get_n_samples()

    @property
    def n_features(self):
        """int: The number of features in the data matrix used to compute the kernel PCA."""
        return self.kpca.get_n_features()

    @property
    def n_components(self):
        """int: The number of components found in the kernel PCA."""
        return self.kpca.get_n_components()

    def fit(self, A):
        r"""
        Computes the kernel principal component analysis on the supplied data matrix.

        Args:
            A (array-like): The data matrix with which to compute the kernel PCA. It has
              shape (:nref:`n_samples`, :nref:`n_features`).

        Returns:
            self (object): Returns the instance itself.
        """
        A, self.order, self.dtype = check_convert_data(
            A, order=self.order, dtype=self.dtype, force_dtype=True
        )

        if self.dtype == "float32":
            self.kpca = self.kpca_single
            self.kpca_double = None
            precision = np.float32
        else:
            precision = np.float64

        self.kpca.pybind_fit(
            A,
            precision(self.gamma),
            precision(self.coef0),
            precision(self.alpha))
        return self

    def transform(self, X):
        r"""
        Transform a data matrix into the kernel PCA feature space.

        Args:
            X (array-like): The data matrix to be transformed. It has shape
              (m_samples, m_features). Note that m\_features must match :nref:`n_features`,
              the number of features in the data matrix originally supplied to ``fit``.

        Returns:
            numpy.ndarray of shape (m_samples, :nref:`n_components`): The transformed
              matrix.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        return self.kpca.pybind_transform(X)

    def inverse_transform(self, Y):
        r"""
        Transform data from the kernel PCA feature space back to the original space.

        Requires ``fit_inverse_transform=True`` to have been set before calling ``fit``.

        Args:
             Y (array-like): The data matrix to be transformed. It has shape
              (k_samples, k_features). Note that k\_features must match n\_components,
              the number of principal components computed by ``fit``.

        Returns:
            numpy.ndarray of shape (k_samples, :nref:`n_features`): The reconstructed
              matrix.
        """
        Y, _, _ = check_convert_data(
            Y, order=self.order, dtype=self.dtype, force_dtype=True
        )
        return self.kpca.pybind_inverse_transform(Y)

    def __getstate__(self):
        """Support for pickle serialization."""
        return {
            'pybind_state': pickle.dumps(self.kpca),
            'order': self.order,
            'dtype': self.dtype,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'alpha': self.alpha,
            'copy_X': self.copy_X,
        }

    def __setstate__(self, state):
        """Support for pickle deserialization."""
        self.kpca = pickle.loads(state['pybind_state'])
        self.order = state['order']
        self.dtype = state['dtype']
        self.gamma = state['gamma']
        self.coef0 = state['coef0']
        self.alpha = state['alpha']
        self.copy_X = state['copy_X']

        if self.dtype == 'float64':
            self.kpca_double = self.kpca
            self.kpca_single = None
        elif self.dtype == 'float32':
            self.kpca_double = None
            self.kpca_single = self.kpca
        else:
            raise ValueError(
                f"Invalid dtype '{self.dtype}' when loading " +
                "model. Expected 'float32' or 'float64'."
            )
        return

    def print_model_versions(self):
        return self.kpca.pybind_print_model_versions()


class NMF():
    """
    Nonnegative Matrix Factorization (NMF).

    Find two nonnegative matrices :math:`W` (transform) and :math:`H` (components)
    whose product approximates the nonnegative input matrix :math:`A`:

    .. math::

        A \\approx W H, \\quad W \\geq 0, \\quad H \\geq 0.

    Args:
        n_components (int or ``'auto'`` or None, optional): Number of components.
            If ``'auto'`` or ``None``, it is set to ``min(n_samples, n_features)``
            when ``fit`` is called. For ``fit_transform`` with custom ``W`` and ``H``,
            it is inferred from ``W``. Default = ``'auto'``.
        solver (str, optional): Numerical solver. Only ``'mu'`` is supported.
            Default = ``'mu'``.
        beta_loss (str, optional): Beta divergence to be minimized. Only
            ``'frobenius'`` is supported. Default = ``'frobenius'``.
        max_iter (int, optional): Maximum number of iterations. Default = 200.
        tol (float, optional): Tolerance for the stopping condition.
            Default = 1.0e-4.
        init (str or None, optional): Initialization method. One of ``'nndsvd'``,
            ``'nndsvda'``, ``'nndsvdar'``, ``'random'``, ``'custom'``, or
            ``'auto'`` (also ``None``). ``'auto'`` uses ``'nndsvda'`` when
            ``n_components < min(n_samples, n_features)``, otherwise ``'random'``.
            Default = ``'auto'``.
        random_state (int or None, optional): Seed for the random number
            generator used by the ``'random'`` and ``'nndsvdar'`` init methods.
            Set to ``None`` for non-deterministic results. Default = ``None``.
        check_every (int, optional): Number of MU iterations between successive
            convergence checks. The relative change of the Frobenius residual is
            evaluated once every ``check_every`` iterations and compared against
            ``tol``. Larger values reduce residual-computation overhead at the
            cost of potentially running extra iterations after convergence.
            Must be at least 1. Only effective when ``tol > 0``. Default = 10.
        check_data (bool, optional): Whether to check the input matrix for NaN
            values before fitting. Enabling this adds a small overhead but can
            help diagnose ill-formed inputs. Default = False.
    """

    def __init__(
            self,
            n_components='auto',
            solver='mu',
            beta_loss='frobenius',
            max_iter=200,
            tol=1.0e-4,
            init='auto',
            random_state=None,
            check_every=10,
            check_data=False):
        if n_components is not None and n_components != 'auto':
            if not isinstance(n_components, (int, np.integer)) or int(n_components) < 1:
                raise ValueError(
                    "n_components must be a positive integer, None, or 'auto', "
                    f"got {n_components!r}.")
            n_components = int(n_components)
        self._n_components = n_components
        # Set the number of components correctly for the C API, which expects 0 to
        # indicate "all components" (min(n_samples, n_features)).
        _n_comp = 0 if (n_components is None or n_components == 'auto') else n_components
        # None is an alias for 'auto' at the C++ layer.
        if init is None:
            init = 'auto'
        # Test validity of random_state
        if random_state is not None and (not isinstance(random_state, (int, np.integer))):
            raise ValueError(
                "random_state must be an integer or None, "
                f"got {random_state!r}.")
        # The C API expects a nonnegative integer seed, with -1 reserved to
        # indicate non-deterministic behavior.
        random_internal = random_state if random_state is not None else -1
        # check_every must be an integer.
        if not isinstance(check_every, (int, np.integer)) or int(check_every) < 1:
            raise ValueError(
                "check_every must be a positive integer, "
                f"got {check_every!r}.")
        self.nmf_double = pybind_NMF(
            _n_comp, solver, beta_loss, max_iter, tol, init,
            random_internal, check_every, 'double', check_data)
        self.nmf_single = pybind_NMF(
            _n_comp, solver, beta_loss, max_iter, tol, init,
            random_internal, check_every, 'single', check_data)
        self.nmf = self.nmf_double
        self.order = 'A'
        self.dtype = 'float'

    @property
    def h(self):
        r"""numpy.ndarray of shape (:nref:`n_components`, :nref:`n_features`): The right
            factor :math:`H` of the factorization :math:`A \approx WH`. Each row
            is a basis vector in feature space. Available after :meth:`fit` or
            :meth:`fit_transform` has been called."""
        return self.nmf.get_components()

    @property
    def w(self):
        r"""numpy.ndarray of shape (:nref:`n_samples`, :nref:`n_components`): The left
            factor :math:`W` of the factorization :math:`A \approx WH`, i.e. the
            representation of the training data in the component space. Available
            after :meth:`fit` or :meth:`fit_transform` has been called."""
        return self.nmf.get_transform()

    @property
    def reconstruction_err(self):
        r"""float: Frobenius reconstruction error :math:`\|A - WH\|_F` evaluated
            at the final iterate. Available after :meth:`fit` or
            :meth:`fit_transform` has been called."""
        return self.nmf.get_reconstruction_error()

    @property
    def n_iter(self):
        """int: Number of multiplicative-update iterations actually performed.
            May be less than ``max_iter`` if convergence was detected earlier.
            Available after :meth:`fit` or :meth:`fit_transform` has been called."""
        return self.nmf.get_n_iter()

    @property
    def n_components(self):
        """int: Rank of the factorization, i.e. the number of columns of :math:`W`
            and rows of :math:`H`. Available after :meth:`fit` or
            :meth:`fit_transform` has been called."""
        return self.nmf.get_n_components()

    @property
    def n_samples(self):
        """int: Number of samples in the training matrix supplied to :meth:`fit`
            or :meth:`fit_transform`."""
        return self.nmf.get_n_samples()

    @property
    def n_features(self):
        """int: Number of features in the training matrix supplied to :meth:`fit`
            or :meth:`fit_transform`."""
        return self.nmf.get_n_features()

    def fit(self, A, W=None, H=None):
        r"""
        Fit the NMF model to the supplied data matrix.

        Computes the factorization :math:`A \approx WH` using multiplicative
        updates. After this call the fitted factors :math:`W` and :math:`H` are
        available via :attr:`w` and :attr:`h` respectively.

        When ``init='custom'``, supply initial factors ``W`` and ``H``. Both
        must be provided together. Supplying only one raises a :class:`ValueError`.
        If ``n_components`` was set to ``'auto'`` or ``None``, the rank is
        inferred from the inner dimension of the supplied ``W``.

        Args:
            A (array-like): Nonnegative data matrix of shape
              (:nref:`n_samples`, :nref:`n_features`). All entries must be
              nonnegative.
            W (array-like, optional): Initial left factor of shape
              (:nref:`n_samples`, :nref:`n_components`). Required when
              ``init='custom'``, ignored otherwise.
            H (array-like, optional): Initial right factor of shape
              (:nref:`n_components`, :nref:`n_features`). Required when
              ``init='custom'``, ignored otherwise.

        Returns:
            self (object): Returns the instance itself.
        """
        A, self.order, self.dtype = check_convert_data(
            A, order=self.order, dtype=self.dtype, force_dtype=True
        )
        if W is not None and H is not None:
            W, _, _ = check_convert_data(
                W, order=self.order, dtype=self.dtype, force_dtype=True
            )
            H, _, _ = check_convert_data(
                H, order=self.order, dtype=self.dtype, force_dtype=True
            )
            if self._n_components is None or self._n_components == 'auto':
                n_comp = W.shape[1]
                for handle in (self.nmf_double, self.nmf_single):
                    if handle is not None:
                        handle.set_n_components(n_comp)
            if self.dtype == "float32":
                self.nmf = self.nmf_single
                self.nmf_double = None
            self.nmf.pybind_fit_custom(A, W, H)
            return self
        if W is not None or H is not None:
            raise ValueError(
                "W and H must both be supplied together for init='custom'.")
        if self.dtype == "float32":
            self.nmf = self.nmf_single
            self.nmf_double = None
        self.nmf.pybind_fit(A)
        return self

    def transform(self, X):
        r"""
        Transform a data matrix into the NMF component space.

        Solves for the left factor :math:`W` that minimizes
        :math:`\|X - W H\|_F` with :math:`H` frozen at the matrix computed
        by :meth:`fit` or :meth:`fit_transform`. Only :math:`W` is updated,
        :math:`H` is not modified.

        Args:
            X (array-like): Data matrix to be transformed. It has shape
              (m_samples, :nref:`n_features`). Note that the number of features
              must match :nref:`n_features`, the number of features in the
              matrix originally supplied to :meth:`fit`.

        Returns:
            numpy.ndarray of shape (m_samples, :nref:`n_components`): Transformed
            matrix representing ``X`` in the component space.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        return self.nmf.pybind_transform(X)

    def inverse_transform(self, Y):
        r"""
        Transform data back to the original feature space.

        Reconstructs approximate data in the original feature space by computing
        :math:`Y H`, where :math:`H` is the components matrix computed by
        :meth:`fit` or :meth:`fit_transform`.

        Args:
            Y (array-like): Transformed data matrix. It has shape
              (k_samples, :nref:`n_components`). Note that the number of columns
              must match :nref:`n_components`.

        Returns:
            numpy.ndarray of shape (k_samples, :nref:`n_features`): Reconstructed
            matrix in the original feature space.
        """
        Y, _, _ = check_convert_data(
            Y, order=self.order, dtype=self.dtype, force_dtype=True
        )
        return self.nmf.pybind_inverse_transform(Y)

    def fit_transform(self, A, W=None, H=None):
        r"""
        Fit the NMF model to ``A`` and return the left factor :math:`W`.

        Equivalent to calling :meth:`fit` followed by reading :attr:`w`.
        Note that this is distinct from calling :meth:`fit` followed by :meth:`transform`:
        :meth:`transform` re-solves for :math:`W` with :math:`H` frozen, performing an
        additional optimization pass, whereas this method returns the :math:`W` already
        computed during fitting.

        When ``init='custom'``, supply initial factors ``W`` and ``H``. Both
        must be provided together. Supplying only one raises a :class:`ValueError`.
        If ``n_components`` was set to ``'auto'`` or ``None``, the rank is
        inferred from the inner dimension of the supplied ``W``.

        Args:
            A (array-like): Nonnegative data matrix of shape
              (:nref:`n_samples`, :nref:`n_features`). All entries must be
              nonnegative.
            W (array-like, optional): Initial left factor of shape
              (:nref:`n_samples`, :nref:`n_components`). Required when
              ``init='custom'``, ignored otherwise.
            H (array-like, optional): Initial right factor of shape
              (:nref:`n_components`, :nref:`n_features`). Required when
              ``init='custom'``, ignored otherwise.

        Returns:
            numpy.ndarray of shape (:nref:`n_samples`, :nref:`n_components`):
            Left factor :math:`W` of the factorization.
        """
        self.fit(A, W=W, H=H)
        return self.nmf.get_transform()

    def __getstate__(self):
        """Support for pickle serialization."""
        return {
            'pybind_state': pickle.dumps(self.nmf),
            'order': self.order,
            'dtype': self.dtype,
            '_n_components': self._n_components,
        }

    def __setstate__(self, state):
        """Support for pickle deserialization."""
        self.nmf = pickle.loads(state['pybind_state'])
        self.order = state['order']
        self.dtype = state['dtype']
        self._n_components = state.get('_n_components', None)
        if self.dtype == 'float64':
            self.nmf_double = self.nmf
            self.nmf_single = None
        elif self.dtype == 'float32':
            self.nmf_double = None
            self.nmf_single = self.nmf
        else:
            raise ValueError(
                f"Invalid dtype '{self.dtype}' when loading " +
                "model. Expected 'float32' or 'float64'."
            )
        return

    def print_model_versions(self):
        return self.nmf.pybind_print_model_versions()
