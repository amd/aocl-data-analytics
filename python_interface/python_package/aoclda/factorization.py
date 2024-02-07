# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from ._aoclda.factorization import pybind_PCA

class PCA(pybind_PCA):
    """
    Principal component analysis (PCA).

    Find all or some of the principal components of a data matrix.

    Args:
        n_components (int, optional): Number of components to keep. Default=1.

        bias (str, optional): Whether to use unbiased or biased estimators for standard deviations
            and variances. It can take the values 'unbiased' or 'biased' (default: 'unbiased').

        method (str, optional): The method used to compute the PCA. Default = 'covariance'.

            - If ``method = 'covariance'`` then the columns are mean-centered so the PCA is based on
              the covariance matrix.

            - If ``method = 'correlation'`` then the columns are mean-centered and scaled by the
              standard deviation so the PCA is based on the correlation matrix.

            - If ``method = 'svd'`` then no normalization occurs so the PCA reduces to the singular
              value decomposition.

        solver (str, optional): Which LAPACK solver to use to compute the underlying singular value
            decomposition, allowed values: 'auto', 'gesdd', 'gesvd', 'gesvdx'. Default='auto'.
            If ``solver = 'auto'`` then ``gesdd`` will be used unless the number of components
            requested is less than 10% of the smallest dimension of your data matrix, in which case
            ``gesvdx`` is used.


        precision (aoclda.precision, optional): Whether to initialize the PCA object in double or
            single precision. It can take the values ``aoclda.single`` or ``aoclda.double``.
            Default = ``aoclda.double``.

    """
    @property
    def principal_components(self):
        """numpy.ndarray of shape (n_components, n_features): Principal axes in feature space,
            representing the directions of maximum variance in the data. Equivalently, the right
            singular vectors of the normalized input data, parallel to its eigenvectors. The
            components are sorted by decreasing ``variance``."""
        return self.get_principal_components()

    @property
    def scores(self):
        """numpy.ndarray of shape (n_samples, n_components): The principal component scores,
            :math:`U\Sigma`."""
        return self.get_scores()

    @property
    def variance(self):
        """numpy.ndarray of shape (n_components, ): The amount of variance explained by each of the
            selected components."""
        return self.get_variance()

    @property
    def total_variance(self):
        """numpy.ndarray of shape (1, ): The total amount of variance within the dataset."""
        return self.get_total_variance()

    @property
    def u(self):
        """nupmy.ndarray of shape (n_samples, n_samples): The matrix :math:`U` from the SVD."""
        return self.get_u()

    @property
    def sigma(self):
        """numpy.ndarray of shape (n_components,): The diagonal values of :math:`\Sigma` from the
            SVD."""
        return self.get_sigma()

    @property
    def vt(self):
        """numpy.ndarray of shape (n_components, n_features): The matrix :math:`V^T` from the
            SVD."""
        return self.get_vt()

    @property
    def column_means(self):
        """numpy.ndarray of shape (n_features, ): The column means of the data matrix.
            These are computed if ``method = 'correlation'`` or ``method = 'correlation'``."""
        return self.get_column_means()

    @property
    def column_sdevs(self):
        """numpy.ndarray of shape (n_features, ): The column standard deviations of the data matrix.
            These are only computed if ``method = 'correlation'``"""
        return self.get_column_sdevs()

    def fit(self, A):
        """
        Computes the principal component analysis on the supplied data matrix.

        Args:
            A (numpy.ndarray): The data matrix with which to compute the PCA. It has shape
              (n_components, n_features).

        Returns:
            self (object): Returns the instance itself.
        """
        return self.pybind_fit(A)

    def transform(self, X):
        """
        Transform a data matrix into new feature space.

        Transforms a data matrix ``X`` from the original coordinate system into the new coordinates
        previously computed by ``pca.fit``.
        The transformation is computed by applying any standardization used on the original data
        matrix to ``X``, then projecting ``X`` into the previously computed principal components.

        Args:
            X (numpy.ndarray): The data matrix to be transformed. It has shape
              (m_samples, m_features). Note that ``m_features`` must match ``n_features``,
              the number of features in the data matrix originally supplied to ``pca.fit``.

        Returns:
            numpy.ndarray of shape (m_samples, n_components): The transformed matrix.
        """
        return self.pybind_transform(X)

    def inverse_transform(self, Y):
        """
        Transform a data matrix into the original coordinate space.

        Transforms a data matrix ``Y`` in the new feature space back into the original coordinate
        space used by the matrix which was supplied to ``pca.fit``. The transformation is computed
        by projecting ``Y`` into the original coordinate space, then inverting any standardization
        used on the original data matrix.

        Args:
            Y (numpy.ndarray): The data matrix to be transformed. It has shape
              (k_samples, k_features). Note that ``k_features`` must match ``n_components``,
              the number of principal components computed by ``pca.fit``.

        Returns:
            numpy.ndarray of shape (k_samples, n_features): The transformed matrix.
        """
        return self.pybind_inverse_transform(Y)
