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


from ._aoclda.factorization import pybind_PCA

class PCA(pybind_PCA):
    """
    This is the PCA class. It does PCA things.

    Parameters:
    -----------

    n_components : int, default=1
        Number of components to keep.

    bias : {'unbiased', 'biased'}, default='unbiased'
        Whether to use unbiased or biased standard deviations

    method : {'covariance', 'correlation', 'svd'}, default='covariance'
        Whether to compute PCA based on covariance, correlation or SVD

    solver : {'auto', 'gesdd', 'gesvd', 'gesvdx'}, default='auto'
        Which LAPACK solver to use

    precision : {aoclda.single, aoclda.double}, default aoclda.double
        Double or single precision


    Attributes:
    -----------

    principal_components : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. Equivalently, the right singular
        vectors of the centered input data, parallel to its eigenvectors.
        The components are sorted by decreasing ``variance``.

    scores : ndarray of shape (n_components, n_features)
        Principal component scores.

    variance : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.

    total_variance : ndarray of shape
        Amount of variance explained by each of the selected components.

    u : ndarray of shape
        U from the SVD

    vt : ndarray of shape
        VT from the SVD

    sigma : ndarray of shape
        Sigma from the SVD

    column_means : ndarray of shape
        Column means

    column_sdevs : ndarray of shape
        Column standard deviations

    Methods:
    ---------
    :meth:`~PCA.fit`
    :meth:`~PCA.transform`
    :meth:`~PCA.inverse_transform`

    """
    @property
    def principal_components(self):
        return self.get_principal_components()

    @property
    def scores(self):
        return self.get_scores()

    @property
    def variance(self):
        return self.get_variance()

    @property
    def total_variance(self):
        return self.get_total_variance()

    @property
    def u(self):
        return self.get_u()

    @property
    def sigma(self):
        return self.get_sigma()

    @property
    def vt(self):
        return self.get_vt()

    @property
    def column_means(self):
        return self.get_column_means()

    @property
    def column_sdevs(self):
        return self.get_column_sdevs()

    def fit(self, A):
        """
        Fits a PCA

        Parameters:
        -----------

        A : ndarray of shape
            Array to use

        Returns:
        --------

        self: object
            Returns the instance itself
        """
        return self.pybind_fit(A)

    def transform(self, X):
        """
        PCA transform

        Parameters:
        -----------

        X : ndarray of shape
            Array to use

        Returns:
        --------

        X_transform: ndarray of shape
            The transformed X


        """
        return self.pybind_transform(X)

    def inverse_transform(self, Y):
        """
        PCA inverse transform

        Parameters:
        -----------

        Y : ndarray of shape
            Array to use

        Returns:
        --------

        Y_inverse_transform: ndarray of shape
            The inverse transformed Y
        """
        return self.pybind_inverse_transform(Y)