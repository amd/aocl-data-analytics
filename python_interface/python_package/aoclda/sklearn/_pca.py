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
Patching scikit learn decomposition: PCA
"""
# pylint: disable = missing-function-docstring, too-many-ancestors, useless-return, super-init-not-called

from sklearn.decomposition import PCA as PCA_sklearn
from aoclda.factorization import PCA as PCA_da
import aoclda as da


class PCA(PCA_sklearn):
    """
    Overwrite sklearn PCA to call DA library
    """

    def __init__(self, n_components=None, *, copy=True, whiten=False, svd_solver='auto',
                 tol=0.0, iterated_power='auto', n_oversamples=10,
                 power_iteration_normalizer='auto', random_state=None):
        # Supported attributes
        self.n_components = n_components

        # Not supported yet
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

        # new internal attributes
        self.aocl = True
        self.pca = PCA_da(n_components=3, method="covariance",
                          solver=self.svd_solver, precision=da.double, bias='unbiased')

    def fit(self, X, y=None):
        self.pca.fit(X)

    # Match all attributes from sklearn
    # return None if not yet written
    @property
    def components_(self):
        return self.pca.get_principal_components()

    @property
    def explained_variance_(self):
        return self.pca.get_variance()

    @property
    def explained_variance_ratio_(self):
        print("NOT IMPLEMENTED!")
        return None

    @property
    def singular_values_(self):
        return self.pca.get_sigma()

    @property
    def mean_(self):
        return self.pca.get_column_means()

    @property
    def n_components_(self):
        print("NOT IMPLEMENTED!")
        return None

    @property
    def n_samples_(self):
        print("NOT IMPLEMENTED!")
        return None

    @property
    def noise_variance_(self):
        print("NOT IMPLEMENTED!")
        return None

    @property
    def n_features_in_(self):
        print("NOT IMPLEMENTED!")
        return None

    @property
    def feature_names_in_(self):
        print("NOT IMPLEMENTED!")
        return None

    # AOCL-DA attributes not yet matched with an sklearn attribute

    @property
    def scores(self):
        return self.pca.get_scores()

    @property
    def total_variance(self):
        return self.pca.get_total_variance()

    @property
    def u(self):
        return self.pca.get_u()

    @property
    def vt(self):
        return self.pca.get_vt()

    @property
    def column_sdevs(self):
        return self.pca.get_column_sdevs()
