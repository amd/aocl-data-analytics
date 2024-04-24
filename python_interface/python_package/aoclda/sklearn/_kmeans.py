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
Patching scikit learn clustering: kmeans
"""
# pylint: disable = missing-function-docstring, too-many-ancestors, useless-return, super-init-not-called, no-member

import warnings
from sklearn.cluster import KMeans as kmeans_sklearn
from aoclda.clustering import kmeans as kmeans_da
import numpy as np


class kmeans(kmeans_sklearn):
    """
    Overwrite sklearn kmeans to call DA library
    """

    def __init__(self, n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001,
                 verbose=None, random_state=None, copy_x=None, algorithm='lloyd'):

        # Supported attributes
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_init = n_init
        self.tol = tol
        self.init = init

        # Not supported yet
        self.copy_x = copy_x
        self.verbose = verbose

        # Check for unsupported attributes
        if (copy_x is not None or verbose is not None):
            warnings.warn(
                "The parameters copy_x and verbose are not supported and have been ignored.",
                category=RuntimeWarning)

        if callable(init):
            raise ValueError("init must be set to 'random', 'random partitions', 'k-means++' "
                             "or an array.")

        # new internal attributes
        self.aocl = True
        self.seed = random_state

        ## guard against some deprecated options in Scikit-learn
        algorithm_internal = self.algorithm
        if algorithm_internal == "full" or algorithm_internal == "auto":
            algorithm_internal = "lloyd"

        if isinstance(random_state, np.random.RandomState):
            raise ValueError("random_state must be an integer or None.")

        if random_state is None:
            self.seed = -1

        if isinstance(init, np.ndarray):
            self.kmeans = kmeans_da(n_clusters, initialization_method = "supplied", n_init = 1,
                                    precision="double", max_iter = self.max_iter, seed = self.seed,
                                    algorithm = algorithm_internal)
        elif n_init == "auto":
            self.kmeans = kmeans_da(n_clusters, initialization_method = self.init, n_init = 10,
                                    precision="double", max_iter = self.max_iter, seed = self.seed,
                                    algorithm = algorithm_internal)
        else:
            self.kmeans = kmeans_da(n_clusters, initialization_method = self.init,
                                    n_init = self.n_init, precision="double",
                                    max_iter = self.max_iter, seed = self.seed,
                                    algorithm = algorithm_internal)

    def fit(self, X, y=None, sample_weight = None):
        if isinstance(self.init, np.ndarray):
            self.kmeans.fit(X, C = self.init, tol = self.tol)
        else:
            self.kmeans.fit(X, tol = self.tol)

    def transform(self, X):
        return self.kmeans.transform(X)

    def predict(self, X):
        return self.kmeans.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.kmeans.transform(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.kmeans.predict(X)

    def get_feature_names_out(self, input_features=None):
        raise RuntimeError("This feature is not implemented")

    def get_metadata_routing(self, *args):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'n_clusters': self.n_clusters,
                  'max_iter': self.max_iter,
                  'algorithm': self.algorithm,
                  'random_state': self.random_state,
                  'n_init': self.n_init,
                  'tol': self.tol,
                  'init': self.init,
                  'copy_x': self.copy_x,
                  'verbose': self.verbose}
        return params

    def score(self, X, y=None):
        raise RuntimeError("This feature is not implemented")

    def set_fit_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_output(self, *, transform=None):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, *, transform=None):
        raise RuntimeError("This feature is not implemented")

    def set_predict_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    # Match all attributes from sklearn
    # return None if not yet written
    @property
    def cluster_centers_(self):
        return self.kmeans.get_cluster_centres()

    @property
    def labels_(self):
        return self.kmeans.get_labels()

    @property
    def inertia_(self):
        return self.kmeans.get_inertia().item(0)

    @property
    def n_iter_(self):
        return self.kmeans.get_n_iter().item(0)

    @property
    def n_features_in_(self):
        print("This attribute is not implemented")
        return None

    @property
    def feature_names_in_(self):
        print("This attribute is not implemented")
        return None

    # AOCL-DA attributes not yet matched with an sklearn attribute
