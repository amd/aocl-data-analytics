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
Patching scikit-learn classifier: KNeighborsClassifier
"""
# pylint: disable = missing-function-docstring, too-many-ancestors, useless-return, super-init-not-called, too-many-instance-attributes, too-many-arguments

import warnings
from aoclda.nearest_neighbors import knn_classifier as knn_classifier_da
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_sklearn

class KNeighborsClassifier(KNeighborsClassifier_sklearn):
    """
    Overwrite scikit-learn KNeighborsClassifier to call AOCL-DA library
    """

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.p = p
        self.metric = metric
        self.precision = "double"
        # new internal attributes
        self.aocl = True

        # Not supported yet
        self.leaf_size = leaf_size
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        # Check for unsupported attributes
        if (leaf_size != 30 or metric_params is not None or n_jobs is not None):
            warnings.warn(
                "The parameters leaf_size, metric_params and n_jobs are not supported \
                 and have been ignored.",
                category=RuntimeWarning)

        # Add warning if any algorithm other than Brute Force is selected
        if algorithm not in ('brute'):
            algorithm = 'brute'
            warnings.warn(
                "invalid algorithm chosen, defaulting to brute.", category=RuntimeWarning)
        if weights not in ('uniform','distance'):
            raise ValueError(
                "invalid weights chosen, available options are 'uniform' and 'distance'.")
        if metric == 'minkowski':
            if p != 2:
                raise ValueError(
                "invalid Minkowski parameter, available option is 2.")
            # if the error is not thrown minkowski with p=2 is requested, so euclidean
            metric = 'euclidean'

        available_metrics = ['euclidean', 'sqeuclidean', 'minkowski']
        if metric not in available_metrics:
            raise ValueError(
                "invalid metric provided, available options are ", available_metrics)

        self.knn_classifier_double = knn_classifier_da(n_neighbors = n_neighbors,
                                                       weights = weights,
                                                       algorithm = algorithm,
                                                       metric = metric,
                                                       precision = "double")

        self.knn_classifier_single = knn_classifier_da(n_neighbors = n_neighbors,
                                                       weights = weights,
                                                       algorithm = algorithm,
                                                       metric = metric,
                                                       precision = "single")

        self.knn_classifier = self.knn_classifier_double

    def fit(self, X, y):
        # If data matrix is in single precision switch internally
        if X.dtype == "float32":
            self.precision = "single"
            self.knn_classifier = self.knn_classifier_single
            self.knn_classifier_double = None

        self.knn_classifier.fit(X, y)

        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if n_neighbors is None:
            n_neigh = 0
        else:
            n_neigh = n_neighbors
        if X is None:
            raise RuntimeError("kneighbors() is not implemented for X=None")
        return self.knn_classifier.kneighbors(X, n_neigh, return_distance)

    def predict_proba(self, X):
        return self.knn_classifier.predict_proba(X)

    def predict(self, X):
        return self.knn_classifier.predict(X)

    def score(self, X, y, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'algorithm': self.algorithm,
                  'leaf_size': self.leaf_size,
                  'metric': self.metric,
                  'metric_params': self.metric_params,
                  'n_jobs': self.n_jobs,
                  'n_neighbors': self.n_neighbors,
                  'p': self.p,
                  'weights': self.weights}
        return params

    def get_metadata_routing(self):
        raise RuntimeError("This feature is not implemented")

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, **kwargs):
        raise RuntimeError("This feature is not implemented")
