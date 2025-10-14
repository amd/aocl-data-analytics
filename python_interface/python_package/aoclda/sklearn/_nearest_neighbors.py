# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
Patching scikit-learn classifier: KNeighborsClassifier, KNeighborsRegressor and NearestNeighbors
"""
# pylint: disable = too-many-ancestors, super-init-not-called,
# too-many-arguments, too-many-instance-attributes

import warnings
from aoclda.nearest_neighbors import knn_classifier as knn_classifier_da
from aoclda.nearest_neighbors import knn_regressor as knn_regressor_da
from aoclda.nearest_neighbors import nearest_neighbors as nearest_neighbors_da
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_sklearn
from sklearn.neighbors import KNeighborsRegressor as KNeighborsRegressor_sklearn
from sklearn.neighbors import NearestNeighbors as NearestNeighbors_sklearn


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
        self.leaf_size = leaf_size
        # new internal attributes
        self.aocl = True

        # Not supported yet
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        # Check for unsupported attributes
        if (metric_params is not None or n_jobs is not None):
            warnings.warn(
                "The parameters metric_params and n_jobs are not supported \
                 and have been ignored.",
                category=RuntimeWarning)

        # Error handling for unsupported parameters
        available_algorithms = ('auto', 'ball_tree', 'brute', 'kd_tree')
        if algorithm not in available_algorithms:
            raise ValueError(
                "Invalid algorithm chosen, available options are ", available_algorithms)
        if weights not in ('uniform', 'distance'):
            raise ValueError(
                "Invalid weights chosen, available options are 'uniform', 'distance' and None.")
        if weights is None:
            self.weights = 'uniform'

        available_metrics = ['euclidean', 'l2', 'sqeuclidean', 'manhattan',
                             'l1', 'cityblock', 'cosine', 'minkowski', 'euclidean_gemm',
                             'sqeuclidean_gemm']
        if metric not in available_metrics:
            raise ValueError(
                "Invalid metric provided, available options are ", available_metrics)

        self.knn_classifier = knn_classifier_da(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric,
            p=self.p)

    def fit(self, X, y):
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


class KNeighborsRegressor(KNeighborsRegressor_sklearn):
    """
    Overwrite scikit-learn KNeighborsRegressor to call AOCL-DA library
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
        self.leaf_size = leaf_size
        # new internal attributes
        self.aocl = True

        # Not supported yet
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        # Check for unsupported attributes
        if (metric_params is not None or n_jobs is not None):
            warnings.warn(
                "The parameters metric_params and n_jobs are not supported \
                 and have been ignored.",
                category=RuntimeWarning)

        # Error handling for unsupported parameters
        available_algorithms = ('auto', 'ball_tree', 'brute', 'kd_tree')
        if algorithm not in available_algorithms:
            raise ValueError(
                "Invalid algorithm chosen, available options are " +
                str(available_algorithms))

        if weights not in ('uniform', 'distance', None):
            raise ValueError(
                "Invalid weights chosen, available options are 'uniform', 'distance' and None.")
        if weights is None:
            self.weights = 'uniform'

        available_metrics = (
            'euclidean',
            'l2',
            'sqeuclidean',
            'manhattan',
            'l1',
            'cityblock',
            'cosine',
            'minkowski',
            'euclidean_gemm',
            'sqeuclidean_gemm')
        if metric not in available_metrics:
            raise ValueError(
                "Invalid metric provided, available options are " +
                str(available_metrics))

        self.knn_regressor = knn_regressor_da(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric,
            p=self.p)

    def fit(self, X, y):
        self.knn_regressor.fit(X, y)
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if n_neighbors is None:
            n_neigh = 0
        else:
            n_neigh = n_neighbors
        if X is None:
            raise RuntimeError("kneighbors() is not implemented for X=None")
        return self.knn_regressor.kneighbors(X, n_neigh, return_distance)

    def predict(self, X):
        return self.knn_regressor.predict(X)

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


class NearestNeighbors(NearestNeighbors_sklearn):
    """
    Overwrite scikit-learn NearestNeighbors to call AOCL-DA library
    """

    def __init__(
        self,
        n_neighbors=5,
        *,
        algorithm='auto',
        leaf_size=30,
        p=2,
        radius=1.0,
        metric='minkowski',
        metric_params=None,
        n_jobs=None
    ):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm

        self.p = p
        self.radius = radius
        self.metric = metric
        self.precision = "double"
        self.leaf_size = leaf_size
        # new internal attributes
        self.aocl = True

        # Not supported yet
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        # Check for unsupported attributes
        if (metric_params is not None or n_jobs is not None):
            warnings.warn(
                "The parameters metric_params and n_jobs are not supported \
                 and have been ignored.",
                category=RuntimeWarning)

        # Error handling for unsupported parameters
        available_algorithms = ('auto', 'ball_tree', 'brute', 'kd_tree')
        if self.algorithm not in available_algorithms:
            raise ValueError(
                "Invalid algorithm chosen, available options are " +
                str(available_algorithms))

        available_metrics = (
            'euclidean',
            'l2',
            'sqeuclidean',
            'manhattan',
            'l1',
            'cityblock',
            'cosine',
            'minkowski',
            'euclidean_gemm',
            'sqeuclidean_gemm')
        if metric not in available_metrics:
            raise ValueError(
                "Invalid metric provided, available options are " +
                str(available_metrics))

        self.nearest_neighbors_obj = nearest_neighbors_da(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            metric=self.metric,
            p=self.p,
            radius=self.radius)

    def fit(self, X, y=None):
        self.nearest_neighbors_obj.fit(X)
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if n_neighbors is None:
            n_neigh = 0
        else:
            n_neigh = n_neighbors
        if X is None:
            raise RuntimeError("kneighbors() is not implemented for X=None")
        return self.nearest_neighbors_obj.kneighbors(X, n_neigh, return_distance)

    def get_params(self, deep=True):
        params = {'algorithm': self.algorithm,
                  'leaf_size': self.leaf_size,
                  'metric': self.metric,
                  'metric_params': self.metric_params,
                  'n_jobs': self.n_jobs,
                  'n_neighbors': self.n_neighbors,
                  'p': self.p,
                  'radius': self.radius}
        return params

    def get_metadata_routing(self):
        raise RuntimeError("This feature is not implemented")

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def radius_neighbors(
            self,
            X=None,
            radius=None,
            return_distance=True,
            sort_results=False):
        raise RuntimeError("This feature is not implemented")

    def radius_neighbors_graph(
            self,
            X=None,
            radius=None,
            mode='connectivity',
            sort_results=False):
        raise RuntimeError("This feature is not implemented")
