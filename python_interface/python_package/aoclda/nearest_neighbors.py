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

# pylint: disable = import-error, invalid-name, too-many-arguments

"""
aoclda.nearest_neighbors module
"""

import numpy as np
from ._aoclda.nearest_neighbors import pybind_knn_classifier
from ._aoclda.nearest_neighbors import pybind_knn_regressor
from ._aoclda.nearest_neighbors import pybind_nearest_neighbors
from ._internal_utils import check_convert_data


class knn_classifier():
    """
    A k-Nearest Neighbors (k-NN) classifier.

    Args:
        n_neighbors (int, optional): The number of nearest neighbors. Default = 5.

        weights (str, optional): The weight function used for prediction.
            Available options are 'uniform' and 'distance'. Default = 'uniform'.

        algorithm (str, optional): The underlying algorithm used to compute
            the k-nearest neighbors. Available options are 'auto', 'ball_tree', 'brute' and
            'kd_tree'. k-d trees are likely to be fastest for lower dimensional datasets, and ball
            trees may be preferred when data is not aligned along coordinate axes, but trees
            cannot not be used with the cosine distance, the squared Euclidean distance, or with the
            Minkowski distance with power less than 1.0. Default = 'auto'.

        leaf_size (int, optional): The leaf size passed to the k-d tree algorithm.
            This affects the construction of the tree and the speed of the nearest neighbor
            queries. Default = 30.

        metric (str, optional): The metric used for the distance computation.
            Available metrics are 'euclidean', 'l2', 'sqeuclidean' (squared euclidean distances),
            'manhattan', 'l1', 'cityblock', 'cosine', 'minkowski', 'euclidean_gemm',
            or 'sqeuclidean_gemm'. Default = 'minkowski'.

        p (float, optional): The power parameter used for the Minkowski metric. For p = 1.0,
            this defaults to 'manhattan' metric and for p = 2.0 this defaults to 'euclidean' metric.
            p is only used for Minkowski distance and will be ignored otherwise. Will return an
            error when p is not positive. Default p = 2.0.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2.0, check_data=False):
        self.knn_classifier_double = pybind_knn_classifier(
            n_neighbors, weights, algorithm, leaf_size, metric, "double", check_data)
        self.knn_classifier_single = pybind_knn_classifier(
            n_neighbors, weights, algorithm, leaf_size, metric, "single", check_data)
        self.knn_classifier = self.knn_classifier_double
        self.order = 'A'
        self.dtype = 'float'
        self.p = p

    def fit(self, X, y):
        """
        Fit the k-NN classifier from the training data set provided.

        Args:
            X (array-like): The feature matrix on which to fit the model.
                Its shape is (n_samples, n_features).

            y (array-like): The vector with the corresponding labels. Its shape is (n_samples).

        Returns:
            self (object): Returns the instance itself.
        """
        X, self.order, self.dtype = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        y, _, _ = check_convert_data(
            y, order=self.order, dtype="da_int", force_dtype=True
        )

        if self.dtype == "float32":
            self.knn_classifier = self.knn_classifier_single
            self.knn_classifier_double = None
            self.p = np.float32(self.p)
        else:
            self.p = np.float64(self.p)

        self.knn_classifier.pybind_fit(X, y, p=self.p)
        return self

    def kneighbors(self, X, n_neighbors=0, return_distance=True):
        """
        Compute the indices of the nearest neighbors for each test data point,
        and optionally the corresponding distances.

        Args:
            X (array-like): The test data for which the nearest neighbors are required.
                Its shape is (n_queries, n_features).

            n_neighbors (int, optional): The number of neighbors. If this is less than or equal to zero,
                the number of neighbors set while fitting the model will be used instead.

            return_distance (bool, optional): Denotes whether to return the distances or not.
                Default=True.

        Returns:
            numpy.ndarray of shape (n_queries, n_neighbors): The matrix with the corresponding
                distances to each neighbor. Only returned if return_distance=True.

            numpy.ndarray of shape (n_queries, n_neighbors): The matrix with the indices to
                each neighbor.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )

        if return_distance:
            return self.knn_classifier.pybind_kneighbors(X, n_neighbors)

        return self.knn_classifier.pybind_kneighbors_indices(X, n_neighbors)

    def predict_proba(self, X):
        """
        Compute the probability estimates for each of the available classes.

        Args:
            X (array-like): The test data for which the nearest neighbors are required.
                Its shape is (n_queries, n_features).

        Returns:
            numpy.ndarray of shape (n_queries, n_classes): The class probabilities of the test data.
            Classes are sorted in ascending order.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        return self.knn_classifier.pybind_predict_proba(X)

    def predict(self, X):
        """
        Compute the predicted labels for the test data.

        Args:
            X (array-like): The test data for which the nearest neighbors are required.
                Its shape is (n_queries, n_features).

        Returns:
            numpy.ndarray of shape (n_queries): The predicted labels of the test data.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        return self.knn_classifier.pybind_predict(X)


class knn_regressor():
    """
    A k-Nearest Neighbors (k-NN) regressor.

    Args:
        n_neighbors (int, optional): The number of nearest neighbors. Default = 5.

        weights (str, optional): The weight function used for prediction.
            Available options are 'uniform' and 'distance'. Default = 'uniform'.

        algorithm (str, optional): The underlying algorithm used to compute
            the k-nearest neighbors. Available options are 'auto', 'ball_tree', 'brute' and
            'kd_tree'. k-d trees are likely to be fastest for lower dimensional datasets, and ball
            trees may be preferred when data is not aligned along coordinate axes, but trees
            cannot not be used with the cosine distance, the squared Euclidean distance, or with the
            Minkowski distance with power less than 1.0. Default = 'auto'.

        leaf_size (int, optional): The leaf size passed to the k-d tree algorithm.
            This affects the construction of the tree and the speed of the nearest neighbor
            queries. Default = 30.

        metric (str, optional): The metric used for the distance computation.
            Available metrics are 'euclidean', 'l2', 'sqeuclidean' (squared Euclidean distances),
            'manhattan', 'l1', 'cityblock', 'cosine', or 'minkowski'. Default = 'euclidean'.

        p (float, optional): The power parameter used for the Minkowski metric. For p = 1.0,
            this defaults to 'manhattan' metric and for p = 2.0 this defaults to 'euclidean' metric.
            p is only used for Minkowski distance and will be ignored otherwise. Will return an
            error when p is not positive. Default p = 2.0.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2.0, check_data=False):
        self.knn_regressor_double = pybind_knn_regressor(
            n_neighbors, weights, algorithm, leaf_size, metric, "double", check_data)
        self.knn_regressor_single = pybind_knn_regressor(
            n_neighbors, weights, algorithm, leaf_size, metric, "single", check_data)
        self.knn_regressor = self.knn_regressor_double
        self.order = 'A'
        self.dtype = 'float'
        self.p = p

    def fit(self, X, y):
        """
        Fit the k-NN regressor from the training data set provided.

        Args:
            X (array-like): The feature matrix on which to fit the model.
                Its shape is (n_samples, n_features).

            y (array-like): The vector with the corresponding target values. Its shape is (n_samples).

        Returns:
            self (object): Returns the instance itself.
        """
        X, self.order, self.dtype = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        y, _, _ = check_convert_data(
            y, order=self.order, dtype=self.dtype, force_dtype=True
        )

        if self.dtype == "float32":
            self.knn_regressor = self.knn_regressor_single
            self.knn_regressor_double = None
            self.p = np.float32(self.p)
        else:
            self.p = np.float64(self.p)

        self.knn_regressor.pybind_fit(X, y, p=self.p)
        return self

    def kneighbors(self, X, n_neighbors=0, return_distance=True):
        """
        Compute the indices of the nearest neighbors for each test data point,
        and optionally the corresponding distances.

        Args:
            X (array-like): The test data for which the nearest neighbors are required.
                Its shape is (n_queries, n_features).

            n_neighbors (int, optional): The number of neighbors. If this is less than or equal to zero,
                the number of neighbors set while fitting the model will be used instead.

            return_distance (bool, optional): Denotes whether to return the distances or not.
                Default=True.

        Returns:
            numpy.ndarray of shape (n_queries, n_neighbors): The matrix with the corresponding
                distances to each neighbor. Only returned if return_distance=True.

            numpy.ndarray of shape (n_queries, n_neighbors): The matrix with the indices to
                each neighbor.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )

        if return_distance:
            return self.knn_regressor.pybind_kneighbors(X, n_neighbors)

        return self.knn_regressor.pybind_kneighbors_indices(X, n_neighbors)

    def predict(self, X):
        """
        Compute the predicted target values for the test data.

        Args:
            X (array-like): The test data for which the nearest neighbors are required.
                Its shape is (n_queries, n_features).

        Returns:
            numpy.ndarray of shape (n_queries): The predicted target values of the test data.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        return self.knn_regressor.pybind_predict(X)


class nearest_neighbors():
    """
    A Nearest Neighbors object.
    Args:
        n_neighbors (int, optional): The number of nearest neighbors. Default = 5.

        algorithm (str, optional): The underlying algorithm used to compute
            the k-nearest neighbors. Available options are 'auto', 'brute' and 'kd_tree'.
            k-d trees are likely to be fastest for lower dimensional datasets, but
            cannot not be used with the cosine distance, the squared Euclidean distance
            or with the Minkowski distance with power less than 1.0. Default = 'auto'.

        leaf_size (int, optional): The leaf size passed to the k-d tree algorithm.
            This affects the construction of the tree and the speed of the nearest neighbor
            queries. Default = 30.

        metric (str, optional): The metric used for the distance computation.
            Available metrics are 'euclidean', 'l2', 'sqeuclidean' (squared Euclidean distances),
            'manhattan', 'l1', 'cityblock', 'cosine', or 'minkowski'. Default = 'euclidean'.

        p (float, optional): The power parameter used for the Minkowski metric. For p = 1.0,
            this defaults to 'manhattan' metric and for p = 2.0 this defaults to 'euclidean' metric.
            p is only used for Miknowski distance and will be ignored otherwise. Will return an
            error when p is not positive. Default p = 2.0.

        radius (float, optional): The radius of the neighborhood to consider for
            :meth:`radius_neighbors` queries. Default = 1.0.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(self, n_neighbors=5, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2.0, radius=1.0, check_data=False):
        self.nearest_neighbors_double = pybind_nearest_neighbors(
            n_neighbors, algorithm, leaf_size, metric, "double", check_data)
        self.nearest_neighbors_single = pybind_nearest_neighbors(
            n_neighbors, algorithm, leaf_size, metric, "single", check_data)
        self.nearest_neighbors = self.nearest_neighbors_double
        self.order = 'A'
        self.dtype = 'float'
        self.p = p
        self.radius = radius

    def fit(self, X):
        """
        Fit the k-NN classifier from the training data set provided.
        Args:
            X (array-like): The feature matrix on which to fit the model.
            Its shape is (n_samples, n_features).

        Returns:
            self (object): Returns the instance itself.
        """
        X, self.order, self.dtype = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )

        if self.dtype == "float32":
            self.nearest_neighbors = self.nearest_neighbors_single
            self.nearest_neighbors_double = None
            self.p = np.float32(self.p)
            self.radius = np.float32(self.radius)
        else:
            self.p = np.float64(self.p)
            self.radius = np.float64(self.radius)

        self.nearest_neighbors.pybind_fit(X, p=self.p, radius=self.radius)
        return self

    def kneighbors(self, X, n_neighbors=0, return_distance=True):
        """
        Compute the indices of the nearest neighbors for each test data point,
        and optionally the corresponding distances.
        Args:
            X (array-like): The test data for which the nearest neighbors are required.
                Its shape is (n_queries, n_features).

            n_neighbors (int, optional): The number of neighbors. If this is less than or equal to zero,
                the number of neighbors set while fitting the model will be used instead.
                return_distance (bool, optional): Denotes whether to return the distances or not.
                Default=True.

        Returns:
            numpy.ndarray of shape (n_queries, n_neighbors): The matrix with the corresponding
                distances to each neighbor. Only returned if return_distance=True.

            numpy.ndarray of shape (n_queries, n_neighbors): The matrix with the indices to
                each neighbor.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )

        if return_distance:
            return self.nearest_neighbors.pybind_kneighbors(X, n_neighbors)

        return self.nearest_neighbors.pybind_kneighbors_indices(X, n_neighbors)
