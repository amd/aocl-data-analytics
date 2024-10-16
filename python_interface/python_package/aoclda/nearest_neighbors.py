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


class knn_classifier():
    """
    A k-Nearest Neighbors (k-NN) classifier.

    Args:
        n_neighbors (int, optional): The number of nearest neighbors. Default = 5.

        weights (str, optional): The weight function used for prediction.
            Available options are 'uniform' and 'distance'. Default = 'uniform'.

        algorithm (str, optional): The underlying algorithm used to compute
            the k-nearest neighbors. Only option is 'brute'.
            This argument is included as a placeholder for more algorithms.

        metric (str, optional): The metric used for the distance computation.
            Available metrics are 'euclidean', 'l2', 'sqeuclidean' (squared euclidean distances),
            'manhattan', 'l1', 'cityblock', 'cosine', or 'minkowski'. Default = 'euclidean'.

        p (float, optional): The power parameter used for the Minkowski metric. For p = 1.0, 
            this defaults to 'manhattan' metric and for p = 2.0 this defaults to 'euclidean' metric.
            p is only used for Miknowski distance and will be ignored otherwise. Will return an
            error when p is not positive. Default p = 2.0.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='brute', metric='euclidean',
                 p=2.0, check_data=False):
        self.knn_classifier_double = pybind_knn_classifier(n_neighbors, weights, algorithm, metric,
                                                           "double", check_data)
        self.knn_classifier_single = pybind_knn_classifier(n_neighbors, weights, algorithm, metric,
                                                           "single", check_data)
        self.knn_classifier = self.knn_classifier_double
        self.p = p

    def fit(self, X, y):
        """
        Fit the k-NN classifier from the training data set provided.

        Args:
            X (numpy.ndarray): The feature matrix on which to fit the model.
                Its shape is (n_samples, n_features).

            y (numpy.ndarray): The vector with the corresponding labels. Its shape is (n_samples).

        Returns:
            self (object): Returns the instance itself.
        """

        if X.dtype == "float32":
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
            X (numpy.ndarray): The test data for which the nearest neighbors are required.
                Its shape is (n_queries, n_features).

            n_neighbors (int, optional): The number of neighbors. If this is less or equal to zero,
                the number of neighbors set while fitting the model will be used instead.

            return_distance (bool, optional): Denotes whether to return the distances or not.
                Default=True.

        Returns:
            numpy.ndarray of shape (n_queries, n_neighbors): The matrix with the corresponding
                distances to each neighbor. Only returned if return_distance=True.

            numpy.ndarray of shape (n_queries, n_neighbors): The matrix with the indices to
                each neighbor.
        """
        if return_distance:
            return self.knn_classifier.pybind_kneighbors(X, n_neighbors)

        return self.knn_classifier.pybind_kneighbors_indices(X, n_neighbors)

    def predict_proba(self, X):
        """
        Compute the probability estimates for each of the available classes.

        Args:
            X (numpy.ndarray): The test data for which the nearest neighbors are required.
                Its shape is (n_queries, n_features).

        Returns:
            numpy.ndarray of shape (n_queries, n_classes): The class probabilities of the test data.
            Classes are sorted in ascending order.
        """
        return self.knn_classifier.pybind_predict_proba(X)

    def predict(self, X):
        """
        Compute the predicted labels for the test data.

        Args:
            X (numpy.ndarray): The test data for which the nearest neighbors are required.
                Its shape is (n_queries, n_features).

        Returns:
            numpy.ndarray of shape (n_queries): The predicted labels of the test data.
        """
        return self.knn_classifier.pybind_predict(X)
