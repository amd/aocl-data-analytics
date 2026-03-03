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

# pylint: disable = import-error, invalid-name, too-many-arguments

"""
aoclda.neighbors module - contains nearest_neighbors and approximate_neighbors classes
"""

import numpy as np
from ._aoclda.neighbors import pybind_nearest_neighbors
from ._aoclda.neighbors import pybind_approximate_neighbors
from ._internal_utils import check_convert_data


class nearest_neighbors():
    r"""
    A Nearest Neighbors object for general nearest neighbor queries and classification/regression.

    Args:
        n_neighbors (int, optional): The number of nearest neighbors for
            :meth:`~nearest_neighbors.kneighbors` queries. Default = 5.

        weights (str, optional): The weight function used for prediction.
            Available options are 'uniform' and 'distance'. Default = 'uniform'.

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
            p is only used for Minkowski distance and will be ignored otherwise. Will return an
            error when p is not positive. Default p = 2.0.

        radius (float, optional): The radius of the neighborhood to consider for
            :meth:`~nearest_neighbors.radius_neighbors` queries. Default = 1.0.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(
            self,
            n_neighbors=5,
            weights='uniform',
            algorithm='auto',
            leaf_size=30,
            metric='euclidean',
            p=2.0,
            radius=1.0,
            check_data=False):

        # Check number of neighbors
        if not isinstance(n_neighbors, int) and n_neighbors is not None:
            raise ValueError(
                "Number of neighbors must be an integer or None. Got n_neighbors=" +
                str(n_neighbors))

        self.n_neighbors = n_neighbors
        if n_neighbors is None:
            n_neighbors = 1

        self.radius = radius

        if not isinstance(self.radius, (int, float)) and self.radius is not None:
            raise ValueError(
                "Radius must be a number or None. Got radius=" +
                str(self.radius))

        elif self.radius is not None and self.radius < 0.0:
            raise ValueError("Radius must be positive. Got radius=" + str(self.radius))

        self.nearest_neighbors_double = pybind_nearest_neighbors(
            n_neighbors, algorithm, leaf_size, metric, weights, "double", check_data)
        self.nearest_neighbors_single = pybind_nearest_neighbors(
            n_neighbors, algorithm, leaf_size, metric, weights, "single", check_data)
        self.nearest_neighbors = self.nearest_neighbors_double
        self.order = 'A'
        self.dtype = 'float'
        self.p = p
        self.y = None

    def fit(self, X, y=None):
        r"""
        Fit the nearest neighbors model from the training data set provided.

        Args:
            X (array-like): The feature matrix on which to fit the model.
                Its shape is (:nref:`n_samples`, :nref:`n_features`).

            y (array-like, optional): Labels for classification or targets for regression.
                Its shape is (:nref:`n_samples`). If None, only unsupervised queries are available.
                That is, only :meth:`~nearest_neighbors.kneighbors` and
                :meth:`~nearest_neighbors.radius_neighbors` can be used.

        Returns:
            self (object): Returns the instance itself.
        """
        if X is None:
            raise ValueError("fit() is not implemented for X=None")
        X, self.order, self.dtype = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )

        self.y = y

        if self.dtype == "float32":
            self.nearest_neighbors = self.nearest_neighbors_single
            self.p = np.float32(self.p)
            if self.radius is not None:
                self.radius = np.float32(self.radius)
            self.nearest_neighbors_double = None
        else:
            self.p = np.float64(self.p)
            if self.radius is not None:
                self.radius = np.float64(self.radius)

        radius = self.radius
        if self.radius is None:
            radius = 0.0

        self.nearest_neighbors.pybind_fit(X, p=self.p, radius=radius)
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        r"""
        Compute the indices of the nearest neighbors for each test data point,
        and optionally the corresponding distances.

        Args:
            X (array-like): The test data for which the nearest neighbors are required.
                Its shape is (n_queries, n_features).

            n_neighbors (int, optional): The number of neighbors. If this is None, then
                the number of neighbors set in the constructor will be used instead.
                Default=None.

            return_distance (bool, optional): Denotes whether to return the distances or not.
                Default=True.

        Returns:
            numpy.ndarray of shape (n_queries, n_neighbors): The matrix with the corresponding
            distances to each neighbor. Only returned if return_distance=True.

            numpy.ndarray of shape (n_queries, n_neighbors): The matrix with the indices to
            each neighbor.
        """

        if X is None:
            raise ValueError("kneighbors() is not implemented for X=None")
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        # Check number of neighbors
        if not isinstance(n_neighbors, int) and n_neighbors is not None:
            raise ValueError(
                "Number of neighbors must be an integer or None. Got n_neighbors=" +
                str(n_neighbors))
        elif n_neighbors is not None and n_neighbors <= 0:
            raise ValueError(
                "Number of neighbors must be positive or None. Got n_neighbors=" +
                str(n_neighbors))
        elif n_neighbors is None and self.n_neighbors is None:
            raise ValueError(
                "n_neighbors cannot be None in kneighbors() if it was None in the constructor."
            )
        if n_neighbors is None:
            n_neighbors = 0
        if return_distance:
            return self.nearest_neighbors.pybind_kneighbors(X, n_neighbors)

        return self.nearest_neighbors.pybind_kneighbors_indices(X, n_neighbors)

    def radius_neighbors(self, X, radius=None, return_distance=True, sort_results=False):
        r"""
        Compute the indices of the radius neighbors for each test data point,
        and optionally the corresponding distances.

        Args:
            X (array-like): The test data for which the radius neighbors are required.
                Its shape is (n_queries, n_features).

            radius (float, optional): The radius within which to search for neighbors.
                Default=None.

            return_distance (bool, optional): Denotes whether to return the distances or not.
                Default=True.

            sort_results (bool, optional): Whether to sort the results by distance.
                Default=False.

        Returns:
            ndarray of shape (n_queries,) of arrays: The matrix with the corresponding
            distances to each neighbor. Only returned if return_distance=True.

            ndarray of shape (n_queries,) of arrays: The matrix with the indices to
            each neighbor.
        """

        if (not return_distance) and sort_results:
            raise ValueError("Cannot sort results if distances are not returned.")

        if X is None:
            raise ValueError("radius_neighbors() is not implemented for X=None")
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )

        if not isinstance(radius, (int, float)) and radius is not None:
            raise ValueError(
                "Radius must be a number or None. Got radius=" +
                str(radius))
        elif (radius is None) and (self.radius is None):
            raise ValueError(
                "radius cannot be None in radius_neighbors() if it was None in the constructor."
            )
        if radius is not None:
            if radius < 0.0:
                raise ValueError(
                    "Radius must be non-negative or None. Got radius=" +
                    str(radius))
        else:
            # if radius is None, set to negative value to indicate
            # that the radius passed in the contructor should be used
            radius = -1.0

        if return_distance:
            dists, inds = self.nearest_neighbors.pybind_radius_neighbors(
                X, radius, sort_results)

            # Handle the case where C++ returns (n_queries, 0) shaped arrays
            if isinstance(dists, np.ndarray) and dists.shape == (len(X), 0):
                # Create object array properly - don't use list comprehension
                result_dists = np.empty(len(X), dtype=object)
                for i in range(len(X)):
                    result_dists[i] = np.array([], dtype=dists.dtype)
                dists = result_dists
            elif isinstance(dists, list):
                # Convert list to proper object array
                result_dists = np.empty(len(dists), dtype=object)
                for i, arr in enumerate(dists):
                    result_dists[i] = arr
                dists = result_dists

            if isinstance(inds, np.ndarray) and inds.shape == (len(X), 0):
                # Create object array properly - don't use list comprehension
                result_inds = np.empty(len(X), dtype=object)
                for i in range(len(X)):
                    result_inds[i] = np.array([], dtype=inds.dtype)
                inds = result_inds
            elif isinstance(inds, list):
                # Convert list to proper object array
                result_inds = np.empty(len(inds), dtype=object)
                for i, arr in enumerate(inds):
                    result_inds[i] = arr
                inds = result_inds

            return (dists, inds)
        else:
            inds = self.nearest_neighbors.pybind_radius_neighbors_indices(X, radius)

            if isinstance(inds, np.ndarray) and inds.shape == (len(X), 0):
                # Create object array properly - don't use list comprehension
                result_inds = np.empty(len(X), dtype=object)
                for i in range(len(X)):
                    result_inds[i] = np.array([], dtype=inds.dtype)
                inds = result_inds
            elif isinstance(inds, list):
                # Convert list to proper object array
                result_inds = np.empty(len(inds), dtype=object)
                for i, arr in enumerate(inds):
                    result_inds[i] = arr
                inds = result_inds

            return inds

    def classifier_predict_proba(self, X, search_mode="knn"):
        r"""
        Compute the probability estimates for each of the available classes.

        Args:
            X (array-like): The test data for which predictions are required.
                Its shape is (n_queries, n_features).

            search_mode (str, optional): The search mode to use for finding neighbors.
                Options are 'knn' (k-nearest neighbors) or 'radius_neighbors'
                (radius-based search). Default = 'knn'.

        Returns:
            numpy.ndarray of shape (n_queries, :nref:`n_classes`): The class probabilities of the
            test data. Classes are sorted in ascending order.
        """
        if X is None:
            raise ValueError("Predicting probabilities is not implemented for X=None")
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        if self.y is None:
            raise ValueError(
                "Labels y were not provided during fit(). Classification is not available.")
        y, _, _ = check_convert_data(
            self.y, order=self.order, dtype="da_int", force_dtype=True
        )
        if (search_mode == "radius_neighbors") and (self.radius is None):
            raise ValueError(
                "radius was None in the constructor."
            )

        self.nearest_neighbors.pybind_set_labels(y)

        return self.nearest_neighbors.pybind_classifier_predict_proba(
            X, search_mode=search_mode)

    def classifier_predict(self, X, search_mode="knn"):
        r"""
        Compute the predicted labels for the test data (classification).

        Args:
            X (array-like): The test data for which predictions are required.
                Its shape is (n_queries, :nref:`n_features`).

            search_mode (str, optional): The search mode to use for finding neighbors.
                Options are 'knn' (k-nearest neighbors) or 'radius_neighbors'
                (radius-based search). Default = 'knn'.

        Returns:
            numpy.ndarray of shape (n_queries): The predicted labels of the test data.
        """
        if X is None:
            raise ValueError("Predicting labels is not implemented for X=None")
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        if self.y is None:
            raise ValueError(
                "Labels y were not provided during fit(). Classification is not available.")
        y, _, _ = check_convert_data(
            self.y, order=self.order, dtype="da_int", force_dtype=True
        )
        if (search_mode == "radius_neighbors") and (self.radius is None):
            raise ValueError(
                "radius was None in the constructor."
            )
        self.nearest_neighbors.pybind_set_labels(y)
        return self.nearest_neighbors.pybind_classifier_predict(
            X, search_mode=search_mode)

    def regressor_predict(self, X, search_mode="knn"):
        r"""
        Compute the predicted target values for the test data (regression).

        Args:
            X (array-like): The test data for which predictions are required.
                Its shape is (n_queries, :nref:`n_features`).

            search_mode (str, optional): The search mode to use for finding neighbors.
                Options are 'knn' (k-nearest neighbors) or 'radius_neighbors'
                (radius-based search). Default = 'knn'.

        Returns:
            numpy.ndarray of shape (n_queries): The predicted target values of the test data.
        """
        if X is None:
            raise ValueError("Predicting target values is not implemented for X=None")
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        if self.y is None:
            raise ValueError(
                "Targets y were not provided during fit(). Regression is not available.")
        y, _, _ = check_convert_data(
            self.y, order=self.order, dtype=self.dtype, force_dtype=True
        )
        if (search_mode == "radius_neighbors") and (self.radius is None):
            raise ValueError(
                "radius was None in the constructor."
            )
        self.nearest_neighbors.pybind_set_targets(y)
        return self.nearest_neighbors.pybind_regressor_predict(X, search_mode=search_mode)


class approximate_neighbors():
    r"""
    An Approximate Nearest Neighbors object.

    Args:
        n_neighbors (int, optional): The number of nearest neighbors for
            :meth:`kneighbors` queries. Default = 5.

        algorithm (str, optional): The algorithm used to compute
            the approximate nearest neighbors. Available options are 'auto' and 'ivfflat'.
            Default = 'ivfflat'.

        metric (str, optional): The metric used for the distance computation.
            Available metrics are 'euclidean', 'sqeuclidean' (squared Euclidean distances),
            'cosine', and 'inner product'. Default = 'sqeuclidean'.

        n_list (int, optional): The number of inverted file lists (centroids) to construct.
            Default = 1.

        n_probe (int, optional): The number of lists to probe at search time.
            Default = 1.

        kmeans_iter (int, optional): Maximum number of k-means iterations to
            perform during training. Default = 10.

        train_fraction (float, optional): Proportion of training data to use for
            k-means clustering. Default = 1.0.

        seed (int, optional): Seed for random number generation. Set to -1 for
            non-deterministic results. Default = 0.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(self, n_neighbors=5, algorithm='ivfflat', metric='sqeuclidean',
                 n_list=1, n_probe=1, kmeans_iter=10, train_fraction=1.0, seed=0,
                 check_data=False):
        self._approx_nn_double = pybind_approximate_neighbors(
            n_neighbors, algorithm, metric, n_list, n_probe, kmeans_iter, seed, "double",
            check_data)
        self._approx_nn_single = pybind_approximate_neighbors(
            n_neighbors, algorithm, metric, n_list, n_probe, kmeans_iter, seed, "single",
            check_data)
        self._approx_nn = self._approx_nn_double
        self._order = 'A'
        self._dtype = 'float'
        self._train_fraction = train_fraction
        self._n_probe = n_probe

    @property
    def n_probe(self):
        """The number of lists to probe at search time."""
        return self._n_probe

    @n_probe.setter
    def n_probe(self, value):
        self._n_probe = value
        self._approx_nn.set_n_probe_opt(n_probe=value)

    def train(self, X_train):
        r"""
        Train the model by computing centroids using k-means clustering.
        Data is not added to the index; use :meth:`add` to populate the index after training.

        Args:
            X_train (array-like): The training data matrix.
                It has shape (:nref:`n_samples`, :nref:`n_features`).

        Returns:
            self (object): Returns the instance itself.
        """
        X_train, self._order, self._dtype = check_convert_data(
            X_train, order=self._order, dtype=self._dtype, force_dtype=True
        )

        if self._dtype == "float32":
            self._approx_nn = self._approx_nn_single
            self._approx_nn_double = None
            train_frac = np.float32(self._train_fraction)
        else:
            train_frac = np.float64(self._train_fraction)

        self._approx_nn.pybind_train(X_train, train_fraction=train_frac)
        return self

    def train_and_add(self, X_train):
        r"""
        Train the model and add the training data to the index.

        Args:
            X_train (array-like): The training data matrix.
                It has shape (:nref:`n_samples`, :nref:`n_features`).

        Returns:
            self (object): Returns the instance itself.
        """
        X_train, self._order, self._dtype = check_convert_data(
            X_train, order=self._order, dtype=self._dtype, force_dtype=True
        )

        if self._dtype == "float32":
            self._approx_nn = self._approx_nn_single
            self._approx_nn_double = None
            train_frac = np.float32(self._train_fraction)
        else:
            train_frac = np.float64(self._train_fraction)

        self._approx_nn.pybind_train_and_add(X_train, train_fraction=train_frac)
        return self

    def add(self, X_add):
        r"""
        Add data points to the index. The model must be trained first.

        Args:
            X_add (array-like): The data matrix to add to the index.
                Its shape is (:nref:`n_samples_add`, :nref:`n_features`).

        Returns:
            self (object): Returns the instance itself.
        """
        X_add, _, _ = check_convert_data(
            X_add, order=self._order, dtype=self._dtype, force_dtype=True
        )

        self._approx_nn.pybind_add(X_add)
        return self

    def kneighbors(self, X_test, n_neighbors=0, return_distance=True):
        r"""
        Compute the approximate k nearest neighbors for each query point.

        Args:
            X_test (array-like): The query data matrix.
                Its shape is (n_queries, :nref:`n_features`).

            n_neighbors (int, optional): The number of neighbors to return.
                If less than or equal to zero, uses the value set during initialization.

            return_distance (bool, optional): Whether to return the distances.
                Default = True.

        Returns:
            numpy.ndarray of shape (n_queries, n_neighbors): The distances to each neighbor.
                Only returned if return_distance=True.

            numpy.ndarray of shape (n_queries, n_neighbors): The indices of each neighbor.
        """
        X_test, _, _ = check_convert_data(
            X_test, order=self._order, dtype=self._dtype, force_dtype=True
        )

        if return_distance:
            return self._approx_nn.pybind_kneighbors(X_test, n_neighbors)

        return self._approx_nn.pybind_kneighbors_indices(X_test, n_neighbors)

    @property
    def cluster_centroids(self):
        r"""numpy.ndarray of shape (n_list, :nref:`n_features`): The centroid vectors computed
            during training."""
        return self._approx_nn.get_cluster_centroids()

    @property
    def list_sizes(self):
        r"""numpy.ndarray of shape (n_list,): The number of vectors assigned to each
            centroid."""
        return self._approx_nn.get_list_sizes()

    @property
    def n_list(self):
        r"""int: The number of inverted file lists (centroids)."""
        return self._approx_nn.get_n_list()

    @property
    def n_index(self):
        r"""int: The number of data points added to the index."""
        return self._approx_nn.get_n_index()

    @property
    def n_features(self):
        r"""int: The number of features in the indexed data."""
        return self._approx_nn.get_n_features()

    @property
    def kmeans_iter(self):
        r"""int: The number of k-means iterations performed during training."""
        return self._approx_nn.get_kmeans_iter()
