# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
import numpy as np
from ._aoclda.clustering import pybind_kmeans, pybind_DBSCAN
from ._internal_utils import check_convert_data


class kmeans():
    r"""
    k-means clustering.

    Partition a data matrix into clusters using k-means clustering.

    Args:
        n_clusters (int, optional): Number of clusters to form. Default=1.

        initialization_method (str, optional): The method used to find the initial cluster centres.
            It can take the values 'k-means++', 'random' (initial clusters are chosen randomly from
            the sample data points), 'random partitions' (sample points are assigned to a random
            cluster and the corresponding cluster centres are computed and used as the starting
            point), or 'afk-mc2' (initial cluster centres are chosen using the AFK-MC^2
            algorithm of Bachem et al. (2016)). Default: 'k-means++'.

        C (array-like, optional): The matrix of initial cluster centres. It has
            shape (n_clusters, :nref:`n_features`). If supplied, these centres will be used as the
            starting point for the first iteration, otherwise the initialization method specified
            above will be used. Default = None.

        n_init (int, optional): Number of runs with different random seeds (ignored if you specify
            initial cluster centres). Default=10.

        max_iter (int, optional): Number of runs with different random seeds (ignored if you specify
            initial cluster centres). Default=300.

        seed (int, optional): Seed for random number generation; set to -1 for non-deterministic
            results. Default=-1.

        algorithm (str, optional): The algorithm used to compute the clusters. It can take the
            values 'elkan', 'lloyd', 'macqueen' or 'hartigan-wong'. Default = 'lloyd'.

        distance (str, optional): The distance metric used for clustering. It can take the
            values 'euclidean' or 'cosine'. If 'cosine' is selected, spherical k-means
            is performed, which optimizes cosine similarity between data points and cluster
            centres. Note that cosine distance is not compatible with the Hartigan-Wong
            algorithm. Default = 'euclidean'.

        normalize_data (bool, optional): Whether to normalize the input data before clustering.
            Only used if distance is set to cosine. Default = False.

        tol (float, optional): The convergence tolerance for the iterations. Default = 1.0-e-4.

        empty_clusters (str, optional): Determines behaviour in the case that all sample points have
            been assigned to fewer than k clusters. If set to 'ignore' then empty clusters are
            allowed and the algorithm proceeds as normal. If set to 'error' then an error is raised
            when an empty cluster is encountered (if 'n_init' > 1 then the next initialization is
            attempted, so an error will only be returned to the calling program if all
            initializations led to empty clusters). If set to 'split' then the point farthest from
            its closest cluster centre is chosen and assigned to the empty cluster. Note that if the
            Hartigan-Wong algorithm is used then 'empty clusters' will be set to 'error' internally.

        afk_mcmc_samples (int, optional): If the AFK-MC^2 initialization method is used, the number
            of MCMC samples to use in the algorithm. Default = 50.

        mixed_precision (bool, optional): Whether to use mixed precision iterative refinement,
            in which lower precision arithmetic is used before switching to the working precision
            for the final iterations. Default = False.

        low_precision_max_iter (int, optional): If mixed precision iterative refinement is enabled,
            maximum number of iterations for the low precision phase. Default = 200.

        low_precision_tol (float, optional): If mixed precision iterative refinement is enabled,
            convergence tolerance for the low precision phase. Default = 1.0-e-2.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.

    """

    def __init__(
            self,
            n_clusters=1,
            initialization_method='k-means++',
            C=None, n_init=10,
            max_iter=300,
            seed=-1,
            algorithm='lloyd',
            distance='euclidean',
            normalize_data=False,
            tol=1.0e-4,
            empty_clusters='ignore',
            afk_mcmc_samples=50,
            mixed_precision=False,
            low_precision_max_iter=200,
            low_precision_tol=1.0e-2,
            check_data=False):

        self.kmeans_double = pybind_kmeans(
            n_clusters,
            initialization_method,
            n_init,
            max_iter,
            seed,
            algorithm,
            distance,
            normalize_data,
            empty_clusters,
            afk_mcmc_samples,
            mixed_precision,
            low_precision_max_iter,
            'double',
            check_data)
        self.kmeans_single = pybind_kmeans(
            n_clusters,
            initialization_method,
            n_init,
            max_iter,
            seed,
            algorithm,
            distance,
            normalize_data,
            empty_clusters,
            afk_mcmc_samples,
            mixed_precision,
            low_precision_max_iter,
            'single',
            check_data)

        self.dtype = 'float'
        self.order = 'A'
        self.C = C
        self.tol = tol
        self.low_precision_tol = low_precision_tol
        self.low_precision_max_iter = low_precision_max_iter
        self.mixed_precision = mixed_precision
        self.afk_mcmc_samples = afk_mcmc_samples
        self.normalize_data = normalize_data
        self.kmeans = self.kmeans_double
        self.distance = distance

    @property
    def cluster_centres(self):
        r"""numpy.ndarray of shape (n_clusters, :nref:`n_features`): The coordinates of the cluster
            centres.
        """
        return self.kmeans.get_cluster_centres()

    @property
    def labels(self):
        r"""numpy.ndarray of shape (:nref:`n_samples`, ): The label (which cluster) of each sample point
           in the data matrix."""
        return self.kmeans.get_labels()

    @property
    def inertia(self):
        """numpy.ndarray of shape (1, ): The inertia (sum of the squared distance of each sample to
           its closest cluster centre)."""
        return self.kmeans.get_inertia()

    @property
    def n_iter(self):
        """int: The number iterations performed in the k-means computation.
        """
        return self.kmeans.get_n_iter()

    @property
    def lp_n_iter(self):
        """int: The number of low precision iterations performed if mixed precision was enabled.
        """
        return self.kmeans.get_lp_n_iter()

    @property
    def n_samples(self):
        """int: The number of samples in the data matrix used. """
        return self.kmeans.get_n_samples()

    @property
    def n_features(self):
        """int: The number of features in the data matrix. """
        return self.kmeans.get_n_features()

    @property
    def n_clusters(self):
        """int: The number of clusters found. """
        return self.kmeans.get_n_clusters()

    def fit(self, A):
        r"""
        Computes k-means clusters for the supplied data matrix, optionally using the supplied
        centres as the starting point.

        Args:
            A (array-like): The data matrix with which to compute the k-means clusters. It has
              shape (:nref:`n_samples`, :nref:`n_features`).

        Returns:
            self (object): Returns the instance itself.

        """
        A, self.order, self.dtype = check_convert_data(
            A, order=self.order, dtype=self.dtype, force_dtype=True
        )
        if self.C is not None:
            self.C, _, _ = check_convert_data(
                self.C, order=self.order, dtype=self.dtype, force_dtype=True)
        if self.dtype == "float32":
            self.kmeans = self.kmeans_single
            self.kmeans_double = None

        self.kmeans.pybind_fit(A, self.C, self.tol, self.low_precision_tol)
        return self

    def transform(self, X):
        r"""
        Transform a data matrix into cluster distance space.

        Transforms a data matrix ``X`` from the original coordinate system into the new coordinates
        in which each dimension is the distance to the cluster centres previously computed by
        ``kmeans.fit``.

        Args:
            X (array-like): The data matrix to be transformed. It has shape
              (m_samples, m_features). Note that :nref:`m_features` must match :nref:`n_features`,
              the number of features in the data matrix originally supplied to ``kmeans.fit``.

        Returns:
            numpy.ndarray of shape (m_samples, :nref:`n_clusters`): The transformed matrix.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )

        return self.kmeans.pybind_transform(X)

    def predict(self, Y):
        r"""
        Predict the cluster each sample in a data matrix belongs to.

        For each sample in the data matrix ``Y`` find the closest cluster centre out of the clusters
        previously computed in ``kmeans.fit``.

        Args:
            Y (array-like): The data matrix to be transformed. It has shape
              (k_samples, k_features). Note that :nref:`k_features` must match :nref:`n_features`,
              the number of features in the data matrix used in ``kmeans.fit``.

        Returns:
            numpy.ndarray of shape (k_samples, ): The labels.
        """
        Y, _, _ = check_convert_data(
            Y, order=self.order, dtype=self.dtype, force_dtype=True
        )

        return self.kmeans.pybind_predict(Y)

    def __getstate__(self):
        """Support for pickle serialization."""
        return {
            'pybind_state': pickle.dumps(self.kmeans),
            'order': self.order,
            'dtype': self.dtype,
            'C': self.C,
            'tol': self.tol,
            'low_precision_tol': self.low_precision_tol,
            'low_precision_max_iter': self.low_precision_max_iter,
            'mixed_precision': self.mixed_precision,
            'afk_mcmc_samples': self.afk_mcmc_samples,
            'normalize_data': self.normalize_data,
            'distance': self.distance
        }

    def __setstate__(self, state):
        """Support for pickle deserialization."""
        self.order = state['order']
        self.dtype = state['dtype']
        self.kmeans = pickle.loads(state['pybind_state'])
        self.C = state['C']
        self.tol = state['tol']
        self.low_precision_tol = state['low_precision_tol']
        self.low_precision_max_iter = state['low_precision_max_iter']
        self.mixed_precision = state['mixed_precision']
        self.afk_mcmc_samples = state['afk_mcmc_samples']
        self.normalize_data = state.get('normalize_data', False)
        self.distance = state.get('distance', 'euclidean')

        if self.dtype == 'float64':
            self.kmeans_double = self.kmeans
            self.kmeans_single = None
        elif self.dtype == 'float32':
            self.kmeans_double = None
            self.kmeans_single = self.kmeans
        else:
            raise ValueError(
                f"Invalid dtype '{self.dtype}' when loading " +
                "model. Expected 'float32' or 'float64'."
            )
        return


class DBSCAN():
    """
    DBSCAN clustering.

    Partition a data matrix into clusters using DBSCAN clustering.

    Args:

        min_samples (int, optional): Minimum number of neighborhood samples for a sample point to be
            considered a core point. Default = 5.

        metric (str, optional): The distance metric used to compare sample points. Available metrics
            are 'euclidean', 'l2', 'sqeuclidean' (squared Euclidean distances), 'manhattan', 'l1',
            'cityblock', 'cosine', or 'minkowski'. Default = 'euclidean'.

        algorithm (str, optional): The algorithm used to compute the clusters. Available options are
            'auto', 'ball_tree', 'brute' and 'kd_tree'. k-d trees are likely to be fastest for lower
            dimensional datasets, and ball trees may be preferred when data is not aligned along
            coordinate axes, but trees cannot not be used with the cosine distance, the squared
            Euclidean distance, or with the Minkowski distance with power less than 1.0.
            Default = 'auto'.

        leaf_size (int, optional): Leaf size for the k-d tree algorithm. Default = 30.

        eps (float, optional): Maximum distance between two samples for them to be considered in
            each other's neighborhood. Default = 0.5.

        power (float, optional): Power used in computing the Minkowski metric. Default = 2.0.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.

    """

    def __init__(
            self,
            min_samples=5,
            metric='euclidean',
            algorithm='auto',
            leaf_size=30,
            eps=0.5,
            power=2.0,
            check_data=False):

        self.DBSCAN_double = pybind_DBSCAN(
            min_samples, metric, algorithm, leaf_size, 'double', check_data)
        self.DBSCAN_single = pybind_DBSCAN(
            min_samples, metric, algorithm, leaf_size, 'single', check_data)

        self.order = 'A'
        self.dtype = 'float'
        self.eps = eps
        self.power = power
        self.DBSCAN = self.DBSCAN_double

    @property
    def labels(self):
        r"""numpy.ndarray of shape (:nref:`n_samples`, ): The label (which cluster) of each sample
           point in the data matrix.  A label of -1 indicates that the point has been classified as
           noise and has not been assigned to a cluster."""
        return self.DBSCAN.get_labels()

    @property
    def core_sample_indices(self):
        """numpy.ndarray of shape (:nref:`n_core_samples`, ): The indices of the core samples in the
           data matrix."""
        return self.DBSCAN.get_core_sample_indices()

    @property
    def n_samples(self):
        """int: The number of samples in the data matrix. """
        return self.DBSCAN.get_n_samples()

    @property
    def n_core_samples(self):
        """int: The number of core samples found in the data matrix. """
        return self.DBSCAN.get_n_core_samples()

    @property
    def n_features(self):
        """int: The number of features in the data matrix. """
        return self.DBSCAN.get_n_features()

    @property
    def n_clusters(self):
        """int: The number of clusters found. """
        return self.DBSCAN.get_n_clusters()

    def fit(self, A):
        r"""
        Computes DBSCAN clusters for the supplied data matrix.

        Args:
            A (array-like): The data matrix with which to compute the DBSCAN clusters. It has
              shape (:nref:`n_samples`, :nref:`n_features`).

        Returns:
            self (object): Returns the instance itself.
        """
        A, self.order, self.dtype = check_convert_data(
            A, order=self.order, dtype=self.dtype, force_dtype=True
        )

        if self.dtype == "float32":
            self.eps = np.float32(self.eps)
            self.power = np.float32(self.power)
            self.DBSCAN = self.DBSCAN_single
            self.DBSCAN_double = None
        else:
            self.eps = np.float64(self.eps)
            self.power = np.float64(self.power)

        self.DBSCAN.pybind_fit(A, self.eps, self.power)
        return self
