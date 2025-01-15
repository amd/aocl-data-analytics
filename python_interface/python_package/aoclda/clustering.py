# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

from ._aoclda.clustering import pybind_kmeans, pybind_DBSCAN


class kmeans():
    """
    k-means clustering.

    Partition a data matrix into clusters using k-means clustering.

    Args:
        n_clusters (int, optional): Number of clusters to form. Default=1.

        initialization_method (str, optional): The method used to find the initial cluster centres.
            It can take the values 'k-means++', 'random' (initial clusters are chosen randomly from
            the sample data points) or 'random partitions' (sample points are assigned to a random
            cluster and the corresponding cluster centres are computed and used as the starting
            point). Default: 'k-means++'.

        C (numpy.ndarray, optional): The matrix of initial cluster centres. It has
            shape (n_clusters, n_features). If supplied, these centres will be used as the starting
            point for the first iteration, otherwise the initialization method specified above will
            be used. Default = None.

        n_init (int, optional): Number of runs with different random seeds (ignored if you specify
            initial cluster centres). Default=10.

        max_iter (int, optional): Number of runs with different random seeds (ignored if you specify
            initial cluster centres). Default=300.

        seed (int, optional): Seed for random number generation; set to -1 for non-deterministic
            results. Default=-1.

        algorithm (str, optional): The algorithm used to compute the clusters. It can take the
            values 'elkan', 'lloyd', 'macqueen' or 'hartigan-wong'. Default = 'lloyd'.

        tol (float, optional): The convergence tolerance for the iterations. Default = 1.0-e-4.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.

    """
    def __init__(self, n_clusters=1, initialization_method='k-means++', C=None, n_init=10,
                 max_iter=300, seed=-1, algorithm='lloyd', tol=1.0e-4, check_data=False):

        self.kmeans_double = pybind_kmeans(n_clusters, initialization_method, n_init, max_iter,
                                           seed, algorithm,  'double', check_data)
        self.kmeans_single = pybind_kmeans(n_clusters, initialization_method, n_init, max_iter,
                                           seed, algorithm,  'single', check_data)

        self.C=C
        self.tol = tol
        self.kmeans = self.kmeans_double

    @property
    def cluster_centres(self):
        """numpy.ndarray of shape (n_clusters, n_features): The coordinates of the cluster centres.
        """
        return self.kmeans.get_cluster_centres()

    @property
    def labels(self):
        """numpy.ndarray of shape (n_samples, ): The label (i.e. which cluster) of each sample point
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
        """
        Computes k-means clusters for the supplied data matrix, optionally using the supplied
        centres as the starting point.

        Args:
            A (numpy.ndarray): The data matrix with which to compute the k-means clusters. It has
              shape (n_samples, n_features).

        Returns:
            self (object): Returns the instance itself.

        """
        if A.dtype == "float32":
            self.kmeans = self.kmeans_single
            self.kmeans_double = None

        self.kmeans.pybind_fit(A, self.C, self.tol)
        return self

    def transform(self, X):
        """
        Transform a data matrix into cluster distance space.

        Transforms a data matrix ``X`` from the original coordinate system into the new coordinates
        in which each dimension is the distance to the cluster centres previously computed by
        ``kmeans.fit``.

        Args:
            X (numpy.ndarray): The data matrix to be transformed. It has shape
              (m_samples, m_features). Note that ``m_features`` must match ``n_features``,
              the number of features in the data matrix originally supplied to ``kmeans.fit``.

        Returns:
            numpy.ndarray of shape (m_samples, n_clusters): The transformed matrix.
        """
        return self.kmeans.pybind_transform(X)

    def predict(self, Y):
        """
        Predict the cluster each sample in a data matrix belongs to.

        For each sample in the data matrix ``Y`` find the closest cluster centre out of the clusters
        previously computed in ``kmeans.fit``.

        Args:
            Y (numpy.ndarray): The data matrix to be transformed. It has shape
              (k_samples, k_features). Note that ``k_features`` must match ``n_features``,
              the number of features in the data matrix used in ``kmeans.fit``.

        Returns:
            numpy.ndarray of shape (k_samples, ): The labels.
        """
        return self.kmeans.pybind_predict(Y)

class DBSCAN():
    """
    DBSCAN clustering.

    Partition a data matrix into clusters using DBSCAN clustering.

    Args:

        min_samples (int, optional): Minimum number of neighborhood samples for a sample point to be
            considered a core point. Default = 5.

        metric (str, optional): The distance metric used to compare sample points. Reserved for
            future use. Default = 'euclidean'.

        algorithm (str, optional): The algorithm used to compute the clusters. Reserved for future
            use. Default = 'brute'.

        leaf_size (int, optional): Leaf size for the KD tree or ball tree algorithms. Reserved for
            future use. Default = 30.

        eps (float, optional): Maximum distance between two samples for them to be considered in
            each other's neighborhood. Default = 0.5.

        power (float, optional): Power used in computing the Minkowski metric. Reserved for future
            use. Default = 2.0.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.

    """
    def __init__(self, min_samples=5, metric='euclidean', algorithm='brute', leaf_size=30, eps=0.5,
                 power=2.0, check_data=False):

        self.DBSCAN_double = pybind_DBSCAN(min_samples, metric, algorithm, leaf_size, 'double',
                                           check_data)
        self.DBSCAN_single = pybind_DBSCAN(min_samples, metric, algorithm, leaf_size, 'single',
                                           check_data)

        self.eps=eps
        self.power = power
        self.DBSCAN = self.DBSCAN_double


    @property
    def labels(self):
        """numpy.ndarray of shape (n_samples, ): The label (i.e. which cluster) of each sample point
           in the data matrix.  A label of -1 indicates that the point has been classified as noise
           and has not been assigned to a cluster."""
        return self.DBSCAN.get_labels()

    @property
    def core_sample_indices(self):
        """numpy.ndarray of shape (n_core_samples, ): The indices of the core samples in the data
           matrix."""
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
        """
        Computes DBSCAN clusters for the supplied data matrix.

        Args:
            A (numpy.ndarray): The data matrix with which to compute the DBSCAN clusters. It has
              shape (n_samples, n_features).

        Returns:
            self (object): Returns the instance itself.
        """

        if A.dtype == "float32":
            self.DBSCAN = self.DBSCAN_single
            self.DBSCAN_double = None

        self.DBSCAN.pybind_fit(A, self.eps, self.power)
        return self
