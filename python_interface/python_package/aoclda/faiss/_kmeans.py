# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
Patching faiss Kmeans
"""

# pylint: disable = missing-function-docstring, too-few-public, too-many-locals
# duplicate-code

import warnings
import numpy as np
from aoclda.clustering import kmeans


class _ClusteringParameters:
    """Stub for faiss.ClusteringParameters, the single struct shared by Kmeans and
    IndexIVF. All fields exist on both; a field a given backend ignores just passes
    through with no effect."""

    _SUPPORTED_ATTRS = frozenset({
        'niter',
        'nredo',
        'seed',
        'spherical',
        'max_points_per_centroid',
    })

    def __init__(self, niter=25, nredo=1, seed=-1, spherical=False,
                 max_points_per_centroid=None):
        self.niter = niter
        self.nredo = nredo
        self.seed = seed
        self.spherical = spherical
        self.max_points_per_centroid = max_points_per_centroid

    def __setattr__(self, name, value):
        if not name.startswith('_') and name not in self._SUPPORTED_ATTRS:
            warnings.warn(
                f"{type(self).__name__}: setting '{name}' has no effect — "
                "this attribute is not supported by the AOCL-DA backend.",
                UserWarning, stacklevel=2)
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise NotImplementedError(
            f"ClusteringParameters.{name} is not supported by the AOCL-DA backend")


class Kmeans:
    """AOCL-DA wrapper for faiss.Kmeans, backed by aoclda.clustering.kmeans."""

    _SUPPORTED_ATTRS = frozenset({'aocl', 'cp'})

    def __init__(
            self,
            d,
            k,
            niter=25,
            nredo=1,
            verbose=False,
            seed=-1,
            spherical=False,
            *,
            algorithm='lloyd',
            initialization_method='k-means++',
            normalize_data=False,
            tol=1.0e-4,
            empty_clusters='ignore',
            afk_mcmc_samples=50,
            mixed_precision=False,
            low_precision_max_iter=10,
            low_precision_tol=1.0e-2,
            **kwargs):
        for kw in kwargs:
            warnings.warn(
                f"Kmeans: '{kw}' is not supported by the AOCL-DA backend.",
                UserWarning, stacklevel=2)
        if verbose:
            warnings.warn(
                "Kmeans: 'verbose' is not supported by the AOCL-DA backend.",
                UserWarning, stacklevel=2)
        self._d = d
        self._k = k
        self._is_trained = False
        self.aocl = True
        self.cp = _ClusteringParameters(
            niter=niter, nredo=nredo, seed=seed, spherical=spherical)
        # AOCL-DA specific options, exposed on the constructor in addition to the
        # native faiss parameters (mirrors the scikit-learn patch). Stored as
        # private attributes so they are not flagged by __setattr__ and are passed
        # through to the backend in train().
        self._algorithm = algorithm
        self._initialization_method = initialization_method
        self._normalize_data = normalize_data
        self._tol = tol
        self._empty_clusters = empty_clusters
        self._afk_mcmc_samples = afk_mcmc_samples
        self._mixed_precision = mixed_precision
        self._low_precision_max_iter = low_precision_max_iter
        self._low_precision_tol = low_precision_tol
        self._kmeans = None

    def train(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self._d:
            raise ValueError(
                f"Kmeans.train: expected array with {self._d} columns, "
                f"got shape {x.shape}")
        self._kmeans = kmeans(
            n_clusters=self._k,
            initialization_method=self._initialization_method,
            max_iter=self.cp.niter,
            n_init=self.cp.nredo,
            seed=self.cp.seed,
            algorithm=self._algorithm,
            distance='cosine' if self.cp.spherical else 'euclidean',
            normalize_data=self._normalize_data,
            tol=self._tol,
            empty_clusters=self._empty_clusters,
            afk_mcmc_samples=self._afk_mcmc_samples,
            mixed_precision=self._mixed_precision,
            low_precision_max_iter=self._low_precision_max_iter,
            low_precision_tol=self._low_precision_tol,
        )
        self._kmeans.fit(x)
        self._is_trained = True

    def assign(self, x):
        """Return (D, I): distance to nearest centroid and centroid index per query."""
        if not self._is_trained or self._kmeans is None:
            raise RuntimeError("Kmeans.assign: not trained, call train() first")
        D_matrix = self._kmeans.transform(x)           # (nsamples, k)
        I = D_matrix.argmin(axis=1).astype(np.int64)   # (nsamples,)
        D = D_matrix[np.arange(len(x)), I]              # (nsamples,)
        return D, I

    @property
    def d(self):
        return self._d

    @property
    def k(self):
        return self._k

    @property
    def is_trained(self):
        return self._is_trained

    @property
    def centroids(self):
        if not self._is_trained:
            return None
        return self._kmeans.cluster_centres

    def __setattr__(self, name, value):
        if not name.startswith('_') and name not in self._SUPPORTED_ATTRS:
            warnings.warn(
                f"Kmeans: setting '{name}' has no effect — "
                "this attribute is not supported by the AOCL-DA backend.",
                UserWarning, stacklevel=2)
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise NotImplementedError(
            f"Kmeans.{name} is not supported by the AOCL-DA backend.")
