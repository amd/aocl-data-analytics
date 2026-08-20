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
Patching faiss IndexIVFFlat
"""

# pylint: disable = missing-function-docstring, too-few-public, too-many-locals
# duplicate-code

import warnings
import numpy as np
from aoclda.neighbors import approximate_neighbors
from aoclda.faiss._kmeans import _ClusteringParameters

_METRIC_MAP = {1: 'sqeuclidean', 0: 'inner product'}


class IndexIVFFlat:
    """AOCL-DA backed replacement for faiss.IndexIVFFlat (IVFFlat approximate search)."""

    _SUPPORTED_ATTRS = frozenset({
        'cp',       # set in __init__
        'aocl',     # marker, set in __init__
        'nprobe',   # property setter — must include to suppress false warning
    })

    def __init__(self, quantizer, d, nlist, metric=1):
        if quantizer.d != d:
            raise ValueError(
                f"quantizer dimension {quantizer.d} does not match d={d}")
        if type(quantizer).__name__ not in ('IndexFlatL2', 'IndexFlatIP'):
            warnings.warn(
                "Non-Flat quantizer passed to IndexIVFFlat; the quantizer will be ignored.",
                UserWarning)
        if metric not in _METRIC_MAP:
            raise ValueError(
                f"Unknown metric integer {metric}; "
                "expected 0 (METRIC_INNER_PRODUCT) or 1 (METRIC_L2)"
            )
        self._d = d
        self._nlist = nlist
        self._metric_int = metric
        self._metric_str = _METRIC_MAP[metric]
        self._ann = None
        self._is_trained = False
        self.cp = _ClusteringParameters(niter=25, seed=0)
        self._nprobe = 1
        self.aocl = True

    @property
    def d(self):
        return self._d

    @property
    def nlist(self):
        return self._nlist

    @property
    def nprobe(self):
        return self._ann.n_probe if self._ann is not None else self._nprobe

    @nprobe.setter
    def nprobe(self, value):
        self._nprobe = value
        if self._ann is not None:
            self._ann.n_probe = value

    @property
    def ntotal(self):
        return self._ann.n_index if self._ann is not None else 0

    @property
    def is_trained(self):
        return self._is_trained

    @property
    def metric_type(self):
        return self._metric_int

    @property
    def quantizer(self):
        return None

    def train(self, X):
        # Check dims here here since we haven't constructed the DA class yet
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self._d:
            raise ValueError(
                f"IndexIVFFlat.train: expected array with {self._d} columns, "
                f"got shape {X.shape}")

        if self.cp.max_points_per_centroid is not None:
            train_frac = min(
                1.0,
                self.cp.max_points_per_centroid *
                self._nlist /
                X.shape[0]
            )
        else:
            train_frac = 1.0
        self._ann = approximate_neighbors(
            n_list=self._nlist,
            metric=self._metric_str,
            n_neighbors=1,
            n_probe=self._nprobe,
            kmeans_iter=self.cp.niter,
            seed=self.cp.seed,
            train_fraction=train_frac,
        )
        self._ann.train(X)
        self._is_trained = True

    def add(self, X):
        if not self._is_trained or self._ann is None:
            raise RuntimeError(
                "IndexIVFFlat.add: index is not trained, call train() first")
        self._ann.add(X)

    def search(self, X, k):
        if not self._is_trained or self._ann is None:
            raise RuntimeError(
                "IndexIVFFlat.search: index is not trained, call train() first")
        distances, indices = self._ann.kneighbors(X, n_neighbors=k)
        return distances, indices.astype(np.int64)

    def reset(self):
        self._ann = None
        self._is_trained = False

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
            f"IndexIVFFlat.{name} is not supported by the AOCL-DA backend")
