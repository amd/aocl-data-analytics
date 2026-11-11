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

# pylint: disable = missing-function-docstring, too-few-public-methods

"""Patching faiss IndexScalarQuantizer."""

import warnings

import numpy as np

from aoclda.neighbors import approximate_neighbors
from ._index_ivfflat import _METRIC_MAP
from ._index_ivfsq import _QTYPE_NAMES, _QTYPE_OPTIONS


class IndexScalarQuantizer:
    """AOCL-DA backed replacement for faiss.IndexScalarQuantizer."""

    _SUPPORTED_ATTRS = frozenset({'aocl'})

    def __init__(self, d, qtype, metric=1):
        if qtype not in _QTYPE_OPTIONS:
            supported = ', '.join(_QTYPE_NAMES.values())
            raise ValueError(
                f"Unsupported scalar quantizer type {qtype}; expected one of {supported}")
        if metric not in _METRIC_MAP:
            raise ValueError(
                f"Unknown metric integer {metric}; "
                "expected 0 (METRIC_INNER_PRODUCT) or 1 (METRIC_L2)"
            )

        self._d = d
        self._qtype = qtype
        self._metric_int = metric
        self._metric_str = _METRIC_MAP[metric]
        self._ann = None
        self._is_trained = False
        self.aocl = True

    @property
    def d(self):
        return self._d

    @property
    def ntotal(self):
        return self._ann.n_index if self._ann is not None else 0

    @property
    def is_trained(self):
        return self._is_trained

    @property
    def metric_type(self):
        return self._metric_int

    def train(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self._d:
            raise ValueError(
                f"IndexScalarQuantizer.train: expected array with {self._d} columns, "
                f"got shape {X.shape}")

        quantization, quantization_scheme = _QTYPE_OPTIONS[self._qtype]
        self._ann = approximate_neighbors(
            algorithm='ivfsq',
            metric=self._metric_str,
            n_neighbors=1,
            n_list=1,
            n_probe=1,
            quantization=quantization,
            quantization_scheme=quantization_scheme,
            quantization_residual=False,
        )
        self._ann.train(X)
        self._is_trained = True

    def add(self, X):
        if not self._is_trained or self._ann is None:
            raise RuntimeError(
                "IndexScalarQuantizer.add: index is not trained, call train() first")
        self._ann.add(X)

    def search(self, X, k):
        if not self._is_trained or self._ann is None:
            raise RuntimeError(
                "IndexScalarQuantizer.search: index is not trained, call train() first")
        distances, indices = self._ann.kneighbors(X, n_neighbors=k)
        return distances, indices.astype(np.int64)

    def reset(self):
        self._ann = None
        self._is_trained = False

    def __setattr__(self, name, value):
        if not name.startswith('_') and name not in self._SUPPORTED_ATTRS:
            warnings.warn(
                f"IndexScalarQuantizer: setting '{name}' has no effect — "
                "this attribute is not supported by the AOCL-DA backend.",
                UserWarning, stacklevel=2)
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise NotImplementedError(
            f"IndexScalarQuantizer.{name} is not supported by the AOCL-DA backend")
