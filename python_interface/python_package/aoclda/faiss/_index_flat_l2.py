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
Patching faiss IndexFlatL2
"""

# pylint: disable = missing-function-docstring

import warnings
import numpy as np
from aoclda.neighbors import nearest_neighbors


class IndexFlatL2:
    """AOCL-DA backed replacement for faiss.IndexFlatL2 (exact brute-force L2 search)."""

    _SUPPORTED_ATTRS = frozenset({
        'aocl',     # marker, set in __init__
    })

    def __init__(self, d):
        self._d = d
        self._nn = None
        self._ntotal = 0
        self._all_data = None
        self.aocl = True

    @property
    def d(self):
        return self._d

    @property
    def ntotal(self):
        return self._ntotal

    @property
    def is_trained(self):
        return True

    @property
    def metric_type(self):
        return 1

    @property
    def metric_arg(self):
        return 0

    def add(self, X):
        # Check dims quickly here, since we haven't constructed the DA class yet
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self._d:
            raise ValueError(
                f"IndexFlatL2.add: expected array with {self._d} columns, "
                f"got shape {X.shape}")
        if self._all_data is None:
            self._all_data = X
        else:
            self._all_data = np.concatenate([self._all_data, X], axis=0)
        self._ntotal += X.shape[0]
        self._nn = nearest_neighbors(metric='sqeuclidean')
        self._nn.fit(self._all_data)

    def train(self, X):
        pass

    def search(self, X, k):
        if self._nn is None:
            raise RuntimeError("IndexFlatL2.search: index is empty, call add() first")
        distances, indices = self._nn.kneighbors(X, n_neighbors=k)
        return distances, indices.astype(np.int64)

    def reset(self):
        self._all_data = None
        self._ntotal = 0
        self._nn = None

    def __getstate__(self):
        return {
            'd': self._d,
            'ntotal': self._ntotal,
            'all_data': self._all_data,
        }

    def __setstate__(self, state):
        self._d = state['d']
        self._ntotal = state['ntotal']
        self._all_data = state['all_data']
        # Rebuild the nn index from the retained training data so that both
        # search() and reconstruct*() work after unpickling. Pickling the data
        # rather than the nn object avoids storing two copies.
        self._nn = None
        if self._all_data is not None:
            self._nn = nearest_neighbors(metric='sqeuclidean')
            self._nn.fit(self._all_data)
        self.aocl = True

    def assign(self, X, k):
        if self._nn is None:
            raise RuntimeError("IndexFlatL2.assign: index is empty, call add() first")
        _, I = self.search(X, k)
        return I

    def _check_reconstruct(self, where):
        if self._ntotal == 0:
            raise RuntimeError(f"IndexFlatL2.{where}: index is empty, call add() first")

    def reconstruct(self, i):
        self._check_reconstruct("reconstruct")
        if i < 0 or i >= self._ntotal:
            raise IndexError(
                f"IndexFlatL2.reconstruct: index {i} out of range [0, {self._ntotal})")
        return self._all_data[i].copy()

    def reconstruct_n(self, i0, ni):
        self._check_reconstruct("reconstruct_n")
        return self._all_data[i0:i0 + ni].copy()

    def reconstruct_batch(self, ids):
        self._check_reconstruct("reconstruct_batch")
        return self._all_data[np.asarray(ids)].copy()

    def search_and_reconstruct(self, X, k):
        self._check_reconstruct("search_and_reconstruct")
        D, I = self.search(X, k)
        I_safe = np.where(I == -1, 0, I)
        R = self._all_data[I_safe]
        R[I == -1] = 0
        return D, I, R.astype(np.float32)

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
            f"IndexFlatL2.{name} is not supported by the AOCL-DA backend")
