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
aoclda.dimension_reduction module
"""

import numpy as np
from ._aoclda.dimension_reduction import pybind_tsne
from ._internal_utils import check_convert_data


class tsne():
    """
    t-SNE embedding.

    Args:
        n_components (int, optional): Number of embedding dimensions. Default=2.
        perplexity (float, optional): Target perplexity. Default=30.0.
        learning_rate (float or str, optional): Gradient descent learning rate.
          Use any non-positive value or "auto" for automatic selection:
          max(n_samples / early_exaggeration / 4, 50). Default=-1 (auto).
        max_iter (int, optional): Maximum number of gradient descent iterations.
          Default=1000.
        n_iter_without_progress (int, optional): Stop if no progress is made for
          this many iterations. Default=300.
        min_grad_norm (float, optional): Stop if the gradient norm is below this
          threshold. Default=1e-7.
        early_exaggeration (float, optional): Early exaggeration factor. Default=12.0.
        theta (float, optional): Barnes-Hut approximation parameter in [0, 1]
          (0 for exact). Default=0.5.
        init (str or numpy.ndarray, optional): Initialization method. Can be 'pca',
          'random', or a numpy array of shape (n_samples, n_components) providing
          the initial embedding. Default='pca'.
        seed (int, optional): Seed for randomness; set to -1 for non-deterministic.
          Default=0.
        mixed_precision (bool, optional): Whether to use mixed precision iterative
          refinement, in which lower precision arithmetic is used before switching
          to the working precision for the final iterations. Default=False.
        low_precision_max_iter (int, optional): If mixed precision iterative
          refinement is enabled, maximum number of iterations for the low precision
          phase. Default=200.
        low_precision_min_grad_norm (float, optional): If mixed precision iterative
          refinement is enabled, gradient norm convergence threshold for the low
          precision phase. Default=1e-4.
        check_data (bool, optional): Whether to check data for NaNs. Default=False.

    """

    def __init__(self, n_components=2, perplexity=30.0, learning_rate=-1.0,
                 max_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-7,
                 early_exaggeration=12.0, theta=0.5, init='pca', seed=0,
                 mixed_precision=False, low_precision_max_iter=200,
                 low_precision_min_grad_norm=1e-4, check_data=False):

        # Handle init: if array-like, store it and use 'supplied' for C++
        self._init_embedding = None
        if isinstance(init, str):
            init_str = init
        else:
            init_arr = np.asarray(init)
            if init_arr.ndim != 2:
                raise ValueError(
                    "init array must be 2-dimensional with shape "
                    "(n_samples, n_components).")
            if init_arr.shape[1] != n_components:
                raise ValueError(
                    f"init array has {init_arr.shape[1]} columns but "
                    f"n_components={n_components}.")
            if not np.all(np.isfinite(init_arr)):
                raise ValueError(
                    "init array must contain only finite values.")
            self._init_embedding = init_arr
            init_str = 'supplied'

        self.tsne_double = pybind_tsne(
            n_components,
            max_iter,
            init_str,
            seed,
            'double',
            check_data,
            mixed_precision,
            low_precision_max_iter)
        self.tsne_single = pybind_tsne(
            n_components,
            max_iter,
            init_str,
            seed,
            'single',
            check_data,
            mixed_precision,
            low_precision_max_iter)

        self.dtype = 'float'
        self.order = 'A'
        self.tsne = self.tsne_double
        self.embedding_ = None
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.early_exaggeration = early_exaggeration
        self.theta = theta
        self.low_precision_min_grad_norm = low_precision_min_grad_norm

    @property
    def embedding(self):
        """numpy.ndarray of shape (n_samples, n_components): The embedding."""
        return self.tsne.get_embedding()

    @property
    def n_samples(self):
        """int: The number of samples in the data matrix used."""
        return self.tsne.get_n_samples()

    @property
    def n_features(self):
        """int: The number of features in the data matrix."""
        return self.tsne.get_n_features()

    @property
    def n_components(self):
        """int: The number of embedding dimensions."""
        return self.tsne.get_n_components()

    @property
    def n_iter(self):
        """int: The number of iterations performed."""
        return self.tsne.get_n_iter()

    @property
    def kl_divergence(self):
        """float: The final KL divergence."""
        return self.tsne.get_kl_divergence()

    def _prepare_fit(self, X):
        """Prepare data and options for fitting."""
        X, self.order, self.dtype = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        if isinstance(self.learning_rate, str):
            if self.learning_rate == "auto":
                self.learning_rate = -1.0
            else:
                raise ValueError(
                    "learning_rate must be a positive float, "
                    "a non-positive value (auto), or 'auto'.")

        if self.dtype == "float32":
            cast = np.float32
            self.tsne = self.tsne_single
            self.tsne_double = None
        else:
            cast = np.float64
        self.perplexity = cast(self.perplexity)
        self.learning_rate = cast(self.learning_rate)
        self.early_exaggeration = cast(self.early_exaggeration)
        self.theta = cast(self.theta)
        self.min_grad_norm = cast(self.min_grad_norm)
        self.low_precision_min_grad_norm = cast(self.low_precision_min_grad_norm)

        init_emb = None
        if self._init_embedding is not None:
            if self._init_embedding.shape[0] != X.shape[0]:
                raise ValueError(
                    f"init array has {self._init_embedding.shape[0]} rows "
                    f"but X has {X.shape[0]} samples.")
            init_emb = np.asarray(self._init_embedding, dtype=X.dtype,
                                  order=self.order)
        return X, init_emb

    def _fit_args(self, init_emb):
        """Return the common argument tuple for pybind_fit / pybind_fit_transform."""
        return (
            self.perplexity,
            self.learning_rate,
            self.early_exaggeration,
            self.theta,
            self.n_iter_without_progress,
            self.min_grad_norm,
            init_emb,
            self.low_precision_min_grad_norm,
        )

    def fit(self, X):
        """
        Compute t-SNE embedding for the supplied data matrix.

        Args:
            X (array-like): The data matrix with shape (n_samples, n_features).

        Returns:
            self (object): Returns the instance itself.
        """
        X, init_emb = self._prepare_fit(X)
        self.tsne.pybind_fit(X, *self._fit_args(init_emb))
        return self

    def fit_transform(self, X):
        """
        Compute t-SNE embedding and return the embedding matrix.

        Args:
            X (array-like): The data matrix with shape (n_samples, n_features).

        Returns:
            numpy.ndarray of shape (n_samples, n_components): The embedding.
        """
        X, init_emb = self._prepare_fit(X)
        self.embedding_ = self.tsne.pybind_fit_transform(
            X, *self._fit_args(init_emb))
        return self.embedding_
