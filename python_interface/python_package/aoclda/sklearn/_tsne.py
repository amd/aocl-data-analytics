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
Patching scikit-learn TSNE
"""
# pylint: disable = missing-function-docstring, too-many-ancestors, too-many-locals,
# useless-return, super-init-not-called, no-member

import warnings
import numpy as np
from sklearn.manifold import TSNE as TSNE_sklearn
from aoclda.dimension_reduction import tsne as tsne_da


class TSNE(TSNE_sklearn):
    """
    Overwrite scikit-learn TSNE to call AOCL-DA library
    """

    def __init__(
            self,
            n_components=2,
            *,
            perplexity=30.0,
            early_exaggeration=12.0,
            learning_rate="auto",
            max_iter=1000,
            n_iter_without_progress=300,
            min_grad_norm=1e-7,
            metric="euclidean",
            metric_params=None,
            init="pca",
            verbose=0,
            random_state=None,
            method="barnes_hut",
            angle=0.5,
            n_jobs=None,
            mixed_precision=False,
            low_precision_max_iter=200,
            low_precision_min_grad_norm=1e-4):

        # Supported attributes
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.method = method
        self.angle = angle

        # Partially supported parameters (kept for sklearn compatibility)
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.mixed_precision = mixed_precision
        self.low_precision_max_iter = low_precision_max_iter
        self.low_precision_min_grad_norm = low_precision_min_grad_norm

        if n_components < 1 or n_components > 3:
            raise ValueError("n_components must be between 1 and 3")

        if isinstance(random_state, np.random.RandomState):
            raise ValueError("random_state must be an integer or None.")

        if not isinstance(init, np.ndarray) and init not in ("pca", "random"):
            raise ValueError(
                "init must be 'pca', 'random', or a numpy array of shape "
                "(n_samples, n_components).")

        if metric != "euclidean":
            warnings.warn(
                "Only euclidean metric is supported and has been enforced.",
                category=RuntimeWarning)

        if any(x is not None for x in (metric_params, n_jobs)) or verbose != 0:
            warnings.warn(
                "Some TSNE parameters are not supported and have been ignored.",
                category=RuntimeWarning)

        # new internal attributes
        self.aocl = True
        self.seed = random_state if random_state is not None else -1

        # Map method/angle to Barnes-Hut theta
        if method == "exact":
            theta = 0.0
        elif method == "barnes_hut":
            theta = angle
        else:
            warnings.warn(
                "method must be set to 'barnes_hut' or 'exact'. Using barnes_hut.",
                category=RuntimeWarning)
            theta = angle

        self.tsne = tsne_da(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            n_iter_without_progress=self.n_iter_without_progress,
            min_grad_norm=self.min_grad_norm,
            early_exaggeration=self.early_exaggeration,
            theta=theta,
            init=self.init,
            seed=self.seed,
            mixed_precision=self.mixed_precision,
            low_precision_max_iter=self.low_precision_max_iter,
            low_precision_min_grad_norm=self.low_precision_min_grad_norm,
            check_data=False)

    def fit(self, X, y=None):
        self.tsne.fit(X)
        return self

    def fit_transform(self, X, y=None):
        return self.tsne.fit_transform(X)

    def get_params(self, deep=True):
        params = {'n_components': self.n_components,
                  'perplexity': self.perplexity,
                  'early_exaggeration': self.early_exaggeration,
                  'learning_rate': self.learning_rate,
                  'max_iter': self.max_iter,
                  'n_iter_without_progress': self.n_iter_without_progress,
                  'min_grad_norm': self.min_grad_norm,
                  'metric': self.metric,
                  'metric_params': self.metric_params,
                  'init': self.init,
                  'verbose': self.verbose,
                  'random_state': self.random_state,
                  'method': self.method,
                  'angle': self.angle,
                  'n_jobs': self.n_jobs}
        return params

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_output(self, *, transform=None):
        raise RuntimeError("This feature is not implemented")

    # Match sklearn attributes
    @property
    def embedding_(self):
        return self.tsne.embedding

    @property
    def kl_divergence_(self):
        return self.tsne.kl_divergence

    @property
    def n_iter_(self):
        return self.tsne.n_iter

    @property
    def n_features_in_(self):
        return self.tsne.n_features
