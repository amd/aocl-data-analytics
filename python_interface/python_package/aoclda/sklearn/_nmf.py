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
Patching scikit-learn decomposition: NMF
"""
# pylint: disable = missing-function-docstring, too-many-ancestors,
# super-init-not-called, too-many-arguments, too-many-instance-attributes

import warnings
import numpy as np
from sklearn.decomposition import NMF as NMF_sklearn
from aoclda.factorization import NMF as NMF_da


class NMF(NMF_sklearn):
    """
    Overwrite sklearn NMF to call DA library
    """

    def __init__(self, n_components='auto', *, init=None, solver='mu',
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None, alpha_W=0.0, alpha_H='same',
                 l1_ratio=0.0, verbose=0, shuffle=False):
        # Supported attributes
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

        # Not supported yet
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle

        # Validate against the only supported configuration
        if solver != 'mu':
            raise ValueError(
                f"solver must be 'mu', got {solver!r}.")
        if beta_loss != 'frobenius':
            raise ValueError(
                f"beta_loss must be 'frobenius', got {beta_loss!r}.")

        if isinstance(random_state, np.random.RandomState):
            raise ValueError("random_state must be an integer or None.")

        if init is None:
            init_da = 'auto'
        elif init in ('nndsvd', 'nndsvda', 'nndsvdar', 'custom', 'random'):
            init_da = init
        else:
            raise ValueError(
                f"Unsupported init={init!r}; expected one of "
                "None, 'nndsvd', 'nndsvda', 'nndsvdar', 'custom', 'random'.")

        if (alpha_W != 0.0 or alpha_H != 'same' or l1_ratio != 0.0 or
                verbose != 0 or shuffle is not False):
            warnings.warn(
                "The parameters alpha_W, alpha_H, l1_ratio, verbose and shuffle "
                "are not supported and have been ignored.",
                category=RuntimeWarning)

        self._n_components_da = 0 if n_components is None else n_components

        # New internal attributes
        self.aocl = True
        self.precision = "double"

        self.nmf = NMF_da(
            n_components=self._n_components_da,
            solver='mu',
            beta_loss='frobenius',
            max_iter=self.max_iter,
            tol=self.tol,
            init=init_da,
            random_state=self.random_state,
            check_every=10,
        )

    def fit(self, X, y=None, **params):
        if self.n_components == 'auto':
            raise ValueError(
                "n_components='auto' requires W and H to be provided via "
                "fit_transform() so that the rank can be inferred from their "
                "shapes. fit() alone is not supported with n_components='auto'.")
        self.nmf.fit(X)
        return self

    def transform(self, X):
        return self.nmf.transform(X)

    def fit_transform(self, X, y=None, W=None, H=None):
        if self.n_components == 'auto':
            # n_components='auto' is only valid when W and H are both supplied
            # so that the rank can be inferred from their inner dimension.
            if W is None or H is None:
                raise ValueError(
                    "n_components='auto' requires both W and H to be provided "
                    "to fit_transform() so that the rank can be inferred from "
                    "their shapes.")
            if self.init != 'custom':
                raise ValueError(
                    "n_components='auto' with W and H requires init='custom'.")
            # Delegate to the DA layer which infers k from W.shape[1].
            return self.nmf.fit_transform(X, W=W, H=H)
        if self.init == 'custom':
            if W is None or H is None:
                raise ValueError(
                    "init='custom' requires both W and H to be supplied to fit_transform().")
            return self.nmf.fit_transform(X, W=W, H=H)
        return self.nmf.fit_transform(X)

    def inverse_transform(self, X):
        return self.nmf.inverse_transform(X)

    def get_params(self, deep=True):
        return {
            'n_components': self.n_components,
            'init': self.init,
            'solver': self.solver,
            'beta_loss': self.beta_loss,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'alpha_W': self.alpha_W,
            'alpha_H': self.alpha_H,
            'l1_ratio': self.l1_ratio,
            'verbose': self.verbose,
            'shuffle': self.shuffle,
        }

    def set_output(self, *, transform=None):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def get_feature_names_out(self, input_features=None):
        raise RuntimeError("This feature is not implemented")

    def get_metadata_routing(self, *args):
        raise RuntimeError("This feature is not implemented")

    # Match sklearn attributes
    @property
    def components_(self):
        return self.nmf.h

    @property
    def n_components_(self):
        return self.nmf.n_components

    @property
    def reconstruction_err_(self):
        return self.nmf.reconstruction_err

    @property
    def n_iter_(self):
        return self.nmf.n_iter

    @property
    def n_features_in_(self):
        return self.nmf.n_features

    @property
    def feature_names_in_(self):
        return None
