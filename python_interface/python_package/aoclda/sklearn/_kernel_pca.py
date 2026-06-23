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
Patching scikit learn decomposition: KernelPCA
"""
# pylint: disable = missing-function-docstring, too-many-ancestors,
# useless-return, super-init-not-called

import warnings
import numpy as np
from sklearn.decomposition import KernelPCA as KernelPCA_sklearn
from aoclda.factorization import KernelPCA as KernelPCA_da


class KernelPCA(KernelPCA_sklearn):
    """
    Overwrite sklearn KernelPCA to call DA library
    """

    def __init__(self, n_components=None, *, kernel='linear', gamma=None, degree=3,
                 coef0=1, kernel_params=None, alpha=1.0, fit_inverse_transform=False,
                 eigen_solver='auto', tol=0, max_iter=None, iterated_power='auto',
                 remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None):
        # Supported attributes
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.fit_inverse_transform = fit_inverse_transform
        self.remove_zero_eig = remove_zero_eig

        # Not supported yet
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.copy_X = copy_X
        self.n_jobs = n_jobs

        # Check for unsupported attributes
        da_n_components = n_components
        if n_components is None:
            da_n_components = 0
        elif n_components < 1:
            raise ValueError("n_components must be a positive integer")

        if callable(kernel):
            raise ValueError("Callable kernels are not supported")

        if kernel == 'cosine':
            raise ValueError("kernel='cosine' is not supported")

        if kernel_params is not None:
            raise ValueError("kernel_params is not supported")

        if eigen_solver not in ('auto', 'dense', 'randomized'):
            raise ValueError(
                "eigen_solver must be set to 'auto', 'dense', or 'randomized'")

        if isinstance(random_state, np.random.RandomState):
            raise ValueError("random_state must be an integer or None.")

        if tol != 0 or max_iter is not None or n_jobs is not None:
            warnings.warn(
                "The parameters tol, max_iter and n_jobs "
                "are not supported and have been ignored.",
                category=RuntimeWarning)

        # new internal attributes
        self.aocl = True
        self.precision = "double"

        # Translate random_state to AOCL-DA seed
        seed = -1 if random_state is None else random_state

        # Translate options to aocl-da ones
        da_gamma = -1.0 if gamma is None else gamma

        solver = eigen_solver
        if eigen_solver == "dense":
            solver = "syevd"
        # 'randomized' and 'auto' pass through as-is

        _power_iterations = -1 if iterated_power == 'auto' else int(iterated_power)

        self.kpca = KernelPCA_da(
            n_components=da_n_components,
            kernel=kernel,
            eigensolver=solver,
            gamma=da_gamma,
            degree=degree,
            coef0=coef0,
            fit_inverse_transform=fit_inverse_transform,
            alpha=alpha,
            remove_zero_eig=remove_zero_eig,
            copy_X=copy_X,
            power_iterations=_power_iterations,
            seed=seed)

    def fit(self, X, y=None):
        self.kpca.fit(X)
        return self

    def transform(self, X):
        return self.kpca.transform(X)

    def inverse_transform(self, X):
        return self.kpca.inverse_transform(X)

    def fit_transform(self, X, y=None):
        self.kpca.fit(X)
        return self.kpca.scores

    def get_feature_names_out(self, input_features=None):
        raise RuntimeError("This feature is not implemented")

    def get_metadata_routing(self, *args):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        return {'n_components': self.n_components,
                'kernel': self.kernel,
                'gamma': self.gamma,
                'degree': self.degree,
                'coef0': self.coef0,
                'kernel_params': self.kernel_params,
                'alpha': self.alpha,
                'fit_inverse_transform': self.fit_inverse_transform,
                'eigen_solver': self.eigen_solver,
                'tol': self.tol,
                'max_iter': self.max_iter,
                'iterated_power': self.iterated_power,
                'remove_zero_eig': self.remove_zero_eig,
                'random_state': self.random_state,
                'copy_X': self.copy_X,
                'n_jobs': self.n_jobs}

    def set_output(self, *, transform=None):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    # Match sklearn attributes
    @property
    def eigenvalues_(self):
        return self.kpca.eigenvalues

    @property
    def eigenvectors_(self):
        return self.kpca.eigenvectors

    @property
    def dual_coef_(self):
        return self.kpca.dual_coef

    @property
    def X_transformed_fit_(self):
        return self.kpca.scores

    @property
    def X_fit_(self):
        return self.kpca.X_fit_

    @property
    def gamma_(self):
        return self.kpca.gamma_

    @property
    def n_features_in_(self):
        return self.kpca.n_features

    @property
    def feature_names_in_(self):
        raise AttributeError("This attribute is not implemented")

    # AOCL-DA attributes not matched with an sklearn attribute
    @property
    def n_components_(self):
        return self.kpca.n_components

    @property
    def n_samples_(self):
        return self.kpca.n_samples
