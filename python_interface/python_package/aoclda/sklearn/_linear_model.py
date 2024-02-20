# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
Patching scikit learn linear models: LinearRegression, Ridge, Lasso
"""
# pylint: disable = super-init-not-called, too-many-ancestors, missing-function-docstring, useless-return

from sklearn.linear_model import LinearRegression as LinearRegression_sklearn
from sklearn.linear_model import Ridge as Ridge_sklearn
from sklearn.linear_model import Lasso as Lasso_sklearn
from aoclda.linear_model import linmod as linmod_da


class LinearRegression(LinearRegression_sklearn):
    """
    Overwrite sklearn LinearRegression to call DA library
    """

    def __init__(self, *, fit_intercept=True, copy_X=True, n_jobs=None, positive=False) -> None:
        # Supported attributes
        self.fit_intercept = fit_intercept

        # Not supported yet
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

        # New attributes used internally
        self.aocl = True
        self.intercept_val = None
        self.lmod = linmod_da("mse", intercept=fit_intercept)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported")
        self.lmod.fit(X, y)

    @property
    def coef_(self):
        coef = self.lmod.get_coef()
        if self.intercept_val is None:
            self.intercept_val = coef[-1]
        return coef[:-1]

    @property
    def rank_(self):
        print("This feature is not implemented")
        return None

    @property
    def singular_(self):
        print("This feature is not implemented")
        return None

    @property
    def intercept_(self):
        if self.intercept_val is None:
            coef = self.lmod.get_coef()
            self.intercept_val = coef[-1]
        return self.intercept_val


class Ridge(Ridge_sklearn):
    """
    Overwrite sklearn Ridge to call DA library
    """

    def __init__(self, alpha=1, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001,
                 solver="auto", positive=False, random_state=None) -> None:
        # supported attributes
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        # currently ignored
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # not supported
        self.solver = solver
        self.positive = positive

        # New attributes used internally
        self.aocl = True
        self.intercept_val = None

        # solver can be in
        # ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
        if solver not in ['auto', 'lbfgs']:
            raise ValueError("Only 'auto' and 'lbfgs' solvers are supported")
        if positive:
            raise ValueError(
                "Constraints on the coefficients are not supported")

        # Initialize aoclda object
        self.lmod = linmod_da("mse", intercept=fit_intercept)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported")
        self.lmod.fit(X, y, reg_lambda=self.alpha, reg_alpha=0.0)

    @property
    def coef_(self):
        coef = self.lmod.get_coef()
        if self.intercept_val is None:
            self.intercept_val = coef[-1]
        return coef[:-1]

    @property
    def n_iter_(self):
        print("This feature is not implemented")
        return None

    @property
    def n_features_in_(self):
        print("This feature is not implemented")
        return None

    @property
    def feature_names_in(self):
        print("This feature is not implemented")
        return None

    @property
    def intercept_(self):
        if self.intercept_val is None:
            coef = self.lmod.get_coef()
            self.intercept_val = coef[-1]
        return self.intercept_val


class Lasso(Lasso_sklearn):
    """
    Overwrite sklearn Lasso to call DA library
    """

    def __init__(self, alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True,
                 max_iter=1000, tol=0.0001, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        # supported attributes
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        # Currently ignored attributes
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection

        # not supported attributes
        self.positive = positive

        # New attributes used internally
        self.aocl = True
        self.intercept_val = None

        # Initialize aoclda object
        self.lmod = linmod_da("mse", intercept=fit_intercept)

    def fit(self, X, y, sample_weight=None, check_input=True):
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported")
        self.lmod.fit(X, y, reg_lambda=self.alpha, reg_alpha=1.0)

    @property
    def coef_(self):
        coef = self.lmod.get_coef()
        if self.intercept_val is None:
            self.intercept_val = coef[-1]
        return coef[:-1]

    @property
    def n_iter_(self):
        print("This feature is not implemented")
        return None

    @property
    def n_features_in_(self):
        print("This feature is not implemented")
        return None

    @property
    def feature_names_in(self):
        print("This feature is not implemented")
        return None

    @property
    def intercept_(self):
        if self.intercept_val is None:
            coef = self.lmod.get_coef()
            self.intercept_val = coef[-1]
        return self.intercept_val

    @property
    def dual_gap_(self):
        print("This feature is not implemented")
        return None
