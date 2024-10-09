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
Patching scikit-learn linear models: LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
"""
# pylint: disable = super-init-not-called, too-many-ancestors, missing-function-docstring, useless-return

import warnings
import numpy as np
from sklearn.linear_model import LinearRegression as LinearRegression_sklearn
from sklearn.linear_model import Ridge as Ridge_sklearn
from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import ElasticNet as ElasticNet_sklearn
from sklearn.linear_model import LogisticRegression as LogisticRegression_sklearn
from aoclda.linear_model import linmod as linmod_da


class LinearRegression(LinearRegression_sklearn):
    """
    Overwrite scikit-learn LinearRegression to call AOCL-DA library
    """

    def __init__(self,
                 *,
                 fit_intercept=True,
                 solver='auto',
                 copy_X=True,
                 n_jobs=None,
                 positive=False) -> None:
        # Supported attributes
        self.fit_intercept = fit_intercept

        # Not supported yet
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

        # New attributes used internally
        self.aocl = True
        self.intercept_val = None
        self.solver = solver
        self.precision = "double"

        # Initialize both single and double precision classes for now
        self.lmod_double = linmod_da(
            "mse", solver=solver, intercept=fit_intercept, precision="double")
        self.lmod_single = linmod_da(
            "mse", solver=solver, intercept=fit_intercept, precision="single")
        self.lmod = self.lmod_double

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported")

        # If data matrix is in single precision switch internally
        if X.dtype == "float32":
            self.precision = "single"
            self.lmod = self.lmod_single
            self.lmod_double = None
        self.lmod.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.lmod.predict(X)

    def get_metadata_routing(self):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'fit_intercept': self.fit_intercept,
                  'solver': self.solver,
                  'copy_X': self.copy_X,
                  'n_jobs': self.n_jobs,
                  'positive': self.positive
                  }
        return params

    def score(self, X, y, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_fit_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    @property
    def coef_(self):
        coef = self.lmod.get_coef()
        if self.fit_intercept:
            if self.intercept_val is None:
                self.intercept_val = coef[-1]
            return coef[:-1]
        else:
            return coef

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
        if self.fit_intercept:
            if self.intercept_val is None:
                coef = self.lmod.get_coef()
                self.intercept_val = coef[-1]
            return self.intercept_val
        else:
            return 0.0

    @property
    def n_features_in_(self):
        print("This feature is not implemented")
        return None

    @property
    def features_names_in_(self):
        print("This feature is not implemented")
        return None


class Ridge(Ridge_sklearn):
    """
    Overwrite scikit-learn Ridge to call AOCL-DA library
    """

    def __init__(self,
                 alpha=1,
                 *,
                 fit_intercept=True,
                 copy_X=True,
                 max_iter=None,
                 tol=0.0001,
                 solver="auto",
                 positive=False,
                 random_state=None) -> None:
        # supported attributes
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

        # currently ignored
        self.copy_X = copy_X
        self.random_state = random_state

        # not supported
        self.positive = positive

        # New attributes used internally
        self.aocl = True
        self.intercept_val = None

        # solver can be in
        # ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
        if self.solver not in ['auto', 'lbfgs', 'cholesky', 'svd', 'sparse_cg']:
            raise ValueError(
                "Only 'auto', 'lbfgs', 'cholesky', 'svd' and 'sparse_cg' solvers are supported"
            )
        if positive:
            raise ValueError(
                "Constraints on the coefficients are not supported")

        # Initialize both single and double precision classes for now
        self.lmod_double = linmod_da(
            "mse", solver=solver, intercept=fit_intercept, precision="double")
        self.lmod_single = linmod_da(
            "mse", solver=solver, intercept=fit_intercept, precision="single")
        self.lmod = self.lmod_double

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported")
        # If data matrix is in single precision switch internally

        if X.dtype == "float32":
            self.precision = "single"
            self.lmod = self.lmod_single
            self.lmod_double = None
            reg_lambda_t = np.float32(self.alpha)
            reg_alpha_t = np.float32(0.0)
            tol_t = np.float32(self.tol)
        else:
            reg_lambda_t = np.float64(self.alpha)
            reg_alpha_t = np.float64(0.0)
            tol_t = np.float64(self.tol)

        self.lmod.fit(X, y, reg_lambda=reg_lambda_t, reg_alpha=reg_alpha_t, tol=tol_t)
        return self

    def predict(self, X) -> np.ndarray:
        return self.lmod.predict(X)

    def get_metadata_routing(self):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'fit_intercept': self.fit_intercept,
                  'solver': self.solver,
                  'copy_X': self.copy_X,
                  'positive': self.positive,
                  'alpha': self.alpha,
                  'max_iter': self.max_iter,
                  'random_state': self.random_state,
                  'tol': self.tol
                  }
        return params

    def score(self, X, y, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_fit_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    @property
    def coef_(self):
        coef = self.lmod.get_coef()
        if self.fit_intercept:
            if self.intercept_val is None:
                self.intercept_val = coef[-1]
            return coef[:-1]
        else:
            return coef

    @property
    def n_iter_(self):
        n_iter = self.lmod.get_n_iter()
        return n_iter

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
        if self.fit_intercept:
            if self.intercept_val is None:
                coef = self.lmod.get_coef()
                self.intercept_val = coef[-1]
            return self.intercept_val
        else:
            return 0.0

    @property
    def solver_(self):
        return self.solver


class Lasso(Lasso_sklearn):
    """
    Overwrite scikit-learn Lasso to call AOCL-DA library
    """

    def __init__(self,
                 alpha=1.0,
                 *,
                 fit_intercept=True,
                 precompute=False,
                 copy_X=True,
                 max_iter=1000,
                 tol=0.0001,
                 warm_start=False,
                 positive=False,
                 random_state=None,
                 selection='cyclic'):
        # supported attributes
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

        # Currently ignored attributes
        self.precompute = precompute
        self.copy_X = copy_X

        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection

        # not supported attributes
        self.positive = positive

        # New attributes used internally
        self.aocl = True
        self.intercept_val = None

        # Initialize both single and double precision classes for now
        self.lmod_double = linmod_da("mse",
                                     intercept=fit_intercept,
                                     max_iter=self.max_iter,
                                     precision="double")
        self.lmod_single = linmod_da("mse",
                                     intercept=fit_intercept,
                                     max_iter=self.max_iter,
                                     precision="single")
        self.lmod = self.lmod_double

    def fit(self, X, y, sample_weight=None, check_input=True):
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported")
        # If data matrix is in single precision switch internally
        if X.dtype == "float32":
            self.precision = "single"
            self.lmod = self.lmod_single
            self.lmod_double = None
            reg_lambda_t = np.float32(self.alpha)
            reg_alpha_t = np.float32(1.0)
            tol_t = np.float32(self.tol)
        else:
            reg_lambda_t = np.float64(self.alpha)
            reg_alpha_t = np.float64(1.0)
            tol_t = np.float64(self.tol)
        self.lmod.fit(X, y, reg_lambda=reg_lambda_t, reg_alpha=reg_alpha_t, tol=tol_t)
        return self

    def predict(self, X) -> np.ndarray:
        return self.lmod.predict(X)

    def get_metadata_routing(self):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'fit_intercept': self.fit_intercept,
                  'precompute': self.precompute,
                  'copy_X': self.copy_X,
                  'positive': self.positive,
                  'alpha': self.alpha,
                  'max_iter': self.max_iter,
                  'random_state': self.random_state,
                  'tol': self.tol,
                  'warm_start': self.warm_start,
                  'selection': self.selection
                  }
        return params

    def score(self, X, y, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_fit_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    @property
    def coef_(self):
        coef = self.lmod.get_coef()
        if self.fit_intercept:
            if self.intercept_val is None:
                self.intercept_val = coef[-1]
            return coef[:-1]
        else:
            return coef

    @property
    def n_iter_(self):
        n_iter = self.lmod.get_n_iter()
        return n_iter

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
        if self.fit_intercept:
            if self.intercept_val is None:
                coef = self.lmod.get_coef()
                self.intercept_val = coef[-1]
            return self.intercept_val
        else:
            return 0.0

    @property
    def dual_gap_(self):
        print("This feature is not implemented")
        return None

    @property
    def sparse_coef_(self):
        print("This feature is not implemented")
        return None


class ElasticNet(ElasticNet_sklearn):
    """
    Overwrite scikit-learn ElasticNet to call AOCL-DA library
    """

    def __init__(
        self,
        alpha=1.0,
        *,
        l1_ratio=0.5,
        fit_intercept=True,
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=1e-4,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        # supported attributes
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

        # Currently ignored attributes
        self.precompute = precompute
        self.copy_X = copy_X

        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection

        # not supported attributes
        self.positive = positive

        # New attributes used internally
        self.aocl = True
        self.intercept_val = None

        # Initialize both single and double precision classes for now
        self.lmod_double = linmod_da("mse",
                                     intercept=fit_intercept,
                                     max_iter=self.max_iter,
                                     precision="double")
        self.lmod_single = linmod_da("mse",
                                     intercept=fit_intercept,
                                     max_iter=self.max_iter,
                                     precision="single")
        self.lmod = self.lmod_double

    def fit(self, X, y, sample_weight=None, check_input=True):
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported")
        # If data matrix is in single precision switch internally
        if X.dtype == "float32":
            self.precision = "single"
            self.lmod = self.lmod_single
            self.lmod_double = None
            reg_lambda_t = np.float32(self.alpha)
            reg_alpha_t = np.float32(self.l1_ratio)
            tol_t = np.float32(self.tol)
        else:
            reg_lambda_t = np.float64(self.alpha)
            reg_alpha_t = np.float64(self.l1_ratio)
            tol_t = np.float64(self.tol)

        self.lmod.fit(X,
                      y,
                      reg_lambda=reg_lambda_t,
                      reg_alpha=reg_alpha_t,
                      tol=tol_t)
        return self

    def predict(self, X) -> np.ndarray:
        return self.lmod.predict(X)

    def get_metadata_routing(self):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'fit_intercept': self.fit_intercept,
                  'precompute': self.precompute,
                  'copy_X': self.copy_X,
                  'positive': self.positive,
                  'alpha': self.alpha,
                  'max_iter': self.max_iter,
                  'random_state': self.random_state,
                  'tol': self.tol,
                  'warm_start': self.warm_start,
                  'selection': self.selection,
                  'l1_ratio': self.l1_ratio
                  }
        return params

    def score(self, X, y, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_fit_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    @property
    def coef_(self):
        coef = self.lmod.get_coef()
        if self.fit_intercept:
            if self.intercept_val is None:
                self.intercept_val = coef[-1]
            return coef[:-1]
        else:
            return coef

    @property
    def n_iter_(self):
        n_iter = self.lmod.get_n_iter()
        return n_iter

    @property
    def dual_gap_(self):
        print("This feature is not implemented")
        return None

    @property
    def sparse_coef_(self):
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
        if self.fit_intercept:
            if self.intercept_val is None:
                coef = self.lmod.get_coef()
                self.intercept_val = coef[-1]
            return self.intercept_val
        else:
            return 0.0


class LogisticRegression(LogisticRegression_sklearn):
    """
    Overwrite scikit-learn LogisticRegression to call AOCL-DA library
    """

    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="deprecated",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=0.0,
        progress_factor=None,
        constraint="ssc"
    ):
        # supported attributes
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

        # Currently ignored attributes
        self.penalty = penalty
        self.dual = dual
        self.solver = solver
        self.verbose = verbose
        self.warm_start = warm_start
        self.l1_ratio = l1_ratio

        if self.penalty != 'l2' and self.penalty is not None:
            raise ValueError(
                "penalty argument currently only supports 'l2' and None")

        if self.dual is not False:
            warnings.warn("dual argument is not supported and will be ignored")

        if self.solver != "lbfgs":
            warnings.warn("currently only lbfgs solver is supported")

        if self.verbose != 0:
            warnings.warn(
                "verbose argument is not supported and will be ignored")

        if self.warm_start is not False:
            warnings.warn(
                "warm_start argument is not supported and will be ignored")

        if self.l1_ratio != 0.0:
            raise ValueError("currently l1_ratio argument is not supported")

        # not supported attributes
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        # We do not warn for this parameter since its becoming deprecated
        self.multi_class = multi_class
        self.n_jobs = n_jobs

        if self.intercept_scaling != 1:
            warnings.warn(
                "intercept_scaling argument is not supported and will be ignored")

        if self.class_weight is not None:
            warnings.warn(
                "class_weight argument is not supported and will be ignored")

        if self.random_state is not None:
            warnings.warn(
                "random_state argument is not supported and will be ignored")

        if self.n_jobs is not None:
            warnings.warn(
                "n_jobs argument is not supported and will be ignored")

        # New attributes used internally
        self.aocl = True
        self.intercept_val = None
        self.progress_factor = progress_factor
        self.constraint = constraint
        if self.penalty == 'l2':
            self.reg_lambda = 1/self.C
        elif self.penalty is None:
            self.reg_lambda = 0
        self.n_class = None

        # Initialize aoclda object
        self.lmod_double = linmod_da("logistic",
                                     intercept=self.fit_intercept,
                                     max_iter=self.max_iter,
                                     constraint=self.constraint,
                                     precision="double")
        self.lmod_single = linmod_da("logistic",
                                     intercept=self.fit_intercept,
                                     max_iter=self.max_iter,
                                     constraint=self.constraint,
                                     precision="single")
        self.lmod = self.lmod_double

    def fit(self, X, y, sample_weight=None, check_input=True):
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported")
        self.n_class = len(np.unique(y))
        if self.n_class == 2:
            self.n_class = 1
        # If data matrix is in single precision switch internally
        if X.dtype == "float32":
            self.precision = "single"
            self.lmod = self.lmod_single
            self.lmod_double = None
            reg_lambda_t = np.float32(self.reg_lambda)
            reg_alpha_t = np.float32(self.l1_ratio)
            tol_t = np.float32(self.tol)
            if self.progress_factor is not None:
                progress_factor_t = np.float32(self.progress_factor)
            else:
                progress_factor_t = None
        else:
            reg_lambda_t = np.float64(self.reg_lambda)
            reg_alpha_t = np.float64(self.l1_ratio)
            tol_t = np.float64(self.tol)
            if self.progress_factor is not None:
                progress_factor_t = np.float64(self.progress_factor)
            else:
                progress_factor_t = None
        self.lmod.fit(X,
                      y,
                      reg_lambda=reg_lambda_t,
                      reg_alpha=reg_alpha_t,
                      tol=tol_t,
                      progress_factor=progress_factor_t)
        return self

    def predict(self, X) -> np.ndarray:
        return self.lmod.predict(X)

    def decision_fuction(self, X):
        raise RuntimeError("This feature is not implemented")

    def densify(self):
        raise RuntimeError("This feature is not implemented")

    def get_metadata_routing(self):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        raise RuntimeError("This feature is not implemented")

    def predict_log_proba(self, X):
        raise RuntimeError("This feature is not implemented")

    def predict_proba(self, X):
        raise RuntimeError("This feature is not implemented")

    def score(self, X, y, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_fit_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def sparsify(self):
        raise RuntimeError("This feature is not implemented")

    @property
    def coef_(self):
        coef = self.lmod.get_coef()
        # We are returning a ndarray of shape (n_class-1, n_feat)
        if self.constraint == 'rsc':
            if self.fit_intercept:
                if self.intercept_val is None:
                    self.intercept_val = coef[-(self.n_class-1):]
                return np.reshape(coef[:-(self.n_class-1)], (self.n_class-1, -1), order="F")
            return np.reshape(coef, (self.n_class-1, -1), order="F")
        # We are returning a ndarray of shape (n_class, n_feat)
        elif self.constraint == 'ssc':
            if self.fit_intercept:
                if self.intercept_val is None:
                    self.intercept_val = coef[-self.n_class:]
                return np.reshape(coef[:-self.n_class], (self.n_class, -1), order="F")
            return np.reshape(coef, (self.n_class, -1), order="F")

    @property
    def intercept_(self):
        if self.constraint == 'rsc':
            if self.fit_intercept:
                if self.intercept_val is None:
                    coef = self.lmod.get_coef()
                    self.intercept_val = coef[-(self.n_class-1):]
                return self.intercept_val
            else:
                return np.zeros(self.n_class-1)
        elif self.constraint == 'ssc':
            if self.fit_intercept:
                if self.intercept_val is None:
                    coef = self.lmod.get_coef()
                    self.intercept_val = coef[-self.n_class:]
                return self.intercept_val
            else:
                return np.zeros(self.n_class)

    @property
    def n_features_in_(self):
        print("This feature is not implemented")
        return None

    @property
    def feature_names_in(self):
        print("This feature is not implemented")
        return None

    @property
    def n_iter_(self):
        n_iter = self.lmod.get_n_iter()
        return n_iter
