# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
Patching scikit learn svm: SVC, SVR, nuSVC, nuSVR
"""
# pylint: disable = missing-function-docstring, too-many-ancestors, useless-return, super-init-not-called

import warnings
from sklearn.svm import SVC as SVC_sklearn, SVR as SVR_sklearn, NuSVC as NuSVC_sklearn, NuSVR as NuSVR_sklearn
from aoclda.svm import SVC as SVC_da, SVR as SVR_da, NuSVC as NuSVC_da, NuSVR as NuSVR_da
import numpy as np


class SVC(SVC_sklearn):
    """
    Overwrite sklearn SVC to call DA library
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=False,
                 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
                 max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
        # Supported attributes
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape

        # Check for unsupported attributes
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.break_ties = break_ties
        self.random_state = random_state

        if kernel == 'precomputed':
            raise RuntimeError("Precomputed kernel is not supported")

        if shrinking is not False:
            warnings.warn(
                "shrinking is not supported and has been ignored.", category=RuntimeWarning)

        if probability is not False:
            warnings.warn(
                "probability is not supported and has been ignored.", category=RuntimeWarning)

        if cache_size != 200:
            warnings.warn(
                "cache_size is not supported and has been ignored.", category=RuntimeWarning)

        if class_weight is not None:
            warnings.warn(
                "class_weight is not supported and has been ignored.", category=RuntimeWarning)

        if verbose is not False:
            warnings.warn(
                "verbose is not supported and has been ignored.", category=RuntimeWarning)

        if break_ties is not False:
            warnings.warn(
                "break_ties is not supported and has been ignored.", category=RuntimeWarning)

        if random_state is not None:
            warnings.warn(
                "random_state is not supported and has been ignored.", category=RuntimeWarning)

        # new internal attributes
        self.aocl = True

        # Translate options to aocl-da ones
        self.svc = SVC_da(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                          coef0=self.coef0, tol=self.tol, max_iter=self.max_iter, decision_function_shape=self.decision_function_shape)

    def fit(self, X, y):
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                self.svc.gamma = -1.0
            elif self.gamma == 'auto':
                self.svc.gamma = 1.0 / X.shape[1]
            else:
                raise ValueError("Unsupported value for gamma")
        self.svc.fit(X, y)
        return self

    def predict(self, X):
        return self.svc.predict(X)

    def decision_function(self, X):
        return self.svc.decision_function(X)

    def score(self, X, y):
        return self.svc.score(X, y)

    def get_metadata_routing(self, *args):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'C': self.C,
                  'kernel': self.kernel,
                  'degree': self.degree,
                  'gamma': self.gamma,
                  'coef0': self.coef0,
                  'tol': self.tol,
                  'max_iter': self.max_iter,
                  'decision_function_shape': self.decision_function_shape,
                  'shrinking': self.shrinking,
                  'probability': self.probability,
                  'cache_size': self.cache_size,
                  'class_weight': self.class_weight,
                  'verbose': self.verbose,
                  'break_ties': self.break_ties,
                  'random_state': self.random_state}
        return params

    def predict_log_proba(self, X):
        raise RuntimeError("This feature is not implemented")

    def predict_proba(self, X):
        raise RuntimeError("This feature is not implemented")

    def set_fit_request(self, *args):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, *args):
        raise RuntimeError("This feature is not implemented")

    @property
    def class_weight_(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def classes_(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def coef_(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def dual_coef_(self):
        return self.svc.dual_coef

    @property
    def fit_status(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def intercept_(self):
        return self.svc.bias

    @property
    def n_features_in_(self):
        return self.svc.n_features

    @property
    def feature_names_in_(self):
        print("This attribute is not implemented")
        return None

    @property
    def n_iter_(self):
        print("This attribute is not implemented")
        return None

    @property
    def support_(self):
        return self.svc.support_vectors_idx

    @property
    def support_vectors_(self):
        return self.svc.support_vectors

    @property
    def n_support_(self):
        return self.svc.n_support_per_class

    @property
    def probA_(self):
        print("This attribute is not implemented")
        return None

    @property
    def probB_(self):
        print("This attribute is not implemented")
        return None

    @property
    def shape_fit_(self):
        print("This attribute is not implemented")
        return None


class SVR(SVR_sklearn):
    """
    Overwrite sklearn SVR to call DA library
    """

    def __init__(self, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=False,
                 cache_size=200, verbose=False, max_iter=-1):
        # Supported attributes
        self.epsilon = epsilon
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

        # Check for unsupported attributes
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose

        if kernel == 'precomputed':
            raise RuntimeError("Precomputed kernel is not supported")

        if shrinking is not False:
            warnings.warn(
                "shrinking is not supported and has been ignored.", category=RuntimeWarning)

        if cache_size != 200:
            warnings.warn(
                "cache_size is not supported and has been ignored.", category=RuntimeWarning)

        if verbose is not False:
            warnings.warn(
                "verbose is not supported and has been ignored.", category=RuntimeWarning)

        # new internal attributes
        self.aocl = True

        # Translate options to aocl-da ones
        self.svr = SVR_da(C=self.C, epsilon=self.epsilon, kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                          coef0=self.coef0, tol=self.tol, max_iter=self.max_iter)

    def fit(self, X, y):
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                self.svr.gamma = -1.0
            elif self.gamma == 'auto':
                self.svr.gamma = 1.0 / X.shape[1]
            else:
                raise ValueError("Unsupported value for gamma")
        self.svr.fit(X, y)
        return self

    def predict(self, X):
        return self.svr.predict(X)

    def score(self, X, y):
        return self.svr.score(X, y)

    def get_metadata_routing(self, *args):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'C': self.C,
                  'epsilon': self.epsilon,
                  'kernel': self.kernel,
                  'degree': self.degree,
                  'gamma': self.gamma,
                  'coef0': self.coef0,
                  'tol': self.tol,
                  'max_iter': self.max_iter,
                  'shrinking': self.shrinking,
                  'cache_size': self.cache_size,
                  'verbose': self.verbose}
        return params

    def set_fit_request(self, *args):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, *args):
        raise RuntimeError("This feature is not implemented")

    @property
    def coef_(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def dual_coef_(self):
        return self.svr.dual_coef

    @property
    def fit_status(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def intercept_(self):
        return self.svr.bias

    @property
    def n_features_in_(self):
        return self.svr.n_features

    @property
    def feature_names_in_(self):
        print("This attribute is not implemented")
        return None

    @property
    def n_iter_(self):
        print("This attribute is not implemented")
        return None

    @property
    def n_support_(self):
        return np.array([self.svr.n_support])

    @property
    def shape_fit_(self):
        print("This attribute is not implemented")
        return None

    @property
    def support_(self):
        return self.svr.support_vectors_idx

    @property
    def support_vectors_(self):
        return self.svr.support_vectors


class NuSVC(NuSVC_sklearn):
    """
    Overwrite sklearn NuSVC to call DA library
    """

    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=False,
                 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
                 max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
        # Supported attributes
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.gamma = gamma
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape

        # Check for unsupported attributes
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.break_ties = break_ties
        self.random_state = random_state

        if kernel == 'precomputed':
            raise RuntimeError("Precomputed kernel is not supported")

        if shrinking is not False:
            warnings.warn(
                "shrinking is not supported and has been ignored.", category=RuntimeWarning)

        if probability is not False:
            warnings.warn(
                "probability is not supported and has been ignored.", category=RuntimeWarning)

        if cache_size != 200:
            warnings.warn(
                "cache_size is not supported and has been ignored.", category=RuntimeWarning)

        if class_weight is not None:
            warnings.warn(
                "class_weight is not supported and has been ignored.", category=RuntimeWarning)

        if verbose is not False:
            warnings.warn(
                "verbose is not supported and has been ignored.", category=RuntimeWarning)

        if break_ties is not False:
            warnings.warn(
                "break_ties is not supported and has been ignored.", category=RuntimeWarning)

        if random_state is not None:
            warnings.warn(
                "random_state is not supported and has been ignored.", category=RuntimeWarning)

        # new internal attributes
        self.aocl = True

        # Translate options to aocl-da ones
        self.nusvc = NuSVC_da(nu=self.nu, kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                              coef0=self.coef0, tol=self.tol, max_iter=self.max_iter, decision_function_shape=self.decision_function_shape)

    def fit(self, X, y):
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                self.nusvc.gamma = -1.0
            elif self.gamma == 'auto':
                self.nusvc.gamma = 1.0 / X.shape[1]
            else:
                raise ValueError("Unsupported value for gamma")
        self.nusvc.fit(X, y)
        return self

    def predict(self, X):
        return self.nusvc.predict(X)

    def decision_function(self, X):
        return self.nusvc.decision_function(X)

    def score(self, X, y):
        return self.nusvc.score(X, y)

    def get_metadata_routing(self, *args):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'nu': self.nu,
                  'kernel': self.kernel,
                  'degree': self.degree,
                  'gamma': self.gamma,
                  'coef0': self.coef0,
                  'tol': self.tol,
                  'max_iter': self.max_iter,
                  'decision_function_shape': self.decision_function_shape,
                  'shrinking': self.shrinking,
                  'probability': self.probability,
                  'cache_size': self.cache_size,
                  'class_weight': self.class_weight,
                  'verbose': self.verbose,
                  'break_ties': self.break_ties,
                  'random_state': self.random_state}
        return params

    def predict_log_proba(self, X):
        raise RuntimeError("This feature is not implemented")

    def predict_proba(self, X):
        raise RuntimeError("This feature is not implemented")

    def set_fit_request(self, *args):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, *args):
        raise RuntimeError("This feature is not implemented")

    @property
    def class_weight_(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def classes_(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def coef_(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def dual_coef_(self):
        return self.nusvc.dual_coef

    @property
    def fit_status(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def intercept_(self):
        return self.nusvc.bias

    @property
    def n_features_in_(self):
        return self.nusvc.n_features

    @property
    def feature_names_in_(self):
        print("This attribute is not implemented")
        return None

    @property
    def n_iter_(self):
        print("This attribute is not implemented")
        return None

    @property
    def support_(self):
        return self.nusvc.support_vectors_idx

    @property
    def support_vectors_(self):
        return self.nusvc.support_vectors

    @property
    def n_support_(self):
        return self.nusvc.n_support_per_class

    @property
    def probA_(self):
        print("This attribute is not implemented")
        return None

    @property
    def probB_(self):
        print("This attribute is not implemented")
        return None

    @property
    def shape_fit_(self):
        print("This attribute is not implemented")
        return None


class NuSVR(NuSVR_sklearn):
    """
    Overwrite sklearn NuSVR to call DA library
    """

    def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=False,
                 tol=0.001, cache_size=200, verbose=False, max_iter=-1):
        # Supported attributes
        self.nu = nu
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.gamma = gamma
        self.max_iter = max_iter

        # Check for unsupported attributes
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose

        if kernel == 'precomputed':
            raise RuntimeError("Precomputed kernel is not supported")

        if shrinking is not False:
            warnings.warn(
                "shrinking is not supported and has been ignored.", category=RuntimeWarning)

        if cache_size != 200:
            warnings.warn(
                "cache_size is not supported and has been ignored.", category=RuntimeWarning)

        if verbose is not False:
            warnings.warn(
                "verbose is not supported and has been ignored.", category=RuntimeWarning)

        # new internal attributes
        self.aocl = True

        # Translate options to aocl-da ones
        self.nusvr = NuSVR_da(nu=self.nu, C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                              coef0=self.coef0, tol=self.tol, max_iter=self.max_iter)

    def fit(self, X, y):
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                self.nusvr.gamma = -1.0
            elif self.gamma == 'auto':
                self.nusvr.gamma = 1.0 / X.shape[1]
            else:
                raise ValueError("Unsupported value for gamma")
        self.nusvr.fit(X, y)
        return self

    def predict(self, X):
        return self.nusvr.predict(X)

    def score(self, X, y):
        return self.nusvr.score(X, y)

    def get_metadata_routing(self, *args):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'nu': self.nu,
                  'C': self.C,
                  'kernel': self.kernel,
                  'degree': self.degree,
                  'gamma': self.gamma,
                  'coef0': self.coef0,
                  'tol': self.tol,
                  'max_iter': self.max_iter,
                  'shrinking': self.shrinking,
                  'cache_size': self.cache_size,
                  'verbose': self.verbose}
        return params

    def set_fit_request(self, *args):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, *args):
        raise RuntimeError("This feature is not implemented")

    @property
    def coef_(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def dual_coef_(self):
        return self.nusvr.dual_coef

    @property
    def fit_status(self):
        print("This attribute is not yet implemented")
        return None

    @property
    def intercept_(self):
        return self.nusvr.bias

    @property
    def n_features_in_(self):
        return self.nusvr.n_features

    @property
    def feature_names_in_(self):
        print("This attribute is not implemented")
        return None

    @property
    def n_iter_(self):
        print("This attribute is not implemented")
        return None

    @property
    def n_support_(self):
        return np.array([self.nusvr.n_support])

    @property
    def shape_fit_(self):
        print("This attribute is not implemented")
        return None

    @property
    def support_(self):
        return self.nusvr.support_vectors_idx

    @property
    def support_vectors_(self):
        return self.nusvr.support_vectors
