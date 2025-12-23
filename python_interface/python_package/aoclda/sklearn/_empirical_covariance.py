# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
Patching scikit-learn linear models: empirical_covariance, EmpiricalCovariance
"""

import warnings
from sklearn.covariance import EmpiricalCovariance as EmpiricalCovariance_sklearn
from aoclda.basic_stats import covariance_matrix, mean
import numpy as np


def empirical_covariance(X, *, assume_centered=False):
    """
    Overwrite scikit-learn empirical_covariance to call AOCL-DA library
    """
    covariance = covariance_matrix(X=X, dof=-1, assume_centered=assume_centered)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance


class EmpiricalCovariance(EmpiricalCovariance_sklearn):
    """
    Overwrite scikit-learn EmpiricalCovariance to call AOCL-DA library
    """

    def __init__(self, *, store_precision=True, assume_centered=False):
        self.assume_centered = assume_centered
        self.store_precision = False

        if store_precision:
            warnings.warn(
                "'store_precision' is not yet implemented and will be set to False."
            )

        self.location_ = None
        self.covariance_ = None

        # New internal attributes
        self.aocl = True

    def _set_covariance(self, covariance):
        self.covariance_ = covariance

    def get_precision(self):
        raise RuntimeError("This feature is not implemented.")

    def fit(self, X, y=None):
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = mean(X)

        covariance = covariance_matrix(X=X, dof=-1, assume_centered=self.assume_centered)
        self._set_covariance(covariance)

        return self

    def score(self, X_test, y=None):
        raise RuntimeError("This feature is not implemented.")

    def error_norm(self, comp_cov, norm="frobenius", scaling=True, squared=True):
        raise RuntimeError("This feature is not implemented.")

    def mahalanobis(self, X):
        raise RuntimeError("This feature is not implemented.")

    @property
    def precision_(self):
        """precision_ not implemented."""
        print("This attribute is not implemented.")
        return None
