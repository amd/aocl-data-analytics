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
Patching scikit-learn clustering: DBSCAN
"""
# pylint: disable = missing-function-docstring, too-many-ancestors, useless-return, super-init-not-called, no-member

import warnings
from sklearn.cluster import DBSCAN as DBSCAN_sklearn
from aoclda.clustering import DBSCAN as DBSCAN_da

class DBSCAN(DBSCAN_sklearn):
    """
    Overwrite scikit-learn DBCSAN to call AOCL-DA library
    """

    def __init__(self, eps=0.5, *, min_samples=5, metric='euclidean', metric_params=None,
                 algorithm='auto', leaf_size=30, p=None, n_jobs=None):

        # Supported attributes
        self.eps = eps
        self.min_samples = min_samples
        self.algorithm = algorithm
        self.metric = metric
        self.leaf_size = leaf_size
        if p is None:
            self.p = 2
        else:
            self.p = p


        # Not supported yet
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        # Check for unsupported attributes
        if (metric_params is not None or n_jobs is not None):
            warnings.warn(
                "The parameters metric_params and n_jobs are not supported and have been ignored.",
                category=RuntimeWarning)

        # new internal attributes
        self.aocl = True

        # Initialize the DBSCAN object
        self.DBSCAN = DBSCAN_da(eps=eps, min_samples=min_samples, metric=metric,
                                algorithm=algorithm, leaf_size=leaf_size, power=p)

    def fit(self, X, y=None, sample_weight=None):
        self.DBSCAN.fit(X)
        return self


    def fit_predict(self,  X, y=None, sample_weight=None):
        self.DBSCAN.fit(X)
        return self.DBSCAN.labels

    def get_metadata_routing(self, *args):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'eps': self.eps,
                  'min_samples': self.min_samples,
                  'algorithm': self.algorithm,
                  'metric': self.metric,
                  'p': self.p,
                  'leaf_size': self.leaf_size}
        return params

    def set_fit_request(self, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, *, transform=None):
        raise RuntimeError("This feature is not implemented")

    # Match all attributes from sklearn
    # return None if not yet written
    @property
    def core_sample_indices_(self):
        return self.DBSCAN.core_sample_indices

    @property
    def labels_(self):
        return self.DBSCAN.labels

    @property
    def components_(self):
        print("This attribute is not implemented")
        return None

    @property
    def n_features_in_(self):
        return self.DBSCAN.n_features

    @property
    def feature_names_in_(self):
        print("This attribute is not implemented")
        return None

    # AOCL-DA attributes not yet matched with an sklearn attribute
