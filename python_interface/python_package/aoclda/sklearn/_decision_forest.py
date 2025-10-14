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
Patching scikit-learn tree: RandomForestClassifier
"""
# pylint: disable = missing-function-docstring, too-many-ancestors,
# super-init-not-called, too-many-instance-attributes, too-many-arguments,
# too-many-locals, super-init-not-called, too-many-locals, too-many-boolean-expressions,
#

import warnings
import numpy as np
from aoclda.decision_forest import decision_forest as decision_forest_da
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_sklearn


class RandomForestClassifier(RandomForestClassifier_sklearn):
    """
    Overwrite scikit-learn RandomForestClassifier to call AOCL-DA library
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        histogram=True,
        maximum_bins=256
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.monotonic_cst = monotonic_cst
        self.precision = "double"
        self.histogram = histogram
        self.maximum_bins = maximum_bins

        self.n_trees = n_estimators
        self.n_obs_per_tree = max_samples

        self.da_max_features = 0

        if criterion in ('log_loss', 'entropy'):
            score_criteria = "cross-entropy"
        else:
            score_criteria = criterion

        if max_depth is None:
            depth = 29
        else:
            depth = max_depth

        if random_state is None:
            seed = -1
        elif not isinstance(random_state, int):
            raise ValueError("random_state must be either None or an int")
        else:
            seed = random_state

        if isinstance(max_features, str):
            self.features_selection = max_features

        if (min_samples_leaf != 1 or
            min_weight_fraction_leaf != 0.0 or
            max_leaf_nodes is not None or
            oob_score is not False or
            (n_jobs is not None and n_jobs != -1) or
            verbose != 0 or
            warm_start is not False or
            class_weight is not None or
            ccp_alpha != 0.0 or
                monotonic_cst is not None):
            warnings.warn(
                "The parameters min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, "
                "oob_score, n_jobs, warm_start, class_weight, ccp_alpha and monotonic_cst "
                "are not supported and have been ignored.", category=RuntimeWarning)

        # new internal attributes
        self.aocl = True

        # Set faeture selection options
        self.da_max_features = 0
        self.da_feature_prop = 0.1
        self.features_selection = "all"
        if isinstance(self.max_features, int):
            self.da_max_features = self.max_features
            self.features_selection = "custom"
        elif isinstance(self.max_features, float):
            self.features_selection = "proportion"
            self.da_feature_prop = self.max_features
        elif isinstance(self.max_features, str):
            self.features_selection = self.max_features

        # Set samples selection options
        self.da_samples_factor = 1.0
        self.update_samples_factor = False
        if isinstance(self.max_samples, int):
            self.update_samples_factor = True
        elif isinstance(self.max_samples, float):
            self.da_samples_factor = self.max_samples

        self.decision_forest = decision_forest_da(
            n_trees=self.n_estimators,
            criterion=score_criteria,
            seed=seed,
            max_depth=depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            features_selection=self.features_selection,
            max_features=self.da_max_features,
            proportion_features=self.da_feature_prop,
            samples_factor=self.da_samples_factor,
            min_impurity_decrease=min_impurity_decrease,
            min_split_score=1.0e-05,
            feat_thresh=1.0e-06,
            histogram=histogram,
            maximum_bins=maximum_bins,
            block_size=256,
            category_split_strategy="ordered",
            check_data=False)

    def fit(self, X, y, sample_weight=None):

        if sample_weight is not None:
            warnings.warn(
                "The parameters sample_weight"
                "is not supported and has been ignored.", category=RuntimeWarning)

        n_samples = X.shape[0]
        if self.update_samples_factor:
            if X.dtype == "float32":
                self.da_samples_factor = np.float32(
                    self.max_samples) / np.float32(n_samples)
            if X.dtype == "float64":
                self.da_samples_factor = np.float64(
                    self.max_samples) / np.float64(n_samples)
            self.decision_forest.samples_factor = self.da_samples_factor

        self.decision_forest.fit(X, y)

        return self

    def predict(self, X):
        return self.decision_forest.predict(X)

    def score(self, X, y, sample_weight=None):

        if sample_weight is not None:
            warnings.warn(
                "The parameter sample_weight"
                "is not supported and has been ignored.", category=RuntimeWarning)

        return self.decision_forest.score(X, y)

    def apply(self, X):
        raise RuntimeError("This feature is not implemented")

    def decision_path(self, X):
        raise RuntimeError("This feature is not implemented")

    def get_metadata_routing(self):
        raise RuntimeError("This feature is not implemented")

    def predict_log_proba(self, X):
        return self.decision_forest.predict_log_proba(X)

    def predict_proba(self, X):
        return self.decision_forest.predict_proba(X)

    def set_fit_request(self, **kwargs):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, **kwargs):
        raise RuntimeError("This feature is not implemented")

    @property
    def estimators_samples_(self):
        print("This attribute is not implemented")

    @property
    def feature_importances_(self):
        print("This attribute is not implemented")
