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
Patching scikit learn tree: RandomForestClassifier
"""
# pylint: disable = missing-function-docstring, too-many-ancestors, useless-return, super-init-not-called

import warnings
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_sklearn
from aoclda.decision_forest import decision_forest as decision_forest_da
import aoclda as da


class RandomForestClassifier(RandomForestClassifier_sklearn):
    """
    Overwrite sklearn RandomForestClassifier to call DA library
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

        self.n_trees = n_estimators
        self.n_obs_per_tree = max_samples

        if criterion in ('log_loss', 'entropy'):
            score_criteria = "cross-entropy"
        else:
            score_criteria = criterion

        if max_depth is None:
            depth = -1
        else:
            depth = max_depth

        if random_state is None:
            seed = -1
        elif not isinstance(random_state, int):
            raise ValueError("random_state must be either None or an int")
        else:
            seed = random_state

        if (min_samples_split != 2 or
            min_samples_leaf != 1 or
            min_weight_fraction_leaf != 0.0 or
            max_leaf_nodes != None or
            min_impurity_decrease != 0.0 or
            bootstrap is not True or
            oob_score is not False or
            n_jobs is not None or
            verbose != 0 or
            warm_start is not False or
            class_weight is not None or
            ccp_alpha != 0.0 or
            monotonic_cst is not None):
            warnings.warn(
                "The parameters min_samples_split, min_samples_leaf, max_leaf_nodes, "
                "min_impurity_decrease, bootstrap, oob_score, n_jobs, verbose, "
                "warm_start, class_weight, ccp_alpha and monotonic_cst"
                "are not supported and have been ignored.", category=RuntimeWarning)

        # new internal attributes
        self.aocl = True

        self.decision_forest_double = decision_forest_da(seed = seed,
                                                  score_criteria = score_criteria,
                                                  n_trees = self.n_trees,
                                                  precision = "double")

        self.decision_forest_single = decision_forest_da(seed = seed,
                                                  score_criteria = score_criteria,
                                                  n_trees = self.n_trees,
                                                  precision = "single")

        self.decision_forest = self.decision_forest_double

    def fit(self, X, y, sample_weight=None):

        if sample_weight is not None:
            warnings.warn(
                "The parameters sample_weight"
                "is not supported and has been ignored.", category=RuntimeWarning)

        # If data matrix is in single precision switch internally
        if X.dtype == "float32":
            self.precision = "single"
            self.decision_forest = self.decision_forest_single
            del self.decision_forest_double

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
        raise RuntimeError("This feature is not implemented")

    def predict_proba(self, X):
        raise RuntimeError("This feature is not implemented")

    def set_fit_request(self, **kwargs):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self, **kwargs):
        raise RuntimeError("This feature is not implemented")

    @property
    def estimators_samples_(self):
        print("This attribute is not implemented")
        return None

    @property
    def feature_importances_(self):
        print("This attribute is not implemented")
        return None