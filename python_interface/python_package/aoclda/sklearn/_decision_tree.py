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
Patching scikit learn tree: DecisionTreeClassifier
"""
# pylint: disable = missing-function-docstring, too-many-ancestors, useless-return, super-init-not-called

import warnings
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier_sklearn
from aoclda.decision_tree import decision_tree as decision_tree_da

class DecisionTreeClassifier(DecisionTreeClassifier_sklearn):
    """
    Overwrite sklearn DecisionTreeClassifier to call DA library
    """

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.precision = "double"

        if criterion in ('log_loss', 'entropy'):
            score_criteria = "cross-entropy"
        elif criterion == "gini":
            score_criteria = "gini"

        if splitter == "random":
            raise ValueError("splitter must be set to best")

        if max_depth is None:
            depth = -1
        else:
            depth = max_depth

        # if max_features is None:
        #     raise ValueError("max_features must be an int and > 0")
        # else:
        #     n_features_to_select = max_features

        if random_state is None:
            seed = -1
        elif not isinstance(random_state, int):
            raise ValueError("random_state must be either None or an int")
        else:
            seed = random_state

        if (min_samples_split != 2 or
            min_samples_leaf != 1 or
            min_weight_fraction_leaf != 0.0 or
            max_leaf_nodes is not None or
            min_impurity_decrease != 0.0 or
            class_weight is not None or
            ccp_alpha != 0.0 or
            monotonic_cst is not None):
            warnings.warn(
                "The parameters min_samples_split, min_samples_leaf, max_leaf_nodes, "
                "min_impurity_decrease, class_weight, ccp_alpha and monotonic_cst"
                "are not supported and have been ignored.", category=RuntimeWarning)

        # new internal attributes
        self.aocl = True

        self.decision_tree_double = decision_tree_da(seed = seed,
                                              score_criteria = score_criteria,
                                              depth = depth,
                                              precision = "double")

        self.decision_tree_single = decision_tree_da(seed = seed,
                                              score_criteria = score_criteria,
                                              depth = depth,
                                              precision = "single")

        self.decision_tree = self.decision_tree_double

    def fit(self, X, y, sample_weight=None, check_input=True):

        if (sample_weight is not None or
            check_input is not True):
            warnings.warn(
                "The parameters sample_weight and check_input"
                "are not supported and have been ignored.", category=RuntimeWarning)

        # If data matrix is in single precision switch internally
        if X.dtype == "float32":
            self.precision = "single"
            self.decision_tree = self.decision_tree_single
            del self.decision_tree_double

        self.decision_tree.fit(X, y)

        return self

    def predict(self, X, check_input=True):

        if check_input is not True:
            warnings.warn(
                "The parameter check_input"
                "is not supported and has been ignored.", category=RuntimeWarning)

        return self.decision_tree.predict(X)

    def score(self, X, y, sample_weight=None):

        if sample_weight is not None:
            warnings.warn(
                "The parameter sample_weight"
                "is not supported and has been ignored.", category=RuntimeWarning)

        return self.decision_tree.score(X, y)

    def apply(self, X, check_input=True):
        raise RuntimeError("This feature is not implemented")

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        raise RuntimeError("This feature is not implemented")

    def get_depth(self):
        raise RuntimeError("This feature is not implemented")

    def get_metadata_routing(self):
        raise RuntimeError("This feature is not implemented")

    def get_n_leaves(self):
        raise RuntimeError("This feature is not implemented")

    def get_params(self, deep=True):
        params = {'random_state': self.random_state,
                  'criterion': self.criterion,
                  'max_depth': self.max_depth}
        return params

    def predict_log_proba(self, X):
        raise RuntimeError("This feature is not implemented")

    def predict_proba(self, X, check_input=True):
        raise RuntimeError("This feature is not implemented")

    def set_fit_request(self):
        raise RuntimeError("This feature is not implemented")

    def set_params(self, **params):
        raise RuntimeError("This feature is not implemented")

    def set_predict_proba_request(self):
        raise RuntimeError("This feature is not implemented")

    def set_predict_request(self):
        raise RuntimeError("This feature is not implemented")

    def set_score_request(self):
        raise RuntimeError("This feature is not implemented")

    @property
    def feature_importances_(self):
        print("This attribute is not implemented")
        return None
