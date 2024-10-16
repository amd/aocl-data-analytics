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

# pylint: disable = import-error, invalid-name, too-many-arguments

"""
aoclda.decision_forest module
"""

from ._aoclda.decision_forest import pybind_decision_forest


class decision_forest():
    """
    Decision forest classifier.

    An ensemble classifier based on decision trees.

    Args:

        seed (int, optional): Set the random seed for the random number generator.
            If the value is -1, a random seed is automatically generated. In this case
            the resulting classification will create non-reproducible results.
            Default = -1.

        n_trees (int, optional): Set the number of trees to compute. Default = 100.

        criterion (str, optional): Select scoring function to use. It can take the values
            'cross-entropy', 'gini', or 'misclassification'

        max_depth (int, optional): Set the maximum depth of the trees. Default = 29.

        features_selection (str, optional): Select how many features to use for each split.
            It can take the values 'all', 'sqrt', 'log2', or 'custom'. If set to 'custom',
            the number of features to consider is set by the optional argument `max_features`.
            Default 'sqrt'.

        max_features (int, optional): Set the number of features to consider when
            splitting a node. Only taken into account when features_selection is set to 'custom'.
            0 means take all the features. Default 0.

        min_samples_split (int, optional): The minimum number of samples required to
            split an internal node. Default 2.

        build_order (str, optional): Select in which order to explore the nodes. It can
            take the values 'breadth first' or 'depth first'. Default 'breadth first'.

        bootstrap (bool, optional): Select whether to bootstrap the samples in the trees.
            Default True.

        samples_factor (float, optional):  Proportion of samples to draw from
            the data set to build each tree if 'bootstrap' was set to True.
            Default 0.8.

        min_impurity_decrease (float, optional): Minimum score improvement
            needed to consider a split from the parent node. Default 0.0

        min_split_score (float, optional): Minimum score needed for a node
            to be considered for splitting. Default 0.0.

        feat_thresh (float, optional): Minimum difference in feature value
            required for splitting. Default 1.0e-06

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(self, n_trees=100, criterion='gini', seed=-1, max_depth=29,
                 min_samples_split=2, build_order='breadth first', bootstrap=True,
                 features_selection='sqrt', max_features=0, samples_factor=0.8,
                 min_impurity_decrease=0.0, min_split_score=0.0, feat_thresh=1.0e-06, check_data=False):

        self.samples_factor = samples_factor
        self.min_impurity_decrease = min_impurity_decrease
        self.min_split_score = min_split_score
        self.feat_thresh = feat_thresh

        self.decision_forest_double = pybind_decision_forest(n_trees=n_trees,
                                                             criterion=criterion, seed=seed,
                                                             max_depth=max_depth,
                                                             min_samples_split=min_samples_split,
                                                             build_order=build_order,
                                                             bootstrap=bootstrap,
                                                             features_selection=features_selection,
                                                             max_features=max_features,
                                                             precision="double",
                                                             check_data=check_data)
        self.decision_forest_single = pybind_decision_forest(n_trees=n_trees,
                                                             criterion=criterion, seed=seed,
                                                             max_depth=max_depth,
                                                             min_samples_split=min_samples_split,
                                                             build_order=build_order,
                                                             bootstrap=bootstrap,
                                                             features_selection=features_selection,
                                                             max_features=max_features,
                                                             precision="single",
                                                             check_data=check_data)
        self.decision_forest = self.decision_forest_double
        self.max_features = max_features
        self.features_selection = features_selection

    @property
    def max_features(self):
        return self.max_features

    @max_features.setter
    def max_features(self, value):
        self.decision_forest.set_max_features_opt(max_features=value)

    @property
    def features_selection(self):
        return self.features_selection

    @features_selection.setter
    def features_selection(self, value):
        self.decision_forest.set_features_selection_opt(
            features_selection=value)

    def fit(self, X, y):
        """
        Computes the decision forest on the feature matrix ``X`` and response vector ``y``

        Args:
            X (numpy.ndarray): The feature matrix on which to compute the model.
                Its shape is (n_samples, n_features).

            y (numpy.ndarray): The response vector. Its shape is (n_samples).

        Returns:
            self (object): Returns the instance itself.
        """
        if X.dtype == "float32":
            self.decision_forest = self.decision_forest_single
            self.decision_forest_double = None

        return self.decision_forest.pybind_fit(X, y, self.samples_factor,
                                               self.min_impurity_decrease, self.min_split_score,
                                               self.feat_thresh)

    def score(self, X, y):
        """
        Calculates score (prediction accuracy) by comparing predicted labels and actual
        labels on a new set of data.

        Args:
            X (numpy.ndarray): The feature matrix to evaluate the model on.
                It must have n_features columns.

            y (numpy.ndarray): The response vector.  It must have shape (n_samples).

        Returns:
            float: The mean accuracy of the model on the test data.
        """
        return self.decision_forest.pybind_score(X, y)

    def predict(self, X):
        """
        Generate labels using fitted decision forest on a new set of data ``X``.

        Args:
            X (numpy.ndarray): The feature matrix to evaluate the model on.
                It must have n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector,
                where n_samples is the number of rows of X.
        """
        return self.decision_forest.pybind_predict(X)

    def predict_proba(self, X):
        """
        Generate class probabilities using fitted decision forest on a new set of data ``X``.

        Args:
            X (numpy.ndarray): The feature matrix to evaluate the model on.
                It must have n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector,
                where n_samples is the number of rows of X.
        """
        return self.decision_forest.pybind_predict_proba(X)

    def predict_log_proba(self, X):
        """
        Generate class log probabilities using fitted decision forest on a new set of data ``X``.

        Args:
            X (numpy.ndarray): The feature matrix to evaluate the model on.
                It must have n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector,
                where n_samples is the number of rows of X.
        """
        return self.decision_forest.pybind_predict_log_proba(X)
