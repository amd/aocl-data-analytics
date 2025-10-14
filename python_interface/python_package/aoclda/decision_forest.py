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

# pylint: disable = import-error, invalid-name, too-many-arguments,
# too-many-positional-arguments

"""
aoclda.decision_forest module
"""

from ._aoclda.decision_forest import pybind_decision_forest
from ._internal_utils import check_convert_data


class decision_forest():
    """
    Decision forest classifier.

    An ensemble classifier based on decision trees.

    Args:

        n_trees (int, optional): Set the number of trees to train. Default = 100.

        criterion (str, optional): Select scoring function to use. It can take the values
            'cross-entropy', 'gini', or 'misclassification'

        max_depth (int, optional): Set the maximum depth of the trees. Default = 29.

        seed (int, optional): Set the random seed for the random number generator.
            If the value is -1, a random seed is automatically generated. In this case
            the resulting classification will create non-reproducible results.
            Default = -1.

        features_selection (str, optional): Select how many features to use for each split.
            'custom' reads the 'maximum features' option, proportion reads the
            'proportion features' option. 'all', 'sqrt' and 'log2' select respectively all,
            the square root or the base-2 logarithm of the total number of features.

        max_features (int, optional): Set the number of features to consider when
            'features selection' is set to 'custom'. 0 means take all the features.
            Default 0.

        proportion_features (float, optional): Proportion of features to consider when
            'features selection' is set to 'proportion'. Default 0.1.

        min_samples_split (int, optional): The minimum number of samples required to
            split an internal node. Default 2.

        bootstrap (bool, optional): Select whether to bootstrap the samples in the trees.
            Default True.

        samples_factor (float, optional):  Proportion of samples to draw from
            the data set to build each tree if 'bootstrap' was set to True.
            Default 1.0.

        min_impurity_decrease (float, optional): Minimum score improvement
            needed to consider a split from the parent node. Default 0.0

        min_split_score (float, optional): Minimum score needed for a node
            to be considered for splitting. Default 0.0.

        feat_thresh (float, optional): Minimum difference in feature value
            required for splitting. Default 1.0e-05

        histogram (bool, optional): Whether to use histogram-based splitting.
            Default = False.

        maximum_bins (int, optional): Maximum number of bins to use for histogram-based splitting.
            Default = 256.

        block_size (int, optional): Block size for internal parallelism. Default = 256.

        category_split_strategy (str, optional): Strategy to use for splitting categorical features.
            For a given categorical feature, 'one-vs-all' tries to split each categorical value
            from all the others while 'ordered' will try to split the smaller categorical from the
            bigger ones. Can be set to "one-vs-all" or "ordered". Default = "ordered".

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(
            self,
            n_trees=100,
            criterion='gini',
            seed=-1,
            max_depth=29,
            min_samples_split=2,
            bootstrap=True,
            features_selection='sqrt',
            max_features=0,
            proportion_features=0.1,
            samples_factor=1.0,
            min_impurity_decrease=0.0,
            min_split_score=0.0,
            feat_thresh=1.0e-06,
            histogram=False,
            maximum_bins=256,
            block_size=256,
            category_split_strategy="ordered",
            check_data=False):

        self._samples_factor = samples_factor
        self._min_impurity_decrease = min_impurity_decrease
        self._min_split_score = min_split_score
        self._feat_thresh = feat_thresh
        self._order = 'A'
        self._dtype = 'float'

        self._decision_forest_double = pybind_decision_forest(
            n_trees=n_trees,
            criterion=criterion,
            seed=seed,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            features_selection=features_selection,
            max_features=max_features,
            precision="double",
            histogram=histogram,
            maximum_bins=maximum_bins,
            block_size=block_size,
            category_split_strategy=category_split_strategy,
            check_data=check_data)
        self._decision_forest_single = pybind_decision_forest(
            n_trees=n_trees,
            criterion=criterion,
            seed=seed,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            features_selection=features_selection,
            max_features=max_features,
            precision="single",
            histogram=histogram,
            maximum_bins=maximum_bins,
            block_size=block_size,
            category_split_strategy=category_split_strategy,
            check_data=check_data)

        self._decision_forest = self._decision_forest_double
        self._max_features = max_features
        self._features_selection = features_selection
        self._proportion_features = proportion_features

    @property
    def max_features(self):
        return self._max_features

    @max_features.setter
    def max_features(self, value):
        self._decision_forest.set_max_features_opt(max_features=value)

    @property
    def features_selection(self):
        return self._features_selection

    @features_selection.setter
    def features_selection(self, value):
        self._decision_forest.set_features_selection_opt(
            features_selection=value)

    @property
    def samples_factor(self):
        return self._samples_factor

    @samples_factor.setter
    def samples_factor(self, value):
        self._samples_factor = value

    def fit(self, X, y, categorical_features=None):
        """
        Computes the decision forest on the feature matrix ``X`` and response vector ``y``

        Args:
            X (array-like): The feature matrix on which to compute the model.
                Its shape is (n_samples, n_features).

            y (array-like): The response vector. Its shape is (n_samples).

            categorical_features (array-like, optional): Integer vector. categorical_features[i]
                should be set to a negative value if feature i is continuous or to the number of
                different categories if feature i if it is categorical. If None, all features are
                considered continuous. Its shape is (n_features).

        Returns:
            self (object): Returns the instance itself.
        """
        X, self._order, self._dtype = check_convert_data(
            X, order=self._order, dtype=self._dtype, force_dtype=True
        )
        y, _, _ = check_convert_data(
            y, order=self._order, dtype="da_int", force_dtype=True
        )
        if categorical_features is not None:
            categorical_features, _, _ = check_convert_data(
                categorical_features, order=self._order, dtype="da_int", force_dtype=True
            )

        if self._dtype == "float32":
            self._decision_forest = self._decision_forest_single
            self._decision_forest_double = None
        return self._decision_forest.pybind_fit(
            X,
            y,
            self._samples_factor,
            self._min_impurity_decrease,
            self._min_split_score,
            self._feat_thresh,
            self._proportion_features,
            categorical_features)

    def score(self, X, y):
        """
        Calculates score (prediction accuracy) by comparing predicted labels and actual
        labels on a new set of data.

        Args:
            X (array-like): The feature matrix to evaluate the model on.
                It must have n_features columns.

            y (array-like): The response vector.  It must have shape (n_samples).

        Returns:
            float: The mean accuracy of the model on the test data.
        """
        X, _, _ = check_convert_data(
            X, order=self._order, dtype=self._dtype, force_dtype=True
        )
        y, _, _ = check_convert_data(
            y, order=self._order, dtype="da_int", force_dtype=True
        )

        return self._decision_forest.pybind_score(X, y)

    def predict(self, X):
        """
        Generate labels using fitted decision forest on a new set of data ``X``.

        Args:
            X (array-like): The feature matrix to evaluate the model on.
                It must have n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector,
                where n_samples is the number of rows of X.
        """
        X, _, _ = check_convert_data(
            X, order=self._order, dtype=self._dtype, force_dtype=True
        )

        return self._decision_forest.pybind_predict(X)

    def predict_proba(self, X):
        """
        Generate class probabilities using fitted decision forest on a new set of data ``X``.

        Args:
            X (array-like): The feature matrix to evaluate the model on.
                It must have n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector,
                where n_samples is the number of rows of X.
        """
        X, _, _ = check_convert_data(
            X, order=self._order, dtype=self._dtype, force_dtype=True
        )

        return self._decision_forest.pybind_predict_proba(X)

    def predict_log_proba(self, X):
        """
        Generate class log probabilities using fitted decision forest on a new set of data ``X``.

        Args:
            X (array-like): The feature matrix to evaluate the model on.
                It must have n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector,
                where n_samples is the number of rows of X.
        """
        X, _, _ = check_convert_data(
            X, order=self._order, dtype=self._dtype, force_dtype=True
        )

        return self._decision_forest.pybind_predict_log_proba(X)
