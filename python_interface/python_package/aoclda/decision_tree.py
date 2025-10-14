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
# too-many-positional-arguments, too-many-instance-attributes

"""
aoclda.decision_tree module
"""

from ._aoclda.decision_tree import pybind_decision_tree
from ._internal_utils import check_convert_data, get_int_info


class decision_tree():
    """
    A decision tree classifier.

    Args:
        max_depth (int, optional): Set the maximum depth of the tree. Default = 29.

        seed (int, optional): Set the random seed for the random number generator.
            If the value is -1, a random seed is automatically generated. In this case
            the resulting classification will create non-reproducible results.
            Default = -1.

        max_features (int, optional): Set the number of features to consider when
            splitting a node. 0 means take all the features. Default 0.

        criterion (str, optional): Select scoring function to use. It can take the values
            'cross-entropy', 'gini', or 'misclassification'

        min_samples_split (int, optional): The minimum number of samples required to
            split an internal node. Default = 2.

        build_order (str, optional): Select in which order to explore the nodes. It can
            take the values 'breadth first' or 'depth first'. Default 'breadth first'.

        min_impurity_decrease (float, optional): Minimum score improvement needed to consider a
            split from the parent node. Default = 0.0

        min_split_score (float, optional): Minimum score needed for a node to be considered for
            splitting. Default 0.0.

        feat_thresh (float, optional): Minimum difference in feature value required for splitting.
            Default = 1.0e-05

        precision (str, optional): Whether to initialize the decision_tree object in double or
            single precision. It can take the values 'single' or 'double'.
            Default = 'double'.

        detect_categorical_data (bool, optional): Whether to check which features are categorical
            in X. Default = False.

        max_category (int, optional): Maximum number of categories for a given feature to be
            considered categorical. Default = 50.

        category_tolerance (float, optional): How far data can be from an integer to be considered
            not categorical. Default = 1.0e-05

        category_split_strategy (str, optional): The strategy to use for splitting categorical features.
            It can take the values 'ordered' or 'one-vs-all'. Default = 'ordered'.

        histogram (bool, optional): Whether to use histograms for continuous features.
            Default = False.

        maximum_bins (int, optional): Maximum number of bins to use for histograms.
            Default = 256.

        predict_proba (bool, optional): Whether to predict class probabilities.
            Default = True.

        check_data (bool, optional): Whether to check the data for NaNs. Default = False.
    """

    def __init__(
            self,
            criterion='gini',
            seed=-1,
            max_depth=29,
            min_samples_split=2,
            max_features=0,
            min_impurity_decrease=0.0,
            min_split_score=0.0,
            feat_thresh=1.0e-05,
            detect_categorical_data=False,
            max_category=50,
            category_tolerance=1.0e-05,
            category_split_strategy="ordered",
            histogram=False,
            maximum_bins=256,
            predict_proba=True,
            check_data=False):

        self.decision_tree_double = pybind_decision_tree(
            criterion=criterion,
            seed=seed,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            precision="double",
            detect_categorical_data=detect_categorical_data,
            max_category=max_category,
            category_split_strategy=category_split_strategy,
            histogram=histogram,
            maximum_bins=maximum_bins,
            predict_proba=predict_proba,
            check_data=check_data)
        self.decision_tree_single = pybind_decision_tree(
            criterion=criterion,
            seed=seed,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            precision="single",
            detect_categorical_data=detect_categorical_data,
            max_category=max_category,
            category_split_strategy=category_split_strategy,
            histogram=histogram,
            maximum_bins=maximum_bins,
            predict_proba=predict_proba,
            check_data=check_data)
        self.decision_tree = self.decision_tree_double

        self.order = 'A'
        self.dtype = 'float'
        self.category_tolerance = category_tolerance
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_split_score = min_split_score
        self.feat_thresh = feat_thresh
        self.model_info = None

    @property
    def max_features(self):
        """Get the maximum number of features to consider when splitting a node."""
        return self.max_features

    @max_features.setter
    def max_features(self, value):
        self.decision_tree.set_max_features_opt(max_features=value)

    def fit(self, X, y, categorical_features=None):
        """
        Computes the decision tree on the feature matrix ``X`` and response vector ``y``

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
        X, self.order, self.dtype = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        y, _, _ = check_convert_data(
            y, order=self.order, dtype='da_int', force_dtype=True
        )
        if categorical_features is not None:
            categorical_features, _, _ = check_convert_data(
                categorical_features, order=self.order, dtype='da_int', force_dtype=True
            )
        if self.dtype == "float32":
            self.decision_tree = self.decision_tree_single
            self.decision_tree_double = None
        self.model_info = None
        return self.decision_tree.pybind_fit(X, y,
                                             self.min_impurity_decrease,
                                             self.min_split_score,
                                             self.feat_thresh,
                                             self.category_tolerance,
                                             categorical_features)

    def score(self, X, y):
        """
        Calculates score (prediction accuracy) by comparing predicted labels and actual
        labels on a new set of data.

        Args:
            X (array-like): The feature matrix to evaluate the model on. It must have
                n_features columns.

            y (array-like): The response vector.  It must have shape (n_samples).

        Returns:
            float: The mean accuracy of the model on the test data.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )
        y, _, _ = check_convert_data(
            y, order=self.order, dtype='da_int', force_dtype=True
        )

        return self.decision_tree.pybind_score(X, y)

    def predict(self, X):
        """
        Generate labels using fitted decision forest on a new set of data ``X``.

        Args:
            X (array-like): The feature matrix to evaluate the model on. It must have
            n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector, where n_samples is
            the number of rows of ``X``.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )

        return self.decision_tree.pybind_predict(X)

    def predict_proba(self, X):
        """
        Generate class probabilities using fitted decision forest on a new set of data ``X``.

        Args:
            X (array-like): The feature matrix to evaluate the model on. It must have
            n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector, where n_samples is
            the number of rows of ``X``.
        """
        X, _, _ = check_convert_data(
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )

        return self.decision_tree.pybind_predict_proba(X)

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
            X, order=self.order, dtype=self.dtype, force_dtype=True
        )

        return self.decision_tree.pybind_predict_log_proba(X)

    def update_model_info(self):
        """
        update the model information (content of rinfo from the C interface)
        """
        self.model_info = self.decision_tree.pybind_get_model_info()

    @property
    def n_samples(self):
        """int: The number of samples used in the trained tree"""
        if self.model_info is None:
            self.update_model_info()
        return self.model_info["n_samples"]

    @property
    def n_obs(self):
        """int: The number of observations used to the trained tree"""
        if self.model_info is None:
            self.update_model_info()
        return self.model_info["n_obs"]

    @property
    def n_features(self):
        """int: The number of features used in the trained tree"""
        if self.model_info is None:
            self.update_model_info()
        return self.model_info["n_features"]

    @property
    def depth(self):
        """int: The depth of the trained tree"""
        if self.model_info is None:
            self.update_model_info()
        return self.model_info["depth"]

    @property
    def n_nodes(self):
        """int: The number of nodes in the trained tree"""
        if self.model_info is None:
            self.update_model_info()
        return self.model_info["n_nodes"]

    @property
    def n_leaves(self):
        """int: The number of leaves in the trained tree"""
        if self.model_info is None:
            self.update_model_info()
        return self.model_info["n_leaves"]
