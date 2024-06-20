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
aoclda.decision_tree module
"""

from ._aoclda.decision_tree import pybind_decision_tree

class decision_tree(pybind_decision_tree):
    """
    A decision tree classifier.

    Args:
        seed (int, optional): Set the random seed for the random number generator.
            If the value is -1, a random seed is automatically generated. In this case
            the resulting classification will create non-reproducible results.
            Default = -1.

        max_depth (int, optional): Set the maximum depth of the tree. Default = 10.

        max_features (int, optional): Set the number of features to consider when
            splitting a node. 0 means take all the features. Default 0.

        criterion (str, optional): Select scoring function to use. It can take the values
            'cross-entropy', 'gini', or 'misclassification'

        min_samples_split (int, optional): The minimum number of samples required to
            split an internal node. Default 2.

        build_order (str, optional): Select in which order to explore the nodes. It can
            take the values 'breadth first' or 'depth first'. Default 'breadth first'.

        precision (str, optional): Whether to initialize the PCA object in double or
            single precision. It can take the values 'single' or 'double'.
            Default = 'double'.
    """

    def fit(self, X, y, min_impurity_decrease=0.03, min_split_score=0.03, feat_thresh=1.0e-06):
        """
        Computes the decision tree on the feature matrix ``X`` and response vector ``y``

        Args:
            X (numpy.ndarray): The feature matrix on which to compute the model.
                Its shape is (n_samples, n_features).

            y (numpy.ndarray): The response vector. Its shape is (n_samples).

            min_impurity_decrease (float, optional): Minimum score improvement
                needed to consider a split from the parent node. Default 0.03

            min_split_score (float, optional): Minimum score needed for a node
                to be considered for splitting. Default 0.03.

            feat_thresh (float, optional): Minimum difference in feature value
                required for splitting. Default 1.0e-06

        Returns:
            self (object): Returns the instance itself.
        """
        return self.pybind_fit(X, y, min_impurity_decrease, min_split_score, feat_thresh)

    def score(self, X, y):
        """
        Calculates score (prediction accuracy) by comparing predicted labels and actual
        labels on a new set of data.

        Args:
            X (numpy.ndarray): The feature matrix to evaluate the model on. It must have
                n_features columns.

            y (numpy.ndarray): The response vector.  It must have shape (n_samples).

        Returns:
            float: The mean accuracy of the model on the test data.
        """
        return self.pybind_score(X, y)

    def predict(self, X):
        """
        Generate labels using fitted decision forest on a new set of data ``X``.

        Args:
            X (numpy.ndarray): The feature matrix to evaluate the model on. It must have
            n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector, where n_samples is
            the number of rows of ``X``.
        """
        return self.pybind_predict(X)
