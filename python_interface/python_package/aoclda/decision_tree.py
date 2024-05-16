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
aoclda.decision_tree module
"""

from ._aoclda.decision_tree import pybind_decision_tree

class decision_tree(pybind_decision_tree):
    """
    Decision Trees

    Args:
        precision (str, optional): Whether to initialize the PCA object in double or
            single precision. It can take the values 'single' or 'double'.
            Default = 'double'.
    """
    def fit(self, X, y):
        """
        Computes the decision tree on the feature matrix X and response vector y

        Args:
            X (numpy.ndarray): The feature matrix on which to compute the model.
                Its shape is (n_samples, n_features).

            y (numpy.ndarray): The response vector. Its shape is (n_samples).
        """
        return self.pybind_fit(X, y)
    def score(self, X, y):
        """
        Calculates score (prediction accuracy) by comparing predicted labels and actual
        labels on a new set of data.

        Args:
            X (numpy.ndarray): The feature matrix to evaluate the model on. It must have
            n_features columns.

            y (numpy.ndarray): The response vector.  It must have shape (n_samples).
        """
        return self.pybind_score(X, y)

    def predict(self, X):
        """
        Generate labels using fitted decision forest on a new set of data X.
                Args:
            X (numpy.ndarray): The feature matrix to evaluate the model on. It must have
            n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector, where n_samples is the number of
            rows of X.
        """
        return self.pybind_predict(X)
