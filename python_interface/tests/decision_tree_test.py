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
Decision tree Python test script
"""

import numpy as np
import pytest
from aoclda.decision_tree import decision_tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time
from aoclda._internal_utils import get_int_info

int_type = "int" + get_int_info()


@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, 'object'])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_dt_run_all_dtypes(numpy_precision, numpy_order):
    """
    Test it runs when supported/unsupported C-interface type is provided.
    """
    X_train = [[1., 1., 1., 0., 0., 0.],
               [0., 1., 2., 0., 1., 2.]]
    y_train = [0, 1, 1, 0, 0, 1]
    X_train = np.array(X_train, dtype=numpy_precision,
                       order=numpy_order).transpose()
    y_train = np.array(y_train, dtype=numpy_precision, order=numpy_order)
    X_test = [[0., 1.],
              [1., 2.]]
    y_test = [0, 1]
    X_test = np.array(X_test, dtype=numpy_precision,
                      order=numpy_order).transpose()
    y_test = np.array(y_test, dtype=numpy_precision, order=numpy_order)

    clf = decision_tree()
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    assert pred.all() == y_test.all()
    score = clf.score(X_test, y_test)
    assert abs(score - 1.0) < 1.0e-04


int_type = "int" + get_int_info()


@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders",
                         [("C", "F"), ("F", "C")])
def test_decision_tree_multiple_orders(numpy_precision, numpy_orders):
    """
    Test it runs when arrays of multiple orders are provided.
    """
    X_train = [[1., 0.], [1., 1.], [1., 2.], [0., 0.],
               [0., 1.], [0., 2.]]
    y_train = [0, 1, 1, 0, 0, 1]
    X_train = np.array(X_train, dtype=numpy_precision,
                       order=numpy_orders[0])
    y_train = np.array(y_train, dtype=numpy_precision, order=numpy_orders[1])
    X_test = [[0., 1.],
              [1., 2.]]
    y_test = [0, 1]
    X_test = np.array(X_test, dtype=numpy_precision,
                      order=numpy_orders[1])
    y_test = np.array(y_test, dtype=numpy_precision, order=numpy_orders[1])

    clf = decision_tree()
    clf.fit(X_train, y_train)

    with pytest.warns(UserWarning):
        pred = clf.predict(X_test)
    assert pred.all() == y_test.all()
    with pytest.warns(UserWarning):
        score = clf.score(X_test, y_test)
    assert abs(score - 1.0) < 1.0e-04

    X_train = np.array(X_train, order=numpy_orders[1])
    with pytest.warns(UserWarning):
        clf.fit(X_train, y_train)


@pytest.mark.parametrize(
    "numpy_precisions", [('float32', 'float64'),
                         ('float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_decision_tree_multiple_dtypes(numpy_precisions, numpy_order):
    """
    Test it runs when arrays of multiple dtypes are provided.
    """

    X_train = [[1., 0.], [1., 1.], [1., 2.], [0., 0.],
               [0., 1.], [0., 2.]]
    y_train = [0, 1, 1, 0, 0, 1]
    X_train = np.array(X_train, dtype=numpy_precisions[0],
                       order=numpy_order)
    y_train = np.array(y_train, dtype=numpy_precisions[1], order=numpy_order)
    X_test = [[0., 1.],
              [1., 2.]]
    y_test = [0, 1]
    X_test = np.array(X_test, dtype=numpy_precisions[1],
                      order=numpy_order)
    y_test = np.array(y_test, dtype=numpy_precisions[0], order=numpy_order)

    clf = decision_tree()
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    assert pred.all() == y_test.all()
    score = clf.score(X_test, y_test)
    assert abs(score - 1.0) < 1.0e-04

    X_train = np.array(X_train, dtype=numpy_precisions[1], order=numpy_order)
    clf.fit(X_train, y_train)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_decision_tree_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the Python wrapper
    """

    X_train = np.array([[0.0, 0.0], [1.0, 1.0]],
                       dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 1],
                       dtype=numpy_precision, order=numpy_order)
    X_test = np.array([[2., 2.]],
                      dtype=numpy_precision, order=numpy_order)
    y_test = np.array([[1]],
                      dtype=numpy_precision, order=numpy_order)

    clf = decision_tree(min_samples_split=1)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)

    assert pred[0] == 1
    assert score == 1.0
    assert clf.n_nodes == 3
    assert clf.n_leaves == 2

    print(f"predictions: [{pred[0]:d}]")
    print(f"probabilities: [{proba[0, 0]:.3f}, {proba[0, 1]:.3f}]")
    print(f"score: {score:.3f}")


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("iseed, expected_score", [(42, 0.918), (66, 0.972)])
def test_decision_tree_synthetic(numpy_precision, numpy_order,
                                 iseed, expected_score):
    """
    Synthetic problem with 5,000 observations and 5 features
    """

    X, y = make_classification(
        n_samples=5_000, random_state=iseed, n_features=5)
    X = X.reshape(X.shape, order=numpy_order).astype(numpy_precision)
    y = y.reshape(y.shape, order=numpy_order).astype(numpy_precision)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    seed = 747
    tree_da = decision_tree(seed=seed, max_depth=5,
                            max_features=5)
    start_da_fit = time.time()
    tree_da.fit(X_train, y_train)
    time_da_fit = time.time() - start_da_fit
    score = tree_da.score(X_test, y_test)

    print("DA score: ", score)
    print("expected score: ", expected_score)
    print("Fit time: ", time_da_fit)
    print("Number of leaves: ", tree_da.n_leaves)

    tol = 1.0e-4
    assert np.abs(score - expected_score) < tol


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_categorical_features(numpy_precision, numpy_order):
    """
    Test decision tree with categorical features
    """

    X_train = np.array([[1, 0], [1, 1], [1, 2], [0, 0], [0, 1], [0, 2]],
                       dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 1, 1, 0, 0, 1], dtype=np.int32)
    categorical_features = np.array(
        [2, 3], dtype=int_type, order=numpy_order)

    # Default parameters
    tree = decision_tree(seed=42)
    tree.fit(X_train, y_train, categorical_features=categorical_features)

    X_test = np.array([[0, 1], [1, 2]],
                      dtype=numpy_precision, order=numpy_order)
    y_test = np.array([0, 1])
    score = tree.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04

    # one vs all split strategy
    tree = decision_tree(seed=42, category_split_strategy="one-vs-all")
    tree.fit(X_train, y_train, categorical_features=categorical_features)
    score = tree.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04

    # automatic detection of categorical data
    tree = decision_tree(seed=42, detect_categorical_data=True)
    tree.fit(X_train, y_train)
    score = tree.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04
    # one vs all
    tree = decision_tree(seed=42, detect_categorical_data=True,
                         category_split_strategy="one-vs-all")
    tree.fit(X_train, y_train)
    score = tree.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04


def test_getters_errors():
    tree = decision_tree(seed=42)
    with pytest.raises(RuntimeError):
        tree.depth
    with pytest.raises(RuntimeError):
        tree.n_leaves
    with pytest.raises(RuntimeError):
        tree.n_nodes
    with pytest.raises(RuntimeError):
        tree.n_samples
    with pytest.raises(RuntimeError):
        tree.n_features


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_split_options(numpy_precision, numpy_order):
    """ Check that the split options correctly stop the splitting at the root node"""
    # autopep8: off
    X_train = np.array([[0,   1,   2,  3, 4, 0.5, 1.5, 2.5, 3.5, 4.5, 6,  7, 8,   9,   5.5],
                        [4, 3,   2,   0,  1, 6.5, 5.5, 7.5, 8.5, 6.,  1.,  2., 4, 3,   2]],
                        dtype=numpy_precision, order=numpy_order).transpose()
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)
    # autopep8: on

    # try different options to force splitting to stop at the root node
    # max depth
    tree = decision_tree(seed=42, max_depth=1)
    tree.fit(X_train, y_train)
    assert tree.depth == 1
    assert tree.n_leaves == 2
    # min_impurity_decrease
    tree = decision_tree(seed=42, min_impurity_decrease=0.99)
    tree.fit(X_train, y_train)
    print(tree.depth)
    assert tree.depth == 0
    assert tree.n_leaves == 1
    # min_split_score
    tree = decision_tree(seed=42, min_split_score=0.99)
    tree.fit(X_train, y_train)
    assert tree.depth == 1
    assert tree.n_leaves == 2


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_small_histograms(numpy_precision, numpy_order):
    """
    Test small problem using histograms
    """
    # autopep8: off
    X_train = np.array([[0,   1,   2,  3, 4, 0.5, 1.5, 2.5, 3.5, 4.5, 6,  7, 8,   9,   5.5],
                        [4, 3,   2,   0,  1, 6.5, 5.5, 7.5, 8.5, 6.,  1.,  2., 4, 3,   2]],
                        dtype=numpy_precision, order=numpy_order).transpose()
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)
    X_test = np.array([[0, 1, 2, 2, 3, 4, 6,   7,   8],
                       [3, 4, 1, 7, 8, 9, 2.5, 3.5, 4.2]],
                       dtype=numpy_precision, order=numpy_order).transpose()
    y_test = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    # autopep8: on

    # Solve first without binning
    tree = decision_tree(seed=42, detect_categorical_data=True)
    tree.fit(X_train, y_train)
    score = tree.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04

    # Solve again, binning the data
    tree = decision_tree(seed=42, histogram=True)
    tree.fit(X_train, y_train)
    score = tree.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04

    # Too few bins
    tree = decision_tree(seed=42, histogram=True, maximum_bins=2)
    tree.fit(X_train, y_train)
    score = tree.score(X_test, y_test)
    assert score > 0.5 and score < 0.9
