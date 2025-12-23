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
from aoclda.decision_forest import decision_forest
from aoclda._internal_utils import get_int_info

int_type = "int" + get_int_info()


@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, 'object'])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_df_all_dtypes(numpy_precision, numpy_order):
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

    clf = decision_forest(n_trees=1, seed=42, bootstrap=False,
                          features_selection="all")
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    pred = clf.predict(X_test)
    assert pred.all() == y_test.all()
    assert np.abs(score - 1.0) < 1.0e-04

    categorical_features = np.array(
        [2, 3], dtype=int_type, order=numpy_order)
    forest = decision_forest(min_samples_split=1)
    forest.fit(X_train, y_train, categorical_features=categorical_features)
    score = forest.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04


int_type = "int" + get_int_info()


@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders",
                         [("C", "F"), ("F", "C")])
def test_decision_forest_multiple_orders(numpy_precision, numpy_orders):
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

    clf = decision_forest(seed=42)
    clf.fit(X_train, y_train)

    with pytest.warns(UserWarning):
        pred = clf.predict(X_test)
    assert pred.all() == y_test.all()
    with pytest.warns(UserWarning):
        score = clf.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04

    categorical_features = np.array(
        [2, 3], dtype=int_type, order=numpy_orders[1])
    forest = decision_forest(min_samples_split=2)
    forest.fit(X_train, y_train, categorical_features=categorical_features)
    with pytest.warns(UserWarning):
        score = forest.score(X_test, y_test)

    X_train = np.array(X_train, order=numpy_orders[1])
    with pytest.warns(UserWarning):
        forest.fit(X_train, y_train)


@pytest.mark.parametrize(
    "numpy_precisions", [('float32', 'float64'),
                         ('float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_decision_forest_multiple_dtypes(numpy_precisions, numpy_order):
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

    clf = decision_forest()
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    assert pred.all() == y_test.all()
    assert np.abs(score - 1.0) < 1.0e-04

    categorical_features = np.array(
        [2, 3], dtype=numpy_precisions[1], order=numpy_order)
    forest = decision_forest(min_samples_split=2)
    forest.fit(X_train, y_train, categorical_features=categorical_features)
    score = forest.score(X_test, y_test)

    X_train = np.array(X_train, dtype=numpy_precisions[1], order=numpy_order)
    clf.fit(X_train, y_train)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_decision_forest_functionality(numpy_precision, numpy_order):
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

    clf = decision_forest(min_samples_split=1)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    assert pred[0] == 1
    assert score == 1.0

    print(f"predictions: [{pred[0]:d}]")
    print(f"score: {score:.3f}")


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_categorical_features(numpy_precision, numpy_order):
    """
    Test decision tree with categorical features
    """

    X_train = np.array([[1, 0], [1, 1], [1, 2], [0, 0], [0, 1], [0, 2]],
                       dtype=numpy_precision, order=numpy_order)
    y_train = np.array(
        [0, 1, 1, 0, 0, 1],
        dtype=numpy_precision, order=numpy_order)
    categorical_features = np.array(
        [2, 3], dtype=int_type, order=numpy_order)
    forest = decision_forest(seed=42, bootstrap=False, n_trees=10,
                             features_selection='all', min_samples_split=2)
    forest.fit(X_train, y_train, categorical_features=categorical_features)

    X_test = np.array([[0, 1], [1, 2]],
                      dtype=numpy_precision, order=numpy_order)
    y_test = np.array([0, 1])
    score = forest.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04

    # one-vs-all
    forest = decision_forest(
        seed=42,
        bootstrap=False,
        n_trees=10,
        features_selection='all',
        min_samples_split=2,
        category_split_strategy='one-vs-all')
    forest.fit(X_train, y_train, categorical_features=categorical_features)


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
    forest = decision_forest(
        seed=42, features_selection='all', n_trees=20)
    forest.fit(X_train, y_train)
    score = forest.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04

    # Solve again, binning the data
    forest = decision_forest(
        seed=42, features_selection='all', histogram=True, n_trees=20)
    forest.fit(X_train, y_train)
    score = forest.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04

    # Too few bins
    forest = decision_forest(
        seed=42, features_selection='all', histogram=True, maximum_bins=2, n_trees=20)
    forest.fit(X_train, y_train)
    score = forest.score(X_test, y_test)
    assert score > 0.5 and score < 0.9
