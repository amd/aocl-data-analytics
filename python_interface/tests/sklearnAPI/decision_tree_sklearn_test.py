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
Decision tree tests, check output of skpatch versus sklearn
"""

# autopep8: off
# pylint: disable = import-outside-toplevel, reimported, no-member, redefined-outer-name, too-many-locals, unexpected-keyword-arg, unexpected-keyword-arg
# autopep8: on
import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
def test_decision_tree(numpy_precision):
    """
    Basic problem with 2 observations and 2 features
    """

    X_train = [[1., 1., 1., 0., 0., 0.],
               [0., 1., 2., 0., 1., 2.]]
    y_train = [0, 1, 1, 0, 0, 1]
    X_train = np.array(X_train, dtype=numpy_precision).transpose()
    y_train = np.array(y_train, dtype=numpy_precision)
    X_test = [[0., 1.],
              [1., 2.]]
    y_test = [0, 1]
    X_test = np.array(X_test, dtype=numpy_precision).transpose()
    y_test = np.array(y_test, dtype=numpy_precision)

    tol = np.sqrt(np.finfo(numpy_precision).eps)

    # patch and import scikit-learn
    skpatch()
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=1)
    clf = clf.fit(X_train, y_train)
    da_yp = clf.predict(X_test)
    da_yprob = clf.predict_proba(X_test)
    da_n_leaves = clf.get_n_leaves()
    assert clf.aocl is True

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    yp = clf.predict(X_test)
    yprob = clf.predict_proba(X_test)
    n_leaves = clf.get_n_leaves()
    assert not hasattr(clf, 'aocl')

    # Check results
    assert da_yp.all() == yp.all()
    assert da_yprob.all() == pytest.approx(yprob.all(), tol)
    assert da_n_leaves == n_leaves

    # print the results if pytest is invoked with the -rA option
    print("Predictions, Leaves")
    print("    aoclda: \n", da_yp, "\n", da_n_leaves)
    print("   sklearn: \n", yp, "\n", n_leaves)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_double_solve(precision):
    """"
    Check that solving the model twice doesn't fail
    """
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=precision)
    Y = np.array([0, 1], dtype=precision)

    # patch and import scikit-learn
    skpatch()
    from sklearn import tree
    # Fit a tree twice, shouldn't raise an error
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    clf.fit(X, Y)
    assert clf.aocl is True


def test_decision_tree_errors():
    '''
    Check we can catch errors in the sklearn decision_tree patch
    '''

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    Y = np.array([0, 1])

    skpatch()
    from sklearn import tree
    with pytest.raises(ValueError):
        clf = tree.DecisionTreeClassifier(splitter='random')

    with pytest.raises(ValueError):
        clf = tree.DecisionTreeClassifier(random_state=np.random.RandomState())

    with pytest.warns(RuntimeWarning):
        clf = tree.DecisionTreeClassifier(min_samples_leaf=10)

    clf = clf.fit(X, Y)

    with pytest.raises(RuntimeError):
        clf.apply(1)
    with pytest.raises(RuntimeError):
        clf.cost_complexity_pruning_path(1, 1)
    with pytest.raises(RuntimeError):
        clf.get_metadata_routing()
    with pytest.raises(RuntimeError):
        clf.set_fit_request()
    with pytest.raises(RuntimeError):
        clf.set_params()
    with pytest.raises(RuntimeError):
        clf.set_predict_proba_request()
    with pytest.raises(RuntimeError):
        clf.set_predict_request()
    with pytest.raises(RuntimeError):
        clf.set_score_request()

    assert clf.feature_importances_ is None


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
    tol = np.sqrt(np.finfo(numpy_precision).eps)

    # Solve first without binning
    skpatch()
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(detect_categorical_data=True)
    clf.fit(X_train, y_train)
    da_yp = clf.predict(X_test)
    da_yprob = clf.predict_proba(X_test)
    da_n_leaves = clf.get_n_leaves()
    assert clf.aocl is True
    # same with vanilla
    undo_skpatch()
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print(clf.get_params())
    sk_yp = clf.predict(X_test)
    sk_yprob = clf.predict_proba(X_test)
    sk_n_leaves = clf.get_n_leaves()
    assert da_yp.any() == sk_yp.any()
    assert da_yprob == pytest.approx(sk_yprob, tol)
    assert da_n_leaves == sk_n_leaves

    # Solve again, binning the data
    skpatch()
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(histogram=True)
    clf.fit(X_train, y_train)
    da_yp = clf.predict(X_test)
    da_yprob = clf.predict_proba(X_test)
    da_n_leaves = clf.get_n_leaves()
    assert clf.aocl is True
    # same with vanilla
    assert da_yp.any() == sk_yp.any()
    assert da_yprob == pytest.approx(sk_yprob, tol)
    assert da_n_leaves == sk_n_leaves
    # Check that histogram parameter fails for Vanilla
    undo_skpatch()
    from sklearn import tree
    with pytest.raises(TypeError):
        clf = tree.DecisionTreeClassifier(histogram=True)

    # Too few bins
    skpatch()
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(histogram=True, maximum_bins=2)
    clf.fit(X_train, y_train)
    assert clf.aocl is True
    score = clf.score(X_test, y_test)
    assert 0.5 < score < 0.9


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
    skpatch()
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(max_depth=1)
    clf.fit(X_train, y_train)
    assert clf.get_depth() == 1
    assert clf.get_n_leaves() == 2
    # min_impurity_decrease
    clf = tree.DecisionTreeClassifier(min_impurity_decrease=0.99)
    clf.fit(X_train, y_train)
    assert clf.get_depth() == 0
    assert clf.get_n_leaves() == 1
    # min_split_score
    clf = tree.DecisionTreeClassifier(min_samples_split=100)
    clf.fit(X_train, y_train)
    assert clf.get_depth() == 0
    assert clf.get_n_leaves() == 1
