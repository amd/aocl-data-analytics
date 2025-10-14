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
Decision forest tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member, unexpected-keyword-arg

import warnings
import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_decision_forest(precision):
    """
    Basic problem with 2 observations and 2 features
    """

    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=precision)
    Y = np.array([0, 1], dtype=precision)
    Xp = np.array([[2., 2.]], dtype=precision)

    # patch and import scikit-learn
    skpatch()
    from sklearn import ensemble
    clf = ensemble.RandomForestClassifier(n_estimators=4,
                                          bootstrap=False,
                                          random_state=0, histogram=False)
    clf = clf.fit(X, Y)
    da_yp = clf.predict(Xp)
    da_yprob = clf.predict_proba(Xp)
    with warnings.catch_warnings(record=True):
        da_ylogprob = clf.predict_log_proba(Xp)
    assert clf.aocl is True
    print(da_yp)

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    from sklearn import ensemble
    clf = ensemble.RandomForestClassifier(n_estimators=4,
                                          bootstrap=False,
                                          random_state=0)
    clf = clf.fit(X, Y)
    yp = clf.predict(Xp)
    yprob = clf.predict_proba(Xp)
    with warnings.catch_warnings(record=True):
        ylogprob = clf.predict_log_proba(Xp)
    assert not hasattr(clf, 'aocl')

    # Check results
    assert da_yp.all() == yp.all()
    assert da_yprob == pytest.approx(yprob, abs=0.15)
    assert da_ylogprob == pytest.approx(ylogprob, abs=0.2)

    # print the results if pytest is invoked with the -rA option
    print("Predictions")
    print("    aoclda: \n", da_yp)
    print("   sklearn: \n", yp)

    print("Probabilities")
    print("    aoclda: \n", da_yprob[0, 0], ", ", da_yprob[0, 1])
    print("   sklearn: \n", yprob[0, 0], ", ", yprob[0, 1])


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_double_solve(precision):
    """"
    Check that solving the model twice doesn't fail
    """
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=precision)
    Y = np.array([0, 1], dtype=precision)

    # patch and import scikit-learn
    skpatch()
    from sklearn import ensemble
    clf = ensemble.RandomForestClassifier(histogram=False)
    clf = clf.fit(X, Y)
    clf.fit(X, Y)
    assert clf.aocl is True


def test_decision_forest_errors():
    '''
    Check we can catch errors in the sklearn decision_forest patch
    '''
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    Y = np.array([0, 1])

    skpatch()
    from sklearn import ensemble

    with pytest.raises(ValueError):
        clf = ensemble.RandomForestClassifier(
            random_state=np.random.RandomState())

    with pytest.warns(RuntimeWarning):
        clf = ensemble.RandomForestClassifier(min_samples_leaf=10, histogram=False)

    clf = clf.fit(X, Y)

    with pytest.raises(RuntimeError):
        clf.apply(1)
    with pytest.raises(RuntimeError):
        clf.decision_path(1)
    with pytest.raises(RuntimeError):
        clf.get_metadata_routing()
    with pytest.raises(RuntimeError):
        clf.set_fit_request()
    with pytest.raises(RuntimeError):
        clf.set_params()
    with pytest.raises(RuntimeError):
        clf.set_score_request()

    assert clf.estimators_samples_ is None
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

    # Solve first without binning
    skpatch()
    from sklearn import ensemble
    forest = ensemble.RandomForestClassifier(
        random_state=42, max_features=None, n_estimators=20, max_samples=12)
    forest.fit(X_train, y_train)
    score = forest.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04
    assert forest.aocl is True
    undo_skpatch()
    from sklearn import ensemble
    forest = ensemble.RandomForestClassifier(
        random_state=42, max_features=None, n_estimators=20, max_samples=12)
    forest.fit(X_train, y_train)
    score = forest.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04

    # Solve again, binning the data
    skpatch()
    from sklearn import ensemble
    forest = ensemble.RandomForestClassifier(
        random_state=42, max_features=None, n_estimators=20, histogram=True)
    forest.fit(X_train, y_train)
    score = forest.score(X_test, y_test)
    assert np.abs(score - 1.0) < 1.0e-04
    assert forest.aocl is True

    # Too few bins
    forest = ensemble.RandomForestClassifier(
        random_state=42,
        max_features=None,
        n_estimators=20,
        histogram=True,
        maximum_bins=2)
    forest.fit(X_train, y_train)
    score = forest.score(X_test, y_test)
    assert 0.9 > score > 0.5
    assert forest.aocl is True
