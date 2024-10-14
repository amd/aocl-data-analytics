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
Decision forest tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member

import numpy as np
import pytest
import warnings
from aoclda.sklearn import skpatch, undo_skpatch

@pytest.mark.parametrize("precision", [np.float64,  np.float32])
@pytest.mark.parametrize("bootstrap", [True, False])
def test_decision_forest(precision, bootstrap):
    """
    Basic problem with 2 observations and 2 features
    """

    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=precision)
    Y = np.array([0, 1], dtype=precision)
    Xp = np.array([[2., 2.]], dtype=precision)

    tol = np.sqrt(np.finfo(precision).eps)

    # patch and import scikit-learn
    skpatch()
    from sklearn import ensemble
    clf = ensemble.RandomForestClassifier(n_estimators=4000, bootstrap=bootstrap)
    clf = clf.fit(X, Y)
    da_yp = clf.predict( Xp )
    da_yprob = clf.predict_proba( Xp )
    with warnings.catch_warnings(record=True):
        da_ylogprob = clf.predict_log_proba( Xp )
    assert clf.aocl is True

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    from sklearn import ensemble
    clf = ensemble.RandomForestClassifier(n_estimators=4000, bootstrap=bootstrap)
    clf = clf.fit(X, Y)
    yp = clf.predict( Xp )
    yprob = clf.predict_proba( Xp )
    with warnings.catch_warnings(record=True):
        ylogprob = clf.predict_log_proba( Xp )
    assert not hasattr(clf, 'aocl')

    # Check results
    assert da_yp == yp
    assert da_yprob == pytest.approx(yprob, abs=0.05)
    assert da_ylogprob == pytest.approx(ylogprob, abs=0.1)

    # print the results if pytest is invoked with the -rA option
    print("Predictions")
    print("    aoclda: \n", da_yp)
    print("   sklearn: \n", yp)

    print("Probabilities")
    print("    aoclda: \n", da_yprob[0,0], ", ", da_yprob[0,1])
    print("   sklearn: \n", yprob[0,0], ", ", yprob[0,1])

@pytest.mark.parametrize("precision", [np.float64,  np.float32])
def test_double_solve(precision):
    """"
    Check that solving the model twice doesn't fail
    """
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=precision)
    Y = np.array([0, 1], dtype=precision)

    # patch and import scikit-learn
    skpatch()
    from sklearn import ensemble
    clf = ensemble.RandomForestClassifier()
    clf = clf.fit(X, Y)
    clf.fit(X,Y)
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
        clf = ensemble.RandomForestClassifier(random_state = np.random.RandomState() )

    with pytest.warns(RuntimeWarning):
        clf = ensemble.RandomForestClassifier(min_samples_leaf = 10)

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

if __name__ == "__main__":
    test_decision_forest(precision=np.float64, bootstrap=True)
    test_decision_forest_errors()
