# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
SVM tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member

import numpy as np
import pytest
import importlib
from aoclda.sklearn import skpatch, undo_skpatch

# Classification dataset
X_train_class = np.array([[1.12, -0.92], [0.7, -2.38], [-0.84, -1.52], [-1.07, -1.44], [-0.68, -1.25], [-0.8, -1.37],
                         [-1.09, 1.15], [1.53, -0.25], [-2.5, 0.16], [-3.08, 2.3], [0.15, -2.21], [-1.2, -0.77], [-2.03, 1.02], [-1.63, 1.9]])
X_test_class = np.array(
    [[1.13, -0.98], [-1.53, 1.16], [-0.32, 1.5], [1.03, -1.11], [-1.82, -0.59], [1.04, -0.73]])
y_train_class = np.array([2, 0, 0, 0, 0, 0, 1, 2, 1, 1, 2, 0, 1, 1])
y_test_class = np.array([2, 1, 1, 2, 0, 2])

# Regression dataset
X_train_reg = np.array([[-0.33, 0.44], [1.86, 1.15], [-0.33, 0.19], [-0.11, 0.0], [
                       0.71, 1.0], [0.6, -0.36], [-0.2, -1.19], [-0.25, 2.43], [-0.86, -0.98], [-0.36, 0.06]])
X_test_reg = np.array([[-0.7, -1.66], [0.64, -1.51],
                      [-0.59, -0.91], [-0.42, -0.87], [1.58, 0.11]])
y_train_reg = np.array([-4.38, 256.73, -20.51, -10.23,
                       133.97, 36.52, -96.51, 130.67, -147.19, -32.02])
y_test_reg = np.array([-175.52, -33.13, -116.49, -97.43, 162.99])


@pytest.mark.parametrize("svm_problem, X_train, X_test, y_train, y_test", [
    ("SVC", X_train_class, X_test_class, y_train_class, y_test_class),
    ("SVR", X_train_reg, X_test_reg, y_train_reg, y_test_reg),
    ("NuSVC", X_train_class, X_test_class, y_train_class, y_test_class),
    ("NuSVR", X_train_reg, X_test_reg, y_train_reg, y_test_reg)
], ids=["SVC", "SVR", "NuSVC", "NuSVR"])
@pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "sigmoid"])
@pytest.mark.parametrize("gamma", [1, "scale", "auto"])
@pytest.mark.parametrize("precision", [np.float64,  np.float32])
def test_svm(svm_problem, kernel, gamma, precision, X_train, X_test, y_train, y_test):
    """
    Basic datasets defined above
    """
    X_train = X_train.astype(precision)
    y_train = y_train.astype(precision)
    X_test = X_test.astype(precision)
    y_test = y_test.astype(precision)

    tol = 1e-4

    # patch and import scikit-learn
    skpatch()
    SVM_module = importlib.import_module('sklearn.svm')
    SVM_model = getattr(SVM_module, svm_problem)
    svm_da = SVM_model(kernel=kernel, gamma=gamma, tol=1e-6)
    svm_da.fit(X_train, y_train)
    da_pred = svm_da.predict(X_test)
    da_score = svm_da.score(X_test, y_test)
    da_params = svm_da.get_params()
    if svm_problem == "SVC" or svm_problem == "NuSVC":
        da_dec_f = svm_da.decision_function(X_test)
    da_dual_coef = svm_da.dual_coef_
    da_intercept = svm_da.intercept_
    da_support = svm_da.support_
    da_support_vectors = svm_da.support_vectors_
    da_n_support = svm_da.n_support_
    da_n_features_in_ = svm_da.n_features_in_
    assert svm_da.aocl is True

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    SVM_module = importlib.import_module('sklearn.svm')
    SVM_model = getattr(SVM_module, svm_problem)
    # Shrinking argument is here to allow correct get_params() comparison
    svm_sk = SVM_model(kernel=kernel, gamma=gamma, tol=1e-6, shrinking=False)
    svm_sk.fit(X_train, y_train)
    sk_pred = svm_sk.predict(X_test)
    sk_score = svm_sk.score(X_test, y_test)
    sk_params = svm_sk.get_params()
    if svm_problem == "SVC" or svm_problem == "NuSVC":
        sk_dec_f = svm_sk.decision_function(X_test)
    sk_dual_coef = svm_sk.dual_coef_
    sk_intercept = svm_sk.intercept_
    sk_support = svm_sk.support_
    sk_support_vectors = svm_sk.support_vectors_
    sk_n_support = svm_sk.n_support_
    sk_n_features_in_ = svm_sk.n_features_in_
    assert not hasattr(svm_sk, 'aocl')

    # Check results
    assert da_pred == pytest.approx(sk_pred, tol)
    assert da_score == pytest.approx(sk_score, tol)
    if svm_problem == "SVC" or svm_problem == "NuSVC":
        assert da_dec_f == pytest.approx(sk_dec_f, 0.02)
    assert da_dual_coef == pytest.approx(sk_dual_coef, 0.03)
    tol_intercept = 4e-4
    if svm_problem == "NuSVC" and precision == np.float32 and kernel == "rbf":
        tol_intercept = 0.34  # This specific case has some discrepancy on the intercept
    assert da_intercept == pytest.approx(sk_intercept, tol_intercept)
    assert da_support == pytest.approx(sk_support, tol)
    assert da_support_vectors == pytest.approx(sk_support_vectors, tol)
    assert da_n_support == pytest.approx(sk_n_support, 1e-10)
    assert da_n_features_in_ == sk_n_features_in_
    assert da_params == sk_params

    # Additional test for OVO decision function shape
    if svm_problem == "SVC" or svm_problem == "NuSVC":
        skpatch()
        SVM_module = importlib.import_module('sklearn.svm')
        SVM_model = getattr(SVM_module, svm_problem)
        svm_da = SVM_model(kernel=kernel, gamma=gamma,
                           decision_function_shape='ovo', tol=1e-6)
        svm_da.fit(X_train, y_train)
        da_dec_f = svm_da.decision_function(X_test)
        undo_skpatch()
        SVM_module = importlib.import_module('sklearn.svm')
        SVM_model = getattr(SVM_module, svm_problem)
        svm_sk = SVM_model(kernel=kernel, gamma=gamma,
                           decision_function_shape='ovo', tol=1e-6)
        svm_sk.fit(X_train, y_train)
        sk_dec_f = svm_sk.decision_function(X_test)
        assert da_dec_f == pytest.approx(sk_dec_f, 0.05)


@pytest.mark.parametrize("svm_problem, X_train, y_train", [
    ("SVC", X_train_class, y_train_class),
    ("SVR", X_train_reg, y_train_reg),
    ("NuSVC", X_train_class, y_train_class),
    ("NuSVR", X_train_reg, y_train_reg)
], ids=["SVC", "SVR", "NuSVC", "NuSVR"])
@pytest.mark.parametrize("precision", [np.float64,  np.float32])
def test_double_solve(svm_problem, precision, X_train, y_train):
    """
    Check that solving the model twice doesn't fail
    """
    X_train = X_train.astype(precision)
    y_train = y_train.astype(precision)

    skpatch()
    SVM_module = importlib.import_module('sklearn.svm')
    SVM_model = getattr(SVM_module, svm_problem)
    svm_da = SVM_model(kernel="linear", tol=1e-6)
    svm_da.fit(X_train, y_train)
    svm_da.fit(X_train, y_train)
    assert svm_da.aocl is True


@pytest.mark.parametrize("svm_problem, X_train, y_train", [
    ("SVC", X_train_class, y_train_class),
    ("SVR", X_train_reg, y_train_reg),
    ("NuSVC", X_train_class, y_train_class),
    ("NuSVR", X_train_reg, y_train_reg)
], ids=["SVC", "SVR", "NuSVC", "NuSVR"])
def test_svm_errors(svm_problem, X_train, y_train):
    '''
    Check we can catch errors in the sklearn svm patch
    '''
    skpatch()
    SVM_module = importlib.import_module('sklearn.svm')
    SVM_model = getattr(SVM_module, svm_problem)

    # Test invalid parameters
    with pytest.raises(RuntimeError):
        svm_da = SVM_model(kernel="precomputed")
        svm_da.fit(X_train, y_train)

    with pytest.raises(RuntimeError):
        svm_da = SVM_model(tol=-1)
        svm_da.fit(X_train, y_train)

    with pytest.raises(TypeError):
        svm_da = SVM_model(max_iter=1.5)
        svm_da.fit(X_train, y_train)

    if svm_problem in ["SVC", "NuSVR"]:
        with pytest.raises(RuntimeError):
            svm_da = SVM_model(C=-1)
            svm_da.fit(X_train, y_train)

    if svm_problem in ["NuSVC", "NuSVR"]:
        with pytest.raises(RuntimeError):
            svm_da = SVM_model(nu=2)
            svm_da.fit(X_train, y_train)

    if svm_problem in ["SVR"]:
        with pytest.raises(RuntimeError):
            svm_da = SVM_model(epsilon=-1)
            svm_da.fit(X_train, y_train)

    # Test unsupported parameters
    with pytest.warns(RuntimeWarning):
        svm_da = SVM_model(cache_size=1)  # non-default cache size

    with pytest.warns(RuntimeWarning):
        svm_da = SVM_model(shrinking=True)

    with pytest.warns(RuntimeWarning):
        svm_da = SVM_model(verbose=True)

    if svm_problem in ["SVC", "NuSVC"]:
        with pytest.warns(RuntimeWarning):
            svm_da = SVM_model(probability=True)

        with pytest.warns(RuntimeWarning):
            svm_da = SVM_model(class_weight='balanced')

        with pytest.warns(RuntimeWarning):
            svm_da = SVM_model(break_ties=True)

        with pytest.warns(RuntimeWarning):
            svm_da = SVM_model(random_state=1)

    # Test unsupported functions
    with pytest.raises(RuntimeError):
        svm_da.get_metadata_routing()

    with pytest.raises(RuntimeError):
        svm_da.set_fit_request()

    with pytest.raises(RuntimeError):
        svm_da.set_params()

    with pytest.raises(RuntimeError):
        svm_da.set_score_request()

    if svm_problem in ["SVC", "NuSVC"]:
        with pytest.raises(RuntimeError):
            svm_da.predict_proba(X_train)
        with pytest.raises(RuntimeError):
            svm_da.predict_log_proba(X_train)

    # Test unsupported attributes

    assert svm_da.coef_ is None
    assert svm_da.fit_status is None
    assert svm_da.feature_names_in_ is None
    assert svm_da.n_iter_ is None
    assert svm_da.shape_fit_ is None

    if svm_problem in ["SVC", "NuSVC"]:
        assert svm_da.class_weight_ is None
        assert svm_da.classes_ is None
        assert svm_da.probA_ is None
        assert svm_da.probB_ is None
