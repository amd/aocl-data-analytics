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
X_test_class = np.array([[1.13, -0.98], [-1.53, 1.16], [-0.32, 1.5],
                         [1.03, -1.11], [-1.82, -0.59], [1.04, -0.73]])
y_train_class = np.array([2, 0, 0, 0, 0, 0, 1, 2, 1, 1, 2, 0, 1, 1])
y_test_class = np.array([2, 1, 1, 2, 0, 2])

# Regression dataset
X_train_reg = np.array([[-0.33, 0.44], [1.86, 1.15], [-0.33, 0.19], [-0.11, 0.0], [0.71, 1.0],
                       [0.6, -0.36], [-0.2, -1.19], [-0.25, 2.43], [-0.86, -0.98], [-0.36, 0.06]])
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
@pytest.mark.parametrize("precision", [np.float64, np.float32])
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
    da_n_iter = svm_da.n_iter_
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
    tol_intercept = 2e-3
    if svm_problem == "NuSVC" and precision == np.float32 and kernel == "rbf":
        tol_intercept = 0.34  # This specific case has some discrepancy on the intercept
    assert da_intercept == pytest.approx(sk_intercept, tol_intercept)
    assert da_support == pytest.approx(sk_support, tol)
    assert da_support_vectors == pytest.approx(sk_support_vectors, tol)
    assert da_n_support == pytest.approx(sk_n_support, 1e-10)
    assert da_n_features_in_ == sk_n_features_in_
    assert da_params == sk_params
    assert da_n_iter.all() > 0

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


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_proba_functionality(numpy_precision, numpy_order):
    """
    Test the probability functionality of SVM in Python wrapper
    """
    # Binary SVC
    X_train = np.array(
        [[-1.2, -1.08, -2.92], [1.84, -0.08, 0.03], [0.91, 0.92, 0.75],
         [3.31, 2.22, -1.48], [-0.01, -0.85, -2.22], [2.14, 0.35, -1.52],
         [-0.54, -1.05, -2.06], [1.46, 1.38, 1.83], [0.44, 0.67, 1.86],
         [-0.99, 1.13, 1.35], [0.09, -1.69, 0.36], [-0.67, 0.01, 0.46],
         [-1.9, -1.64, 0.6], [0.47, 0.92, 0.44], [-1.53, -0.3, -0.93],
         [-1.81, 0.92, 0.65], [0.32, 0.99, 1.41]],
        dtype=numpy_precision, order=numpy_order)
    X_test = np.array([[-0.14, -0.01, 1.93], [1.98, 1.55, 0.69],
                       [-0.66, -0.61, 0.41], [1.89, -1.54, -0.08],
                       [-1.48, -2.81, 0.77], [0.62, 0.8, 2.02],
                       [0.81, 0.8, 0.44], [-0.8, 0.64, 0.56]],
                      dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                       dtype=numpy_precision, order=numpy_order)
    skpatch()
    from sklearn.svm import SVC, NuSVC
    svc = SVC(kernel='rbf', tol=1e-5, probability=True, random_state=1)
    svc.fit(X_train, y_train)

    expected_probaA = -3.31093987
    expected_probaB = 0.36261956
    expected_proba = np.array(
        [[0.187326, 0.812674], [0.968047, 0.031953],
         [0.01328, 0.98672], [0.766831, 0.233169],
         [0.093423, 0.906577], [0.747862, 0.252138],
         [0.65693, 0.34307], [0.012665, 0.987335]])
    expected_log_proba = np.array(
        [[-1.674904, -0.207425], [-0.032474, -3.443504],
         [-4.321526, -0.013369], [-0.265489, -1.455991],
         [-2.37062, -0.098079], [-0.290537, -1.377779],
         [-0.420177, -1.069822], [-4.368891, -0.012746]])

    tol = 3e-5 if numpy_precision == np.float64 else 1e-3
    assert np.isclose(svc.probA_, expected_probaA, atol=tol)
    assert np.isclose(svc.probB_, expected_probaB, atol=tol)

    norm = np.linalg.norm(svc.predict_proba(X_test) - expected_proba)
    assert norm < tol

    norm = np.linalg.norm(svc.predict_log_proba(X_test) - expected_log_proba)
    assert norm < tol

    # Multi class NuSVC
    X_train = np.array(
        [[0.54, 3.1, 1.66], [0.44, 0.58, 0.54], [0.3, 1.74, 0.6],
         [0.31, 1.14, 0.69], [0.27, 0.82, 1.36], [-1.91, 0.67, 1.01],
         [0.85, 1.35, 0.39], [-0.57, 1.3, 2.71], [-3.13, 0.13, -1.34],
         [-1.14, 0.54, -2.41], [-1.22, 0.14, -2.53], [-0.64, -1.2, -0.32],
         [-1.55, -0.76, -1.66], [-0.25, -1.68, -0.05], [-1.48, 3.85, -1.61],
         [-1.88, 1.26, -1.61], [-2.06, -2.82, 2.05], [-3.45, -1.49, 2.26],
         [0.48, -0.16, -1.11], [1.88, -2.43, -0.72], [-1.86, -1.93, 2.47]],
        dtype=numpy_precision, order=numpy_order)
    X_test = np.array([[-1.37, -1.49, -1.52], [-1.35, 1.19, -1.11],
                       [1.43, -1.55, -1.06], [0.53, -1.22, -0.7],
                       [-0.85, 2.46, -0.97], [0.18, 0.11, -0.64],
                       [2.49, -0.93, -1.84], [-0.05, 0.28, 0.06], [-1.2, 1.64, 1.66]],
                      dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                       dtype=numpy_precision, order=numpy_order)

    nusvc = NuSVC(kernel='rbf', nu=0.4, tol=1e-5,
                  probability=True, random_state=1)
    nusvc.fit(X_train, y_train)

    expected_probaA = np.array([-1.9521972, -2.58923649, -1.53862029])
    expected_probaB = np.array([-0.29152272, 0.21993318, 0.17880565])
    expected_proba = np.array(
        [[0.103926, 0.745962, 0.150112], [0.303237, 0.593313, 0.10345],
         [0.052564, 0.161176, 0.78626], [0.08279, 0.408456, 0.508755],
         [0.314321, 0.552387, 0.133292], [0.314429, 0.342879, 0.342692],
         [0.082065, 0.092287, 0.825648], [0.603143, 0.237038, 0.159819],
         [0.902306, 0.051694, 0.046]])
    expected_log_proba = np.array(
        [[-2.264073, -0.293081, -1.896373], [-1.19324, -0.522033, -2.268669],
         [-2.945724, -1.82526, -0.240467], [-2.491454, -0.895372, -0.675789],
         [-1.15734, -0.593507, -2.015213], [-1.156997, -1.070376, -1.070924],
         [-2.500242, -2.382848, -0.191587], [-0.505602, -1.439534, -1.833712],
         [-0.102801, -2.962417, -3.079117]])

    tol = 5e-5 if numpy_precision == np.float64 else 1e-3

    norm = np.linalg.norm(nusvc.probA_ - expected_probaA)
    assert norm < tol

    norm = np.linalg.norm(nusvc.probB_ - expected_probaB)
    assert norm < tol

    norm = np.linalg.norm(nusvc.predict_proba(X_test) - expected_proba)
    assert norm < tol

    norm = np.linalg.norm(nusvc.predict_log_proba(X_test) - expected_log_proba)
    assert norm < tol
    undo_skpatch()


@pytest.mark.parametrize("svm_problem, X_train, y_train", [
    ("SVC", X_train_class, y_train_class),
    ("SVR", X_train_reg, y_train_reg),
    ("NuSVC", X_train_class, y_train_class),
    ("NuSVR", X_train_reg, y_train_reg)
], ids=["SVC", "SVR", "NuSVC", "NuSVR"])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
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
    ("NuSVC", X_train_class, y_train_class),
    ("SVR", X_train_reg, y_train_reg),
    ("NuSVR", X_train_reg, y_train_reg)
], ids=["SVC", "NuSVC", "SVR", "NuSVR"])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_svm_errors(svm_problem, X_train, y_train, precision):
    '''
    Check we can catch errors in the sklearn svm patch
    '''
    X_train = X_train.astype(precision)
    y_train = y_train.astype(precision)

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

    with pytest.raises(RuntimeError):
        svm_da = SVM_model(cache_size=-2.0)
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
        svm_da = SVM_model(shrinking=True)

    with pytest.warns(RuntimeWarning):
        svm_da = SVM_model(verbose=True)

    if svm_problem in ["SVC", "NuSVC"]:
        with pytest.warns(RuntimeWarning):
            svm_da = SVM_model(class_weight='balanced')

        with pytest.warns(RuntimeWarning):
            svm_da = SVM_model(break_ties=True)

    # Test unsupported functions
    with pytest.raises(RuntimeError):
        svm_da.get_metadata_routing()

    with pytest.raises(RuntimeError):
        svm_da.set_fit_request()

    with pytest.raises(RuntimeError):
        svm_da.set_params()

    with pytest.raises(RuntimeError):
        svm_da.set_score_request()

    # Test unsupported attributes

    assert svm_da.coef_ is None
    assert svm_da.fit_status is None
    assert svm_da.feature_names_in_ is None
    assert svm_da.shape_fit_ is None

    if svm_problem in ["SVC", "NuSVC"]:
        assert svm_da.class_weight_ is None
        assert svm_da.classes_ is None
