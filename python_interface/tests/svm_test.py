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
Support Vector Machine Python test script
"""

from aoclda.svm import SVC, SVR, NuSVC, NuSVR
import numpy as np
import pytest


@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, 'object'])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_svc_all_dtypes(numpy_precision, numpy_order):
    """
    Test it runs when supported/unsupported C-interface type is provided.
    """

    X_train = np.array(
        [[1.92, -0.52],
         [1.76, 0.84],
         [-1.02, 0.94],
         [0.79, 1.34],
         [2.86, -0.43]],
        dtype=numpy_precision, order=numpy_order)
    X_test = np.array([[-0.15, 2.33], [-1.14, 1.11]],
                      dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 0, 2, 1, 0],
                       dtype=numpy_precision, order=numpy_order)
    y_test = np.array([1, 2],
                      dtype=numpy_precision, order=numpy_order)

    svc = SVC(kernel='rbf', gamma=1, tol=1e-6)
    svc.fit(X_train, y_train)
    predictions = svc.predict(X_test)

    df1 = svc.decision_function(X_test)
    score = svc.score(X_test, y_test)


@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders",
                         [("C", "F"), ("F", "C")])
def test_svc_multiple_orders(numpy_precision, numpy_orders):
    """
    Test it runs when arrays of multiple orders are provided.
    """

    X_train = np.array(
        [[1.92, -0.52],
         [1.76, 0.84],
         [-1.02, 0.94],
         [0.79, 1.34],
         [2.86, -0.43]],
        dtype=numpy_precision, order=numpy_orders[0])
    X_test = np.array([[-0.15, 2.33], [-1.14, 1.11]],
                      dtype=numpy_precision, order=numpy_orders[1])
    y_train = np.array([0, 0, 2, 1, 0],
                       dtype=numpy_precision, order=numpy_orders[1])
    y_test = np.array([1, 2],
                      dtype=numpy_precision, order=numpy_orders[1])

    svc = SVC(kernel='rbf', gamma=1, tol=1e-6)

    # No warning due to y_train being 1d
    svc.fit(X_train, y_train)
    with pytest.warns(UserWarning):
        predictions = svc.predict(X_test)

    with pytest.warns(UserWarning):
        df1 = svc.decision_function(X_test)
    with pytest.warns(UserWarning):
        score = svc.score(X_test, y_test)

    X_train = np.array(X_train, order=numpy_orders[1])
    with pytest.warns(UserWarning):
        svc.fit(X_train, y_train)


@pytest.mark.parametrize(
    "numpy_precisions", [('float32', 'float64'),
                         ('float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_svc_multiple_dtypes(numpy_precisions, numpy_order):
    """
    Test it runs when arrays of multiple dtypes are provided.
    """

    X_train = np.array(
        [[1.92, -0.52],
         [1.76, 0.84],
         [-1.02, 0.94],
         [0.79, 1.34],
         [2.86, -0.43]],
        dtype=numpy_precisions[0], order=numpy_order)
    X_test = np.array([[-0.15, 2.33], [-1.14, 1.11]],
                      dtype=numpy_precisions[1], order=numpy_order)
    y_train = np.array([0, 0, 2, 1, 0],
                       dtype=numpy_precisions[1], order=numpy_order)
    y_test = np.array([1, 2],
                      dtype=numpy_precisions[1], order=numpy_order)

    svc = SVC(kernel='rbf', gamma=1, tol=1e-6)

    # No warning due to y_train being 1d
    svc.fit(X_train, y_train)
    predictions = svc.predict(X_test)

    df1 = svc.decision_function(X_test)
    score = svc.score(X_test, y_test)

    X_train = np.array(X_train, dtype=numpy_precisions[1])
    svc.fit(X_train, y_train)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_svc_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the Python wrapper
    """

    X_train = np.array(
        [[1.92, -0.52],
         [1.76, 0.84],
         [-1.02, 0.94],
         [0.79, 1.34],
         [2.86, -0.43]],
        dtype=numpy_precision, order=numpy_order)
    X_test = np.array([[-0.15, 2.33], [-1.14, 1.11]],
                      dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 0, 2, 1, 0],
                       dtype=numpy_precision, order=numpy_order)
    y_test = np.array([1, 2],
                      dtype=numpy_precision, order=numpy_order)

    svc = SVC(kernel='rbf', gamma=1, tol=1e-6)
    svc.fit(X_train, y_train)

    expected_bias = np.array([0.6614310488724978, 0.5365371320535124, 0.0])
    expected_n_support_per_class = np.array([3, 1, 1])
    expected_support_idx = np.array([0, 1, 4, 3, 2])
    expected_dual_coef = np.array([[0.1567003658615869, 0.6042853359149776,
                                    0.23901429822343534, -1.0, -1.0],
                                   [0.2661922344693693, 0.4034513140618167,
                                    0.330356451468814, 1.0, -1.0]])
    expected_support_vectors = np.array(
        [[1.92, -0.52],
         [1.76, 0.84],
         [2.86, -0.43],
         [0.79, 1.34],
         [-1.02, 0.94]])
    expected_predict = np.array([0, 2])
    expected_score = 0.5
    expected_decision_ovr = np.array(
        [[2.1647936191078374, 0.9012607489781155, -0.1192298001005942],
         [1.0595892420507806, -0.20380478276768527, 2.191835754976765]])
    expected_decision_ovo = np.array(
        [[0.5080428914586723, 0.4697303287293784, 0.08714885163887187],
         [0.638684111555492, -0.42100181649081425, -0.9347511031340464]])

    tol = np.sqrt(np.finfo(numpy_precision).eps)

    norm = np.linalg.norm(svc.bias - expected_bias)
    assert norm < tol * 10

    assert not np.any(svc.n_support_per_class - expected_n_support_per_class)

    assert not np.any(svc.support_vectors_idx - expected_support_idx)

    norm = np.linalg.norm(svc.dual_coef - expected_dual_coef)
    assert norm < tol * 200

    norm = np.linalg.norm(svc.support_vectors - expected_support_vectors)
    assert norm < tol

    predictions = svc.predict(X_test)
    not np.any(predictions - expected_predict)

    score = svc.score(X_test, y_test)
    assert score == expected_score

    norm = np.linalg.norm(svc.decision_function(
        X_test) - expected_decision_ovr)
    assert norm < tol

    norm = np.linalg.norm(svc.decision_function(
        X_test, 'ovo') - expected_decision_ovo)
    assert norm < tol * 10

    assert svc.n_classes == 3

    assert svc.n_support == 5

    assert svc.n_samples == 5

    assert svc.n_features == 2


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_svr_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the Python wrapper
    """

    X_train = np.array(
        [[1.29, -0.73],
         [2.3, -0.47],
         [-1.73, 1.66],
         [0.11, -0.38],
         [-1.03, 0.48]],
        dtype=numpy_precision, order=numpy_order)
    X_test = np.array([[-1.61, 0.99], [0.11, 0.35]],
                      dtype=numpy_precision, order=numpy_order)
    y_train = np.array([-5.66, 45.94, 53.25, -21.96, -2.46],
                       dtype=numpy_precision, order=numpy_order)
    y_test = np.array([12.5, 27.19],
                      dtype=numpy_precision, order=numpy_order)

    svr = SVR(kernel='linear', epsilon=0.1, tol=1e-6)
    svr.fit(X_train, y_train)

    expected_bias = -3.664383850678176
    expected_n_support = 5
    expected_support_idx = np.array([0, 1, 2, 3, 4])
    expected_dual_coef = np.array(
        [[-0.7796538205625038, 1.0, 1.0, -1.0, -0.22034617943749626]])
    expected_support_vectors = np.array(
        [[1.29, -0.73],
         [2.3, -0.47],
         [-1.73, 1.66],
         [0.11, -0.38],
         [-1.03, 0.48]])
    expected_predict = np.array([-1.1380735884612876, -2.9877681126775064])
    expected_score = -9.16417571463822

    tol = np.sqrt(np.finfo(numpy_precision).eps)

    assert np.isclose(svr.bias, expected_bias,
                      atol=np.finfo(numpy_precision).eps)

    assert not np.any(svr.support_vectors_idx - expected_support_idx)

    norm = np.linalg.norm(svr.dual_coef - expected_dual_coef)
    assert norm < tol * 10

    norm = np.linalg.norm(svr.support_vectors - expected_support_vectors)
    assert norm < tol

    predictions = svr.predict(X_test)
    not np.any(predictions - expected_predict)

    score = svr.score(X_test, y_test)
    assert np.isclose(score, expected_score,
                      atol=np.finfo(numpy_precision).eps)

    assert svr.n_support == expected_n_support

    assert svr.n_samples == 5

    assert svr.n_features == 2


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_nusvc_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the Python wrapper
    """

    X_train = np.array([[1.2, 0.92], [1.39, 1.06], [1.24, 0.91], [1.21, 1.05], [
                       0.65, -0.75]], dtype=numpy_precision, order=numpy_order)
    X_test = np.array([[-0.02, 0.57], [0.89, -0.77]],
                      dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 0, 0, 0, 1],
                       dtype=numpy_precision, order=numpy_order)
    y_test = np.array([1, 1],
                      dtype=numpy_precision, order=numpy_order)

    nusvc = NuSVC(kernel='sigmoid', coef0=1, nu=0.3, tol=1e-6)
    nusvc.fit(X_train, y_train)

    expected_bias = np.array([0.04340625628910816])
    expected_n_support_per_class = np.array([1, 1])
    expected_support_idx = np.array([2, 4])
    expected_dual_coef = np.array([[-6.089160022858101, 6.089160022858101]])
    expected_support_vectors = np.array([[1.24, 0.91], [0.65, -0.75]])
    expected_predict = np.array([0, 1])
    expected_score = 0.5
    expected_decision_ovr = np.array([-3.642897716170645, 0.4697240910420319])
    expected_decision_ovo = np.array([-3.642897716170645, 0.4697240910420319])

    tol = np.sqrt(np.finfo(numpy_precision).eps)

    assert np.isclose(nusvc.bias, expected_bias,
                      atol=np.finfo(numpy_precision).eps)

    assert not np.any(nusvc.n_support_per_class - expected_n_support_per_class)

    assert not np.any(nusvc.support_vectors_idx - expected_support_idx)

    norm = np.linalg.norm(nusvc.dual_coef - expected_dual_coef)
    assert norm < tol * 200

    norm = np.linalg.norm(nusvc.support_vectors - expected_support_vectors)
    assert norm < tol

    predictions = nusvc.predict(X_test)
    not np.any(predictions - expected_predict)

    score = nusvc.score(X_test, y_test)
    assert score == expected_score

    norm = np.linalg.norm(nusvc.decision_function(
        X_test) - expected_decision_ovr)
    assert norm < tol * 100

    norm = np.linalg.norm(nusvc.decision_function(
        X_test, 'ovo') - expected_decision_ovo)
    assert norm < tol * 100

    assert nusvc.n_classes == 2

    assert nusvc.n_support == 2

    assert nusvc.n_samples == 5

    assert nusvc.n_features == 2


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_nusvr_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the Python wrapper
    """

    X_train = np.array(
        [[2.07, 0.43],
         [-0.43, 0.3],
         [-0.04, 0.39],
         [0.46, 0.09],
         [-1.38, -0.54]],
        dtype=numpy_precision, order=numpy_order)
    X_test = np.array([[-0.87, 1.73], [-1.1, -0.82]],
                      dtype=numpy_precision, order=numpy_order)
    y_train = np.array([125.09, -1.89, 22.09, 27.03, -98.74],
                       dtype=numpy_precision, order=numpy_order)
    y_test = np.array([65.67, -103.03],
                      dtype=numpy_precision, order=numpy_order)

    nusvr = NuSVR(kernel='poly', tol=1e-6)
    nusvr.fit(X_train, y_train)

    expected_bias = 12.437607185762609
    expected_n_support = 4
    expected_support_idx = np.array([0, 1, 3, 4])
    expected_dual_coef = np.array(
        [[1.0, -0.25, 0.25, -1.0]])
    expected_support_vectors = np.array(
        [[2.07, 0.43], [-0.43, 0.3], [0.46, 0.09], [-1.38, -0.54]])
    expected_predict = np.array([11.96152679112588, 3.5562811749938152])
    expected_score = -0.001081559858395753

    tol = np.sqrt(np.finfo(numpy_precision).eps)

    assert np.isclose(nusvr.bias, expected_bias,
                      atol=np.finfo(numpy_precision).eps)

    assert not np.any(nusvr.support_vectors_idx - expected_support_idx)

    norm = np.linalg.norm(nusvr.dual_coef - expected_dual_coef)
    assert norm < tol

    norm = np.linalg.norm(nusvr.support_vectors - expected_support_vectors)
    assert norm < tol

    predictions = nusvr.predict(X_test)
    not np.any(predictions - expected_predict)

    score = nusvr.score(X_test, y_test)
    assert np.isclose(score, expected_score,
                      atol=np.finfo(numpy_precision).eps)

    assert nusvr.n_support == expected_n_support

    assert nusvr.n_samples == 5

    assert nusvr.n_features == 2


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

    tol = 1e-5 if numpy_precision == np.float64 else 1e-3
    assert np.isclose(svc.probA, expected_probaA, atol=tol)
    assert np.isclose(svc.probB, expected_probaB, atol=tol)

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

    norm = np.linalg.norm(nusvc.probA - expected_probaA)
    assert norm < tol

    norm = np.linalg.norm(nusvc.probB - expected_probaB)
    assert norm < tol

    norm = np.linalg.norm(nusvc.predict_proba(X_test) - expected_proba)
    assert norm < tol

    norm = np.linalg.norm(nusvc.predict_log_proba(X_test) - expected_log_proba)
    assert norm < tol


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("svm_model", [SVC, SVR, NuSVC, NuSVR])
def test_svm_error_exits(numpy_precision, svm_model):
    """
    Test error exits in the Python wrapper
    """
    x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision)
    y = np.array([0, 1, 2], dtype=numpy_precision)  # Incorrect shape

    # Test wrong arguments
    with pytest.raises(RuntimeError):
        model = svm_model(kernel="fbr")
    with pytest.raises(RuntimeError):
        model = svm_model(degree=-1)
    with pytest.raises(ValueError):
        model = svm_model(gamma="a")
        model.fit(x, y)
    with pytest.raises(RuntimeError):
        model = svm_model(max_iter=-2)  # -1 allowed for unlimited
    with pytest.raises(RuntimeError):
        model = svm_model(tol=-1)
        model.fit(x, y)
    if svm_model in [SVC, NuSVC]:
        with pytest.raises(ValueError):
            model = svm_model()
            model.fit(x, y)
            model.decision_function(x, shape="ooo")
    if svm_model in [NuSVC, NuSVR]:
        with pytest.raises(RuntimeError):
            model = svm_model(nu=1.1)
            model.fit(x, y)
    if svm_model in [SVC, NuSVR]:
        with pytest.raises(RuntimeError):
            model = svm_model(C=0)
            model.fit(x, y)
    if svm_model in [SVR]:
        with pytest.raises(RuntimeError):
            model = svm_model(epsilon=-0.1)
            model.fit(x, y)
    model = svm_model()
    model.fit(x, y)
    # Wrong number of features
    a = np.array([[1, 1], [2, 2], [3, 3]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        model.predict(a)
    with pytest.raises(RuntimeError):
        model.score(a, y)
    if svm_model in [SVC, NuSVC]:
        with pytest.raises(RuntimeError):
            model.decision_function(a)
