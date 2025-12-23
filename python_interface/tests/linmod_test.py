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
# pylint: disable = missing-module-docstring,missing-function-docstring
# pylint: disable = invalid-name,consider-using-in
"""
Linear models Python test script
"""

import numpy as np
import pytest
from aoclda.linear_model import linmod


@pytest.fixture(scope="function")
def no_fortran(request):
    return request.config.no_fortran


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_linear_regression(numpy_precision, numpy_order):
    X = np.array([[1, 1, None, None], [2, 3, None, None],
                  [3, 5, None, None], [4, 8, None, None],
                  [5, 7, None, None], [6, 9, None, None]],
                 dtype=numpy_precision, order=numpy_order)
    y = np.array([3., 6.5, 10., 12., 13., 19.], dtype=numpy_precision)
    tol = np.sqrt(np.finfo(numpy_precision).eps)

    # compute linear regression without intercept
    lmod = linmod("mse")
    lmod.fit(X[:, 0:2], y)  # exercise ldx

    # check expected results
    expected_coef = np.array([2.45, 0.35], dtype=numpy_precision)
    norm = np.linalg.norm(np.abs(lmod.coef) - np.abs(expected_coef))
    assert norm < tol

    # same test with intercept
    lmod = linmod("mse", intercept=True)
    lmod.fit(X[:, 0:2], y)

    # check expected results
    expected_coef = np.array(
        [2.35, 0.35, 0.43333333333333535], dtype=numpy_precision)
    norm = np.linalg.norm(np.abs(lmod.coef) - np.abs(expected_coef))
    assert norm < tol


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_linear_regression_error_exits(numpy_precision, numpy_order):
    X = np.array([[1, 1], [2, 3], [3, 5], [4, 8], [5, 7], [6, 9]],
                 dtype=numpy_precision, order=numpy_order)
    y = np.array([3., 6.5, 10., 12., 13., 19.], dtype=numpy_precision)

    # lambda out of bounds
    with pytest.raises(RuntimeError):
        lmod = linmod("mse", reg_lambda=-1)
        lmod.fit(X, y)

    # alpha out of bounds
    with pytest.raises(RuntimeError):
        lmod = linmod("mse", reg_alpha=-1)
        lmod.fit(X, y)
    with pytest.raises(RuntimeError):
        lmod = linmod("mse", reg_alpha=1.1)
        lmod.fit(X, y)

    # max_iter out of bounds
    with pytest.raises(RuntimeError):
        lmod = linmod("mse", max_iter=-1)
        lmod.fit(X, y)

    # solving lasso with cholesky
    with pytest.raises(RuntimeError):
        lmod = linmod("mse", solver='cholesky', reg_alpha=1, reg_lambda=1)
        lmod.fit(X, y)

    # solving ridge with qr
    with pytest.raises(RuntimeError):
        lmod = linmod("mse", solver='qr', reg_alpha=0, reg_lambda=1)
        lmod.fit(X, y)

    # NaN checking
    X = np.array([[1, 1], [2, 3], [3, 5], [4, 8], [5, 7], [6, 9]],
                 dtype=numpy_precision, order=numpy_order)
    y = np.array([3., 6.5, 10., 12., np.nan, 19.], dtype=numpy_precision)
    lmod2 = linmod("mse", check_data=True)
    with pytest.raises(RuntimeError):
        lmod2.fit(X, y)


elastic_net_answer = np.array(
    [30.42698430323763, 47.92432984823038, 0.6201200680560284])
lasso_answer = np.array([29.751069, 51.37881, -0.3529817356770124])
ridge_answer = np.array([28.49233212, 40.06618967, 3.7353163594626566])
parameters_list_linreg = [
    {"mod": "mse", "reg_lambda": 0.1, "reg_alpha": 0.5, "tol": 1e-6,
     "intercept": True, "actual_coef": elastic_net_answer},
    {"mod": "mse", "reg_lambda": 1.0, "reg_alpha": 1.0, "tol": 1e-6,
     "intercept": True, "actual_coef": lasso_answer},
    {"mod": "mse", "reg_lambda": 1.0, "reg_alpha": 0.0, "tol": 1e-6,
     "intercept": True, "solver": "sparse_cg", "actual_coef": ridge_answer}]


@pytest.mark.parametrize("problem", parameters_list_linreg,
                         ids=["ElasticNet", "Lasso", "Ridge"])
@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_warmstart_l1_l2(problem, numpy_precision, numpy_order):
    X = np.array([[0.497, -0.138], [0.648, 1.523], [-0.469, 0.543],
                  [1.579, 0.767], [-0.234, -0.234], [-0.463, -0.466]],
                 dtype=numpy_precision, order=numpy_order)
    y = np.array([6.132, 99.065, 13.28, 88.632, -
                 20.423, -39.951], dtype=numpy_precision)
    problem_copy = problem.copy()

    true_coefs = problem_copy["actual_coef"]
    del problem_copy["actual_coef"]

    # Do first run with warm start (usual case)
    model = linmod(**problem_copy, warm_start=True)
    model.fit(X, y)
    assert model.n_iter > 1
    first_coef = model.coef
    assert first_coef == pytest.approx(true_coefs, abs=1e-4)

    # Do second run with warm start (reusing last coefficients automatically)
    model.fit(X, y)
    assert (model.n_iter == 1) or (model.n_iter == 0)
    assert model.coef == pytest.approx(first_coef, abs=1e-4)

    # Confirm that passing x0 takes precedence over warm start
    zero_coef = np.zeros_like(true_coefs)
    model.fit(X, y, x0=zero_coef)
    assert model.n_iter > 1
    assert model.coef == pytest.approx(first_coef, abs=1e-4)

    # Confirm that passsing correct x0 is equal to second run warm start
    model = linmod(**problem_copy, warm_start=False)
    model.fit(X, y, x0=first_coef)
    assert (model.n_iter == 1) or (model.n_iter == 0)
    assert model.coef == pytest.approx(true_coefs, abs=1e-4)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_warmstart_underdetermined(numpy_precision, numpy_order):
    X = np.array([[0.497, -0.138, -0.421], [0.648, 1.523, 2.123]],
                 dtype=numpy_precision, order=numpy_order)
    y = np.array([6.132, 99.065], dtype=numpy_precision)
    # Do first run
    model = linmod(mod="mse", solver="sparse_cg", reg_lambda=2.0,
                   intercept=False, warm_start=True)
    model.fit(X, y)
    primal_coef = model.coef
    dual_coef = model.dual_coef
    primal_coef_expected = np.array(
        [10.31036367, 16.26075041, 21.27136156], dtype=numpy_precision)
    dual_coef_expected = np.array(
        [6.10348802, 11.22983044], dtype=numpy_precision)
    assert primal_coef == pytest.approx(primal_coef_expected, abs=1e-4)
    assert dual_coef == pytest.approx(dual_coef_expected, abs=1e-4)
    assert model.n_iter > 1
    # Do second run with warm start
    model.fit(X, y)
    assert model.coef == pytest.approx(primal_coef_expected, abs=1e-4)
    assert model.dual_coef == pytest.approx(dual_coef_expected, abs=1e-4)
    assert model.n_iter == 0
    # Create new object and pass solution as starting point
    model = linmod(mod="mse", solver="sparse_cg", reg_lambda=2.0,
                   intercept=False, warm_start=False)
    model.fit(X, y, x0=dual_coef_expected)
    assert model.coef == pytest.approx(primal_coef_expected, abs=1e-4)
    assert model.dual_coef == pytest.approx(dual_coef_expected, abs=1e-4)
    assert model.n_iter == 0


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_warm_start_diff_col_l1_l2(numpy_precision, numpy_order):
    # DO THE TEST WHEN COLUMN COUNT HAS DECREASED
    X_initial = np.array([[0.497, -0.138, 0.132], [0.648, 1.523, -1.023],
                          [-0.469, 0.543, -3.731], [1.579, 0.767, 0.123]],
                         dtype=numpy_precision, order=numpy_order)
    y_initial = np.array([6.132, 99.065, 13.28, 88.632], dtype=numpy_precision)
    # Do first run with warm start
    model = linmod(mod="mse", reg_lambda=1.0, reg_alpha=0.5,
                   intercept=True, warm_start=True, tol=1e-6)
    model.fit(X_initial, y_initial)
    coef_expected = np.array(
        [14.15518376, 23.6555845, 4.02668761, 32.38833198],
        dtype=numpy_precision)
    assert model.coef == pytest.approx(coef_expected, abs=1e-4)
    assert model.n_iter > 1
    # Do second run with data without the third column
    X_smaller = np.array([[0.497, -0.138], [0.648, 1.523],
                          [-0.469, 0.543], [1.579, 0.767]],
                         dtype=numpy_precision, order=numpy_order)
    y_smaller = np.array([6.132, 99.065, 13.28, 88.632], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        model.fit(X_smaller, y_smaller)

    # NOW DO THE SAME WHEN COLUMN COUNT IS INCREASING
    X_bigger = np.array(
        [[0.497, -0.138, 0.132, 1.0],
         [0.648, 1.523, -1.023, 1.0],
         [-0.469, 0.543, -3.731, 1.0],
         [1.579, 0.767, 0.123, 1.0]],
        dtype=numpy_precision, order=numpy_order)
    y_bigger = np.array([6.132, 99.065, 13.28, 88.632], dtype=numpy_precision)
    # Do first run with warm start
    model = linmod(mod="mse", reg_lambda=1.0, reg_alpha=0.5,
                   intercept=True, warm_start=True, tol=1e-6)
    model.fit(X_initial, y_initial)
    assert model.coef == pytest.approx(coef_expected, abs=1e-4)
    assert model.n_iter > 1
    # Do second run with data with the additional fourth column
    with pytest.raises(RuntimeError):
        model.fit(X_bigger, y_bigger)


logistic_regr_answer_ssc = np.array(
    [[-0.25164543762030384, -0.3429889469402158, 2.81856299],
     [-0.5082127481640433, 0.16869816164326473, 1.32964205],
     [0.7598581857843467, 0.1742907852969505, -4.14820504]])
logistic_regr_answer_single_prec_ssc = np.array(
    [[-0.328682, -0.310989, 2.783613],
     [-0.414201, 0.228109, 0.54265785],
     [0.742886, 0.082883, -3.3262687]])
logistic_regr_answer_rsc = np.array(
    [[-0.551207644383338, 5.808431799389836, 14.94856184],
     [-0.6393035441433154, -31.5701270058952, 20.52009586]])
logistic_regr_answer_single_prec_rsc = np.array(
    [[-0.5811843872070312, 6.347817897796631, 4.0107155],
     [-0.6911936402320862, -9.66157054901123, 9.869258]])

parameters_list = [
    {"mod": "logistic", "reg_lambda": 1.0, "tol": 1e-6, "intercept": True,
     "constraint": "ssc", "actual_coef": logistic_regr_answer_ssc,
     "actual_coef_single_prec": logistic_regr_answer_single_prec_ssc},
    {"mod": "logistic", "reg_lambda": 1.0, "tol": 1e-6, "intercept": True,
     "constraint": "rsc", "actual_coef": logistic_regr_answer_rsc,
     "actual_coef_single_prec": logistic_regr_answer_single_prec_rsc}]


@pytest.mark.parametrize("problem", parameters_list,
                         ids=["Logistic_ssc", "Logistic_rsc"])
@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_warmstart_logistic(no_fortran, problem, numpy_precision, numpy_order):
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    X = np.array([[1, 1], [2, 3], [3, 5], [4, 8], [5, 7], [6, 9]],
                 dtype=numpy_precision, order=numpy_order)
    y = np.array([0, 1, 0, 1, 2, 2], dtype=numpy_precision)
    problem_copy = problem.copy()

    # single precision logistic regression (lbfgs internally) can produce
    # vastly different results
    if numpy_precision == np.float32:
        true_coefs = problem_copy["actual_coef_single_prec"]
    else:
        true_coefs = problem_copy["actual_coef"]
    del problem_copy["actual_coef"]
    del problem_copy["actual_coef_single_prec"]

    # Do first run with warm start (usual case)
    model = linmod(**problem_copy, warm_start=True)
    model.fit(X, y)
    assert model.n_iter > 1
    first_coef = model.coef
    assert first_coef == pytest.approx(true_coefs, abs=1e-4)

    # Do second run with warm start (reusing last coefficients automatically)
    model.fit(X, y)
    assert model.n_iter == 1
    assert model.coef == pytest.approx(first_coef, abs=0.02)

    # Confirm that passing x0 takes precedence over warm start
    zero_coef = np.zeros_like(true_coefs, order=numpy_order)
    model.fit(X, y, x0=zero_coef)
    assert model.n_iter > 1
    assert model.coef == pytest.approx(first_coef, abs=1e-4)

    # Confirm that passsing correct x0 is equal to second run warm start
    model = linmod(**problem_copy, warm_start=False)
    model.fit(X, y, x0=first_coef)
    assert model.n_iter == 1
    assert model.coef == pytest.approx(true_coefs, abs=0.02)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_warm_start_diff_col_logistic(no_fortran, numpy_precision, numpy_order):
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    # DO THE TEST WHEN COLUMN COUNT HAS DECREASED
    X_initial = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0], [1.0, 0.1, 0.1]],
                         dtype=numpy_precision, order=numpy_order)
    y_initial = np.array([0, 1, 2, 0], dtype=numpy_precision)
    # Do first run with warm start
    model = linmod(mod="logistic", reg_lambda=1.0,
                   intercept=True, warm_start=True, tol=1e-4)
    model.fit(X_initial, y_initial)
    coef_expected = np.array(
        [[0.653542930622, -0.293128447445, -0.293128447445, 0.31997256],
         [-0.326771465311, 0.5249460481913, -0.231817600746, -0.15998628],
         [-0.326771465311, -0.23181760074, 0.524946048191, -0.15998628]],
        dtype=numpy_precision)
    assert model.coef == pytest.approx(coef_expected, abs=0.05)
    assert model.n_iter > 1
    # Decrease number of columns
    X_smaller = np.array([[1.0, 0.0], [0.0, 1.0],
                          [0.0, 0.0], [1.0, 0.1]],
                         dtype=numpy_precision, order=numpy_order)
    y_smaller = np.array([0, 1, 2, 0], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        model.fit(X_smaller, y_smaller)
    # Decrease number of classes
    y_less_class = np.array([0, 1, 1, 0], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        model.fit(X_initial, y_less_class)

    # NOW DO THE SAME WHEN COLUMN COUNT IS INCREASING
    X_bigger = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0], [1.0, 0.1, 0.1, 0.0]],
                        dtype=numpy_precision, order=numpy_order)
    y_bigger = np.array([0, 1, 2, 0], dtype=numpy_precision)
    # Do first run with warm start
    model = linmod(mod="logistic", reg_lambda=1.0,
                   intercept=True, warm_start=True, tol=1e-4)
    model.fit(X_initial, y_initial)
    assert model.coef == pytest.approx(coef_expected, abs=0.05)
    assert model.n_iter > 1
    # Do second run with data with the additional fourth column
    with pytest.raises(RuntimeError):
        model.fit(X_bigger, y_bigger)
    # Increase number of classes
    y_more_class = np.array([0, 1, 2, 3], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        model.fit(X_initial, y_more_class)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_warmstart_error(no_fortran, numpy_precision, numpy_order):
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    X = np.array(
        [[0.497, -0.138],
         [0.648, 1.523],
         [-0.469, 0.543],
         [1.579, 0.767],
         [-0.234, -0.234],
         [-0.463, -0.466]],
        dtype=numpy_precision, order=numpy_order)
    y = np.array([0, 1, 0, 1, 1, 2], dtype=numpy_precision)

    # Do first run of logistic with dataset with 2 features and 3 classes
    model = linmod("logistic", reg_lambda=0.5,
                   tol=1e-6, intercept=True, warm_start=True)
    model.fit(X, y)

    # Do second run with dataset with 2 features and 4 classes (expect error)
    X = np.array([[0.497, 0.1], [0.648, 0.2], [-0.469, 0.3],
                  [1.579, 0.4], [-0.234, 0.5], [-0.463, 0.6]],
                 dtype=numpy_precision, order=numpy_order)
    y = np.array([0, 1, 3, 1, 1, 2], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        model.fit(X, y)

    # First fit well-determined problem with CG and then try to
    # use warm start on underdetermined problem
    X = np.array([[0.497, 0.1], [0.648, 0.2], [-0.469, 0.3],
                  [1.579, 0.4], [-0.234, 0.5], [-0.463, 0.6]],
                 dtype=numpy_precision, order=numpy_order)
    y = np.array([1.21, -0.12, 3.55, 2.0, -1.99, 2.12], dtype=numpy_precision)
    model = linmod("mse", solver="cg", reg_lambda=1,
                   tol=1e-6, intercept=True, warm_start=True)
    model.fit(X, y)
    X = np.array([[0.497, 0.1, 0.991], [0.648, 0.2, -1.121]],
                 dtype=numpy_precision, order=numpy_order)
    y = np.array([1.21, -0.12], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        model.fit(X, y)


@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, 'object'])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_linmod_all_dtypes(numpy_precision, numpy_order):
    """
    Test it runs when supported/unsupported C-interface type is provided.
    """

    x_train = np.array([[-1.5, -1.2, 2.6],
                        [-2.5, -1.3, 3.8],
                        [-3.6, -2.8, -1.7],
                        [1.6, 3.4, 1.2],
                        [2.8, 5.3, 1.3],
                        [3.0, -1.2, 2.4]],
                       dtype=numpy_precision, order=numpy_order)

    y_train = np.array([1, 2, 0, 1, 2, 2],
                       dtype=numpy_precision, order=numpy_order)

    x_test = np.array([[-2.5, 2.2, 3.6],
                       [-1.4, -2.7, -1.0],
                       [2.6, 1.2, -3.7]],
                      dtype=numpy_precision, order=numpy_order)

    lmod = linmod("mse")
    lmod.fit(x_train, y_train)
    preds = lmod.predict(x_test)


@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders",
                         [("C", "F"), ("F", "C")])
def test_linmod_multiple_orders(numpy_precision, numpy_orders):
    """
    Test it runs when arrays of multiple orders are provided.
    """

    x_train = np.array([[-1.5, -1.2, 2.6],
                        [-2.5, -1.3, 3.8],
                        [-3.6, -2.8, -1.7],
                        [1.6, 3.4, 1.2],
                        [2.8, 5.3, 1.3],
                        [3.0, -1.2, 2.4]],
                       dtype=numpy_precision, order=numpy_orders[0])

    y_train = np.array([1, 2, 0, 1, 2, 2],
                       dtype=numpy_precision, order=numpy_orders[1])

    x_test = np.array([[-2.5, 2.2, 3.6],
                       [-1.4, -2.7, -1.0],
                       [2.6, 1.2, -3.7]],
                      dtype=numpy_precision, order=numpy_orders[1])

    lmod = linmod("mse")
    lmod.fit(x_train, y_train)
    with pytest.warns(UserWarning):
        preds = lmod.predict(x_test)
    x_train = np.array(x_train, order=numpy_orders[1])
    with pytest.warns(UserWarning):
        lmod.fit(x_train, y_train)


@pytest.mark.parametrize(
    "numpy_precisions", [('float32', 'float64'),
                         ('float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_linmod_multiple_dtypes(numpy_precisions, numpy_order):
    """
    Test it runs when arrays of multiple dtypes are provided.
    """

    x_train = np.array([[-1.5, -1.2, 2.6],
                        [-2.5, -1.3, 3.8],
                        [-3.6, -2.8, -1.7],
                        [1.6, 3.4, 1.2],
                        [2.8, 5.3, 1.3],
                        [3.0, -1.2, 2.4]],
                       dtype=numpy_precisions[0], order=numpy_order)

    y_train = np.array([1, 2, 0, 1, 2, 2],
                       dtype=numpy_precisions[1], order=numpy_order)

    x_test = np.array([[-2.5, 2.2, 3.6],
                       [-1.4, -2.7, -1.0],
                       [2.6, 1.2, -3.7]],
                      dtype=numpy_precisions[1], order=numpy_order)

    lmod = linmod("mse")
    lmod.fit(x_train, y_train)
    preds = lmod.predict(x_test)
    x_train = np.array(x_train, dtype=numpy_precisions[1])
    lmod.fit(x_train, y_train)
