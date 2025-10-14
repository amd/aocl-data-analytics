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
# pylint: disable = missing-module-docstring,missing-function-docstring,invalid-name,unused-argument,unused-variable,no-member

import numpy as np
import pytest
from aoclda.nonlinear_model import nlls


@pytest.fixture(scope="function")
def no_fortran(request):
    return request.config.no_fortran

# Common functions
# Attempts to fit the model y_i = x_1 e^(x_2 t_i)
# For parameters x_1 and x_2, and input data (t_i, y_i)

# Calculate r_i(x; t_i, y_i) = x_1 e^(x_2 * t_i) - y_i


def exp_r(x, r, data) -> int:
    t = data['t']
    y = data['y']
    x1 = x[0]
    x2 = x[1]
    r[:] = x1 * np.exp(x2 * t) - y
    return 0

# Calculate:
# J_i1 = e^(x_2 * t_i)
# J_i2 = t_i x_1 e^(x_2 * t_i)


def exp_J(x, J, data) -> int:
    x1 = x[0]
    x2 = x[1]
    J[:] = np.column_stack((
        np.exp(x2 * data['t']),
        data['t'] * x1 * np.exp(x2 * data['t'])
    ))
    return 0

# Calculate:
# Hr = sum_i r_i H_i
# Where H_i = [ 0                     t_i x_1 e^(x_2 t_i)    ]
#             [ t_i x_1 e^(x_2 t_i)   x_1 t_i^2 e^(x_2 t_i)  ]


def exp_Hr(x, r, Hr, data) -> int:
    x1 = x[0]
    x2 = x[1]
    Hr[:] = np.zeros((2, 2))
    v = data['t'] * np.exp(x2 * data['t'])
    Hr[1, 0] = np.dot(r, v)                 # H_21
    Hr[0, 1] = Hr[1, 0]                     # H_21
    Hr[1, 1] = np.dot(r, (data['t'] * x1) * v)  # H_22
    return 0


# Data to be fitted
exp_data = {"t": np.array([1.0, 2.0, 4.0, 5.0, 8.0]),
            "y": np.array([3.0, 4.0, 6.0, 11.0, 20.0])}


def r_fail(x, r, data) -> int:
    return 1


def J_fail(x, r, data) -> int:
    return 1


@pytest.mark.parametrize("opt_params", [True, False])
def test_functionality(no_fortran, opt_params):
    """Test correct functionality while solving a simple problem"""
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    numpy_precision = np.float32
    n_coef = 2
    n_res = 5
    x = np.array([2.5, 0.25], dtype=numpy_precision)
    w = 0.12 * np.array([1, 1, 1, 1, 1], dtype=numpy_precision)
    blx = np.array([0.0, 0.0], dtype=numpy_precision)
    bux = np.array([5.0, 3.0], dtype=numpy_precision)
    ndf = nlls(n_coef, n_res, weights=w,
               lower_bounds=blx, upper_bounds=bux)

    try:
        if opt_params:
            ndf.fit(x, r_fail, J_fail, data=exp_data,
                    abs_gtol=1e-7, gtol=1.e-9, maxit=20)
        else:
            ndf.fit(x, r_fail, J_fail)
    except RuntimeError as e:
        print('Ok')
    except Exception as e:
        raise AssertionError(
            "Did not catch the expected exception for FLOAT test") from e
    else:
        raise AssertionError(
            "Did not catch the expected exception for FLOAT test")


@pytest.mark.parametrize("numpy_order, use_fd",
                         [("C", False),
                          ("C", True),
                          ("F", False),
                          ("F", True)])
def test_functionality_order(no_fortran, numpy_order, use_fd):
    """Test correct functionality while solving a simple problem"""
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    if use_fd:
        tol = 1e-4
        abs_gtol = 1e-7
        gtol = 1.e-7
        maxit = 20
    else:
        tol = 1e-7
        abs_gtol = 1e-7
        gtol = 1.e-9
        maxit = 20
    n_coef = 2
    n_res = 5
    expected_x = np.array([2.54104549, 0.25950481])
    x = np.array([2.5, 0.25])
    w = 0.12 * np.array([1, 1, 1, 1, 1])
    blx = np.array([0.0, 0.0])
    bux = np.array([5.0, 3.0])
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx, upper_bounds=bux,
               order=numpy_order, verbose=0, check_derivatives='yes')
    if use_fd:
        ndf.fit(x, exp_r, data=exp_data,
                abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)
    else:
        ndf.fit(x, exp_r, exp_J, exp_Hr, data=exp_data,
                abs_gtol=abs_gtol, gtol=gtol, maxit=maxit)

    # check expected results
    expected_x = np.array([2.5410455, 0.25950481])
    norm = np.linalg.norm(np.abs(x) - np.abs(expected_x))
    assert norm < tol

    # check the @properties
    if use_fd:
        assert ndf.n_eval['fd_f'] >= 10
    else:
        assert ndf.n_iter == 17
        assert ndf.n_eval == {'f': 20, 'j': 18, 'h': 17, 'hp': 0, 'fd_f': 2}

    assert ndf.metrics['obj'] < 0.04
    assert ndf.metrics['norm_g'] < 1e-6
    assert ndf.metrics['scl_norm_g'] < 1e-6

# Interface checks


def test_iface_too_tight(no_fortran):
    """Finite difference test tolerance too tight"""
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    tol = 1e-10
    n_coef = 2
    n_res = 5
    x = np.array([2.5, 0.25])
    w = 0.12 * np.array([1, 1, 1, 1, 1])
    ndf = nlls(n_coef, n_res, weights=w, verbose=0,
               check_derivatives='yes')
    try:
        ndf.fit(x, exp_r, exp_J, data=exp_data, fd_ttol=tol)

    # check expected results
    except RuntimeError as e:
        print('Ok')
    except Exception as e:
        raise AssertionError(
            "Did not catch the expected exception for FLOAT test") from e
    else:
        raise AssertionError(
            "Did not catch the expected exception for FLOAT test")


def test_iface_bad_weights(no_fortran):
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    n_coef = 2
    n_res = 5
    x = np.array([2.5, 0.25])
    w = -0.12 * np.array([1, 1, 1, 1, 1], dtype=np.float64)
    ndf = nlls(n_coef, n_res, weights=w)
    try:
        ndf.fit(x, exp_r, data=exp_data)
    except RuntimeError as e:
        print('Ok')
    except Exception as e:
        raise AssertionError(
            "Did not catch the expected exception for FLOAT test") from e
    else:
        raise AssertionError(
            "Did not catch the expected exception for FLOAT test")


def test_iface_wrong_option(no_fortran):
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    n_coef = 2
    n_res = 5
    x = np.array([2.5, 0.25])
    w = 0.12 * np.array([1, 1, 1, 1, 1])
    blx = np.array([0.0, 0.0])
    bux = np.array([5.0, 3.0])
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx,
               upper_bounds=bux, model="invalid")
    try:
        ndf.fit(x, exp_r, data=exp_data)
    except RuntimeError as e:
        print("Ok")
    except Exception as e:
        raise AssertionError(
            "Did not catch the expected exception") from e
    else:
        raise AssertionError(
            "Did not catch the expected exception")


def test_iface_x(no_fortran):
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    n_coef = 2
    n_res = 5
    x = np.array([2.5, 0.25], dtype=np.float32)
    w = 0.12 * np.array([1, 1, 1, 1, 1])
    blx = np.array([0.0, 0.0])
    bux = np.array([5.0, 3.0])
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx, upper_bounds=bux)
    try:
        ndf.fit(x, r_fail, J_fail)
    except RuntimeError as e:
        print('Ok')
    except Exception as e:
        raise AssertionError(
            "Did not catch the expected exception") from e
    else:
        raise AssertionError(
            "Did not catch the expected exception")


def test_warning(no_fortran):
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    n_coef = 2
    n_res = 5
    x = np.array([2.5, 0.25])
    w = 0.12 * np.array([1, 1, 1, 1, 1])
    blx = np.array([0.0, 0.0])
    bux = np.array([5.0, 3.0])
    ndf = nlls(n_coef, n_res, weights=w,
               lower_bounds=blx, upper_bounds=bux)

    with pytest.warns(RuntimeWarning):
        ndf.fit(x, exp_r, exp_J, exp_Hr, data=exp_data,
                abs_gtol=1e-7, gtol=1.e-9, maxit=1)


def test_nan(no_fortran):
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    abs_gtol = 1e-7
    gtol = 1.e-9
    maxit = 20
    n_coef = 2
    n_res = 5
    x = np.array([2.5, np.nan])
    w = 0.12 * np.array([1, 1, 1, 1, 1])
    blx = np.array([np.nan, 0.0])
    bux = np.array([5.0, 3.0])
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx, upper_bounds=bux,
               verbose=0, check_derivatives='yes', check_data=True)
    with pytest.raises(RuntimeError):
        ndf.fit(x, exp_r, exp_J, exp_Hr, data=exp_data,
                abs_gtol=abs_gtol, gtol=gtol, maxit=maxit)
    blx = np.array([0.0, 0.0])
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx, upper_bounds=bux,
               verbose=0, check_derivatives='yes', check_data=True)
    with pytest.raises(RuntimeError):
        ndf.fit(x, exp_r, exp_J, exp_Hr, data=exp_data,
                abs_gtol=abs_gtol, gtol=gtol, maxit=maxit)


@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, 'object'])
def test_nlls_all_dtypes(no_fortran, numpy_precision):
    """
    Test it runs when supported/unsupported C-interface type is provided.
    """

    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    abs_gtol = 1e-7
    gtol = 1.e-7
    maxit = 100
    n_coef = 2
    n_res = 5
    x = np.array([0.5, 2.5], dtype=numpy_precision)
    w = 0.12 * np.array([1, 1, 1, 1, 1], dtype=numpy_precision)
    blx = np.array([0.0, 0.0], dtype=numpy_precision)
    bux = np.array([7.0, 9.0], dtype=numpy_precision)
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx, upper_bounds=bux,
               verbose=0, check_derivatives='yes')
    ndf.fit(x, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)


@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders",
                         [("F", "C", "C", "C", "F"),
                          ("F", "C", "C", "F", "C"),
                          ("F", "C", "C", "F", "F"),
                          ("F", "C", "F", "C", "C"),
                          ("F", "C", "F", "C", "F"),
                          ("F", "C", "F", "F", "C"),
                          ("F", "C", "F", "F", "F"),
                          ("C", "F", "C", "C", "C"),
                          ("C", "F", "C", "C", "F"),
                          ("C", "F", "C", "F", "C"),
                          ("C", "F", "C", "F", "F"),
                          ("C", "F", "F", "C", "C"),
                          ("C", "F", "F", "C", "F"),
                          ("C", "F", "F", "F", "C")])
def test_nlls_multiple_orders(no_fortran, numpy_precision, numpy_orders):
    """
    Test it runs when arrays of multiple orders are provided.
    """

    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    abs_gtol = 1e-7
    gtol = 1.e-4
    maxit = 100
    n_coef = 2
    n_res = 5
    x = np.array([0.5, 2.5], dtype=numpy_precision, order=numpy_orders[1])
    w = 0.12 * np.array([1, 1, 1, 1, 1], dtype=numpy_precision, order=numpy_orders[2])
    blx = np.array([0.0, 0.0], dtype=numpy_precision, order=numpy_orders[3])
    bux = np.array([7.0, 9.0], dtype=numpy_precision, order=numpy_orders[4])

    # No warnings expected due to arrays being 1d
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx, upper_bounds=bux,
               verbose=0, check_derivatives='yes')
    ndf.fit(x, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)
    x2 = np.array([0.5, 2.5], dtype=numpy_precision, order=numpy_orders[0])
    ndf.fit(x2, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)

    bux = np.array([0, 8.0], dtype=numpy_precision, order=numpy_orders[4])
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=None, upper_bounds=bux,
               verbose=0, check_derivatives='yes')
    ndf.fit(x, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)
    x2 = np.array([0.5, 2.5], dtype=numpy_precision, order=numpy_orders[0])
    ndf.fit(x2, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)

    x = np.array([0.5, 2], dtype=numpy_precision, order=numpy_orders[1])
    w = 0.01 * np.array([1, 1, 1, 1, 1], dtype=numpy_precision, order=numpy_orders[2])
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx, upper_bounds=None,
               verbose=0, check_derivatives='yes')
    ndf.fit(x, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-4)
    x2 = np.array([0.5, 2], dtype=numpy_precision, order=numpy_orders[0])
    ndf.fit(x2, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-4)

    x = np.array([0.5, 2.5], dtype=numpy_precision, order=numpy_orders[1])
    ndf = nlls(n_coef, n_res, weights=None, lower_bounds=None, upper_bounds=None,
               verbose=0, check_derivatives='yes')
    ndf.fit(x, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)
    x2 = np.array([0.5, 2.5], dtype=numpy_precision, order=numpy_orders[0])
    ndf.fit(x2, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)


@pytest.mark.parametrize(
    "numpy_precisions", [('float64', 'float32', 'float32', 'float32', 'float64'),
                         ('float64', 'float32', 'float32', 'float64', 'float32'),
                         ('float64', 'float32', 'float32', 'float64', 'float64'),
                         ('float64', 'float32', 'float64', 'float32', 'float32'),
                         ('float64', 'float32', 'float64', 'float32', 'float64'),
                         ('float64', 'float32', 'float64', 'float64', 'float32'),
                         ('float64', 'float32', 'float64', 'float64', 'float64'),
                         ('float32', 'float64', 'float32', 'float32', 'float32'),
                         ('float32', 'float64', 'float32', 'float32', 'float64'),
                         ('float32', 'float64', 'float32', 'float64', 'float32'),
                         ('float32', 'float64', 'float32', 'float64', 'float64'),
                         ('float32', 'float64', 'float64', 'float32', 'float32'),
                         ('float32', 'float64', 'float64', 'float32', 'float64'),
                         ('float32', 'float64', 'float64', 'float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_nlls_multiple_dtypes(no_fortran, numpy_precisions, numpy_order):
    """
    Test it runs when arrays of multiple dtypes are provided.
    """

    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    abs_gtol = 1e-7
    gtol = 1.e-4
    maxit = 100
    n_coef = 2
    n_res = 5
    x = np.array([0.5, 2.5], dtype=numpy_precisions[1], order=numpy_order)
    w = 0.12 * np.array([1, 1, 1, 1, 1], dtype=numpy_precisions[2], order=numpy_order)
    blx = np.array([0.0, 0.0], dtype=numpy_precisions[3], order=numpy_order)
    bux = np.array([7.0, 9.0], dtype=numpy_precisions[4], order=numpy_order)

    ndf = nlls(n_coef, n_res, weights=None, lower_bounds=blx, upper_bounds=bux,
               verbose=0, check_derivatives='yes')
    ndf.fit(x, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)
    x2 = np.array([0.5, 2.5], dtype=numpy_precisions[0], order=numpy_order)
    ndf.fit(x2, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)

    bux = np.array([0, 8.0], dtype=numpy_precisions[4], order=numpy_order)
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=None, upper_bounds=bux,
               verbose=0, check_derivatives='yes')
    ndf.fit(x, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)
    x2 = np.array([0.5, 2.5], dtype=numpy_precisions[0], order=numpy_order)
    ndf.fit(x2, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)

    x = np.array([0.5, 2], dtype=numpy_precisions[1], order=numpy_order)
    w = 0.02 * np.array([1, 1, 1, 1, 1], dtype=numpy_precisions[2], order=numpy_order)
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx, upper_bounds=None,
               verbose=0, check_derivatives='yes')
    ndf.fit(x, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)
    x2 = np.array([0.5, 5], dtype=numpy_precisions[0], order=numpy_order)
    ndf.fit(x, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-10)

    x = np.array([0.5, 2.5], dtype=numpy_precisions[1], order=numpy_order)
    ndf = nlls(n_coef, n_res, weights=None, lower_bounds=None, upper_bounds=None,
               verbose=0, check_derivatives='yes')
    ndf.fit(x, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)
    x2 = np.array([0.5, 2.5], dtype=numpy_precisions[0], order=numpy_order)
    ndf.fit(x2, exp_r, data=exp_data,
            abs_gtol=abs_gtol, gtol=gtol, maxit=maxit, fd_step=2.e-7)
