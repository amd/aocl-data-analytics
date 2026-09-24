# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
Nonlinear least-squares tests, check output of sppatch versus scipy.optimize
"""

# pylint: disable = import-outside-toplevel, reimported, no-member, invalid-name
# pylint: disable = consider-using-from-import, too-many-locals, unused-argument
# pylint: disable = unexpected-keyword-arg
# pylint: disable = missing-function-docstring, redefined-outer-name

import numpy as np
import pytest
from aoclda.scipy import sppatch, undo_sppatch


@pytest.fixture(scope="function")
def no_fortran(request):
    return request.config.no_fortran


# Fit the model y_i = x_1 * exp(x_2 * t_i) to the data below
_T = np.array([1.0, 2.0, 4.0, 5.0, 8.0])
_Y = np.array([3.0, 4.0, 6.0, 11.0, 20.0])


def _fun(x):
    return x[0] * np.exp(x[1] * _T) - _Y


def _jac(x):
    return np.column_stack((np.exp(x[1] * _T),
                            _T * x[0] * np.exp(x[1] * _T)))


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_least_squares_analytic_jac(no_fortran, precision):
    """
    Solve with a user-supplied Jacobian and compare against scipy.
    """
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    tol = 1.0e-4 if precision == np.float32 else 1.0e-5
    solver_tol = 1.0e-4 if precision == np.float32 else 1.0e-8

    x0 = np.array([2.5, 0.25], dtype=precision)

    sppatch()
    import scipy.optimize as optimize
    da_res = optimize.least_squares(_fun, x0.copy(), jac=_jac,
                                    ftol=solver_tol, xtol=solver_tol,
                                    gtol=solver_tol)

    undo_sppatch()
    import scipy.optimize as optimize
    sk_res = optimize.least_squares(_fun, x0.copy(), jac=_jac,
                                    ftol=solver_tol, xtol=solver_tol,
                                    gtol=solver_tol)

    assert da_res.success
    assert da_res.x == pytest.approx(sk_res.x, abs=tol)
    assert da_res.cost == pytest.approx(sk_res.cost, rel=tol, abs=tol)


def test_least_squares_finite_difference(no_fortran):
    """
    Solve using the solver's finite-difference Jacobian
    """
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    x0 = np.array([2.5, 0.25])

    sppatch()
    import scipy.optimize as optimize
    da_res = optimize.least_squares(_fun, x0.copy())

    undo_sppatch()
    import scipy.optimize as optimize
    sk_res = optimize.least_squares(_fun, x0.copy())

    assert da_res.success
    assert da_res.x == pytest.approx(sk_res.x, abs=1.0e-4)


def test_least_squares_bounds(no_fortran):
    """
    Solve a bound-constrained problem and compare against scipy
    """
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    x0 = np.array([2.5, 0.25])
    bounds = ([0.0, 0.0], [5.0, 3.0])

    sppatch()
    import scipy.optimize as optimize
    da_res = optimize.least_squares(_fun, x0.copy(), jac=_jac, bounds=bounds)

    undo_sppatch()
    import scipy.optimize as optimize
    sk_res = optimize.least_squares(_fun, x0.copy(), jac=_jac, bounds=bounds)

    assert da_res.success
    assert da_res.x == pytest.approx(sk_res.x, abs=1.0e-4)


@pytest.mark.parametrize("reg_power, power", [('quadratic', 2), ('cubic', 3)])
def test_least_squares_args(no_fortran, reg_power, power):
    """
    Check forwarded arguments and all result fields for regularized problems.
    """
    if no_fortran:
        pytest.skip("Skipping test due to no_fortran flag")

    target = 2.0
    weight = 1.5
    reg_term = 0.2

    def fun(x, target):
        return np.array([x[0] - target])

    def jac(x, target):
        return np.ones((1, 1), dtype=x.dtype)

    sppatch()
    import scipy.optimize as optimize
    da_res = optimize.least_squares(
        fun, np.array([1.0]), jac=jac, args=(target,),
        x_scale=np.array([weight]),
        reg_term=reg_term, reg_power=reg_power)
    undo_sppatch()

    weight_squared = weight**2
    if power == 2:
        expected_x = weight_squared * target / (weight_squared + reg_term)
    else:
        discriminant = weight_squared**2 + 4 * reg_term * weight_squared * target
        expected_x = (-weight_squared + np.sqrt(discriminant)) / (2 * reg_term)

    expected_fun = np.array([expected_x - target])
    expected_jac = np.ones((1, 1))
    expected_cost = (0.5 * weight_squared * np.dot(expected_fun, expected_fun)
                     + reg_term / power * expected_x**power)
    expected_grad = np.array([
        weight_squared * (expected_x - target)
        + reg_term * expected_x**(power - 1)
    ])

    assert set(da_res.keys()) == {
        'x', 'cost', 'fun', 'jac', 'grad', 'optimality', 'active_mask',
        'nfev', 'njev', 'status', 'message', 'success'
    }
    assert da_res.x == pytest.approx([expected_x], abs=1.0e-5)
    assert da_res.cost == pytest.approx(expected_cost, abs=1.0e-5)
    assert da_res.fun == pytest.approx(expected_fun, abs=1.0e-5)
    assert da_res.jac == pytest.approx(expected_jac)
    assert da_res.grad == pytest.approx(expected_grad, abs=1.0e-5)
    assert da_res.optimality == pytest.approx(np.max(np.abs(expected_grad)),
                                              abs=1.0e-5)
    assert da_res.active_mask == pytest.approx([0])
    assert isinstance(da_res.nfev, int) and da_res.nfev > 0
    assert isinstance(da_res.njev, int) and da_res.njev > 0
    assert da_res.status == 1
    assert da_res.message == "The solver converged to a solution."
    assert da_res.success


def test_patching():
    """
    Test we can patch and unpatch least_squares
    """

    sppatch("least_squares")
    import scipy.optimize as optimize
    assert optimize.least_squares.__module__.startswith("aoclda")

    undo_sppatch("least_squares")
    import scipy.optimize as optimize
    assert not optimize.least_squares.__module__.startswith("aoclda")


if __name__ == "__main__":
    test_patching()
