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
Patching scipy.optimize nonlinear least-squares: least_squares
"""
# pylint: disable = too-many-positional-arguments, invalid-name, too-many-arguments
# pylint: disable = too-many-locals, too-many-branches, too-many-statements

import numpy as np
from scipy.optimize import OptimizeResult

from aoclda.nonlinear_model import nlls as nlls_da
from ._utils import _warn_if


def _process_bounds(bounds, n, dtype):
    """
    Convert scipy-style ``bounds`` into AOCL-DA ``lower_bounds`` / ``upper_bounds``.

    scipy accepts a 2-tuple ``(lb, ub)`` of scalars or arrays, or a ``Bounds``
    object. AOCL-DA expects 1D arrays (or ``None`` when a side is unbounded).
    """
    if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
        lb, ub = bounds.lb, bounds.ub
    else:
        lb, ub = bounds

    lb = np.broadcast_to(np.asarray(lb, dtype=dtype), (n,)).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=dtype), (n,)).copy()

    lower = None if np.all(np.isneginf(lb)) else lb
    upper = None if np.all(np.isposinf(ub)) else ub
    return lower, upper

# pylint: disable=unused-argument
def least_squares(fun, x0, jac='2-point', bounds=(-np.inf, np.inf),
                  method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8,
                  x_scale=None, loss='linear', f_scale=1.0, diff_step=None,
                  tr_solver=None, tr_options=None, jac_sparsity=None,
                  max_nfev=None, verbose=0, args=(), kwargs=None,
                  callback=None, workers=None,
                  model='hybrid', glob_strategy='tr',
                  subproblem_solver='galahad', reg_power='quadratic',
                  check_derivatives='no', reg_term=0.0, fd_ttol=1.0e-4):
    """
    Overwrite scipy.optimize.least_squares to call AOCL-DA library.

    Some of the standard SciPy arguments are silently ignored, these are:
    method, f_scale, tr_solver, tr_options.
    In addition, the jac_sparsity, callback, and workers arguments must be None.
    The following AOCL-DA specific options are exposed as keyword arguments 
    (they have no SciPy equivalent):

    model (str): nonlinear model, one of ``'gauss-newton'``,
        ``'quasi-newton'``, ``'hybrid'`` (default) or ``'tensor-newton'``.
    glob_strategy (str): globalization strategy, ``'tr'`` (trust-region,
        default) or ``'reg'`` (regularization).
    subproblem_solver (str): solver for the subproblem (maps to the AOCL-DA
        ``method`` option), e.g. ``'galahad'`` (default), ``'more-sorensen'``,
        ``'powell-dogleg'``, ``'aint'`` or ``'linear solver'``.
    reg_power (str): regularization power, ``'quadratic'`` (default) or
        ``'cubic'``. Only relevant when ``reg_term`` > 0.
    check_derivatives (str): verify the user-supplied Jacobian via
        finite-differences, ``'no'`` (default) or ``'yes'``.
    reg_term (float): regularization penalty term. Default 0 (no
        regularization).
    fd_ttol (float): derivative-test tolerance used together with
        ``check_derivatives``.
        
    Currently the storage order of the Jacobian is assumed to always be 
    'C' (row-major). This may change in the future.
    
    """
    # Warn about parameters that the AOCL-DA solver does not support
    _warn_if(diff_step is None and jac in ('3-point', 'cs'), "jac")
    _warn_if(jac_sparsity is not None, "jac_sparsity")
    _warn_if(callback is not None, "callback")
    _warn_if(workers is not None, "workers")
    if x_scale == 'jac':
        raise ValueError("x_scale='jac' is not supported by AOCL-DA.")

    if kwargs is None:
        kwargs = {}

    # Coerce the initial guess to a contiguous float array. This keeps the
    # array AOCL-DA writes the solution into identical to the one we read back
    # afterwards (the solver updates the coefficients in-place).
    x = np.atleast_1d(np.asarray(x0))
    dtype = np.float32 if x.dtype == np.float32 else np.float64
    x = np.ascontiguousarray(x, dtype=dtype)
    n_coef = x.shape[0]

    # Probe the residual function once to determine the number of residuals.
    r0 = np.asarray(fun(x, *args, **kwargs), dtype=dtype)
    n_res = r0.shape[0]

    # Get the residual weights
    weights = None if x_scale is None else np.atleast_1d(np.asarray(x_scale))
    if weights is not None:
        weights = np.ascontiguousarray(weights, dtype=dtype)

    jac_callable = callable(jac)

    def _fun_cb(x, residuals, data) -> int:
        residuals[:] = fun(x, *args, **kwargs)
        return 0

    def _jac_cb(x, jacobian, data) -> int:
        jacobian[:] = jac(x, *args, **kwargs)
        return 0

    lower_bounds, upper_bounds = _process_bounds(bounds, n_coef, dtype)

    ndf = nlls_da(n_coef, n_res,
                  weights=weights,
                  lower_bounds=lower_bounds,
                  upper_bounds=upper_bounds,
                  order='c',
                  model=model,
                  method=subproblem_solver,
                  glob_strategy=glob_strategy,
                  reg_power=reg_power,
                  check_derivatives=check_derivatives,
                  verbose=min(int(verbose), 3))

    fit_kwargs = {'ftol': ftol, 'gtol': gtol, 'xtol': xtol,
                  'reg_term': reg_term, 'fd_ttol': fd_ttol}
    if diff_step is not None:
        fit_kwargs['fd_step'] = diff_step
    if max_nfev is not None:
        fit_kwargs['maxit'] = max_nfev

    success = True
    status = 1
    message = "The solver converged to a solution."
    try:
        ndf.fit(x, _fun_cb, jac=_jac_cb if jac_callable else None,
                data=None, **fit_kwargs)
    except RuntimeError as exc:
        success = False
        status = 0
        message = str(exc)

    # Residuals (and optionally the Jacobian) at the returned solution
    res = np.asarray(fun(x, *args, **kwargs), dtype=dtype)

    try:
        cost = ndf.metrics.get('obj')
    except RuntimeError:
        cost = None

    if jac_callable:
        jac_sol = np.asarray(jac(x, *args, **kwargs), dtype=dtype)
        grad_res = res if weights is None else weights**2 * res
        grad = jac_sol.T @ grad_res
        if 0 < reg_term:
            if reg_power == 'quadratic':
                grad += reg_term * x
            else:
                grad += reg_term * np.linalg.norm(x) * x
    else:
        jac_sol = None
        grad = None

    try:
        optimality = ndf.metrics.get('norm_g')
    except RuntimeError:
        optimality = None

    try:
        nfev = ndf.n_eval.get('f')
    except RuntimeError:
        nfev = None

    try:
        njev = ndf.n_eval.get('j')
    except RuntimeError:
        njev = None


    return OptimizeResult(
        x=x,
        cost=cost,
        fun=res,
        jac=jac_sol,
        grad=grad,
        optimality=optimality,
        active_mask=np.zeros(n_coef, dtype=int),
        nfev=nfev,
        njev=njev,
        status=status,
        message=message,
        success=success,
    )
