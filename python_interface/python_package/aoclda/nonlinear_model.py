# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and / or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

# pylint: disable=import-error, invalid-name, too-many-arguments, missing-module-docstring,too-many-locals
from inspect import signature
from ._aoclda.nlls import pybind_nlls


class nlls(pybind_nlls):
    r"""
    Nonlinear data fitting.

    This class defines a model and solves the problem

    .. math::
         \underset{\text{subject to} x \in R^{n_{coef}}}{\text{minimize}}
         F(x) = \frac{1}{2} \sum_{i=0}^{n_{res-1}} r_i(x)^2_W + \frac{\sigma}{p} ||x||_2^p

    where

    - :math:`x` is the :math:`n_{coef}`-vector of coefficients (parameters) to be trained,

    - :math:`r_i(x)` is the :math:`n_{res}`-vector of model residuals evaluated at :math:`x`,

    - :math:`\sigma > 0`, and :math:`p=2,3` are the regularization hyperparameters, and

    - :math:`W` is a vector of the diagonal elements of the weight norm matrix.

    Args:

        n_coef (int): Number of coefficient in the model, must be positive.

        n_res (int): Number of residuals in the model, must also be positive.

        weights (float, optional): Vector containing the values of the diagonal weight matrix ``W``.
            It is expected that the values are all non-negative and normalized. Default = ``None``.

        lower_bounds (float, optional): Vector of lower bounds of the coefficient vector.
            Default = ``None``.

        upper_bounds (float, optional): Vector of upper bounds of the coefficient vector.
            Default = ``None``.

        order (str, optional): defines the storage scheme for the matrices.
            Default = ``'c'``.

            - ``'c'`` (row-major),
            - ``'fortran'`` (column-major)

        prec (str, optional): defines the data precision to use.
            Default = ``'double'``.

            - ``'double'`` uses floats in `float64` format,
            - ``'single'`` uses floats in `float32` format.

        model (str, optional): Choose the nonlinear model to use.  Default = ``'hybrid'``.

            - ``'gauss-newton'`` Gauss-Newton method,
            - ``'quasi-newton'`` Quasi-Newton method,
            - ``'hybrid'`` uses a strategy that allows to switch between Gauss-Newton
              and Quasi-Newton methods,
            - ``'tensor-newton'`` higher order method that uses Hessians. This option
              requires to have call-backs for second order derivatives Hessians.

        glob_strategy (str, optional): Globalization strategy. Only relevant when ``model`` =
            ``'hybrid'``, ``'gauss-newton'``, or ``'quasi-newton'``.  Default = ``'tr'``.

            - ``'tr'`` uses trust-region methods,
            - ``'reg'`` uses regularization techniques.

        method (str, optional): Optimization solver for the subproblem.  Default = ``'galahad'``.

            - Valid option values for when ``glob_strategy`` = ``'tr'``:

              - ``'galahad'`` GALAHAD's DTRS,
              - ``'more-sorensen'`` variant of More-Sorensen's method,
              - ``'powell-dogleg'`` Powell's dog-leg method.
              - ``'aint'`` Generalized eigen value method.

            - Valid option values for when ``glob_strategy`` = ``'reg'``:

              - ``'galahad'`` GALAHAD's DRQS.
              - ``'linear solver'`` solve step system using a linear solver,

            Refer to :cite:p:`ralfit` for details on these subproblem solvers.

        reg_power (str, optional): Regularization power (p).  Default = ``'quadratic'``.
            This option is only relevant if the `regularization term` (parameter ``reg_term`` in
            :py:meth:`~nlls.fit`) is positive.

            - ``'quadratic'`` uses second order regularization,
            - ``'cubic'`` uses third order regularization.

        check_derivatives (str, optional): Verify the user-supplied jacobian
            derivatives using finite-differences method.

            - ``'no'``  will not perform any verifiations
            - ``'yes'`` will trigger the verification on each element of the
              Jacobian matrix. The verification requires one call to the
              residual function for each column of the Jacobian.
              The verification process will print one line for each Jacobian
              entry that is deemed to be wrong in a similar format to

              ``Jac[2,6] =   1.28402 ~ 1.582  [ 2.097E-01], ( 0.250E-06) XT``

              informing with an ``X`` that the entry is likely to be wrong.
              It also informs with a ``T`` that the entry at ``Jac[2,6]``
              matches the correct value in the transposed Jacobian,
              indicating that the storage scheme should be reviewed (storing
              Fortran format instead of C format, or vice-versa).
              Storage scheme is defined with the ``order`` argument and
              defaults to ``c``.
              The If verbose level is high then one line for each entry
              of the whole Jacobian is printed regardless of its correctness.

              The finite-difference and derivative checker machinery
              is controlled via two arguments of the fit function:

              ``fd_step`` which defines the overall step size for the
              finite-difference approximation, and

              ``fd_ttol`` the derivative test tolerance which sets the
              threshold for deciding if a user-supplied derivative is
              close to the finite-difference approximation.

              If false positives are reported by the derivative checker or if
              the optimization process fails, then trialing
              different values for this parameter may help.

        verbose (int, optional): Set verbosity level (0 to 3) of the solver. Default = 0.
    """

    def __init__(self, n_coef, n_res, weights=None, lower_bounds=None, upper_bounds=None,
                 order='c', prec='double', model='hybrid', method='galahad', glob_strategy='tr',
                 reg_power='quadratic', check_derivatives='no', verbose=0):
        super().__init__(n_coef=n_coef, n_res=n_res, weights=weights,
                         lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                         order=order, prec=prec, model=model, method=method,
                         glob_strategy=glob_strategy, reg_power=reg_power,
                         check_derivatives=check_derivatives, verbose=verbose)

    def fit(self, x, fun, jac=None, hes=None, hep=None, data=None, ftol=1.0e-8, abs_ftol=1.0e-8,
            gtol=1.0e-8, abs_gtol=1.0e-5, xtol=2.22e-16, reg_term=0, maxit=100,
            fd_step=1.0e-7, fd_ttol=1.0e-4):
        """
        Fit data to a nonlinear model using regularized least-squares.

        Args:

            x (NDArray): initial guess to start optimizing from

            fun (method): function that calculates the ``n_res`` residuals.
                This function must return the residual vector of the model evaluated
                at the point ``x``.
                This function has the interface

                :code:`def fun(x, residuals, data) -> int:`

                .. note::

                    * The function should only modify **in-place** the residual vector.

                    * Must return 0 on success and nonzero otherwise.

                Example:

                .. code-block:: python

                    def fun(x, r, data) -> int:
                        r[:] = data.residuals(x)
                        return 0;

            jac (method): function that calculates the residual Jacobian matrix:
                This function must return the Jacobian matrix of size
                ``n_res`` by ``n_coef`` of the model evaluated
                at the point ``x``.
                This function has the interface

                :code:`def jac(x, jacobian, data) -> int:`

                .. note::

                    * The function should only modify **in-place** the Jacobian matrix.

                    * Must return 0 on success and nonzero otherwise.

                Example:

                .. code-block:: python

                    def jac(x, j, data) -> int:
                        j[:] = data.Jacobian(x)
                        return 0;

            hes (method, optional): function that calculates the ``n_coef`` by
                ``n_coef`` symmetric residual Hessian matrix: H = sum_i r_i H_i.
                This function has the interface

                :code:`def hes(x, r, h, data) -> int:`

                Default = `None`.

                .. note::

                    * The function should only modify **in-place** the Hessian matrix.

                    * Must return 0 on success and nonzero otherwise.


                Example:

                .. code-block:: python

                    def hes(x, r, h, data) -> int:
                        n = data['n_coef'];
                        Hi = data['hessians']
                        h[:] = Hi.sum(n, x, r)
                        return 0;

            hep (method, optional): function that calculates the matrix-vector
                product of the symmetric residual Hessian matrices with a vector y:
                hp[:] = [ H1(x)*y, H2(x)*y, ..., Hn(x)*y]. This resulting matrix
                is of size number of coefficients by number of residuals.
                This function has the interface

                :code:`def hep(x, y, hp, data) -> int:`


                Default = ``None``.

                .. note::

                    * The function should only modify **in-place** the ``hp``
                      Hessian-vector product matrix.

                    * Must return 0 on success and nonzero otherwise.

                Example:

                .. code-block:: python

                    def hep(x, y, hp, data) -> int:
                        n = data['n_coef'];
                        m = data['n_res'];
                        Hi = data['hessians']
                        hp[:] = Hi.prod(n, m, x, y)
                        return 0

            data (optional) user data object to pass to the user functions.
                The solver does not read or write to this object.
                Default = ``None``.

            ftol (float, optional): Defines the relative tolerance for the
                residual norm to declare convergence. Default = 1.0e-8.

            abs_ftol (float, optional): Defines the absolute tolerance for the
                residual norm to declare convergence. Default = 1.0e-8.

            gtol (float, optional): Defines the relative tolerance of the
                gradient norm to declare convergence. Default = 1.0e-8.

            abs_gtol (float, optional): Defines the absolute tolerance of the
                gradient norm to declare convergence. Default = 1.0e-5.

            xtol (float, optional): Defines the tolerance of the step length of
                two consecutive iterates to declare convergence. Default = 2.22e-16.

            reg_term (float, optional): Defines the regularization penalty term
                (sigma).  Default = 0.

            maxit (int, optional): Defines the iteration limit.
                Default = 100.

            fd_step (float, optional): Defines the overall step size to use in
            the finite-difference approximation. If the optimization process
            stagnates or fails, then trialing different values for this
            parameter may help.

            fd_ttol (float, optional): Derivative test tolerance, sets the
            threshold for deciding if a user-supplied derivative is acceptably
            close to the finite-difference approximation.

        Returns:
            self (object): Returns the fitted model, instance itself.
        """

        # inspect some parameter before entering c++
        self.__cb_inspect(fun, 3)
        if jac is not None:
            self.__cb_inspect(jac, 3)
        if hes is not None:
            self.__cb_inspect(hes, 4)
        if hep is not None:
            self.__cb_inspect(hep, 4)
        # Separate calls, pybind is calling the wrong specialization
        if pybind_nlls._get_precision(self) == "double":
            pybind_nlls.fit_d(self, x=x, fun=fun, jac=jac, hes=hes,
                              hep=hep, data=data, ftol=ftol, abs_ftol=abs_ftol,
                              gtol=gtol, abs_gtol=abs_gtol, xtol=xtol,
                              reg_term=reg_term, maxit=maxit, fd_step=fd_step,
                              fd_ttol=fd_ttol)
        else:
            pybind_nlls.fit_s(self, x=x, fun=fun, jac=jac, hes=hes,
                              hep=hep, data=data, ftol=ftol, abs_ftol=abs_ftol,
                              gtol=gtol, abs_gtol=abs_gtol, xtol=xtol,
                              reg_term=reg_term, maxit=maxit, fd_step=fd_step,
                              fd_ttol=fd_ttol)
        return self

    @property
    def n_iter(self):
        """
        int: number of iterations the solver made.
        """
        return pybind_nlls.get_info_iter(self)

    @property
    def n_eval(self):
        """
        (dict) [nevalf, nevalg, nevalh, nevalhp, nevalfd]: dictionary with the
        number of function calls for:

        - ``f`` (int): Residual function ``fun`` (includes ``fd_f`` count as well).
        - ``j`` (int): Gradient function ``jac``.
        - ``h`` (int): Hessian function ``hes``.
        - ``hp`` (int): Hessian-vector product function ``hep``.
        - ``fd_f`` (int): Residual function ``fun`` due to finite-differences or derivative checker.
        """
        return pybind_nlls.get_info_evals(self)

    @property
    def metrics(self):
        """
        (dict) [obj, nrmg, sclg]: dictionary with the metrics:

        - ``obj`` (float): Objective value at iterate ``x``.
        - ``norm_g`` (float): Norm of the residual gradient at iterate ``x``.
        - ``scl_norm_g`` (float): Norm of the scaled residual gradient at ``x``.
        """
        return pybind_nlls.get_info_optim(self)

    def __cb_inspect(self, f: callable, narg: int):
        """Internal helper function"""
        sig = signature(f)
        pos_cnt = sum(1 for param in sig.parameters.values()
                      if param.kind == param.POSITIONAL_OR_KEYWORD)
        if pos_cnt < narg:
            raise ValueError(f"Function {f} has {pos_cnt} positional arguments " +
                             "but must have at least {narg}.")
        if sig.return_annotation is not int:
            raise ValueError(f"Function {f} must return an integer.")
