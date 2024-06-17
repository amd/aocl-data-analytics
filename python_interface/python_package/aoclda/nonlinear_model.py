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

# pylint: disable=import-error, invalid-name, too-many-arguments, missing-module-docstring
from inspect import signature
from typing import Optional, Any, Callable
from numpy.typing import NDArray
from ._aoclda.nlls import pybind_nlls


class nlls(pybind_nlls):
    """
    Nonlinear data fitting.

    Data Fitting for nonlinear least-squares
    =========================================

    This function defines the model and solves the problem

    :math:`minimize F(x) = 1/2 \\sum_{i=0}^{n_res-1} ri(x)^2_W + sigma/p ||x||_2^p`
    :math:`subject to x \\in R^{n_coef}`

    where

        `x` is the `n_coef`-vector of parameters to be trained,

        `ri(x)` is the `n_res`-vector of model residuals evaluated at x

        `sigma` > 0, and `p`=2,3 are the regularization hyperparams

        `W` is the (weight) norm marix

    Parameters
    ----------

        ``n_coef`` (int): Number of coefficient in the model, must be positive.

        ``n_res`` (int): Number of residuals in the model, must also be positive.

        ``weights`` (float, optional): Vector of weights containing the values of W.
            It is expected that the values are all non-negative and normalized. Default: None.

        ``lower_bounds`` (float, optional): Vector of lower bounds of the coefficient vector.
            Default = ``None``.

        ``upper_bounds (float, optional): Vector of upper bounds of the coefficient vector.
            Default = ``None``.

        ``order`` (str, optional): defines the storage scheme for the matrices.
            Default = "c" (row-major).

        ``prec`` (str, optional): defines the data precion to use. Default = "double".

        ``model`` (str, optional): Chose the nonlinear model to use. Default = "hybrid".

        ``method`` (str, optional): Optimization solver for the subproblem. Default = "galahad".

        ``glob_strategy`` (str, optional): Globalization strategy. Default = "tr".

        ``reg_power`` (str, optional): Regularization power. Default = "quadratic".

        ``verbose`` (int, optional): Set verbosity level (0 to 3) of the solver. Default = 0.

    Returns
    -------

        This function does not return a value.

    See Also
    --------

        nlls.fit(...)

        nlls.n_iter(...)

        nlls.metrics(...)

        nlls.n_eval(...)
    """

    def __init__(self, n_coef: int, n_res: int,
                 weights: Optional[NDArray] = None,
                 lower_bounds: Optional[NDArray] = None,
                 upper_bounds: Optional[NDArray] = None,
                 order: Optional[str] = "c",
                 prec: Optional[str] = "double",
                 model: Optional[str] = "hybrid",
                 method: Optional[str] = "galahad",
                 glob_strategy: Optional[str] = "tr",
                 reg_power: Optional[str] = "quadratic",
                 verbose: Optional[int] = 0):
        super().__init__(n_coef=n_coef, n_res=n_res, weights=weights,
                         lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                         order=order, prec=prec, model=model, method=method,
                         glob_strategy=glob_strategy, reg_power=reg_power,
                         verbose=verbose)

    def fit(self, x: NDArray,
            fun: Callable[[NDArray, NDArray, Any], int],
            jac: Callable[[NDArray, NDArray, Any], int],
            hes: Optional[Callable[[
                NDArray, NDArray, NDArray, Any], int]] = None,
            hep: Optional[Callable[[
                NDArray, NDArray, NDArray, Any], int]] = None,
            data: Any = None,
            ftol: float = 1.0e-8, abs_ftol: float = 1.0e-8,
            gtol: float = 1.0e-8, abs_gtol: float = 1.0e-5,
            xtol: float = 2.22e-16, reg_term: float = 0,
            maxit: int = 100):
        """
        Train a nonlinear data model.

        Fit data to a nonlinear model using regularized least-squares.

        Parameters
        ----------

            ``x`` (NDArray) initial guess to start optimizing from

            ``fun`` (method): function that calculates the n_res residuals.
                This function has the interface

                ``def fun(x, residuals, data):``

                This function must return the residual vector of the model evaluated
                at the point x.
                The function should only modify the residual vector.
                Must return 0 on success and nonzero otherwise.

                Example:

                ```{python}
                    def fun(x, r, data):
                        r[:] = data.residuals(x)
                        return 0;
                ```

            ``jac`` (method): function that calculates the residual Jacobian matrix:
                This function has the interface

                ``def jac(x, jacobian, data):``

                The function should only modify the Jacobian matrix.
                Must return 0 on success and nonzero otherwise.

                Example:

                ```{python}
                    def jac(x, j, data):
                        j[:] = data.Jacobian(x)
                        return 0;
                ```

            ``hes`` (method, optional): function that calculates the n_coef by n_coef
                symmetric residual Hessian matrix: H = sum_i r_i H_i.
                This function has the interface

                ``def hes(x, r, h, data):``

                The function should only modify the Hessian matrix.
                Must return 0 on success and nonzero otherwise.

                Default = `None`.

                Example:

                ```{python}
                    def hes(x, r, h, data):
                        n = data['n_coef'];
                        Hi = data['hessians']
                        h[:] = Hi.sum(x,r)
                        return 0;
                ```

            ``hep`` (method, optional): function that calculates the matrix-vector
                product of the symmetric residual Hessian matrices with a vector y.
                This function has the interface

                ``def hep(x, y, hp, data):``

                The function should only modify **in-place** the ``hp``
                Hessian-vector product matrix. Must return 0 on success and
                nonzero otherwise.

                Default = ``None``.

                Example:

                ```{python}
                    def hep(x, y, hp, data):
                        n = data['n_coef'];
                        m = data['n_res'];
                        Hi = data['hessians']
                        hp[:] = Hi.prod(x,y)
                        return 0
                ```

            ``data`` (optional) user data object to pass to the user functions.
                The solver does not read or write to this object.
                Default = ``None``.

            ``ftol`` (float, optional): Defines the relative tolerance for the
                residual norm to declare convergence. Default = 1.0e-8.

            ``abs_ftol`` (float, optional): Defines the absolute tolerance for the
                residual norm to declare convergence. Default = 1.0e-8.

            ``gtol`` (float, optional): Defines the relative tolerance of the
                gradient norm to declare convergence. Default = 1.0e-8.

            ``abs_gtol`` (float, optional): Defines the absolute tolerance of the
                gradient norm to declare convergence. Default = 1.0e-5.

            ``xtol`` (float, optional): Defines the tolerance of the step length of
                two consecutive iterates to declare convergence. Default = 2.22e-16.

            ``reg_term`` (float, optional): Defines the tolerance to declare
                convergence.  Default = 0.

            ``maxit`` (int, optional): Defines the tolerance to declare convergence.
                Default = 100.

        Returns
        -------

            This function does not return a value.

        See Also
        --------

            nlls(...)

            nlls.n_iter(...)

            nlls.metrics(...)

            nlls.n_eval(...)
        """

        # inspect some parameter before entering c++
        self.__cb_inspect(fun, 3)
        if jac is not None:
            self.__cb_inspect(jac, 3)
        if hes is not None:
            self.__cb_inspect(hes, 4)
        if hep is not None:
            self.__cb_inspect(hep, 4)

        pybind_nlls.fit(self, x=x, fun=fun, jac=jac, hes=hes,
                        hep=hep, data=data, ftol=ftol, abs_ftol=abs_ftol,
                        gtol=gtol, abs_gtol=abs_gtol, xtol=xtol,
                        reg_term=reg_term, maxit=maxit)

    @property
    def n_iter(self):
        """
        Property
        --------

        (int): Returns the number of iterations the solver made.

        See Also
        --------

        nlls(...)

        nlls.fit(...)

        nlls.metrics(...)

        nlls.n_eval(...)
        """
        return pybind_nlls.get_info_iter(self)

    @property
    def n_eval(self):
        """
        Property
        --------

        (dict) [nevalf, nevalg, nevalh, nevalhp].

            Returns a dictionary with the number of function calls for

            ``nevalf`` (int): Resifual function `fun`.

            ``nevalg`` (int): Gradient function `jac`.

            ``nevalh`` (int): Hessian function `hes`.

            ``nevalhp`` (int): Hessian-vector product function `hep`.

        See Also
        --------

        nlls(...)

        nlls.fit(...)

        nlls.n_iter(...)

        nlls.metrics(...)
        """
        return pybind_nlls.get_info_evals(self)

    @property
    def metrics(self):
        """
        Property
        --------

        (dict) [obj, nrmg, sclg].

            Returns a dictionary with the metrics:

            ``obj`` (float): Objective value at iterate `x`.

            ``nrmg`` (float): Norm of the residual gradient at iterate `x`.

            ``sclg`` (float): Norm of the scaled residual gradient at `x`.
        See Also
        --------

        nlls(...)

        nlls.fit(...)

        nlls.n_iter(...)

        nlls.n_eval(...)
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
