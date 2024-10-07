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

# pylint: disable = import-error, anomalous-backslash-in-string, invalid-name, too-many-arguments

"""
aoclda.linear_model module
"""

from ._aoclda.linear_model import pybind_linmod


class linmod(pybind_linmod):
    """
    Linear models.

    Args:

        mod (str): Which linear model to compute.

            - If ``linmod_model='mse'`` then :math:`\ell_2` norm linear regression is calculated.

            - If ``linmod_model='logistic'`` then logistic regression is calculated.

        intercept (bool, optional): Controls whether to add an intercept variable to the model.
            Default=False.

        solver (str, optional): Which solver to use to compute the coefficients. It can \
            take values 'auto', 'svd', 'cholesky', 'sparse_cg', 'qr', 'coord', 'lbfgs'. \
            Some solvers are not suitable for some regularization types.

            - ``'auto'`` chooses the best solver based on regression type and data.

            - ``'svd'`` works with normal and ridge regression. The most robust solver but the \
                least efficient.

            - ``'cholesky'`` works with normal and ridge regression. Will return an error when a \
                singular matrix is encountered.

            - ``'sparse_cg'`` works with normal and ridge regression. You may need to set a \
                smaller `tol` if a badly conditioned matrix is encountered.

            - ``'qr'`` works with normal linear regression only. Will return an error when an \
                underdetermined system is encountered.

            - ``'coord'`` works with all regression types. Requires data to have variance of 1 \
                column-wise (this can be achieved with the `scaling` option set to `scale only`). \
                In the case of normal linear regression and an underdetermined system it\
                will converge to a solution that is not necessarily a minimum norm solution.

            - ``'lbfgs'`` works with normal and ridge regression. In the case of normal linear \
                regression and an underdetermined system it will converge to a solution that is \
                not necessarily a minimum norm solution.

        scaling (str, optional): What type of preprocessing you want to apply on the dataset. \
            Available options are: 'none', 'centering', 'scale_only', 'standardize'.

        max_iter (int, optional): Maximum number of iterations. Applies only to iterative \
            solvers: 'sparse_cg', 'coord', 'lbfgs'. The default value depends on the solver. For \
            'sparse_cg' it is 500, for 'lbfgs' and 'coord' it is 10000.

        constraint (str, optional): Affects only multinomial logistic regression. \
            The type of constraint put on coefficients. This will affect the number of \
            coefficients returned.

            - ``'rsc'`` means we choose a reference category whose coefficients \
                will be set to all 0. This results in K-1 class coefficients for K \
                class problems.

            - ``'ssc'`` means the sum of coefficients class-wise for each feature \
                is 0. It will result in K class coefficients for K class problems.

        precision (str, optional): Whether to compute the linear model in double or
            single precision. It can take the values 'single' or 'double'.
            Default = 'double'.
    """

    # This is done to change the order of parameters, in pybind usage of std::optional for max_iter
    # made it necessary to put it in the front, but more natural position is near the end.
    def __init__(self,
                 mod,
                 intercept=False,
                 solver='auto',
                 scaling='auto',
                 max_iter=None,
                 constraint='ssc',
                 precision='double'):
        super().__init__(mod=mod,
                         max_iter=max_iter,
                         intercept=intercept,
                         solver=solver,
                         scaling=scaling,
                         constraint=constraint,
                         precision=precision)

    def fit(self,
            X,
            y,
            reg_lambda=0.0,
            reg_alpha=0.0,
            x0=None,
            tol=0.0001,
            progress_factor=None):
        """
        Computes the chosen linear model on the feature matrix ``X`` and response vector ``y``

        Args:
            X (numpy.ndarray): The feature matrix on which to compute the model.
                Its shape is (n_samples, n_features).

            y (numpy.ndarray): The response vector. Its shape is (n_samples).

            reg_lambda (float, optional): :math:`\lambda`, the magnitude of the regularization term.
                Default=0.0.

            reg_alpha (float, optional): :math:`\\alpha`, the share of the :math:`\ell_1` \
                term in the regularization.

            x0 (numpy.ndarray, optional): Initial guess for solution. Applies only to iterative \
                solvers

            tol (float, optional): Convergence tolerance for iterative solvers. Applies only \
                to iterative solvers: 'sparse_cg', 'coord', 'lbfgs'.

            progress_factor (float, optional): Applies only to 'lbfgs' and 'coord' solver. \
                Factor used to detect convergence of the iterative optimization step.

        Returns:
            self (object): Returns the instance itself.
        """
        self.pybind_fit(X,
                        y,
                        x0=x0,
                        progress_factor=progress_factor,
                        reg_lambda=reg_lambda,
                        reg_alpha=reg_alpha,
                        tol=tol)
        return self

    def predict(self, X):
        """
        Evaluate the model on a data set ``X``.

        Args:
            X (numpy.ndarray): The feature matrix to evaluate the model on. It must have \
                n_features columns.

        Returns:
            numpy.ndarray of length n_samples: The prediction vector, where n_samples is \
                the number of rows of ``X``.
        """
        return self.pybind_predict(X)

    @property
    def coef(self):
        """
        numpy.ndarray: contains the output coefficients of the model. If an intercept variable was
            required, it corresponds to the last element.
        """
        return self.get_coef()

    @property
    def loss(self):
        """numpy.ndarray of shape (1, ): The value of loss function :math:`L(\\beta_0, \\beta)`.
        """
        return self.get_loss()

    @property
    def nrm_gradient_loss(self):
        """numpy.ndarray of shape (1, ): The norm of the gradient of the loss function. Only valid \
            for iterative solvers.
        """
        return self.get_norm_gradient_loss()

    @property
    def n_iter(self):
        """int: The number iterations performed to find the solution.
        Only valid for iterative solvers.
        """
        return self.get_n_iter()

    @property
    def time(self):
        """numpy.ndarray of shape (1, ): Compute time (wall clock time in seconds).
        """
        return self.get_time()
