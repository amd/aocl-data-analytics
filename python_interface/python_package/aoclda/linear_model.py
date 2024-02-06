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

# pylint: disable = missing-module-docstring, unused-import

from ._aoclda.linear_model import pybind_linmod, linmod_model


class linmod(pybind_linmod):
    """
    Linear models.

    Args:

        linmod_model (aoclda.linear_model.linmod_model): Which linear model to compute.
            Can take the value 'mse' (mean squared error) or 'logistic'.

        intercept (bool, optional): Controls whether to add an intercept variable to the model.
            Default=False.

        precision (aoclda.precision, optional): Whether to compute the linear model in double or
            single precision. It can take the values ``aoclda.single`` or ``aoclda.double``.
            Default = ``aoclda.double``.
    """

    def fit(self, X, y, reg_lambda=0.0, reg_alpha=0.0):
        """
        Computes the chosen linear model on the feature matrix X and response vector y

        Args:
            X (numpy.ndarray): The feature matrix on which to compute the model.
                Its shape is (n_samples, n_features).

            y (numpy.ndarray): The response vector. Its shape is (n_samples).

            reg_lambda (float, optional): :math:`lambda`, the magnitude of the regularization term.
                Default=0.0.

            reg_alpha (float, optional): :math:`alpha`, the share of the :math:`\ell_1` term in the
                regularization.
        """
        self.pybind_fit(X, y, reg_lambda=reg_lambda, reg_alpha=reg_alpha)

    @property
    def coef(self):
        """
        numpy.ndarray: contains the output coefficients of the model. If an intercept variable was
            required, it corresponds to the last element.
        """
        return self.get_coef()
