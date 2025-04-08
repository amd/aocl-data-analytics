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

# pylint: disable = import-error, anomalous-backslash-in-string, invalid-name, too-many-arguments
"""
aoclda.svm module
"""
import numpy as np
from ._aoclda.svm import pybind_svc, pybind_svr, pybind_nusvc, pybind_nusvr


class BaseSVM:
    def __init__(
        self,
        kernel="rbf",
        degree=3,
        gamma=-1.0,
        coef0=0.0,
        tol=0.001,
        max_iter=0,
        tau=None,
        check_data=False,
    ):
        if max_iter == -1:
            max_iter = 0
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.tau = tau
        self.check_data = check_data
        self.precision = "double"  # default precision
        # Objects to bind with C++ backend (assigned by subclasses)
        self._model_double = None
        self._model_single = None
        self._model = None

    def fit(self, X, y, **kwargs):
        if kwargs.get("tau") is None:
            del kwargs["tau"]
        if X.dtype == np.float32:
            self._model = self._model_single
            self.precision = "single"
            for key in kwargs:
                kwargs[key] = np.float32(kwargs[key])
        else:
            self._model = self._model_double
            self.precision = "double"
            for key in kwargs:
                kwargs[key] = np.float64(kwargs[key])

        if y.dtype.kind in np.typecodes["AllInteger"]:
            y = y.astype(X.dtype, copy=False)

        self._model.pybind_fit(X, y, **kwargs)

        return self

    def predict(self, X):
        preds = self._model.pybind_predict(X)
        return preds

    def score(self, X, y):
        if y.dtype.kind in np.typecodes["AllInteger"]:
            y = y.astype(X.dtype, copy=False)
        return self._model.pybind_score(X, y)

    @property
    def n_samples(self):
        """
        int: The number of training samples used to fit the model.
        """
        return self._model.get_n_samples()

    @property
    def n_features(self):
        """
        int: The number of features in the training data.
        """
        return self._model.get_n_features()

    @property
    def n_iter(self):
        """
        int: The number of features in the training data.
        """
        return self._model.get_n_iterations()

    @property
    def n_support(self):
        """
        int: The total number of support vectors.
        """
        return self._model.get_n_sv()

    @property
    def n_support_per_class(self):
        """
        numpy.ndarray of shape (n_classes,): The number of support vectors for each class.
        """
        return self._model.get_n_sv_per_class()

    @property
    def dual_coef(self):
        """
        numpy.ndarray of shape (n_classes-1, n_support): The dual coefficients of the support vectors.
        """
        return self._model.get_dual_coef()

    @property
    def support_vectors_idx(self):
        """
        numpy.ndarray of shape (n_support,): The indices of the support vectors.
        """
        return self._model.get_support_vectors_idx()

    @property
    def support_vectors(self):
        """
        numpy.ndarray of shape (n_support, n_features): The support vectors used by the model.
        """
        return self._model.get_sv()

    @property
    def bias(self):
        """
        numpy.ndarray or float: The bias term(s) of the model (also known as the intercept).
        """
        return self._model.get_bias()


class SVC(BaseSVM):
    """
    Support Vector Classification.

    Train a C-Support Vector Classification model.

    Args:
        C (float, optional): Regularization parameter. Controls the trade-off between maximizing \
            the margin between classes and minimizing classification errors. A larger \
            value means higher penalty to the loss function on misclassified observations. Must be \
            strictly positive. Default=1.0.
        kernel (str, optional): Kernel type to use in the algorithm. Possible values: \
            'linear', 'poly', 'rbf', 'sigmoid'. Default='rbf'.
        degree (int, optional): Degree of the polynomial kernel function. Ignored by \
            all other kernels. Default=3.
        gamma (float, optional): Kernel coefficient. If set to -1, it is calculated as \
            :math:`1/(Var(X) * n\_features)`. Default=-1.0.
        coef0 (float, optional): Independent term in kernel function (check :func:`kernel functions \
            <aoclda.kernel_functions.polynomial_kernel>` for more details). It is only used \
            in 'poly' and 'sigmoid' kernel functions. Default=0.0.
        probability (bool, optional): Currently not supported. Whether to enable \
            probability estimates. Default=False.
        tol (float, optional): Tolerance for stopping criterion. Default=0.001.
        max_iter (int, optional): Hard limit on iterations within solver, or 0 for no limit. \
            Default=0.

        tau (float, optional): Numerical stability parameter. If it is None then machine \
            epsilon is used. Default=None.
        check_data (bool, optional): Whether to check data for NaNs. Default=False.
    """

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma=-1.0,
        coef0=0.0,
        probability=False,
        tol=0.001,
        max_iter=0,
        tau=None,
        check_data=False,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            max_iter=max_iter,
            tau=tau,
            check_data=check_data,
        )
        # Create backend objects with chosen precision
        self._model_double = pybind_svc(
            kernel=self.kernel,
            degree=self.degree,
            max_iter=self.max_iter,
            precision="double",
            check_data=self.check_data,
        )
        self._model_single = pybind_svc(
            kernel=self.kernel,
            degree=self.degree,
            max_iter=self.max_iter,
            precision="single",
            check_data=self.check_data,
        )
        self._model = self._model_double
        # Not supported yet
        self.probability = probability
        # Supported
        self.C = C

    def fit(self, X, y):
        """
        Fit the SVC model according to the given training data.

        Args:
            X (numpy.ndarray): Training vectors of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,). They are expected to range from 0 to \p n_class - 1.

        Returns:
            self (object): Returns the instance itself.
        """
        parameters = {"C": self.C, "gamma": self.gamma,
                      "coef0": self.coef0, "tol": self.tol, "tau": self.tau}
        super().fit(X, y, **parameters)
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted class labels for samples in X.
        """
        preds = super().predict(X)
        if self.precision == "double":
            preds = preds.astype(np.int64)
        else:
            preds = preds.astype(np.int32)
        return preds

    def decision_function(self, X, shape="ovr"):
        """
        Evaluate the decision function for the samples in X.

        In multi-class problems, you can use the 'shape' parameter to choose \
        between one-vs-rest ('ovr') and one-vs-one ('ovo') decision function shape. \
        Note that in our case, OvR decision values are derived from OvO. \
        For binary problems, this parameter is ignored.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).
            shape (str, optional): Whether to return a one-vs-rest ('ovr') \
            decision function or the original one-vs-one ('ovo'). Default='ovr'.

        Returns:
            numpy.ndarray: Decision function values for each sample.
        """
        return self._model.pybind_decision_function(X, shape)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X (numpy.ndarray): Test samples of shape (n_samples, n_features).
            y (numpy.ndarray): True labels for X.

        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        return super().score(X, y)

    @property
    def n_classes(self):
        """
        int: The number of classes in the classification problem.
        """
        return self._model.get_n_classes()


class SVR(BaseSVM):
    """
    Support Vector Regression.

    Train an epsilon-Support Vector Regression model.

    Args:
        C (float, optional): Regularization parameter. Controls the trade-off between maximizing \
            the margin between classes and minimizing classification errors. A larger \
            value means higher penalty to the loss function on misclassified observations. Must be \
            strictly positive. Default=1.0.
        epsilon (float, optional): Epsilon in the SVR model. Defines the tolerance for errors in \
            predictions by creating an acceptable margin (tube) within which errors are not \
            penalized. Default=0.1.
        kernel (str, optional): Kernel type to use in the algorithm. Possible values: \
            'linear', 'poly', 'rbf', 'sigmoid'. Default='rbf'.
        degree (int, optional): Degree of the polynomial kernel function. Ignored by \
            all other kernels. Default=3.
        gamma (float, optional): Kernel coefficient. If set to -1, it is calculated as \
            :math:`1/(Var(X) * n\_features)`. Default=-1.0.
        coef0 (float, optional): Independent term in kernel function (check :func:`kernel functions \
            <aoclda.kernel_functions.polynomial_kernel>` for more details). It is only used \
            in 'poly' and 'sigmoid' kernel functions. Default=0.0.
        probability (bool, optional): Currently not supported. Whether to enable \
            probability estimates. Default=False.
        tol (float, optional): Tolerance for stopping criterion. Default=0.001.
        max_iter (int, optional): Hard limit on iterations within solver, or 0 for no limit. \
            Default=0.
        tau (float, optional): Numerical stability parameter. If it is None then machine \
            epsilon is used. Default=None.
        check_data (bool, optional): Whether to check data for NaNs. Default=False.
    """

    def __init__(
        self,
        C=1.0,
        epsilon=0.1,
        kernel="rbf",
        degree=3,
        gamma=-1.0,
        coef0=0.0,
        tol=0.001,
        max_iter=0,
        tau=None,
        check_data=False,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            max_iter=max_iter,
            tau=tau,
            check_data=check_data,
        )
        self._model_double = pybind_svr(
            kernel=self.kernel,
            degree=self.degree,
            max_iter=self.max_iter,
            precision="double",
            check_data=self.check_data,
        )
        self._model_single = pybind_svr(
            kernel=self.kernel,
            degree=self.degree,
            max_iter=self.max_iter,
            precision="single",
            check_data=self.check_data,
        )
        self._model = self._model_double
        # Supported
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):
        """
        Fit the SVR model according to the given training data.

        Args:
            X (numpy.ndarray): Training vectors of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,).

        Returns:
            self (object): Returns the instance itself.
        """
        parameters = {"C": self.C, "epsilon": self.epsilon, "gamma": self.gamma,
                      "coef0": self.coef0, "tol": self.tol, "tau": self.tau}
        super().fit(X, y, **parameters)
        return self

    def predict(self, X):
        """
        Predict regression values for samples in X.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted values.
        """
        return super().predict(X)

    def score(self, X, y):
        """
        Return the coefficient of determination :math:`R^2` of the prediction.

        Args:
            X (numpy.ndarray): Test samples of shape (n_samples, n_features).
            y (numpy.ndarray): True values for X.

        Returns:
            float: :math:`R^2` of self.predict(X) wrt. y.
        """
        return super().score(X, y)


class NuSVC(BaseSVM):
    """
    Nu-Support Vector Classification.

    Train a Nu-Support Vector Classification model.

    Args:
        nu (float, optional): An upper bound on the fraction of training errors and a lower \
            bound of the fraction of support vectors. Default=0.5.
        kernel (str, optional): Kernel type to use in the algorithm. Possible values: \
            'linear', 'poly', 'rbf', 'sigmoid'. Default='rbf'.
        degree (int, optional): Degree of the polynomial kernel function. Ignored by \
            all other kernels. Default=3.
        gamma (float, optional): Kernel coefficient. If set to -1, it is calculated as \
            :math:`1/(Var(X) * n\_features)`. Default=-1.0.
        coef0 (float, optional): Independent term in kernel function (check :func:`kernel functions \
            <aoclda.kernel_functions.polynomial_kernel>` for more details). It is only used \
            in 'poly' and 'sigmoid' kernel functions. Default=0.0.
        probability (bool, optional): Currently not supported. Whether to enable \
            probability estimates. Default=False.
        tol (float, optional): Tolerance for stopping criterion. Default=0.001.
        max_iter (int, optional): Hard limit on iterations within solver, or 0 for no limit. \
            Default=0.
        tau (float, optional): Numerical stability parameter. If it is None then machine \
            epsilon is used. Default=None.
        check_data (bool, optional): Whether to check data for NaNs. Default=False.
    """

    def __init__(
        self,
        nu=0.5,
        kernel="rbf",
        degree=3,
        gamma=-1.0,
        coef0=0.0,
        probability=False,
        tol=0.001,
        max_iter=0,
        tau=None,
        check_data=False,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            max_iter=max_iter,
            tau=tau,
            check_data=check_data,
        )
        self._model_double = pybind_nusvc(
            kernel=self.kernel,
            degree=self.degree,
            max_iter=self.max_iter,
            precision="double",
            check_data=self.check_data,
        )
        self._model_single = pybind_nusvc(
            kernel=self.kernel,
            degree=self.degree,
            max_iter=self.max_iter,
            precision="single",
            check_data=self.check_data,
        )
        self._model = self._model_double
        # Not supported yet
        self.probability = probability
        # Supported
        self.nu = nu

    def fit(self, X, y):
        """
        Fit the NuSVC model according to the given training data.

        Args:
            X (numpy.ndarray): Training vectors of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,). They are expected to range from 0 to \p n_class - 1.

        Returns:
            self (object): Returns the instance itself.
        """
        parameters = {"nu": self.nu, "gamma": self.gamma,
                      "coef0": self.coef0, "tol": self.tol, "tau": self.tau}
        super().fit(X, y, **parameters)
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted class labels for samples in X.
        """
        preds = super().predict(X)
        if self.precision == "double":
            preds = preds.astype(np.int64)
        else:
            preds = preds.astype(np.int32)
        return preds

    def decision_function(self, X, shape="ovr"):
        """
        Evaluate the decision function for the samples in X.

        In multi-class problems, you can use the 'shape' parameter to choose \
        between one-vs-rest ('ovr') and one-vs-one ('ovo') decision function shape. \
        Note that in our case, OvR decision values are derived from OvO. \
        For binary problems, this parameter is ignored.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).
            shape (str, optional): Whether to return a one-vs-rest ('ovr') \
            decision function or the original one-vs-one ('ovo'). Default='ovr'.

        Returns:
            numpy.ndarray: Decision function values for each sample.
        """
        return self._model.pybind_decision_function(X, shape)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X (numpy.ndarray): Test samples of shape (n_samples, n_features).
            y (numpy.ndarray): True labels for X.

        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        return super().score(X, y)

    @property
    def n_classes(self):
        """
        int: The number of classes in the classification problem.
        """
        return self._model.get_n_classes()


class NuSVR(BaseSVM):
    """
    Nu-Support Vector Regression.

    Train a Nu-Support Vector Regression model.

    Args:
        nu (float, optional): An upper bound on the fraction of training errors and a lower \
            bound of the fraction of support vectors. Default=0.5.
        C (float, optional): Regularization parameter. Controls the trade-off between maximizing \
            the margin between classes and minimizing classification errors. A larger \
            value means higher penalty to the loss function on misclassified observations. Must be \
            strictly positive. Default=1.0.
        kernel (str, optional): Kernel type to use in the algorithm. Possible values: \
            'linear', 'poly', 'rbf', 'sigmoid'. Default='rbf'.
        degree (int, optional): Degree of the polynomial kernel function. Ignored by \
            all other kernels. Default=3.
        gamma (float, optional): Kernel coefficient. If set to -1, it is calculated as \
            :math:`1/(Var(X) * n\_features)`. Default=-1.0.
        coef0 (float, optional): Independent term in kernel function (check :func:`kernel functions \
            <aoclda.kernel_functions.polynomial_kernel>` for more details). It is only used \
            in 'poly' and 'sigmoid' kernel functions. Default=0.0.
        probability (bool, optional): Currently not supported. Whether to enable \
            probability estimates. Default=False.
        tol (float, optional): Tolerance for stopping criterion. Default=0.001.
        max_iter (int, optional): Hard limit on iterations within solver, or 0 for no limit. \
            Default=0.
        tau (float, optional): Numerical stability parameter. If it is None then machine \
            epsilon is used. Default=None.
        check_data (bool, optional): Whether to check data for NaNs. Default=False.
    """

    def __init__(
        self,
        nu=0.5,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma=-1.0,
        coef0=0.0,
        tol=0.001,
        max_iter=0,
        tau=None,
        check_data=False,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            max_iter=max_iter,
            tau=tau,
            check_data=check_data,
        )
        self._model_double = pybind_nusvr(
            kernel=self.kernel,
            degree=self.degree,
            max_iter=self.max_iter,
            precision="double",
            check_data=self.check_data,
        )
        self._model_single = pybind_nusvr(
            kernel=self.kernel,
            degree=self.degree,
            max_iter=self.max_iter,
            precision="single",
            check_data=self.check_data,
        )
        self._model = self._model_double
        # Supported
        self.nu = nu
        self.C = C

    def fit(self, X, y):
        """
        Fit the NuSVR model according to the given training data.

        Args:
            X (numpy.ndarray): Training vectors of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,).

        Returns:
            self (object): Returns the instance itself.
        """
        parameters = {"C": self.C, "nu": self.nu, "gamma": self.gamma,
                      "coef0": self.coef0, "tol": self.tol, "tau": self.tau}
        super().fit(X, y, **parameters)
        return self

    def predict(self, X):
        """
        Predict regression values for samples in X.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted values.
        """
        return super().predict(X)

    def score(self, X, y):
        """
        Return the coefficient of determination :math:`R^2` of the prediction.

        Args:
            X (numpy.ndarray): Test samples of shape (n_samples, n_features).
            y (numpy.ndarray): True values for X.

        Returns:
            float: :math:`R^2` of self.predict(X) wrt. y.
        """
        return super().score(X, y)
