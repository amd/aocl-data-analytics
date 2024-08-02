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


class SVC:
    """
    Support Vector Classification.

    Train a C-Support Vector Classification model.

    Args:
        C (float, optional): Regularization parameter. Controls the trade-off between maximizing \
            the margin between classes and minimizing classification errors. The larger \
            value means higher penalty to the loss function on misclassified observations. Must be \
            strictly positive. Default=1.0.
        kernel (str, optional): Kernel type to use in the algorithm. Possible values: \
            'linear', 'poly', 'rbf', 'sigmoid'. Default='rbf'.
        degree (int, optional): Degree of the polynomial kernel function. Ignored by \
            all other kernels. Default=3.
        gamma (float, optional): Kernel coefficient. If set to -1, it is calculated as \
            :math:`1/(Var(X) * n\_features)`. Default=-1.0.
        coef0 (float, optional): Independent term in kernel function. It is only used \
            in 'poly' and 'sigmoid' kernel functions. Default=0.0.
        probability (bool, optional): Currently not supported. Whether to enable \
            probability estimates. Default=False.
        tol (float, optional): Tolerance for stopping criterion. Default=0.001.
        max_iter (int, optional): Hard limit on iterations within solver, or -1 for no limit. \
            Default=-1.
        decision_function_shape (str, optional): Whether to return a one-vs-rest ('ovr') \
            decision function or the original one-vs-one ('ovo'). Default='ovr'.
        tau (float, optional): Numerical stability parameter. Default=1.0e-12.
        check_data (bool, optional): Whether to check data for NaNs. Default=False.
    """

    def __init__(self,
                 C=1.0,
                 kernel="rbf",
                 degree=3,
                 gamma=-1.0,
                 coef0=0.0,
                 probability=False,
                 tol=0.001,
                 max_iter=-1,
                 decision_function_shape="ovr",
                 tau=1.0e-12,
                 check_data=False):
        self.svc_double = pybind_svc(kernel=kernel,
                                     degree=degree,
                                     max_iter=max_iter,
                                     dec_f_shape=decision_function_shape,
                                     precision="double",
                                     check_data=check_data)
        self.svc_single = pybind_svc(kernel=kernel,
                                     degree=degree,
                                     max_iter=max_iter,
                                     dec_f_shape=decision_function_shape,
                                     precision="single",
                                     check_data=check_data)
        self.C = C
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.tau = tau
        self.svc = self.svc_double
        self.precision = "double"

    def fit(self, X, y):
        """
        Fit the SVC model according to the given training data.

        Args:
            X (numpy.ndarray): Training vectors of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,). They are expected to range from 0 to \p n_class - 1.

        Returns:
            self (object): Returns the instance itself.
        """
        if X.dtype == 'float32':
            self.svc = self.svc_single
            self.svc_double = None
            self.C = np.float32(self.C)
            self.gamma = np.float32(self.gamma)
            self.coef0 = np.float32(self.coef0)
            self.tol = np.float32(self.tol)
            self.tau = np.float32(self.tau)
            self.precision = "single"
        else:
            self.C = np.float64(self.C)
            self.gamma = np.float64(self.gamma)
            self.coef0 = np.float64(self.coef0)
            self.tol = np.float64(self.tol)
            self.tau = np.float64(self.tau)

        if y.dtype.kind in np.typecodes["AllInteger"]:
            y = y.astype(X.dtype, copy=False)

        self.svc.pybind_fit(X,
                            y,
                            C=self.C,
                            gamma=self.gamma,
                            coef0=self.coef0,
                            tol=self.tol,
                            tau=self.tau)
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted class labels for samples in X.
        """
        preds = self.svc.pybind_predict(X)
        if self.precision == "double":
            preds = preds.astype(np.int64)
        else:
            preds = preds.astype(np.int32)
        return preds

    def decision_function(self, X):
        """
        Evaluate the decision function for the samples in X.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Decision function values for each sample.
        """
        return self.svc.pybind_decision_function(X)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X (numpy.ndarray): Test samples of shape (n_samples, n_features).
            y (numpy.ndarray): True labels for X.

        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        if y.dtype.kind in np.typecodes["AllInteger"]:
            y = y.astype(X.dtype, copy=False)

        return self.svc.pybind_score(X, y)

    @property
    def n_samples(self):
        """
        int: The number of training samples used to fit the model.
        """
        return self.svc.get_n_samples()

    @property
    def n_features(self):
        """
        int: The number of features in the training data.
        """
        return self.svc.get_n_features()

    @property
    def n_classes(self):
        """
        int: The number of classes in the classification problem.
        """
        return self.svc.get_n_classes()

    @property
    def n_support(self):
        """
        int: The total number of support vectors.
        """
        return self.svc.get_n_sv()

    @property
    def n_support_per_class(self):
        """
        numpy.ndarray of shape (n_classes,):
            The number of support vectors for each class.
        """
        return self.svc.get_n_sv_per_class()

    @property
    def dual_coef(self):
        """
        numpy.ndarray of shape (n_classes-1, n_support):
            The dual coefficients of the support vectors.
        """
        return self.svc.get_dual_coef()

    @property
    def support_vectors_idx(self):
        """
        numpy.ndarray of shape (n_support,):
            The indices of the support vectors.
        """
        return self.svc.get_support_vectors_idx()

    @property
    def support_vectors(self):
        """
        numpy.ndarray of shape (n_support, n_features):
            The support vectors used by the model.
        """
        return self.svc.get_sv()

    @property
    def bias(self):
        """
        numpy.ndarray or float:
            The bias term(s) of the model (also known as intercept).
        """
        return self.svc.get_bias()


class SVR():
    """
    Support Vector Regression.

    Train an epsilon-Support Vector Regression model.

    Args:
        C (float, optional): Regularization parameter. Controls the trade-off between maximizing \
            the margin between classes and minimizing classification errors. The larger \
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
        coef0 (float, optional): Independent term in kernel function. It is only used \
            in 'poly' and 'sigmoid' kernel functions. Default=0.0.
        probability (bool, optional): Currently not supported. Whether to enable \
            probability estimates. Default=False.
        tol (float, optional): Tolerance for stopping criterion. Default=0.001.
        max_iter (int, optional): Hard limit on iterations within solver, or -1 for no limit. \
            Default=-1.
        tau (float, optional): Numerical stability parameter. Default=1.0e-12.
        check_data (bool, optional): Whether to check data for NaNs. Default=False.
    """

    def __init__(self,
                 C=1.0,
                 epsilon=0.1,
                 kernel="rbf",
                 degree=3,
                 gamma=-1.0,
                 coef0=0.0,
                 probability=False,
                 tol=0.001,
                 max_iter=-1,
                 tau=1.0e-12,
                 check_data=False):
        self.svr_double = pybind_svr(kernel=kernel,
                                     degree=degree,
                                     max_iter=max_iter,
                                     precision="double",
                                     check_data=check_data)
        self.svr_single = pybind_svr(kernel=kernel,
                                     degree=degree,
                                     max_iter=max_iter,
                                     precision="single",
                                     check_data=check_data)
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.tau = tau
        self.svr = self.svr_double
        self.precision = "double"

    def fit(self, X, y):
        """
        Fit the SVR model according to the given training data.

        Args:
            X (numpy.ndarray): Training vectors of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,).

        Returns:
            self (object): Returns the instance itself.
        """
        if X.dtype == 'float32':
            self.svr = self.svr_single
            self.svr_double = None
            self.C = np.float32(self.C)
            self.epsilon = np.float32(self.epsilon)
            self.gamma = np.float32(self.gamma)
            self.coef0 = np.float32(self.coef0)
            self.tol = np.float32(self.tol)
            self.tau = np.float32(self.tau)
            self.precision = "single"
        else:
            self.C = np.float64(self.C)
            self.epsilon = np.float64(self.epsilon)
            self.gamma = np.float64(self.gamma)
            self.coef0 = np.float64(self.coef0)
            self.tol = np.float64(self.tol)
            self.tau = np.float64(self.tau)

        if y.dtype.kind in np.typecodes["AllInteger"]:
            y = y.astype(X.dtype, copy=False)

        self.svr.pybind_fit(X,
                            y,
                            C=self.C,
                            epsilon=self.epsilon,
                            gamma=self.gamma,
                            coef0=self.coef0,
                            tol=self.tol,
                            tau=self.tau)
        return self

    def predict(self, X):
        """
        Predict regression values for samples in X.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted values.
        """
        return self.svr.pybind_predict(X)

    def score(self, X, y):
        """
        Return the coefficient of determination :math:`R^2` of the prediction.

        Args:
            X (numpy.ndarray): Test samples of shape (n_samples, n_features).
            y (numpy.ndarray): True values for X.

        Returns:
            float: :math:`R^2` of self.predict(X) wrt. y.
        """
        if y.dtype.kind in np.typecodes["AllInteger"]:
            y = y.astype(X.dtype, copy=False)

        return self.svr.pybind_score(X, y)

    @property
    def n_samples(self):
        """
        int: The number of training samples used to fit the model.
        """
        return self.svr.get_n_samples()

    @property
    def n_features(self):
        """
        int: The number of features in the training data.
        """
        return self.svr.get_n_features()

    @property
    def n_support(self):
        """
        int: The total number of support vectors.
        """
        return self.svr.get_n_sv()

    @property
    def dual_coef(self):
        """
        numpy.ndarray of shape (1, n_support):
            The dual coefficients of the support vectors.
        """
        return self.svr.get_dual_coef()

    @property
    def support_vectors_idx(self):
        """
        numpy.ndarray of indices:
            The indices of the support vectors.
        """
        return self.svr.get_support_vectors_idx()

    @property
    def support_vectors(self):
        """
        numpy.ndarray of shape (n_support, n_features):
            The support vectors used by the model.
        """
        return self.svr.get_sv()

    @property
    def bias(self):
        """
        float:
            The bias term of the model (also known as the intercept).
        """
        return self.svr.get_bias()


class NuSVC():
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
        coef0 (float, optional): Independent term in kernel function. It is only used \
            in 'poly' and 'sigmoid' kernel functions. Default=0.0.
        probability (bool, optional): Currently not supported. Whether to enable \
            probability estimates. Default=False.
        tol (float, optional): Tolerance for stopping criterion. Default=0.001.
        max_iter (int, optional): Hard limit on iterations within solver, or -1 for no limit. \
            Default=-1.
        decision_function_shape (str, optional): Whether to return a one-vs-rest ('ovr') \
            decision function or the original one-vs-one ('ovo'). Default='ovr'.
        tau (float, optional): Numerical stability parameter. Default=1.0e-12.
        check_data (bool, optional): Whether to check data for NaNs. Default=False.
    """

    def __init__(self,
                 nu=0.5,
                 kernel="rbf",
                 degree=3,
                 gamma=-1.0,
                 coef0=0.0,
                 probability=False,
                 tol=0.001,
                 max_iter=-1,
                 decision_function_shape="ovr",
                 tau=1.0e-12,
                 check_data=False):
        self.nusvc_double = pybind_nusvc(kernel=kernel,
                                         degree=degree,
                                         max_iter=max_iter,
                                         dec_f_shape=decision_function_shape,
                                         precision="double",
                                         check_data=check_data)
        self.nusvc_single = pybind_nusvc(kernel=kernel,
                                         degree=degree,
                                         max_iter=max_iter,
                                         dec_f_shape=decision_function_shape,
                                         precision="single",
                                         check_data=check_data)
        self.nu = nu
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.tau = tau
        self.nusvc = self.nusvc_double
        self.precision = "double"

    def fit(self, X, y):
        """
        Fit the NuSVC model according to the given training data.

        Args:
            X (numpy.ndarray): Training vectors of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,). They are expected to range from 0 to \p n_class - 1.

        Returns:
            self (object): Returns the instance itself.
        """
        if X.dtype == 'float32':
            self.nusvc = self.nusvc_single
            self.nusvc_double = None
            self.nu = np.float32(self.nu)
            self.gamma = np.float32(self.gamma)
            self.coef0 = np.float32(self.coef0)
            self.tol = np.float32(self.tol)
            self.tau = np.float32(self.tau)
            self.precision = "single"
        else:
            self.nu = np.float64(self.nu)
            self.gamma = np.float64(self.gamma)
            self.coef0 = np.float64(self.coef0)
            self.tol = np.float64(self.tol)
            self.tau = np.float64(self.tau)

        if y.dtype.kind in np.typecodes["AllInteger"]:
            y = y.astype(X.dtype, copy=False)

        self.nusvc.pybind_fit(X,
                              y,
                              nu=self.nu,
                              gamma=self.gamma,
                              coef0=self.coef0,
                              tol=self.tol,
                              tau=self.tau)
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted class labels for samples in X.
        """
        preds = self.nusvc.pybind_predict(X)
        if self.precision == "double":
            preds = preds.astype(np.int64)
        else:
            preds = preds.astype(np.int32)
        return preds

    def decision_function(self, X):
        """
        Evaluate the decision function for the samples in X.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Decision function values for each sample.
        """
        return self.nusvc.pybind_decision_function(X)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X (numpy.ndarray): Test samples of shape (n_samples, n_features).
            y (numpy.ndarray): True labels for X.

        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        if y.dtype.kind in np.typecodes["AllInteger"]:
            y = y.astype(X.dtype, copy=False)

        return self.nusvc.pybind_score(X, y)

    @property
    def n_samples(self):
        """
        int: The number of training samples used to fit the model.
        """
        return self.nusvc.get_n_samples()

    @property
    def n_features(self):
        """
        int: The number of features in the training data.
        """
        return self.nusvc.get_n_features()

    @property
    def n_classes(self):
        """
        int: The number of classes in the classification problem.
        """
        return self.nusvc.get_n_classes()

    @property
    def n_support(self):
        """
        int: The total number of support vectors.
        """
        return self.nusvc.get_n_sv()

    @property
    def n_support_per_class(self):
        """
        numpy.ndarray of shape (n_classes,):
            The number of support vectors for each class.
        """
        return self.nusvc.get_n_sv_per_class()

    @property
    def dual_coef(self):
        """
        numpy.ndarray of shape (n_classes-1, n_support):
            The dual coefficients of the support vectors.
        """
        return self.nusvc.get_dual_coef()

    @property
    def support_vectors_idx(self):
        """
        numpy.ndarray of size (n_support,):
            The indices of the support vectors.
        """
        return self.nusvc.get_support_vectors_idx()

    @property
    def support_vectors(self):
        """
        numpy.ndarray of shape (n_support, n_features):
            The support vectors used by the model.
        """
        return self.nusvc.get_sv()

    @property
    def bias(self):
        """
        numpy.ndarray or float:
            The bias term(s) of the model (also known as the intercept).
        """
        return self.nusvc.get_bias()


class NuSVR():
    """
    Nu-Support Vector Regression.

    Train a Nu-Support Vector Regression model.

    Args:
        nu (float, optional): An upper bound on the fraction of training errors and a lower \
            bound of the fraction of support vectors. Default=0.5.
        C (float, optional): Regularization parameter. Controls the trade-off between maximizing \
            the margin between classes and minimizing classification errors. The larger \
            value means higher penalty to the loss function on misclassified observations. Must be \
            strictly positive. Default=1.0.
        kernel (str, optional): Kernel type to use in the algorithm. Possible values: \
            'linear', 'poly', 'rbf', 'sigmoid'. Default='rbf'.
        degree (int, optional): Degree of the polynomial kernel function. Ignored by \
            all other kernels. Default=3.
        gamma (float, optional): Kernel coefficient. If set to -1, it is calculated as \
            :math:`1/(Var(X) * n\_features)`. Default=-1.0.
        coef0 (float, optional): Independent term in kernel function. It is only used \
            in 'poly' and 'sigmoid' kernel functions. Default=0.0.
        probability (bool, optional): Currently not supported. Whether to enable \
            probability estimates. Default=False.
        tol (float, optional): Tolerance for stopping criterion. Default=0.001.
        max_iter (int, optional): Hard limit on iterations within solver, or -1 for no limit. \
            Default=-1.
        tau (float, optional): Numerical stability parameter. Default=1.0e-12.
        check_data (bool, optional): Whether to check data for NaNs. Default=False.
    """

    def __init__(self,
                 nu=0.5,
                 C=1.0,
                 kernel="rbf",
                 degree=3,
                 gamma=-1.0,
                 coef0=0.0,
                 probability=False,
                 tol=0.001,
                 max_iter=-1,
                 tau=1.0e-12,
                 check_data=False):
        self.nusvr_double = pybind_nusvr(kernel=kernel,
                                         degree=degree,
                                         max_iter=max_iter,
                                         precision="double",
                                         check_data=check_data)
        self.nusvr_single = pybind_nusvr(kernel=kernel,
                                         degree=degree,
                                         max_iter=max_iter,
                                         precision="single",
                                         check_data=check_data)
        self.nu = nu
        self.C = C
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.tau = tau
        self.nusvr = self.nusvr_double
        self.precision = "double"

    def fit(self, X, y):
        """
        Fit the NuSVR model according to the given training data.

        Args:
            X (numpy.ndarray): Training vectors of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,).

        Returns:
            self (object): Returns the instance itself.
        """
        if X.dtype == 'float32':
            self.nusvr = self.nusvr_single
            self.nusvr_double = None
            self.nu = np.float32(self.nu)
            self.C = np.float32(self.C)
            self.gamma = np.float32(self.gamma)
            self.coef0 = np.float32(self.coef0)
            self.tol = np.float32(self.tol)
            self.tau = np.float32(self.tau)
            self.precision = "single"
        else:
            self.nu = np.float64(self.nu)
            self.C = np.float64(self.C)
            self.gamma = np.float64(self.gamma)
            self.coef0 = np.float64(self.coef0)
            self.tol = np.float64(self.tol)
            self.tau = np.float64(self.tau)

        if y.dtype.kind in np.typecodes["AllInteger"]:
            y = y.astype(X.dtype, copy=False)

        self.nusvr.pybind_fit(X,
                              y,
                              nu=self.nu,
                              C=self.C,
                              gamma=self.gamma,
                              coef0=self.coef0,
                              tol=self.tol,
                              tau=self.tau)
        return self

    def predict(self, X):
        """
        Predict regression values for samples in X.

        Args:
            X (numpy.ndarray): Input vectors of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted values.
        """
        return self.nusvr.pybind_predict(X)

    def score(self, X, y):
        """
        Return the coefficient of determination :math:`R^2` of the prediction.

        Args:
            X (numpy.ndarray): Test samples of shape (n_samples, n_features).
            y (numpy.ndarray): True values for X.

        Returns:
            float: :math:`R^2` of self.predict(X) wrt. y.
        """
        if y.dtype.kind in np.typecodes["AllInteger"]:
            y = y.astype(X.dtype, copy=False)

        return self.nusvr.pybind_score(X, y)

    @property
    def n_samples(self):
        """
        int: The number of training samples used to fit the model.
        """
        return self.nusvr.get_n_samples()

    @property
    def n_features(self):
        """
        int: The number of features in the training data.
        """
        return self.nusvr.get_n_features()

    @property
    def n_support(self):
        """
        int: The total number of support vectors.
        """
        return self.nusvr.get_n_sv()

    @property
    def dual_coef(self):
        """
        numpy.ndarray of shape (1, n_support):
            The dual coefficients of the support vectors.
        """
        return self.nusvr.get_dual_coef()

    @property
    def support_vectors_idx(self):
        """
        numpy.ndarray of indices:
            The indices of the support vectors.
        """
        return self.nusvr.get_support_vectors_idx()

    @property
    def support_vectors(self):
        """
        numpy.ndarray of shape (n_support, n_features):
            The support vectors used by the model.
        """
        return self.nusvr.get_sv()

    @property
    def bias(self):
        """
        float:
            The bias term of the model (also known as intercept).
        """
        return self.nusvr.get_bias()
