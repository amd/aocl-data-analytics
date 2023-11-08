..
    Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.



.. _chapter_linmod:

Linear models
*************

The topic of linear models encompasses a range of commonly used statistical models and fitting algorithms, including
Multiple linear regression, logistic regression, polynomial regression, and nonparametric regression.

The general form of a linear model fitting problem is as follows:

.. math::

    \min_{\beta}\left[ C_{\theta}\left( y, g^{-1}(\ \beta \, \phi(X)\ ) \right) \right],

where :math:`X` is an array of :math:`n_{\text{samples}}` observations with :math:`n_{\text{feat}}` features, :math:`y` is an array of :math:`n_{\text{samples}}` responses
/ labels, :math:`\phi` is a set of (possibly nonlinear) basis functions, :math:`\beta` is a set of weights /
coefficients, :math:`g^{-1}` is a (possibly nonlinear) activation / link function, and :math:`C_{\theta}` is a cost /
error function, which may depend on a set of (fixed) hyperparameters, :math:`\theta`.

Most linear models satisfy a form where the cost function can be split into a Loss function (which measures quality
of fit to the data) and a penalty term (which regularizes the model parameters).  Regularization is also referred to as
shrinkage because it tends to shrink the size of the parameter values and/or shrink the number of non-zero parameters.

The Loss function is typically a function of the responses or labels (:math:`y`), the features (:math:`X`), and the model
parameters (:math:`\beta`), whereas the penalty term is typically only a function of the 1-norm and/or the 2-norm of the
model parameters (:math:`\beta`).  Such linear models are often referred to as Elastic-Nets.

Typically, in addition to the conditions above, :math:`\phi` and :math:`g^{-1}` are identity mappings and the general form
for the cost function becomes,

.. math::

   C_{\{\lambda,\alpha\}} \left( \beta \right) = L(y, \beta X)
   + \lambda \bigg( \alpha \lVert \beta \rVert_1 + (1 - \alpha) \lVert \beta \rVert_2^2  \bigg),

where :math:`0\le\lambda, 0\le\alpha\le1` are hyperparameters, :math:`\lVert \beta \rVert_1` is the 1-norm
and :math:`\lVert \beta \rVert_2` is the 2-norm of :math:`\beta`, while :math:`\lambda` sets the magnitude of the overall penalization,
:math:`\alpha` distributes its share across the :math:`\ell_1` and :math:`\ell_2` regularization terms. :math:`L` is known as the
*Loss function*. Linear models
where :math:`\alpha=0` are called Ridge Regression. Conversely, when :math:`\alpha=1` the model is called Lasso.

**Intercept**


If a linear model contains an intercept term, the parameter vector :math:`\beta` has dimension :math:`n_{\text{feat}}+1` and the
observations :math:`X_i` are extended with a constant of :math:`1`.  For example, if there is a single feature and the
linear model has an intercept term the Mean Square Error Loss cost function becomes,

.. math::

   C\left( \beta_0, \beta_1 \right) = \sum_{i=0}^n (y_i - \beta_0 - \beta_1 x_i)^2,

where :math:`x_i` represents a single (scalar) observation and without any regularization terms.

In general, the intercept can be added or not by setting an option in the linear regression handle.

Available Models
================

Models can be classified by their loss function, the following subsections present the supported loss functions.

Mean Square Error
-----------------

This is the most basic model and its use is widespread across many applications.
The cost function for a linear regression Model where the fit (loss) is measured by the Mean Square Error (MSE) is,

.. math::

   C_{\{0<\lambda,0\le\alpha\le1\}}\left( \beta \right) = L(y, \beta X) =\text{MSE}(y, \beta X)= \sum_{i=1}^n (y_i - \beta X_i)^2
   + \lambda \bigg( \alpha \lVert \beta \rVert_1 + (1 - \alpha) \lVert \beta \rVert_2^2  \bigg),

where :math:`X_i` represents a single (multi-dimensional) observation, i.e., a row in a table of observations.

logistic regression
-------------------

logistic regression is a type of supervised classification model aiming at assigning labels.
In AOCL-DA, the labels are expected to be provided in a categorical response variable, :math:`y`, encoded by :math:`\{0, 1, 2, \ldots, K-1 \}`.
The fit is based on maximizing the log-likelihood (loss function) of the probabilities that each observation :math:`i` belongs to a given class,
in turn defined by,

.. math::
   p(y_i=k\,|\,X_i, \beta) = \frac{ \exp(\beta_k X_i) }{ 1 + \sum_{l=0}^{K-2}\exp(\beta_l X_i) }, \text{ for } 0 \leq k < K-1,\\
   p(y_i=K-1\,|\,X_i, \beta) = \frac{ 1 }{ 1 + \sum_{l=0}^{K-2}\exp(\beta_l X_i) }.

As an example, if :math:`K=2`, the loss function simplifies to,

.. math::

   C\left( \beta \right) = -L(y, \beta X) = \sum_{i=0}^n \bigg( y_i \log p(X_i, \beta) + (1 - y_i) \log \big( 1 - p(X_i, \beta) \big) \bigg).

As in the Linear Regession Model, :math:`\ell_1` or :math:`\ell_2` regularization can be applied by adding the corresponding
penalty term to the cost function.

.. only:: internal

    Extensions [Internal]
    =====================

    Beyond MSE regression, ridge regression, the Lasso, and logistic regression, there are other classes of Linear
    Model which are not currently supported by AOCL-DA.  This includes,

    * Weighted residuals - Loss function is of form :math:`\sum_{i=1}^n w_i r_i = \sum_{i=1}^n w_i (y_i - \beta X_i)^2`
    * Additional loss functions - for example Huber, Cauchy, or Quantile in addition to MSE and Log Loss,
    * Basis expansions - for example addition of polynomial terms or extension to nonparametric regression, e.g., Loss
      functions of the form :math:`\sum_{i=1}^n \big(y_i - \beta \, \phi(X_i) \big)^2`


Fitting Methods
===============

Different methods are available to compute the models. The method is chosen automatically by default but can be set manually using the optional parameter ``linmod optim method`` (see the :ref:`options section <linmod_options>`).

**Direct solvers**

* QR (``linmod optim method = QR``). The standard MSE linear regression model can be computed using the QR factorization of the data matrix if no regularization term is required.

.. math::

   X = QR,

where :math:`Q` is an :math:`n_{\text{samples}} \times n_{\text{feat}}` matrix with orthogonal columns and :math:`R` is a :math:`n_{\text{feat}}\times n_{\text{feat}}` triangular matrix.

**Iterative solvers**

* L-BFGS-B (``linmod optim method = lbfgs``) is a solver aimed at minimizing smooth nonlinear functions (:cite:t:`lbfgsb`). It can be used to compute both MSE and logistic models with or without :math:`\ell_2` regularization. It is not suitable when an :math:`\ell_1` regularization term is required.

* Coordinate Descent (``linmod optim method = coord``) is a solver aimed at minimizing nonlinear functions. It is particularly suitable for linear models with an :math:`\ell_1` regularization term (:cite:t:`coord_elastic`).


Available outputs
=================

Once a model is computed, some elements can be retrieved using :cpp:func:`da_handle_get_result_d` or :cpp:func:`da_handle_get_result_s`:

* coefficients (:cpp:enumerator:`da_linmod_coeff`): The optimal coefficients of the fitted model
* rinfo[100] (:cpp:enumerator:`da_linmod_rinfo`): a set of values of interest
   * rinfo[0]: :math:`n_{feat}`, the number of features in the model.
   * rinfo[1]: :math:`n_{samples}`, the number of samples the model has been trained on.
   * rinfo[2]: :math:`n_{coef}`, the number of model coefficients.
   * rinfo[3]: intercept, 1 if an intercept term is present in the model, 0 otherwise.
   * rinfo[4]: :math:`\alpha`, share of the :math:`\ell_1` term in the regularization.
   * rinfo[5]: :math:`\lambda`, the magnitude of the regularization term.
   * rinfo[6-99]: reserved for future use.


Typical workflow for linear models
==================================

The standard way of computing a linear model using AOCL-DA is as follows.

1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_linmod``.
2. Pass data to the handle using either :cpp:func:`da_linmod_define_features_s` or :cpp:func:`da_linmod_define_features_d`.
3. Customize the model using :cpp:func:`da_options_set_int`, :cpp:func:`da_options_set_real_d`, :cpp:func:`da_options_set_real_s` and :cpp:func:`da_options_set_string` (see :ref:`below <linmod_options>` for a list of the available options).
4. Compute the linear mdoel using :cpp:func:`da_linmod_fit_d` or :cpp:func:`da_linmod_fit_s`.
5. Evaluate the model on new data using :cpp:func:`da_linmod_evaluate_model_d` or :cpp:func:`da_linmod_evaluate_model_s`
6. Extract results using :cpp:func:`da_handle_get_result_d` or :cpp:func:`da_handle_get_result_s`.


.. _linmod_options:

Linear Model Options
====================

Various options can be set to customize the linear models by calling one of these
:ref:`functions <api_handle_options>`. The following table details the available options, where :math:`\epsilon` represents the machine precision.

.. csv-table:: Linear models options
   :header: "Option Name", "Type", "Default", "Description", "Constraints"

   "linmod optim method", "string", ":math:`s =` `'auto'`", "Select the optimization method to use.", "`s = 'auto', 'coord', 'lbfgs' or 'qr'`"
   "linmod optim progress factor", "real", ":math:`r = \frac{10}{\sqrt{2\epsilon}}`", "Factor used to detect convergence of the iterative optimization step. See option in the corresponding optimization solver documentation.",  ":math:`0 \le r`"
   "linmod optim convergence tol", "real", ":math:`r = \sqrt{2\epsilon}`", "Tolerance to declare convergence for the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 < r < 1`"
   "print options", "string", ":math:`s =` `'no'`", "Print options.", ":math:`s =` `'no'`, or `'yes'`."
   "linmod lambda", "real", ":math:`r = 0`", "Penalty coefficient for the regularization terms :math:`\lambda ( (1-\alpha ) \ell_2 + \alpha \ell_1 )`", ":math:`0 \le r`"
   "linmod alpha", "real", ":math:`r = 0`", "Coefficient of alpha in the regularization terms :math:`\lambda ( (1-\alpha) \ell_2 + \alpha \ell_1 )`", ":math:`0 \le r \le 1`"
   "linmod optim iteration limit", "da_int", ":math:`i = \inf`", "Maximum number of iterations to perform in the optimization phase. Valid only for iterative solvers, e.g. L-BFGS-B, Coordinate Descent, etc.", ":math:`1 \le i`"
   "linmod intercept", "da_int", ":math:`i = 0`", "Add intercept variable to the model", ":math:`0 \le i \le 1`"
   "print level", "da_int", ":math:`i = 0`", "Set the level of verbosity for the solver", ":math:`0 \le i \le 5`"


Further Reading
===============

An introduction to linear models for Regression and Classification can be found in Chapters 3, 4 of :cite:t:`bishop`, or
in Chapters 3-5 of :cite:t:`hastie`.

.. toctree::
    :maxdepth: 1
    :hidden:

    linmod_api
