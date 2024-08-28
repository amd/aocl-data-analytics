..
    Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

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

Linear Models
*************

The topic of linear models encompasses a range of commonly used statistical models and fitting algorithms, including
multiple linear regression, logistic regression, polynomial regression, and nonparametric regression.

The general form of a linear model fitting problem is as follows:

.. math::

    \min_{\beta}\left[ C_{\theta}\left( y, g^{-1}(\ \beta \, \phi(X)\ ) \right) \right],

where :math:`X` is an array of :math:`n_{\mathrm{samples}}` observations with :math:`n_{\mathrm{features}}` features, :math:`y` is an array of :math:`n_{\mathrm{samples}}` responses
/ labels, :math:`\phi` is a set of (possibly nonlinear) basis functions, :math:`\beta` is a set of weights /
coefficients, :math:`g^{-1}` is a (possibly nonlinear) activation / link function, and :math:`C_{\theta}` is a cost /
error function, which may depend on a set of (fixed) hyperparameters, :math:`\theta`.

Most linear models satisfy a form where the cost function can be split into a loss function (which measures quality
of fit to the data) and a penalty term (which regularizes the model parameters).  Regularization is also referred to as
shrinkage because it tends to shrink the size of the parameter values and/or shrink the number of non-zero parameters.

The loss function is typically a function of the responses or labels (:math:`y`), the features (:math:`X`), and the model
parameters (:math:`\beta`), whereas the penalty term is typically only a function of the 1-norm and/or the 2-norm of the
model parameters (:math:`\beta`).  Such linear models are often referred to as elastic nets.

Typically, in addition to the conditions above, :math:`\phi` and :math:`g^{-1}` are identity mappings and the general form
for the cost function becomes,


.. math::
   C_{\{\lambda,\alpha\}} \left( \beta \right) = L(y, \beta X)
   + \lambda \bigg( \alpha \lVert \beta \rVert_1 + \frac{(1 - \alpha)}{2} \lVert \beta \rVert_2^2  \bigg),
   :label: loss

where :math:`0\le\lambda, 0\le\alpha\le1` are hyperparameters with :math:`\lambda` setting the magnitude of the overall penalization and
:math:`\alpha` distributing its share across the :math:`\ell_1` and :math:`\ell_2` regularization terms, :math:`\lVert \beta \rVert_1` and :math:`\lVert \beta \rVert_2`. :math:`L` is known as the
*loss function*. Linear models
where :math:`\alpha=0` are called *ridge regression* models.
When :math:`\alpha=1` the model is a *lasso* model, whereas if :math:`0 < \alpha < 1` the model is an *elastic net regression* model.

**Intercept**


If a linear model contains an intercept term, the parameter vector :math:`\beta` has dimension :math:`n_{\mathrm{features}}+1` and the
observations :math:`X_i` are extended with a constant of :math:`1`.  For example, if there is a single feature and the
linear model has an intercept term the mean square error loss cost function becomes,

.. math::

   C\left( \beta_0, \beta_1 \right) = \sum_{i=0}^n (y_i - \beta_0 - \beta_1 x_i)^2,

where :math:`x_i` represents a single (scalar) observation and without any regularization terms.

The intercept can be added by setting the linear regression handle option `intercept` to `1`.
For further details see :ref:`linmod_options`.

.. note::
    The intercept term is never regularized. In the example above only :math:`\beta_1` is subject to regularization.

Available models
================

Models can be classified by their loss function, the following subsections present the supported loss functions.

Mean square error
-----------------

This is the most basic model and its use is widespread across many applications.
The cost function for a linear regression model where the fit (loss) is measured by the mean square error (MSE) is,

.. math::

   C_{\{0<\lambda,0\le\alpha\le1\}}\left( \beta_0, \beta \right) & = L(y, \beta_0, \beta X) \\
   & = \mathrm{MSE}(y, \beta_0, \beta X)= \frac{1}{2n} \sum_{i=1}^n (y_i - \beta_0 - \beta X_i)^2
   + \lambda \bigg( \alpha \lVert \beta \rVert_1 + \frac{(1 - \alpha)}{2} \lVert \beta \rVert_2^2  \bigg),

where :math:`X_i` represents a single (multi-dimensional) observation, i.e., a row in a table of observations.
Note that the intercept term :math:`\beta_0` is not regularized.

Logistic regression
-------------------

Logistic regression is a type of supervised classification model aimed at assigning labels.
In AOCL-DA, the labels are expected to be provided in a categorical response variable, :math:`y`, encoded by :math:`\{0, 1, 2, \ldots, K-1 \}`.
The fit is based on maximizing the log-likelihood (loss function) of the probabilities that each observation :math:`i` belongs to a given class,
in turn defined by,

.. math::
   p(y_i=k\,|\,X_i, \beta) = \frac{ \exp(\beta_k X_i) }{ 1 + \sum_{l=0}^{K-2}\exp(\beta_l X_i) }, \mathrm{ for } 0 \leq k < K-1,\\
   p(y_i=K-1\,|\,X_i, \beta) = \frac{ 1 }{ 1 + \sum_{l=0}^{K-2}\exp(\beta_l X_i) }.

As an example, if :math:`K=2`, the loss function simplifies to,

.. math::

   C\left( \beta \right) = -L(y, \beta X) = \sum_{i=0}^n \bigg( y_i \log p(X_i, \beta) + (1 - y_i) \log \big( 1 - p(X_i, \beta) \big) \bigg).

As in the linear regression model, :math:`\ell_1` or :math:`\ell_2` regularization can be applied by adding the corresponding
penalty term to the cost function.

.. only:: internal

    Extensions
    ==========

    Beyond MSE regression, ridge regression, lasso, and logistic regression, there are other classes of linear
    models which are not currently supported by AOCL-DA.  These include,

    * Weighted residuals - loss function is of form :math:`\sum_{i=1}^n w_i r_i = \sum_{i=1}^n w_i (y_i - \beta X_i)^2`
    * Additional loss functions - for example Huber, Cauchy, or Quantile in addition to MSE and Log Loss,
    * Basis expansions - for example addition of polynomial terms or extension to nonparametric regression, e.g., Loss
      functions of the form :math:`\sum_{i=1}^n \big(y_i - \beta \, \phi(X_i) \big)^2`


Fitting methods
===============

Different methods are available to compute the models. The method is chosen automatically by default but can be set manually using the optional parameter `optim method` (see :ref:`linmod_options`).

**Direct solvers**

* QR: the standard MSE linear regression model can be computed using the QR factorization of the data matrix if no regularization term is required.

  .. math::

   X = QR,

  where :math:`Q` is a :math:`n_{\mathrm{samples}} \times n_{\mathrm{features}}` matrix with orthogonal columns and :math:`R` is a :math:`n_{\mathrm{features}}\times n_{\mathrm{features}}` triangular matrix.

* SVD: the singular value decomposition can be used to compute standard MSE and ridge regression models.

  .. math::

   X = UDV^T,

  where :math:`U` is an orthogonal matrix of size :math:`n_{\mathrm{samples}}\times n_{\mathrm{samples}}`, :math:`D` is a :math:`n_{\mathrm{samples}}\times n_{\mathrm{features}}`
  diagonal matrix whose elements are the non-negative singular values of :math:`X` and :math:`V^T` is an orthogonal matrix of size :math:`n_{\mathrm{features}} \times n_{\mathrm{features}}`.

* Cholesky: the Cholesky decomposition can be used for normal and ridge regression when the data matrix is full-rank. It factorizes
  the symmetric positive-definite normal equations matrix :math:`X^TX` into two triangular matrices. In linear models it can be used to find coefficients expressed as:

  .. math::

   \beta = (X^TX+\lambda)^{-1}X^Ty,

  where, after left multiplying by the expression inside of the inverse, we end up with a system of linear equations in the form :math:`Ax=B`.
  The left hand side can be factorized using the Cholesky decomposition as follows:

  .. math::

   X^TX+\lambda = LL^T,

  where :math:`L` is a lower triangular matrix with real and positive diagonal entries. This matrix is then used to find a solution to the system of linear equations.


**Iterative solvers**

* L-BFGS-B: a solver aimed at minimizing smooth nonlinear functions (:cite:t:`lbfgsb`). It can be used to compute both MSE and logistic models with or without :math:`\ell_2` regularization. It is not suitable when an :math:`\ell_1` regularization term is required.

* Coordinate descent: a solver aimed at minimizing nonlinear functions.
  It is suitable for linear models with an :math:`\ell_1` regularization term and elastic nets (:cite:t:`coord_elastic`, :cite:t:`elnet1`).

  .. note::

    The coordinate descent method implemented is optimized to solve lasso or elastic net problems.
    For ridge or unregularized problems the use of any alternative methods is recommended.

  .. warning::

    The implemented coordinate descent method is designed to fit standardized data. As such, it should not be used with the scaling method
    set to anything other than :code:`scaling only` (default), or :code:`standardize` (see next section). Using any other type of scaling will result in lack of
    convergence or unexpected results.

* Conjugate gradient: a solver aimed at finding a solution to a system of linear equations.
  It can be used to compute linear regression with or without :math:`\ell_2` regularization.


Scaling the data
================

In many circumstances, the data used to perform a fit is badly scaled and rescaling can have numerical benefits.
Furthermore, iterative solvers can show improved quality of the solution when fitting on rescaled data.

If scaling is requested, then the fitting routine takes care of rescaling the problem
data. Once the model is trained, it reverts the scaling on the trained
coefficients, so this process is transparent to the user.

The optional parameter *scaling* controls the scaling of the data. It defaults to :code:`auto`, where it selects the type of scaling according to the optimization method used. The following table shows the default for each solver, while the table :ref:`tbl_scaling_types` shows the supported scaling types.

.. csv-table:: Default scaling type when optional parameter *scaling* = :code:`auto`
    :header: Method, option *optim method*, option *scaling*

    Conjugate Gradient Method,    :code:`cg`,     :code:`centering` or :code:`none`
    Coordinate Descent Method,    :code:`coord`,  :code:`scale only`
    Singular Value Decomposition, :code:`svd`,    :code:`centering`
    QR Factorization,             :code:`qr`,     :code:`centering`
    Cholesky Factorization,       :code:`cholesky`,  :code:`centering` or :code:`none`
    L-BFGS-B Solver,              :code:`lbfgsb`, :code:`centering` or :code:`none`

When a solver provides two default options for *scaling*, then it is chosen based on problem characteristics, e.g.,
if the problem is underdetermined or if the model has an intercept.

.. note::
    Scaling is applied prior to solving the problem and hence the regularization (if any) is done over the scaled problem and **not** on the
    unscaled version. This has implications on the trained coefficients and may differ from the regularized model trained with unscaled data.

The following table shows the supported scaling types.
In the table, :math:`N` is a shorthand for :math:`n_{\mathrm{samples}}`; :math:`\sigma_Z` refers to the sample standard deviation
of :math:`Z`; and :math:`\mu_Z` represents the sample mean of :math:`Z`. Finally, :math:`\hat Z` represents the scaled
version of :math:`Z`. Specifically, in the table below columns labeled :math:`\hat Y` and :math:`\hat X_j` report the transforms performed on the response vector :math:`Y` and the
columns of the predictor matrix :math:`X`.

.. _tbl_scaling_types:

.. csv-table:: Linear model data scaling types
    :header: *scaling* value , model intercept, :math:`m_Y`, :math:`s_Y`, :math:`\hat Y`, :math:`m_{X_j}`, :math:`s_{X_j}`, :math:`\hat X_j`

    :code:`none`       , yes/no,:math:`0`,    :math:`1`,                                                            :math:`Y`,                                           :math:`0`,        :math:`1`,           :math:`X_j`
    :code:`centering`  , yes,   :math:`\mu_Y`,:math:`1`,                                                            :math:`Y - \mu_Y`,                                   :math:`\mu_{X_j}`,:math:`1`,           :math:`X_j - \mu_{X_j}`
    :code:`centering`  , no,    :math:`0`,    :math:`1`,                                                            :math:`Y`,                                           :math:`0`,        :math:`1`,           :math:`X_j`
    :code:`scale only` , yes,   :math:`\mu_Y`,:math:`\sigma_Y`,                                                     :math:`\frac{\frac{1}{\sqrt{N}}(Y-\mu_Y)}{\sigma_Y}`,:math:`\mu_{X_j}`,:math:`1`,           :math:`\frac{1}{\sqrt{N}}(X_j-\mu_{X_j})`
    :code:`scale only` , no,    :math:`0`,    :math:`\left\|\frac{1}{\sqrt{N}} Y\right\|`,                          :math:`\frac{\frac{1}{\sqrt{N}} Y}{s_Y}`,            :math:`0`,        :math:`1`,           :math:`\frac{1}{\sqrt{N}} X_j`
    :code:`standardize`, yes,   :math:`\mu_Y`,:math:`\sigma_Y`,                                                     :math:`\frac{\frac{1}{\sqrt{N}}(Y-\mu_Y)}{\sigma_Y}`,:math:`\mu_{X_j}`,:math:`\sigma_{X_j}`,:math:`\frac{\frac{1}{\sqrt{N}} (X_j-\mu_{X_j})}{\sigma_{X_j}}`
    :code:`standardize`, no ,   :math:`0`,    :math:`\left\|\frac{1}{\sqrt{N}} Y\right\|`,                          :math:`\frac{\frac{1}{\sqrt{N}} Y}{s_Y}`,            :math:`0`,        :math:`\sigma_{X_j}`,:math:`\frac{\frac{1}{\sqrt{N}} X_j}{\sigma_{X_j}}`

.. only:: internal

    This is the complete table, containing the scaling factors used in the step function for the coordinate descent method (:math:`\eta`).

    .. csv-table:: Linear model data scaling types (full version)
        :header: *scaling* value , model intercept, :math:`m_Y`, :math:`s_Y`, :math:`\hat Y`, :math:`m_{X_j}`, :math:`s_{X_j}`, :math:`x_v(j) =\eta_j`, :math:`\hat X_j`

        :code:`none`       , yes/no,:math:`0`,    :math:`1`,                                                                 :math:`Y`,                                                   :math:`0`,        :math:`1`,                                                                       :math:`1`,                                                                           :math:`X_j`
        :code:`centering`  , yes,   :math:`\mu_Y`,:math:`1`,                                                                 :math:`Y - \mu_Y`,                                           :math:`\mu_{X_j}`,:math:`1`,                                                                       :math:`1`,                                                                           :math:`X_j - \mu_{X_j}`
        :code:`centering`  , no,    :math:`0`,    :math:`1`,                                                                 :math:`Y`,                                                   :math:`0`,        :math:`1`,                                                                       :math:`1`,                                                                           :math:`X_j`
        :code:`scale only` , yes,   :math:`\mu_Y`,:math:`\sigma_Y`,  :math:`\frac{\frac{1}{\sqrt{N}}(Y-\mu_Y)}{\sigma_Y}`,   :math:`\mu_{X_j}`,                                           :math:`1`,        :math:`\sigma^2_{X_j}=\left\|{\frac{1}{\sqrt{N}} \big(X_j-\mu_{X_j}\big)}\right\|^2`,:math:`\frac{1}{\sqrt{N}}(X_j-\mu_{X_j})`
        :code:`scale only` , no,    :math:`0`,    :math:`\left\|\frac{1}{\sqrt{N}} Y\right\|`,                               :math:`\frac{Y}{\|Y\|}`,                                     :math:`0`,        :math:`1`, :math:`\frac{1}{N}X_j^TX_j`,                                          :math:`\frac{1}{\sqrt{N}} X_j`
        :code:`standardize`, yes,   :math:`\mu_Y`,:math:`\sigma_Y=\left\|\frac{1}{\sqrt{N}} \big(Y-\mu_Y\big)\right\|`,      :math:`\frac{\frac{1}{\sqrt{N}}(Y-\mu_Y)}{\sigma_Y}`,        :math:`\mu_{X_j}`,:math:`\sigma_{X_j}=\left\|{\frac{1}{\sqrt{N}} \big(X_j-\mu_{X_j}\big)}\right\|`,:math:`1`,:math:`\frac{\frac{1}{\sqrt{N}} (X_j-\mu_{X_j})}{s_{X_j}}`
        :code:`standardize`, no ,   :math:`0`,    :math:`\left\|\frac{1}{\sqrt{N}} Y\right\|`,     :math:`\frac{\frac{1}{\sqrt{N}} Y}{s_Y}= \frac{Y}{\|Y\|}`,   :math:`0`,                :math:`\sigma_{X_j}`, :math:`\frac{\frac{1}{N} X_j^TX_j}{\sigma^2_{X_j}}`,                        :math:`\frac{\frac{1}{\sqrt{N}} X_j}{s_{X_j}}`

.. note::

    Scaling types :code:`scale only` and :code:`standardize` also affect the regularization penalty term :math:`\lambda` in :eq:`loss`.

    Iterative solvers can show the convergence progress by printing to standard output some progress metric for each iteration (as specified by the optional parameter *print level*).
    It is important to note that all reported metrics are based on the rescaled data.

    When requesting information from the handle, after training the model, the reported metrics are also based on the scaling type used.

Initial coefficients
====================

For iterative solvers, it is possible to provide a warm-start for the process via an initial guess for the coefficients (for example, these may have been obtained via previous fit).
When provided with too many coefficients, only the first :math:`k` coefficients will be used, where :math:`k` is the expected number of coefficients. The intercept will always be
the last member of the provided array. If too few coefficients are provided, the initial guess will be ignored.

.. warning::

    For the conjugate gradient solver in the underdetermined case (:math:`n_{\mathrm{features}} > n_{\mathrm{samples}}`) we are solving the dual problem, and thus the initial coefficients
    should also be dual (so that the expected number of coefficients :math:`k` is equal to :math:`n_{\mathrm{samples}}` instead of :math:`n_{\mathrm{features}}`).

Initial coefficients can be provided in the following way:

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      Provide :code:`x0` parameter when calling :func:`aoclda.linear_model.linmod.fit`.


   .. tab-item:: C
      :sync: C

      Provide :code:`coefs` pointer to initial coefficients while calling :ref:`da_linmod_fit_start_? <da_linmod_fit_start>`


Typical workflow for linear models
==================================

The standard way of computing a linear model using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.linear_model.linmod` object with options set in the class constructor.
      2. Fit the linear model to your data using :func:`aoclda.linear_model.linmod.fit`.
      3. Extract results from the :func:`aoclda.linear_model.linmod` object via its class attributes.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_linmod``.
      2. Pass data to the handle using :ref:`da_linmod_define_features_? <da_linmod_define_features>`.
      3. Customize the model using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <linmod_options>` for a list of the available options).
      4. Compute the linear model using :ref:`da_linmod_fit_? <da_linmod_fit>`.
      5. Evaluate the model on new data using :ref:`da_linmod_evaluate_model_? <da_linmod_evaluate_model>`.
      6. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`. The following results are available:

         * Coefficients (:cpp:enumerator:`da_linmod_coef`): the optimal coefficients of the fitted model

         * Some solvers provide extra information. (:cpp:enumerator:`da_linmod_rinfo`), when available, contains the
           info[100] array with the following values:

           * info[0]: loss value at current iterate, :math:`L(\beta_0, \beta)`,
           * info[1]: norm of the gradient of the loss function,
           * info[2]: number of iterations made (only for iterative solvers),
           * info[3]: compute time (wall clock time in seconds),
           * info[4]: number of model evaluations performed,
           * info[5]: infinity norm of the optimization metric (varies with the method used),
           * info[6]: infinity norm of of a given metric at the initial iterate (varies with the method used),
           * info[7]: number of *cheap* model evaluations (only relevant for Coordinate Descent Method) and indicates the number of low-rank updates used to evaluate model,
           * info[8-99]: reserved for future use.

           .. note::
                When information is not available, -1 is returned.

.. _linmod_options:

Linear model options
====================

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.linear_model.linmod` class constructor.

   .. tab-item:: C
      :sync: C

      Various options can be set to customize the linear models by calling one of these
      :ref:`functions <api_handle_options>`. The following table details the available options, where :math:`\epsilon` represents the machine precision.

      .. update options using table _opts_linearmodels

      .. csv-table:: Linear models options
         :header: "Option name", "Type", "Default", "Description", "Constraints"

         "optim method", "string", ":math:`s=` `auto`", "Select optimization method to use.", ":math:`s=` `auto`, `bfgs`, `cg`, `chol`, `cholesky`, `coord`, `lbfgs`, `lbfgsb`, `qr`, `sparse_cg`, or `svd`."
         "scaling", "string", ":math:`s=` `auto`", "Scale or standardize feature matrix and response vector. Matrix is copied and then rescaled. Option key value auto indicates that rescaling type is chosen by the solver (this also includes no scaling).", ":math:`s=` `auto`, `centering`, `no`, `none`, `scale`, `scale only`, `standardise`, or `standardize`."
         "optim progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "Factor used to detect convergence of the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 \le r`"
         "optim convergence tol", "real", ":math:`r=10/2\sqrt{2\,\varepsilon}`", "Tolerance to declare convergence for the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 < r < 1`"
         "print options", "string", ":math:`s=` `no`", "Print options.", ":math:`s=` `no`, or `yes`."
         "lambda", "real", ":math:`r=0`", "Penalty coefficient for the regularization terms: lambda( (1-alpha)/2 L2 + alpha L1 )", ":math:`0 \le r`"
         "alpha", "real", ":math:`r=0`", "Coefficient of alpha in the regularization terms: lambda( (1-alpha)/2 L2 + alpha L1 )", ":math:`0 \le r \le 1`"
         "optim iteration limit", "integer", ":math:`i=10000`", "Maximum number of iterations to perform in the optimization phase. Valid only for iterative solvers, e.g. L-BFGS-B, Coordinate Descent, etc.", ":math:`1 \le i`"
         "intercept", "integer", ":math:`i=0`", "Add intercept variable to the model", ":math:`0 \le i \le 1`"
         "print level", "integer", ":math:`i=0`", "Set level of verbosity for the solver", ":math:`0 \le i \le 5`"

      For the complete list of optional parameters see :ref:`linear model options <opts_linearmodels>`.

Examples
========

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. collapse:: Linear Model Example

          .. literalinclude:: ../../python_interface/python_package/aoclda/examples/linmod_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The showcased examples sources and more, can be found in the ``examples`` folder of your installation.

      .. collapse:: Linear Model Example

          .. literalinclude:: ../../tests/examples/linear_model.cpp
              :language: C++
              :linenos:

      .. collapse:: Elastic Net Example

          .. literalinclude:: ../../tests/examples/linmod_diabetes.cpp
              :language: C++
              :linenos:



Further reading
===============

An introduction to linear models for Regression and Classification can be found in Chapters 3, 4 of :cite:t:`bishop`, or
in Chapters 3-5 of :cite:t:`hastie`.

.. toctree::
    :maxdepth: 1
    :hidden:

    linmod_api
