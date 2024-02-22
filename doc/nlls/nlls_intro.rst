..
    Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

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



.. _chapter_nlls:

Nonlinear Data Fitting
**********************

The topic of linear models encompasses a range of commonly used statistical models and fitting algorithms, including
Multiple linear regression, logistic regression, polynomial regression, and nonparametric regression.

The general form of a linear model fitting problem is as follows:

.. math::

    \min_{\beta}\left[ C_{\theta}\left( y, g^{-1}(\ \beta \, \phi(X)\ ) \right) \right],

where :math:`X` is an array of :math:`n_{\mathrm{samples}}` observations with :math:`n_{\mathrm{features}}` features, :math:`y` is an array of :math:`n_{\mathrm{samples}}` responses
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


Fitting Methods
===============


Typical workflow for linear models
==================================

The standard way of computing a linear model using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.linear_model.linmod` object with options set in the class constructor.
      2. Fit the linear model for your data using :func:`aoclda.linear_model.linmod.fit`.
      3. Extract results from the :func:`aoclda.linear_model.linmod` object via its class attributes.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_linmod``.
      2. Pass data to the handle using either :ref:`da_linmod_define_features_? <da_linmod_define_features>`.
      3. Customize the model using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <linmod_options>` for a list of the available options).
      4. Compute the linear model using :ref:`da_linmod_fit_? <da_linmod_fit>`.
      5. Evaluate the model on new data using :ref:`da_linmod_evaluate_model_? <da_linmod_evaluate_model>`.
      6. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`. The following results are available:

         * coefficients (:cpp:enumerator:`da_linmod_coef`): the optimal coefficients of the fitted model

         * rinfo[100] (:cpp:enumerator:`da_linmod_rinfo`): a set of values of interest
            * rinfo[0]: :math:`n_{features}`, the number of features in the model.
            * rinfo[1]: :math:`n_{samples}`, the number of samples the model has been trained on.
            * rinfo[2]: :math:`n_{coef}`, the number of model coefficients.
            * rinfo[3]: intercept, 1 if an intercept term is present in the model, 0 otherwise.
            * rinfo[4]: :math:`\alpha`, share of the :math:`\ell_1` term in the regularization.
            * rinfo[5]: :math:`\lambda`, the magnitude of the regularization term.
            * rinfo[6-99]: reserved for future use.


.. _nlls_options:

Nonlinear Least-Squares Options
===============================

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.linear_model.linmod` class constructor.

   .. tab-item:: C
      :sync: C

      Various options can be set to customize the linear models by calling one of these
      :ref:`functions <api_handle_options>`. The following table details the available options, where :math:`\epsilon` represents the machine precision.

      .. update options using table _opts_linearmodel

      .. csv-table:: Linear models options
         :header: "Option name", "Type", "Default", "Description", "Constraints"

         "optim method", "string", ":math:`s=` `auto`", "Select optimization method to use.", ":math:`s=` `auto`, `bfgs`, `cg`, `chol`, `cholesky`, `coord`, `lbfgs`, `lbfgsb`, `qr`, `sparse_cg`, or `svd`."
         "optim progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "factor used to detect convergence of the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 \le r`"
         "optim convergence tol", "real", ":math:`r=10/2\sqrt{2\,\varepsilon}`", "tolerance to declare convergence for the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 < r < 1`"
         "print options", "string", ":math:`s=` `no`", "Print options.", ":math:`s=` `no`, or `yes`."
         "lambda", "real", ":math:`r=0`", "penalty coefficient for the regularization terms: lambda( (1-alpha)/2 L2 + alpha L1 )", ":math:`0 \le r`"
         "optim iteration limit", "integer", ":math:`i=10000`", "Maximum number of iterations to perform in the optimization phase. Valid only for iterative solvers, e.g. L-BFGS-B, Coordinate Descent, etc.", ":math:`1 \le i`"
         "intercept", "integer", ":math:`i=0`", "Add intercept variable to the model", ":math:`0 \le i \le 1`"
         "print level", "integer", ":math:`i=0`", "set level of verbosity for the solver", ":math:`0 \le i \le 5`"

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

      The code below can be found in ``linear_model.cpp`` in the ``examples`` folder of your installation.

      .. collapse:: Linear Model Example

          .. literalinclude:: ../../tests/examples/pca.cpp
              :language: C++
              :linenos:



Further Reading
===============

An introduction to linear models for Regression and Classification can be found in Chapters 3, 4 of :cite:t:`bishop`, or
in Chapters 3-5 of :cite:t:`hastie`.

.. toctree::
    :maxdepth: 1
    :hidden:

    nlls_api
