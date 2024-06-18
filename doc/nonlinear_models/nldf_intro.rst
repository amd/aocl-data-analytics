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

      1. Initialize a :func:`aoclda.nonlinear_model.nlls` object with some options set in the class constructor.
      2. Fit a nonlinear model to the data using :func:`aoclda.nonlinear_model.nlls.fit`. Here you will have to provide some functions that
         define the nonlinear model's residual vector and a function to return the residual Jacobian matrix. The optimized parameters,
         \f$x\f$, are modified in-place and returned on the interface of :func:`aoclda.nonlinear_model.nlls.fit`.
      3. Extract results from the :func:`aoclda.nonlinear_model.nlls` object via its class attributes.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_nlls``.
      2. Pass model to the handle using :ref:`da_nlls_define_residuals_? <da_nlls_define_residuals>`.
      3. Customize the model using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <nlls_options>` for a list of the available options).
      4. Train the nonlinear model using :ref:`da_nlls_fit_? <da_nlls_fit>` (you will have to provide an initial guess).
      5. Optimized parameters, \f$x\f$ are returned on the interface.
      6. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`
         using :cpp:enum:`da_result::da_rinfo`.
         The following results are available in the :code:`info[100]` array:

            * info[0] objective value,
            * info[1] gradient norm of objective,
            * info[2] number of iterations,
            * info[3] reserved for future use,
            * info[4] number of function callback evaluations,
            * info[5] reserved for future use,
            * info[6] reserved for future use,
            * info[7] reserved for future use,
            * info[8] number of gradient callback evaluations,
            * info[9] number of Hessian callback evaluations,
            * info[10] number of Hessian-vector callback evaluations,
            * info[11] scaled gradient norm of objective,
            * info[12-99]: reserved for future use.

.. _nlls_options:

Nonlinear Least-Squares Options
===============================

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.nonlinear_model.nlls` class constructor.

   .. tab-item:: C
      :sync: C

      Various options can be set to customize the nonlinear models by calling one of these
      :ref:`functions <api_handle_options>`. The following table details the available options, where :math:`\epsilon` represents the machine precision.

      .. update options using table _opts_optimizationsolvers

      .. csv-table:: Nonlinear data fitting options
         :header: "Option name", "Type", "Default", "Description", "Constraints"

         "ralfit model", "string", ":math:`s=` `hybrid`", "NLLS model to solve.", ":math:`s=` `gauss-newton`, `hybrid`, `quasi-newton`, or `tensor-newton`."
         "ralfit nlls method", "string", ":math:`s=` `galahad`", "NLLS solver to use.", ":math:`s=` `aint`, `galahad`, `linear solver`, `more-sorensen`, or `powell-dogleg`."
         "ralfit globalization method", "string", ":math:`s=` `trust-region`", "Globalization method to use. This parameter makes use of the regularization term and power option values.", ":math:`s=` `reg`, `regularization`, `tr`, or `trust-region`."
         "regularization power", "string", ":math:`s=` `quadratic`", "Value for the regularization power term.", ":math:`s=` `cubic`, or `quadratic`."
         "regularization term", "real", ":math:`r=0`", "Value for the regularization term. A value of 0 disables regularization.", ":math:`0 \le r`"
         "ralfit iteration limit", "integer", ":math:`i=100`", "Maximum number of iterations to perform.", ":math:`1 \le i`"
         "ralfit convergence rel tol fun", "real", ":math:`r=10^{-8}`", "relative tolerance to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
         "ralfit convergence abs tol fun", "real", ":math:`r=10^{-8}`", "absolute tolerance to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
         "ralfit convergence rel tol grd", "real", ":math:`r=10^{-8}`", "relative tolerance on the gradient norm to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
         "ralfit convergence abs tol grd", "real", ":math:`r=10^{-5}`", "absolute tolerance on the gradient norm to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
         "ralfit convergence step size", "real", ":math:`r=\varepsilon/2`", "absolute tolerance over the step size to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
         "print level", "integer", ":math:`i=1`", "set level of verbosity for the solver 0 indicates no output while 5 is a very verbose printing", ":math:`0 \le i \le 5`"
         "print options", "string", ":math:`s=` `no`", "Print options list", ":math:`s=` `no`, or `yes`."
         "storage scheme", "string", ":math:`s=` `c`", "Define the storage scheme used to store multi-dimensional arrays (Jacobian matrix, etc).", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."

Examples
========

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. collapse:: Nonlinear Data Fitting Example

          .. literalinclude:: ../../python_interface/python_package/aoclda/examples/nlls_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The code below can be found in ``nlls.cpp`` in the ``examples`` folder of your installation.

      .. collapse:: Nonlinear Data Fitting Example

          .. literalinclude:: ../../tests/examples/nlls.cpp
              :language: C++
              :linenos:



Further Reading
===============

An introduction to nonlinear least-squares methods can be found in
:cite:t:`NocWri06NumOpt`.
Indepth literature on modern trust-region solvers can be reviewed in:
:cite:t:`ralfit`,
:cite:t:`kanzow`,
:cite:t:`adachi`,
:cite:t:`ConGouToi00TR`, and
:cite:t:`galahad`.

.. toctree::
    :maxdepth: 1
    :hidden:

    nldf_api
