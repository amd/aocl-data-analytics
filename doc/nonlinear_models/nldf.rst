..
    Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.

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

The topic of nonlinear data fitting encompasses a range of commonly used statistical models and fitting algorithms, including
nonlinear least-squares and its variations.

The general form of a nonlinear least-squares fitting problem is as follows:

.. math::

    \underset{x \in \Omega}{\text{minimize }} F(x) := \frac{1}{2} \|\psi(r(x))\|_W^2 + \frac{\sigma}{p} \|x\|_2^p,

where
:math:`x` is the coefficient vector of size :math:`n_{coef}` to be optimized,
:math:`\Omega` is a constraint set, :math:`r(x): R^{n_{coef}} \rightarrow R^{n_{res}}` with
:math:`\psi` a `loss` function, :math:`W` is a diagonal matrix of weights that defines the residual norm,
and :math:`\sigma` and :math:`p` are the regularization parameters.

In most common use cases, the set :math:`\Omega` is either absent or describes simple bound constraints (i.e.,
:math:`\ell_x \le x \le u_x`), the loss function is absent
and the weight matrix is the identity (making the residual norm :math:`\ell_2`). The regularization is also absent,
thus reducing the problem to

.. math::

    \underset{\ell_x \le x \le u_x}{\text{minimize }} F(x) := \frac{1}{2} \|r(x)\|_2^2.


Given the nonlinear nature of the residual function, :math:`r(x)`, there is no closed formula
as is the case for linear models. Here, the approach is to use an iterative optimization method that improves on an
initial guess at every iteration.

  .. note::

   Our prebuilt Windows Python wheels (https://www.amd.com/en/developer/aocl.html) do not include the nonlinear least squares solver.
   To access it, building from source is required. Source code and compilation instructions are available at https://github.com/amd/aocl-data-analytics/.
   If you encounter issues, please e-mail us on toolchainsupport@amd.com.

Defining a nonlinear model
============================

A model is defined inside a :ref:`handle<intro_handle>`, in which all the components of the model are configured.
In particular, any model is defined via the residual function, :math:`r(x) = \theta(t, x) - y`, where
the pair :math:`(t, y)` are the data points used to evaluate the model's residual vector.

**Residual functions**

To train the model, the optimizer needs to make calls to the residual function which is
`defined` using :cpp:func:`da_nlls_define_residuals<da_nlls_define_residuals_s>`.
Some solvers require further information such as the
first order derivatives (residual Jacobian matrix) or even second order ones.
These are also defined with this function.
Refer to :ref:`nonlinear least-squares callbacks<da_nlls_callbacks>` for further details on the
residual function signatures.

**Derivatives**

A key requirement of this iterative optimizer is to have access to first order derivatives (residual Jacobian matrix)
in order to calculate an improved solution.
There is a strong relationship between the quality of the derivatives and the
performance of the solver. If the user does not provide a call-back derivative function,
either because it is not available or by choice, then the solver will approximate the derivatives matrix using the
single-sided finite-differences method.

.. collapse:: Details

    Finite-differences is a well established and numerically effective method to estimate missing derivatives.
    The method is expensive, requiring a number of residual function calls proportional to the number of
    variables (coefficients) in the model.

    The implementation provides a single optional parameter (``'finite differences step'``) that defines the perturbation step used
    to estimate a derivative. The value of this step plays a crucial role in the quality of the approximation. The default
    is a judicious value that works for most applications.

It is strongly recommended to relax the convergence tolerances (see options) when approximating derivatives. If it is
observed that the solver "stagnates" or fails during the optimization process, tweaking the step value is encouraged.


**Verifying derivatives**

One of the most common problems while trying to train a model is having incorrect derivatives.
Writing the derivative call-back function is error-prone and to address this, a **derivative
checker** can be activated (set option ``'Check derivatives'`` to ``'yes'``) for checking the
derivatives provided by the call-back. The checker produces a table similar to

.. code::

    Begin Derivative Checker

       Jacobian storage scheme (Fortran_Jacobian) = C (row-major)

       Jac[     0,     0] =  -1.681939915944E-01 ~  -1.681939717973E-01  [ 1.177E-07], ( 0.165E-06)
       Jac[     2,     5] =   1.000000000000E+01 ~  -1.318047255339E+02  [ 1.076E+00], ( 0.100E-06)  XT
       ...
       Jac[     3,    40] =  -1.528131154992E+02 ~  -1.597095836470E+02  [ 4.318E-02], ( 0.100E-06)  XT   Skip

       Derivative checker detected    106 likely error(s)

       Note: derivative checker detected that     66 entries may correspond to the transpose.
       Verify the Jacobian storage ordering is correct.

    End Derivative Checker

.. collapse:: Details

    The reported table has a few sections. The first column after the equal sign (``=``), is the derivative
    returned by the user-supplied call-back. The column after the ``~`` sign is the approximated finite-difference
    derivative. The value inside the brackets is the relative threshold
    :math:`\frac{|\mathrm{approx} - \mathrm{exact}|}{\max(|\mathrm{approx}|,\; \mathrm{fd_ttol})}`,
    (``fd_ttol`` is defined by the option ``Derivative test tol``). The value inside the parenthesis is the relative tolerance
    to compare the relative threshold against.
    The last column provides some flags: ``X`` to indicate that the threshold is larger than the tolerance and is deemed likely
    to be wrong. ``T`` indicates that the value stored in :math:`J(i,j)` corresponds the to the value belonging to the transposed Jacobian matrix,
    providing a hint that possibly the storage sequence is incorrect. This implies that you should check in case the matrix is being stored in row-major format and that
    the solver option ``'Storage scheme'`` is set to column-major or vice-versa. Finally, ``Skip`` indicates that either the
    associated variable is fixed (constrained to a fixed value) or the bounds on it are too tight to perform a finite-difference
    approximation and thus the check for this entry cannot be performed and is skipped.

    The derivative checker uses finite-differences to compare with the user-provided derivatives and as such the
    quality of the approximation depends on the finite-difference step used (see option ``'Finite difference step'``).

    The option ``'Derivative test tol'`` is involved in defining the relative tolerance to decide if the user-supplied
    derivative is correct. A smaller value implies a more stringent test.

    Under certain circumstances the checker may signal false-positives. Tweaking the options ``'Finite difference step'``
    and ``'Derivative test tol'`` can help prevent this.

It is highly recommended that during the writing or development of the derivative call-back, you set the option
``'Check derivatives'`` to ``'yes'``.
After validating the residual Jacobian matrix, and to avoid performance impact, the option can then be reset to ``'no'``.

**Residual weights**

Under certain circumstances it is known that some residuals are more reliable than others. In such cases it is
desirable to give more importance to these. This is done by :ref:`defining the weighting matrix<da_nlls_define_weights>`, :math:`W`, using
:cpp:func:`da_nlls_define_weights<da_nlls_define_weights_s>`. Note that  :math:`W` is a diagonal matrix with
positive elements. These elements
should correspond to the inverse of the variance of each residual.

**Constraining the model**

Some models aim to explain real-life phenomena where some coefficients may not make physical sense if
they take certain invalid
values, e.g. coefficient :math:`x_j` representing a distance may not take negative values. For these cases, parameter
optimization needs to be constrained to valid values. In the previous distance example, the coefficient would be
`bound constrained` to the non-negative real half-space: :math:`0 \le x_j`.
These constraints are added to the model using :cpp:func:`da_nlls_define_bounds<da_nlls_define_bounds_s>`.

**Adding regularization**

Nonlinear models can have multiple local-minima that are undesirable, provide a biased solution or
even show signs of overfitting.
A practical way to tackle these scenarios is to introduce regularization.
Typically quadratic or cubic regularization (i.e., :math:`p=2, 3`) yield best results. Note that :math:`\sigma` and
:math:`p` are hyperparameters and are not optimized by this model, so they have to be provided by the caller.
:math:`\sigma` provides a transition between an unregularized local solution (:math:`\sigma=0`) and the
zero-coefficient vector (:math:`\sigma \gg 0`). Striking the correct balance may require trial and error
or a good understanding of the underlying model. Regularization is added by using the
optional parameters ``Regularization term`` (:math:`\sigma`) and ``Regularization power`` (:math:`p`),
see :ref:`nlls_options`.

**Training the model**

Once the model has been set up, the iterative training process is performed by calling the optimizer :cpp:func:`da_nlls_fit<da_nlls_fit_s>`.

Typical workflow for nonlinear models
=====================================

The standard way of computing a nonlinear model using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.nonlinear_model.nlls` object with some options set in the class constructor.
      2. Fit a nonlinear model to the data using :func:`aoclda.nonlinear_model.nlls.fit`. Here you will have to provide some functions that
         define the nonlinear model's residual vector and a function to return the residual Jacobian matrix. The optimized parameters,
         :math:`x`, are modified in-place and returned on the interface of :func:`aoclda.nonlinear_model.nlls.fit`.
      3. Extract results from the :func:`aoclda.nonlinear_model.nlls` object via its class attributes.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_nlls``.
      2. Pass the model to the handle using :ref:`da_nlls_define_residuals_? <da_nlls_define_residuals>`.
      3. Customize the model using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <nlls_options>` for a list of the available options).
      4. Train the nonlinear model using :ref:`da_nlls_fit_? <da_nlls_fit>` (you will have to provide an initial guess).
      5. Optimized coefficients :math:`x` are returned on the interface.
      6. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`
         using :cpp:enumerator:`da_result_::da_rinfo`.

         * The following results are available in the :code:`info[100]` array:

            * info[0]: objective value,
            * info[1]: norm of objective gradient,
            * info[2]: number of iterations,
            * info[3]: reserved for future use,
            * info[4]: number of function callback evaluations (includes ``info[9]``),
            * info[5]: number of gradient callback evaluations,
            * info[6]: number of Hessian callback evaluations,
            * info[7]: number of Hessian-vector callback evaluations,
            * info[8]: scaled gradient norm of objective,
            * info[9]: number of objective function callback evaluations used
              for approximating the derivatives or due to derivative checker,
            * info[10-99]: reserved for future use.

.. _nlls_options:

Nonlinear least-squares options
===============================

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.nonlinear_model.nlls` class constructor and
      the :func:`aoclda.nonlinear_model.nlls.fit` method.

   .. tab-item:: C
      :sync: C

      Various options can be set to customize the nonlinear models by calling one of these
      :ref:`functions <api_handle_options>`. The following table details the available options, where :math:`\epsilon` represents the machine precision.

      .. update options using table _opts_nonlinearleastsquares

      .. csv-table:: Nonlinear data fitting options
         :header: "Option name", "Type", "Default", "Description", "Constraints"

         "ralfit model", "string", ":math:`s=` `hybrid`", "NLLS model to solve.", ":math:`s=` `gauss-newton`, `hybrid`, `quasi-newton`, or `tensor-newton`."
         "ralfit nlls method", "string", ":math:`s=` `galahad`", "NLLS solver to use.", ":math:`s=` `aint`, `galahad`, `linear solver`, `more-sorensen`, or `powell-dogleg`."
         "ralfit globalization method", "string", ":math:`s=` `trust-region`", "Globalization method to use. This parameter makes use of the regularization term and power option values.", ":math:`s=` `reg`, `regularization`, `tr`, or `trust-region`."
         "regularization power", "string", ":math:`s=` `quadratic`", "Value of the regularization power term.", ":math:`s=` `cubic`, or `quadratic`."
         "regularization term", "real", ":math:`r=0`", "Value of the regularization term. A value of 0 disables regularization.", ":math:`0 \le r`"
         "ralfit iteration limit", "integer", ":math:`i=100`", "Maximum number of iterations to perform.", ":math:`1 \le i`"
         "ralfit convergence rel tol fun", "real", ":math:`r=10/21\sqrt{2\,\varepsilon}`", "Relative tolerance to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
         "ralfit convergence abs tol fun", "real", ":math:`r=10/21\sqrt{2\,\varepsilon}`", "Absolute tolerance to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
         "ralfit convergence rel tol grd", "real", ":math:`r=10/21\sqrt{2\,\varepsilon}`", "Relative tolerance on the gradient norm to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
         "ralfit convergence abs tol grd", "real", ":math:`r=500\;\sqrt{2\,\varepsilon}`", "Absolute tolerance on the gradient norm to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
         "ralfit convergence step size", "real", ":math:`r=\varepsilon/2`", "Absolute tolerance over the step size to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
         "print level", "integer", ":math:`i=1`", "Set level of verbosity for the solver: from 0, indicating no output, to 5, which is very verbose.", ":math:`0 \le i \le 5`"
         "print options", "string", ":math:`s=` `no`", "Print options list.", ":math:`s=` `no`, or `yes`."
         "check derivatives", "string", ":math:`s=` `no`", "Check user-provided derivatives using finite-differences.", ":math:`s=` `no`, or `yes`."
         "finite differences step", "real", ":math:`r=10\;\sqrt{2\,\varepsilon}`", "Size of step to use for estimating derivatives using finite-differences.", ":math:`0 < r < 10`"
         "derivative test tol", "real", ":math:`r=10^{-4}`", "Tolerance used to check user-provided derivatives by finite-differences. If <print level> is 1, then only the entries with larger discrepancy are reported, and if print level is greater than or equal to 2, then all entries are printed.", ":math:`0 < r \le 10`"
         "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
         "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."

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

      .. collapse:: Nonlinear Data Fitting Example (using finite-differences)

          .. literalinclude:: ../../python_interface/python_package/aoclda/examples/nlls_fd_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The code below can be found in ``nlls.cpp`` in the ``examples`` folder of your installation.

      .. collapse:: Nonlinear Data Fitting Example

          .. literalinclude:: ../../tests/examples/nlls.cpp
              :language: C++
              :linenos:

      .. collapse:: Nonlinear Data Fitting Example (Lanczos)

          .. literalinclude:: ../../tests/examples/nlls_lanczos_fd.cpp
              :language: C++
              :linenos:



Further reading
===============

An introduction to nonlinear least-squares methods can be found in
:cite:t:`da_NocWri06NumOpt`.
In-depth literature on modern trust-region solvers can be reviewed in:
:cite:t:`da_ralfit`,
:cite:t:`da_kanzow`,
:cite:t:`da_adachi`,
:cite:t:`da_ConGouToi00TR`, and
:cite:t:`da_galahad`.




Nonlinear Data Fitting APIs
=============================


.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.nonlinear_model.nlls(n_coef, n_res, weights=None, lower_bounds=None, upper_bounds=None, order='c', model='hybrid', method='galahad', glob_strategy='tr', reg_power='quadratic', verbose=0, check_data=false)
         :members:

   .. tab-item:: C

      .. _da_nlls_callbacks:

      .. doxygentypedef:: da_resfun_t_s
         :project: da
         :outline:
      .. doxygentypedef:: da_resfun_t_d
         :project: da

      .. _da_nlls_callbacks_j:

      .. doxygentypedef:: da_resgrd_t_s
         :project: da
         :outline:
      .. doxygentypedef:: da_resgrd_t_d
         :project: da

      .. _da_nlls_callbacks_hf:

      .. doxygentypedef:: da_reshes_t_s
         :project: da
         :outline:
      .. doxygentypedef:: da_reshes_t_d
         :project: da

      .. _da_nlls_callbacks_hp:

      .. doxygentypedef:: da_reshp_t_s
         :project: da
         :outline:
      .. doxygentypedef:: da_reshp_t_d
         :project: da

      .. _da_nlls_define_residuals:

      .. doxygenfunction:: da_nlls_define_residuals_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nlls_define_residuals_d
         :project: da

      .. _da_nlls_define_weights:

      .. doxygenfunction:: da_nlls_define_weights_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nlls_define_weights_d
         :project: da

      .. _da_nlls_define_bounds:

      .. doxygenfunction:: da_nlls_define_bounds_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nlls_define_bounds_d
         :project: da

      .. _da_nlls_fit:

      .. doxygenfunction:: da_nlls_fit_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nlls_fit_d
         :project: da

      .. _da_optim_info_t:

      .. doxygenenum:: da_optim_info_t_
         :project: da
