..
    Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

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



.. _chapter_svm:

Support Vector Machines (SVM)
*****************************

Introduction
============

Given a feature matrix :math:`X` and corresponding labels :math:`y`, a support vector machine (SVM) aims to find an optimal separating hyperplane that
best classifies or fits the data. Optimal here means that it maximizes the distance (also called the margin) from nearest datapoint to the hyperplane.

Mathematical formulation
------------------------
SVM can have different formulations depending on the task at hand. The most common ones are:

- **Support Vector Classification (SVC)**:
  Uses a parameter :math:`C` to balance margin maximization and classification errors, seeking a decision boundary that maximizes the margin while penalizing misclassifications.

- **Nu-Support Vector Classification (NuSVC)**:
  Employs a parameter :math:`\nu` to constrain the fraction of training errors and the fraction of support vectors, providing an alternative classification formulation to SVC.

- **Support Vector Regression (SVR)**:
  Fits a function that allows for a margin of tolerance :math:`\epsilon` around the predicted values, penalizing data points lying outside this tolerance to achieve a trade-off between model complexity and fitting errors.

- **Nu-Support Vector Regression (NuSVR)**:
  Similar to SVR but utilizes a parameter :math:`\nu` to control the allowable support vectors and fitting errors, offering a different mechanism to balance model complexity and error tolerance.

In the simplest case (binary classification), given training samples :math:`\{ (x_i, y_i) \}_{i=1}^{n}` with :math:`y_i \in \{-1, +1\}`,
the SVC in its primal form, poses the following problem:

.. math::

   \min_{\beta,\,b,\,\xi_i} \frac{1}{2} \|\beta\|^2 &+ C \sum_{i=1}^{n} \xi_i \\
   \text{subject to}
   \quad
   y_i (\beta^\top \phi(x_i) + b ) &\ge 1 - \xi_i,\quad
   \xi_i \ge 0,

where :math:`\beta` represents the vector of coefficients, also known as the weights, associated with each input feature, :math:`b` is the bias term, 
:math:`\phi` is a (possibly nonlinear) mapping defined by a kernel, :math:`C > 0` is a regularization parameter controlling the trade-off between 
margin maximization and misclassification, and :math:`\xi_i` (known as a slack variable) is the distance from the :math:`i`-th sample to the correct boundary.

Minimizing the norm of the coefficients leads to maximizing the margin between classes. Meanwhile, slack variables introduce a penalty for misclassification 
errors; if a data point is correctly classified, :math:`\xi_i` is 0, otherwise, :math:`\xi_i` is greater than 0. The optimal hyperplane is determined by 
a subset of the training samples known as *support vectors*, which lie on or within the margin.

Since the feature space can be high-dimensional or data might not be linearly separable, it is more convenient to solve the dual version of this problem. 
This formulation focuses on the dual coefficients :math:`\alpha_i` rather than explicit feature transformations. The corresponding objective problem is:

.. math::

   \min_{\alpha} \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \sum_{i=1}^{n} \alpha_i \\
   \text{subject to}
   \quad
   0 \le \alpha_i \le C,\quad
   \sum_{i=1}^{n} \alpha_i y_i = 0,

where :math:`K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)` is the kernel function. This eliminates the need to perform an explicit mapping :math:`\phi(x)`, allowing SVMs to handle 
both linear and nonlinear relationships in the data. The decision function is then given by

.. math::

   f(x) = \sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b,

and predictions are made based on the sign of :math:`f(x)`.

Multi-class classification with SVMs is approached using a **one-vs-one** strategy, which decomposes the multi-class task in a way that each class is paired with each 
other to form :math:`\frac{n_{\mathrm{class}} \times (n_{\mathrm{class}}-1)}{2}` binary classification submodels. Each submodel learns to distinguish between two 
classes. The final label is determined by aggregating the results of these binary decisions, by a voting mechanism.

.. note::

   In :ref:`da_svm_decision_function_? <da_svm_decision_function>` you can access decision function values in both OvO (one-vs-one) and OvR (one-vs-rest) shapes. 
   In the one-vs-rest (OvR) approach, :math:`(n_{\mathrm{class}} - 1)` subproblems are defined, each separating a single class from all remaining classes. 
   Note that in our case, OvR decision values are derived from OvO, so this setting does not affect the underlying training process.

For regression, a similar formulation is used, minimizing errors within an margin of tolerance :math:`\epsilon` around the regression function.

Implementation details
----------------------
We implement ThunderSVM (see :cite:t:`wenthundersvm18`), a specialized variant of the Sequential Minimal Optimization (SMO) algorithm, to solve the dual problem. 
This approach iteratively decomposes the dual problem into smaller subproblems of certain size, and solves each with SMO until the overall solution converges.


Typical workflow for SVM
------------------------

The standard way of computing SVM using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The following description relates to SVC but analogous steps can be applied to other SVM models.

      1. Initialize a :func:`aoclda.svm.SVC` object with options set in the class constructor.
      2. Compute SVM on your data using :func:`aoclda.svm.SVC.fit`.
      3. Call :func:`aoclda.svm.SVC.predict` to evaluate the model on new data.
      4. Extract results from the :func:`aoclda.svm.SVC` object via its class attributes.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_svm``.
      2. Select the SVM model :cpp:type:`da_svm_model` with :ref:`da_svm_select_model_? <da_svm_select_model>`.
      3. Pass data to the handle using :ref:`da_svm_set_data_? <da_svm_set_data>`.
      4. Customize the model using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <svm_options>` for a list of the available options).
      5. Compute the SVM using :ref:`da_svm_compute_? <da_svm_compute>`.
      6. Evaluate the model on new data using :ref:`da_svm_predict_? <da_svm_predict>`.
      7. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`. The following results are available:

         * Total number of support vectors (:cpp:enumerator:`da_svm_n_support_vectors`), :math:`(n_{\mathrm{support\_vectors}},\,)`. Integer.

         * Number of support vectors per class (:cpp:enumerator:`da_svm_n_support_vectors_per_class`). Vector of size :math:`(n_{\mathrm{class}},\,)`.

         * Support vectors (:cpp:enumerator:`da_svm_support_vectors`): The subset of training samples that lie on or within the margin. Matrix of size :math:`(n_{\mathrm{support\_vectors}},\, n_{\mathrm{features}})`.
  
         * Bias (intercept) (:cpp:enumerator:`da_svm_bias`): The bias term in the decision function. Vector of size :math:`(n_{\mathrm{class}}-1,\,)`.

         * Dual coefficients (:cpp:enumerator:`da_svm_dual_coef`): :math:`\alpha` in the dual problem. Weights assigned to each support vector, reflecting their importance in defining the optimal decision boundary. Matrix of size :math:`(n_{\mathrm{support\_vectors}},\, n_{\mathrm{class}}-1)`.

         * Indexes to support vectors (:cpp:enumerator:`da_svm_idx_support_vectors`). Vector of size :math:`(n_{\mathrm{support\_vectors}},\,)`.
         
         * Number of iterations (:cpp:enumerator:`da_svm_n_iterations`). In this context it counts the number of SMO subproblems solved, for each classifier. Vector of size :math:`(n_{\mathrm{classifiers}},\,)`.

         * Some solvers provide extra information. :cpp:enumerator:`da_svm_rinfo`, when available, contains the
           info[100] array with the following values:

           * info[0]: number of rows in the input matrix,
           * info[1]: number of columns in the input matrix,
           * info[2]: number of detected classes, for regression returns 2,
           * info[3-99]: reserved for future use.

.. _svm_options:

SVM options
===========

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the respective class constructor :func:`aoclda.svm.SVC`, :func:`aoclda.svm.SVR`, :func:`aoclda.svm.NuSVC`, or :func:`aoclda.svm.NuSVR`.

   .. tab-item:: C
      :sync: C

      Various options can be set to customize the SVM models by calling one of these
      :ref:`functions <api_handle_options>`. The following table details the available options, where :math:`\epsilon` represents the machine precision.

      .. update options using table _opts_supportvectormachines

      .. csv-table:: SVM options
         :header: "Option name", "Type", "Default", "Description", "Constraints"

         "kernel", "string", ":math:`s=` `rbf`", "Kernel function to use for the calculations.", ":math:`s=` `linear`, `poly`, `polynomial`, `rbf`, or `sigmoid`."
         "coef0", "real", ":math:`r=0`", "Constant in 'polynomial' and 'sigmoid' kernels.", "There are no constraints on :math:`r`."
         "gamma", "real", ":math:`r=-1`", "Parameter for 'rbf', 'polynomial', and 'sigmoid' kernels. If the value is less than 0, it is set to 1/(n_features * Var(X)).", ":math:`-1 \le r`"
         "epsilon", "real", ":math:`r=0.1`", "Defines the tolerance for errors in predictions by creating an acceptable margin (tube) within which errors are not penalized. Applies to SVR", ":math:`0 \le r`"
         "tau", "real", ":math:`r=\varepsilon`", "Numerical stability parameter used in working set selection when kernel is not positive semi definite.", ":math:`0 \le r`"
         "tolerance", "real", ":math:`r=10^{-3}`", "Convergence tolerance.", ":math:`0 < r`"
         "nu", "real", ":math:`r=0.5`", "An upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors. Applies to NuSVC and NuSVR.", ":math:`0 < r \le 1`"
         "max_iter", "integer", ":math:`i=0`", "Sets the maximum number of iterations. Use 0 to specify no limit.", ":math:`0 \le i`"
         "c", "real", ":math:`r=1`", "Regularization parameter. Controls the trade-off between maximizing the margin between classes and minimizing classification errors. A larger value means higher penalty to the loss function on misclassified observations. Applies to SVC, SVR and NuSVR.", ":math:`0 < r`"
         "degree", "integer", ":math:`i=3`", "Parameter for 'polynomial' kernel.", ":math:`1 \le i`"
         "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
         "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."

Examples
========

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. collapse:: SVM Example

          .. literalinclude:: ../../python_interface/python_package/aoclda/examples/svm_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The example sources can be found in the ``examples`` folder of your installation.

      .. collapse:: SVC Example (column-major)

          .. literalinclude:: ../../tests/examples/svc.cpp
              :language: C++
              :linenos:

      .. collapse:: Nu-SVR Example (row-major)

          .. literalinclude:: ../../tests/examples/nusvr.cpp
              :language: C++
              :linenos:

.. toctree::
    :maxdepth: 1
    :hidden:

    svm_api
