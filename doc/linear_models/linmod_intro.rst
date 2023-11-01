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

Linear Models
*************

The topic of Linear Models encompasses a range of commonly used statistical models and fitting algorithms, including
Multiple Linear Regression, Logistic Regression, Polynomial Regression, and Nonparametric Regression.

The general form of a Linear Model fitting problem is as follows:

.. math::

    \min_{\beta}\left[ C_{\theta}\left( y, g^{-1}(\ \beta \, \phi(X)\ ) \right) \right],

where :math:`X` is an array of :math:`n_{\text{sample}}` observations with :math:`n_{\text{feat}}` features, :math:`y` is an array of :math:`n_{sample}` responses
/ labels, :math:`\phi` is a set of (possibly nonlinear) basis functions, :math:`\beta` is a set of weights /
coefficients, :math:`g^{-1}` is a (possibly nonlinear) activation / link function, and :math:`C_{\theta}` is a cost /
error function, which may depend on a set of (fixed) hyperparameters, :math:`\theta`.

Several Linear Models satisfy a form where the cost function can be split into a Loss function (which measures quality
of fit to the data) and a penalty term (which regularizes the model parameters).  Regularization is also referred to as
shrinkage because it tends to shrink the size of the parameter values and/or shrink the number of non-zero parameters.

The Loss function is typically a function of the responses or labels (:math:`y`), the features (:math:`X`), and the model
parameters (:math:`\beta`), whereas the penalty term is typically only a function of the 1-norm and/or the 2-norm of the
model parameters (:math:`\beta`).  Such Linear Models are often referred to as Elastic-Nets.

If, in addition to the conditions above, :math:`\phi` and :math:`g^{-1}` are identity mappings we get the following form
for the cost function,

.. math::

   C_{\{\lambda,\alpha\}} \left( \beta \right) = L(y, \beta X) 
   + \lambda \bigg( \alpha \lVert \beta \rVert_1 + (1 - \alpha) \lVert \beta \rVert_2^2  \bigg),

where :math:`0\le\lambda, 0\le\alpha\le1` are hyperparameters, :math:`\lVert \beta \rVert_1` is the 1-norm of :math:`\beta`
and :math:`\lVert \beta \rVert_2` is the 2-norm, while :math:`\lambda` sets the magnitude of the overall penalization,
:math:`\alpha` distributes its share across the :math:`\ell_1` and :math:`\ell_2` regularization terms. :math:`L` is known as the
*Loss function*. Linear Models
where :math:`\alpha=0` are called Ridge Regression, conversely, when :math:`\alpha=1` the model is called Lasso.

**Intercept**


If a Linear Model contains an intercept term, the parameter vector :math:`\beta` has dimension :math:`n_{\text{feat}}+1` and the
observations :math:`X_i` are extended with a constant of :math:`1`.  For example, if there is a single feature and the
Linear Model has an intercept term the cost function becomes,

.. math::

   C\left( \beta_0, \beta_1 \right) = \sum_{i=0}^n (y_i - \beta_0 - \beta_1 x_i)^2,

where :math:`x_i` represents a single (scalar) observation.

In general, the intercept can be added or not by setting an option in the linear regression handle.

**Regularization**

Ridge regression is a shrinkage method that penalizes large parameter values.  More specifically, Ridge Regression is an
extension of the basic Linear Model with :math:`\ell_2`` regularization.  The cost function for Ridge
regression with MSE loss function is,

.. math::
   C_{\{0<\lambda,\alpha=0\}} \left( \beta \right) &= \text{MSE}(y, \beta X) 
   + \lambda \bigg( \alpha \lVert \beta \rVert_1 + (1 - \alpha) \lVert \beta \rVert_2^2  \bigg)\\
   C_{\lambda}\left( \beta \right) &= \sum_{i=1}^n (y_i - \beta X_i)^2 + \lambda \lVert \beta \rVert_2^2,

where :math:`\lambda` is a user-defined hyperparameter controlling the amount of regularization, and :math:`\lVert \beta
\rVert_2` is the :math:`\ell_2` norm of :math:`\beta`,

.. math::

   \lVert \beta \rVert_2^2 = \sum_{i=1}^d \beta_j^2.

The Lasso is another shrinkage method similar to Ridge regression but the :math:`\ell_1` norm is used instead of the :math:`\ell_2` norm.
It can be thought of as a kind of continuous subset selection as the penalty term causes some of the coefficients to be
exactly zero when :math:`\lambda` is sufficiently large, these are also known as *sparse solutions*.  The cost function for the Lasso (with a general Loss function)
is,

.. math::

   C_{\lambda}\left( \beta \right) = L(y, \beta X) + \lambda \sum_{i=1}^d \left| \beta_j \right|,

where :math:`\lambda` is again a user-defined hyperparameter controlling the amount of regularization.



Linear Regression Models
========================

Models can be classified by their loss function, these TODO

The following subsection presents the supported loss functions.

Mean Square Error
-----------------

The cost function for a Linear Regression Model where the fit (loss) is measured by the Mean Square Error (MSE) is,

.. math::

   C_{\{0<\lambda,0\le\alpha\le1\}}\left( \beta \right) = L(y, \beta X) =\text{MSE}(y, \beta X)= \sum_{i=1}^n (y_i - \beta X_i)^2
   + \lambda \bigg( \alpha \lVert \beta \rVert_1 + (1 - \alpha) \lVert \beta \rVert_2^2  \bigg),

where :math:`X_i` represents a single (multi-dimensional) observation, i.e., a row in a table of observations.
In the following sections we describe each component of the model.

Logistic Regression
-------------------

Logistic Regression is a type of Linear Classification Model. Its main use is to classify 2 or more classes 
provided by labels in a categorical response variable, :math:`y`, encoded by :math:`\{0, 1, 2, \ldots, K-1 \}`. 
The fit is based on maximizing the log-likelihood (loss function) for the probabilities that each observation :math:`i` belongs to a given class,
inturn defined by,

.. math::
   p(y_i=k\,|\,X_i, \beta) = \frac{ \exp(\beta_k X_i) }{ 1 + \sum_{l=0}^{K-2}\exp(\beta_l X_i) }, \text{ for } 0 \leq k < K-1,\\
   p(y_i=K-1\,|\,X_i, \beta) = \frac{ 1 }{ 1 + \sum_{l=0}^{K-2}\exp(\beta_l X_i) }.

As an example, if :math:`K=2` classes, the loss function simplifies to, 

.. math::

   C\left( \beta \right) = -L(y, \beta X) = \sum_{i=0}^n \bigg( y_i \log p(X_i, \beta) + (1 - y_i) \log \big( (1 - p(X_i, \beta) \big) \bigg).

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

Direct solver
 QR: TODO

Iterative solver
 L-BFGS-B: TODO

 COORD: TODO


Linear Model Options
====================

TODO

Further Reading
===============

An introduction to Linear Models for Regression and Classification can be found in Chapters 3, 4 of :cite:t:`bishop`, or
in Chapters 3-5 of :cite:t:`hastie`.

.. toctree::
    :maxdepth: 1
    :hidden:

    linmod_api
