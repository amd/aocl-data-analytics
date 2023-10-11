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
Multiple Linear Regression, Logistic Regression, and Polynomial Regression.

The general form of a Linear Model fitting problem is as follows:

.. math::

    \min_{\beta}\left[ C_{\theta}\left( y, g^{-1}(\ \beta \, \phi(X)\ ) \right) \right]

where :math:`X` is an array of :math:`n` features with dimension :math:`d`, :math:`y` is an array of :math:`n` responses
/ labels, :math:`\phi` is a set of (possibly nonlinear) basis functions, :math:`\beta` is a set of weights /
coefficients, :math:`g^{-1}` is a (possibly nonlinear) activation / link function, and :math:`C_{\theta}` is a cost /
error function, which may depend on a set of (fixed) hyperparameters, :math:`\theta`.

An introduction to Linear Models for Regression and Classification can be found in Chapters 3, 4 of :cite:t:`bishop`, or
in Chapters 3-5 of :cite:t:`hastie`.

[Generalize to add hyperparameters / regularization.]

.. toctree::
    :maxdepth: 1
    :hidden:
    
    linmod_api
