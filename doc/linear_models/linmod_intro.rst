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
