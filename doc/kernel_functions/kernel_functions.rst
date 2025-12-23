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



.. _chapter_kernel_functions:

Kernel Functions
****************

A kernel function computes the inner product of two data points in some (potentially high-dimensional) feature space without requiring an explicit mapping.
By carefully choosing the kernel, one can capture complex relationships or nonlinear separations in the data.

Common Kernel Functions
=========================

1. **Linear Kernel**:
    Computes the standard dot product of two vectors in the original feature space.
    Useful if the data are approximately linearly separable.

    .. math::
        K(x, y) = x \cdot y.

2. **RBF (Radial Basis Function) Kernel**:
    Projects every data point into an infinite-dimensional feature space, enabling nonlinear decision boundaries.
    Due to its flexibility, the RBF kernel often works well on a wide range of datasets and is a common default method.

    .. math::
        K(x, y) = \exp(-\gamma \|x - y\|^2).

3. **Polynomial Kernel**:
    Maps data points into higher-dimensional feature space via polynomial terms.
    Useful for capturing polynomial relationships between features.

    .. math::
        K(x, y) = (\gamma x \cdot y + c)^d.

4. **Sigmoid Kernel**:
    Similar to neural network activation functions.
    Sometimes used for certain data distributions, though not as common as RBF or polynomial.

    .. math::
        K(x, y) = \tanh(\gamma x \cdot y + c).


Examples
========

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. collapse:: Kernel Functions Example

          .. literalinclude:: ../../python_interface/python_package/aoclda/examples/kernel_functions_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The example sources can be found in the ``examples`` folder of your installation.

      .. collapse:: Kernel Functions Example

          .. literalinclude:: ../../tests/examples/kernel_functions.cpp
              :language: C++
              :linenos:


Kernel Functions APIs
========================

.. tab-set::

   .. tab-item:: Python

      .. autofunction:: aoclda.kernel_functions.rbf_kernel()
      .. autofunction:: aoclda.kernel_functions.linear_kernel()
      .. autofunction:: aoclda.kernel_functions.polynomial_kernel()
      .. autofunction:: aoclda.kernel_functions.sigmoid_kernel()

   .. tab-item:: C

      .. _da_rbf_kernel:

      .. doxygenfunction:: da_rbf_kernel_s
         :project: da
         :outline:
      .. doxygenfunction:: da_rbf_kernel_d
         :project: da

      .. da_linear_kernel:

      .. doxygenfunction:: da_linear_kernel_s
         :project: da
         :outline:
      .. doxygenfunction:: da_linear_kernel_d
         :project: da

      .. _da_polynomial_kernel:

      .. doxygenfunction:: da_polynomial_kernel_s
         :project: da
         :outline:
      .. doxygenfunction:: da_polynomial_kernel_d
         :project: da

      .. _da_sigmoid_kernel:

      .. doxygenfunction:: da_sigmoid_kernel_s
         :project: da
         :outline:
      .. doxygenfunction:: da_sigmoid_kernel_d
         :project: da

