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




Distance Metrics
****************

This chapter contains functions for the computation of distance metrics.

.. _pairwise_intro:

Pairwise distances
============================

A pairwise distance is a measure of *similarity* between samples. Consider an :math:`m \times k` matrix :math:`X`,
and an :math:`n \times k` matrix :math:`Y`.
The distance between the row vectors of :math:`X` and the row vectors of :math:`Y` is represented by a matrix :math:`D` of size :math:`m \times n`.
In addition, the distance matrix :math:`D` between each row vector of :math:`X` can be computed and is of size :math:`m \times m`.
There are different metrics used to compute the distance between two samples. The available metrics are listed below.

**Euclidean distance**

Each element :math:`D_{ij}` of the Euclidean distance matrix :math:`D`, between two matrices :math:`X` and :math:`Y`, is defined as

.. math::
    D_{ij} = \sqrt{(X_i-Y_j)^\mathrm{T}(X_i-Y_j)},

where :math:`X_i` and :math:`Y_j` denote the :math:`i`-th and :math:`j`-th rows of :math:`X` and :math:`Y`, respectively.

In AOCL-DA, we compute the Euclidean distances using the equivalent formula

.. math::
    D_{ij} = \sqrt{X_i^\mathrm{T}X_i + Y_j^\mathrm{T}Y_j - 2X_i^\mathrm{T}Y_j}.

The term :math:`X_i^\mathrm{T}Y_j` can be computed for all :math:`i,j` using highly optimized `gemm` operations, which yields better performance.

**Squared Euclidean distance**

Similarly to Euclidean distance above, each element :math:`D_{ij}` of the distance matrix :math:`D`, between two matrices :math:`X` and :math:`Y`, is defined as

.. math::
    D_{ij} = (X_i-Y_j)^\mathrm{T}(X_i-Y_j).


Examples
========

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. collapse:: Metrics Example

          .. literalinclude:: ../../python_interface/python_package/aoclda/examples/metrics_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The code below can be found in ``metrics.cpp`` in the ``examples`` folder of your installation.

      .. collapse:: Metrics Example

          .. literalinclude:: ../../tests/examples/metrics.cpp
              :language: C++
              :linenos:

.. toctree::
    :maxdepth: 1
    :hidden:

    metrics_api

