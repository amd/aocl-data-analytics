..
    Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.

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


.. _train_test_split_api:

Train Test Split
*****************

Split a dataset into training and test sets, which can optionally be shuffled and stratified.

Stratify
--------

If stratification is enabled, ``train_test_split`` will try to ensure that the relative proportions of classes in the training and test sets is the same as in the original dataset. This is particularly useful for classification tasks where you want to ensure that both sets have a representative distribution of classes.

Examples
--------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. dropdown:: train_test_split Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/train_test_split.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The code below can be found in ``train_test_split.cpp`` in the ``examples`` folder of your installation.

      .. dropdown:: train_test_split Example

          .. literalinclude:: ../../../tests/examples/train_test_split.cpp
              :language: C++
              :linenos:

Train Test Split APIs
---------------------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      .. py:function:: aoclda.utils.train_test_split(*arrays, test_size=None, train_size=None, seed=None, shuffle=True, stratify=None)

   .. tab-item:: C
      :sync: C

      .. doxygenfunction:: da_get_shuffled_indices_int
         :project: da
         :outline:
      .. doxygenfunction:: da_get_shuffled_indices_s
         :project: da
         :outline:
      .. doxygenfunction:: da_get_shuffled_indices_d
         :project: da

      .. doxygenfunction:: da_train_test_split_int
         :project: da
         :outline:
      .. doxygenfunction:: da_train_test_split_s
         :project: da
         :outline:
      .. doxygenfunction:: da_train_test_split_d
         :project: da


