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


*k*-Nearest Neighbors (*k*-NN)
******************************

This chapter contains functions for computing the *k*-nearest neighbors of a test data set and a training data set.
Nearest neighbors functionality provided in this chapter can be used for classification.

.. _knn_intro:

The *k*-nearest neighbors algorithm is a supervised learning method used for classification and regression.
For each point :math:`x_i` of a test data set :math:`X_{test}`, this algorithm computes the :math:`k` points of a given training set :math:`X_{train}`
which are the most similar to :math:`x_i`.

In addition, when a vector :math:`y_{train}` with the associated labels for each data point in :math:`X_{train}` is provided, this algorithm
computes the predicted labels of the test data :math:`X_{test}`. A query point :math:`x_i` of :math:`X_{test}` is labeled using the majority vote of the neighbors.
In case of a tie, the first class label is returned by convension.

Outputs from *k*-Nearest Neighbors
----------------------------------
The following results can be computed with this algorithm:

- **indices of nearest neighbors** - the indices of the nearest neighbors for each data point, provided in order of distance (closest first).
- **distances to the nearest neighbors** - the corresponding distances to each data point.
- **number of classes** - the number of available class labels. Used to allocate memory for the probability prediction.
- **class labels** - the available class labels, sorted in ascending order.
- **probability estimates** - the probability that data points are labeled according to the available classes.
- **predicted labels** - the predicted label for each of the queries in :math:`X_{test}`.

Typical workflow for *k*-NN
---------------------------

The standard way of computing the *k*-nearest neighbors using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_knn``.
      2. Pass data to the handle using :ref:`da_knn_set_training_data_? <da_knn_set_training_data>`.
      3. Set the number of neighbors required and the metric, or weights used in *k*-NN using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <knn_options>`).
      4. Compute the indices to the nearest neighbors and optionally the corresponding distances using :ref:`da_knn_kneighbors_? <da_knn_kneighbors>`.
      5. If only the labels of the test data are required, use :ref:`da_knn_predict_? <da_knn_predict>`.
      6. If the probability estimates for each label are required, use :ref:`da_knn_predict_proba_? <da_knn_predict_proba>`. To allocate the appropriate memory space for the predicted probabilities, use :ref:`da_knn_classes_? <da_knn_classes>`.

.. _knn_options:

Options
-------

.. tab-set::

   .. tab-item:: C
      :sync: C

      For details of optional parameters see the :ref:`options section <opts_k-nearestneighbors>`.

Examples
========

.. tab-set::

   .. tab-item:: C
      :sync: C

      The code below can be found in ``knn.cpp`` in the ``examples`` folder of your installation.

      .. collapse:: k-NN Example

          .. literalinclude:: ../../tests/examples/knn.cpp
              :language: C++
              :linenos:

.. toctree::
    :maxdepth: 1
    :hidden:

    knn_api
