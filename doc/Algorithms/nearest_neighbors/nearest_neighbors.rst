..
    Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.

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

.. _chapter_nearest_neighbors:

Nearest Neighbors
*****************

Introduction
============

This chapter contains functions for computing the :math:`k`-nearest neighbors (:math:`k`-NN) and the radius neighbors of a test data set and a training data set.
Nearest neighbors functionality provided in this chapter can be used for classification or regression problems and offers different algorithms to compute the neighbors,
namely brute force, :math:`k`-d tree, and ball tree.
These algorithms use various distance metrics to compute the similarity between data points and based on this similarity, they identify either the :math:`k` most similar
observations in a sample (in the case of :math:`k`-NN) or all observations within a specified radius (in the case of radius neighbors). Then, using the neighbors,
they predict the label or target value for each observation in the test data set.

For classification problems, when a vector :math:`y_{train}` with the associated labels for each data point in :math:`X_{train}` is provided, this algorithm
computes the predicted labels of the test data :math:`X_{test}`. A query point :math:`x_i` of :math:`X_{test}` is labeled using the majority vote of the neighbors.
In case of a tie, the first class label is returned by convention.

For regression problems, the predicted target value of a query point :math:`x_i` is computed as the mean of the target values of the neighbors.


*k*-Nearest Neighbors (*k*-NN)
==============================

The *k*-nearest neighbors algorithm is a supervised learning method used for classification and regression.
For each point :math:`x_i` of a test data set :math:`X_{test}`, this algorithm computes the :math:`k` points of a given training set :math:`X_{train}`
which are the most similar to :math:`x_i`.

Outputs from *k*-nearest neighbors
----------------------------------
The following results can be computed with this algorithm:

- **indices of nearest neighbors** - the indices of the nearest neighbors for each data point, provided in order of distance (closest first).
- **distances to the nearest neighbors** - the corresponding distances to each data point.
- **number of classes** - the number of available class labels. Used to allocate memory for the probability prediction.
- **class labels** - the available class labels, sorted in ascending order.
- **probability estimates** - the probability that data points are labeled according to the available classes.
- **predicted labels** - the predicted label for each of the queries in :math:`X_{test}` in classification problems.
- **predicted target values** - the predicted target value for each of the queries in :math:`X_{test}` in regression problems.

Typical workflow for *k*-nearest neighbors
------------------------------------------

The standard way of computing the *k*-nearest neighbors is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.neighbors.nearest_neighbors` object with options set in the class constructor.
      2. Fit the *k*-NN for your training data set using :func:`aoclda.neighbors.nearest_neighbors.fit`.
      3. Compute the indices of the nearest neighbors and optionally the corresponding distances using :func:`aoclda.neighbors.nearest_neighbors.kneighbors`.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_nn``.
      2. Pass data to the handle using :ref:`da_nn_set_data_? <da_nn_set_data>`.
      3. Compute the indices of the nearest neighbors and optionally the corresponding distances using :ref:`da_nn_kneighbors_? <da_nn_kneighbors>`.

Radius Neighbors
================

The radius neighbors algorithm is a supervised learning method used for classification and regression.
For each point :math:`x_i` of a test data set :math:`X_{test}`, this algorithm computes the points of a given training set :math:`X_{train}`
which are within the specified radius :math:`r` of :math:`x_i`.

Outputs from radius neighbors
-----------------------------
The following results can be computed with this algorithm:

- **number of neighbors** - the number of neighbors within radius :math:`r` of each data point.
- **indices of neighbors** - the indices of the neighbors within radius :math:`r` of each data point, provided in order of distance (closest first).
- **distances to the neighbors** - the corresponding distances to each data point.
- **number of classes** - the number of available class labels. Used to allocate memory for the probability prediction.
- **class labels** - the available class labels, sorted in ascending order.
- **probability estimates** - the probability that data points are labeled according to the available classes.
- **predicted labels** - the predicted label for each of the queries in :math:`X_{test}` in classification problems.
- **predicted target values** - the predicted target value for each of the queries in :math:`X_{test}` in regression problems.

Typical workflow for radius neighbors
-------------------------------------

The standard way of computing the radius neighbors using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.neighbors.nearest_neighbors` object with options set in the class constructor.
      2. Fit the nearest neighbors for your training data set using :func:`aoclda.neighbors.nearest_neighbors.fit`.
      3. Compute the indices of the neighbors and optionally the corresponding distances using :func:`aoclda.neighbors.nearest_neighbors.radius_neighbors`.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_nn``.
      2. Pass data to the handle using :ref:`da_nn_set_data_? <da_nn_set_data>`.
      3. Set the radius of neighbors required and the metric or weights used in radius neighbors using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <nn_options>`).
      4. Compute the radius neighbors and optionally the corresponding distances for each data point using :ref:`da_nn_radius_neighbors_? <da_nn_radius_neighbors>`.
      5. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`. The following results are available:

         * Number of neighbors for each query point using :cpp:enumerator:`da_nn_radius_neighbors_count`. The size of the output array is equal to the number of query points plus one. The last element contains the total number of neighbors found for all query points and can be used for memory allocation.
         * Indices of neighbors for each query point using :cpp:enumerator:`da_nn_radius_neighbors_indices_index`. The size of the output array is equal to the number of neighbors found for the specific query point.
         * Distances to neighbors for each query point using :cpp:enumerator:`da_nn_radius_neighbors_distances_index`, if available. The size of the output array is equal to the number of neighbors found for the specific query point.
         * Offsets to locate the neighbors for each query point using :cpp:enumerator:`da_nn_radius_neighbors_offsets`. The size of the output array is equal to the number of query points plus one. For query points where no neighbors were found, the offsets will be -1.
         * Indices of neighbors for all query points using :cpp:enumerator:`da_nn_radius_neighbors_indices`. The size of the output array is equal to the total number of neighbors found for all query points.
         * Distances to neighbors for all query points using :cpp:enumerator:`da_nn_radius_neighbors_distances`, if available. The size of the output array is equal to the total number of neighbors found for all query points.

         .. note::
            We can extract all indices and distances of neighbors for all query points at once using :cpp:enumerator:`da_nn_radius_neighbors_offsets`, :cpp:enumerator:`da_nn_radius_neighbors_indices` and :cpp:enumerator:`da_nn_radius_neighbors_distances`.
            Otherwise, we can extract the indices and distances of neighbors for a specific query point using :cpp:enumerator:`da_nn_radius_neighbors_count` to find out how much memory to allocate and :cpp:enumerator:`da_nn_radius_neighbors_indices_index` and :cpp:enumerator:`da_nn_radius_neighbors_distances_index`.


Typical Workflow for Classification Problems
--------------------------------------------

The standard way using nearest neighbors for classification using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.neighbors.nearest_neighbors` object with options set in the class constructor.
      2. Fit your training data set using :func:`aoclda.neighbors.nearest_neighbors.fit`.
      3. If only the labels of the test data are required, use :func:`aoclda.neighbors.nearest_neighbors.classifier_predict` with search method set to `knn` or `radius_neighbors`. Note that previous calls to :func:`aoclda.neighbors.nearest_neighbors.kneighbors` or :func:`aoclda.neighbors.nearest_neighbors.radius_neighbors` are not required.
      4. If the probability estimates for each label are required, use :func:`aoclda.neighbors.nearest_neighbors.classifier_predict_proba` with search method set to `knn` or `radius_neighbors`. Note that previous calls to :func:`aoclda.neighbors.nearest_neighbors.kneighbors` or :func:`aoclda.neighbors.nearest_neighbors.radius_neighbors` are not required.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_nn``.
      2. Pass data to the handle using :ref:`da_nn_set_data_? <da_nn_set_data>`.
      3. Set the number of neighbors, the radius, the metric or weights used in nearest neighbors using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <nn_options>`).
      4. Set the labels corresponding to the training data using :ref:`da_nn_set_labels_? <da_nn_set_labels>`. This is only required for predicting labels or probabilities.
      5. If only the labels of the test data are required, use :ref:`da_nn_classifier_predict_? <da_nn_classifier_predict>` with search method set to `knn_search_mode` or `radius_search_mode`. Note that previous calls to :ref:`da_nn_kneighbors_? <da_nn_kneighbors>` or :ref:`da_nn_radius_neighbors_? <da_nn_radius_neighbors>` are not required.
      6. If the probability estimates for each label are required, use :ref:`da_nn_classifier_predict_proba_? <da_nn_classifier_predict_proba>` with search method set to `knn_search_mode` or `radius_search_mode`. To allocate the appropriate memory space for the predicted probabilities, use :ref:`da_nn_classes_? <da_nn_classes>`. Note that previous calls to :ref:`da_nn_kneighbors_? <da_nn_kneighbors>` or :ref:`da_nn_radius_neighbors_? <da_nn_radius_neighbors>` are not required.

Typical Workflow for Regression Problems
----------------------------------------

The standard way using nearest neighbors for regression using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.neighbors.nearest_neighbors` object with options set in the class constructor.
      2. Fit your training data set using :func:`aoclda.neighbors.nearest_neighbors.fit`.
      3. If only the target values of the test data are required, use :func:`aoclda.neighbors.nearest_neighbors.regressor_predict` with search method set to `knn` or `radius_neighbors`. Note that previous calls to :func:`aoclda.neighbors.nearest_neighbors.kneighbors` or :func:`aoclda.neighbors.nearest_neighbors.radius_neighbors` are not required.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_nn``.
      2. Pass data to the handle using :ref:`da_nn_set_data_? <da_nn_set_data>`.
      3. Set the number of neighbors, the radius, the metric or weights used in nearest neighbors using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <nn_options>`).
      4. Set the targets corresponding to the training data using :ref:`da_nn_set_targets_? <da_nn_set_targets>`. This is only required for predicting target values.
      5. If only the target values of the test data are required, use :ref:`da_nn_regressor_predict_? <da_nn_regressor_predict>` with search method set to `knn_search_mode` or `radius_search_mode`. Note that previous calls to :ref:`da_nn_kneighbors_? <da_nn_kneighbors>` or :ref:`da_nn_radius_neighbors_? <da_nn_radius_neighbors>` are not required.

.. _nn_options:

Options
-------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.neighbors.nearest_neighbors` class constructor.

   .. tab-item:: C
      :sync: C

      The following options can be set using :ref:`da_options_set_? <da_options_set>`:

      .. update options using table _opts_k-nearestneighbors

      .. csv-table:: :strong:`Table of options for k-Nearest Neighbors.`
         :escape: ~
         :header: "Option name", "Type", "Default", "Description", "Constraints"

         "weights", "string", ":math:`s=` `uniform`", "Weight function used to compute the k-nearest neighbors.", ":math:`s=` `distance`, or `uniform`."
         "metric", "string", ":math:`s=` `euclidean`", "Metric used to compute the pairwise distance matrix.", ":math:`s=` `cityblock`, `cosine`, `euclidean`, `euclidean_gemm`, `l1`, `l2`, `manhattan`, `minkowski`, `sqeuclidean`, or `sqeuclidean_gemm`."
         "algorithm", "string", ":math:`s=` `auto`", "Algorithm used to compute the k-nearest neighbors.", ":math:`s=` `auto`, `ball tree`, `brute`, or `kd tree`."
         "radius", "real", ":math:`r=1`", "Maximum distance for the radius neighbors computation.", ":math:`0 \le r`"
         "minkowski parameter", "real", ":math:`r=2`", "Minkowski parameter for metric used for the computation of k-nearest neighbors.", ":math:`0 < r`"
         "number of neighbors", "integer", ":math:`i=5`", "Number of neighbors considered for k-nearest neighbors.", ":math:`1 \le i`"
         "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
         "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."

Examples
========

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. dropdown:: k-NN  Classification Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/knn_classification_ex.py
              :language: Python
              :linenos:

      .. dropdown:: k-NN  Regression Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/knn_regression_ex.py
              :language: Python
              :linenos:

      .. dropdown:: Radius Neighbors Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/nearest_neighbors_ex.py
              :language: Python
              :linenos:

      .. dropdown:: Radius Neighbors Classification Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/radius_neighbors_classification_ex.py
              :language: Python
              :linenos:

      .. dropdown:: Radius Neighbors Regression Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/radius_neighbors_regression_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The code below can be found in ``knn_classification.cpp``, ``knn_regression.cpp``, ``radius_neighbors.cpp``, ``radius_neighbors_classification.cpp``, and ``radius_neighbors_regression.cpp`` in the ``examples`` folder of your installation.

      .. dropdown:: k-NN Classification Example

          .. literalinclude:: ../../../tests/examples/knn_classification.cpp
              :language: C++
              :linenos:

      .. dropdown:: k-NN Regression Example

          .. literalinclude:: ../../../tests/examples/knn_regression.cpp
              :language: C++
              :linenos:

      .. dropdown:: Radius Neighbors Example

          .. literalinclude:: ../../../tests/examples/radius_neighbors.cpp
              :language: C++
              :linenos:

      .. dropdown:: Radius Neighbors Classification Example

          .. literalinclude:: ../../../tests/examples/radius_neighbors_classification.cpp
              :language: C++
              :linenos:

      .. dropdown:: Radius Neighbors Regression Example

          .. literalinclude:: ../../../tests/examples/radius_neighbors_regression.cpp
              :language: C++
              :linenos:


Nearest Neighbors APIs
======================

.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.neighbors.nearest_neighbors(n_neighbors=5, radius=1.0, weights='uniform', algorithm='auto', metric='minkowski', p=2.0, leaf_size=30, check_data=false)
         :members:
         :no-index:

   .. tab-item:: C

      .. _da_nn_set_data:

      .. doxygenfunction:: da_nn_set_data_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nn_set_data_d
         :project: da

      .. _da_nn_set_labels:

      .. doxygenfunction:: da_nn_set_labels_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nn_set_labels_d
         :project: da

      .. _da_nn_set_targets:

      .. doxygenfunction:: da_nn_set_targets_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nn_set_targets_d
         :project: da

      .. _da_nn_kneighbors:

      .. doxygenfunction:: da_nn_kneighbors_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nn_kneighbors_d
         :project: da

      .. _da_nn_radius_neighbors:

      .. doxygenfunction:: da_nn_radius_neighbors_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nn_radius_neighbors_d
         :project: da

      .. _da_nn_classes:

      .. doxygenfunction:: da_nn_classes_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nn_classes_d
         :project: da

      .. _da_nn_classifier_predict_proba:

      .. doxygenfunction:: da_nn_classifier_predict_proba_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nn_classifier_predict_proba_d
         :project: da

      .. _da_nn_classifier_predict:

      .. doxygenfunction:: da_nn_classifier_predict_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nn_classifier_predict_d
         :project: da

      .. _da_nn_regressor_predict:

      .. doxygenfunction:: da_nn_regressor_predict_s
         :project: da
         :outline:
      .. doxygenfunction:: da_nn_regressor_predict_d
         :project: da

      .. _da_nn_search_mode:

      .. doxygentypedef:: da_nn_search_mode
         :project: da
      .. doxygenenum:: da_nn_search_mode_
         :project: da