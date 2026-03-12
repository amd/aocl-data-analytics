..
    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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


Approximate Nearest Neighbors
*****************************



This chapter contains functions for computing the approximate nearest neighbors of a test data set and a training data set.

.. _ann_intro:

Approximate nearest neighbors (ANN) is a technique for efficiently finding points in a dataset that are close to a given query point, without requiring an exhaustive search through all data points.
Unlike exact nearest neighbor search, which guarantees finding the true nearest neighbors, ANN algorithms trade some accuracy for improvements in runtime.
For each point :math:`x_i` in a test set :math:`X_{test}`, these algorithms compute the approximate :math:`k` nearest points to :math:`x_i` from a reference dataset :math:`X`.

AOCL-DA implements the IVFFlat algorithm for approximate nearest neighbor search (:cite:t:`da_sizi03`).
This algorithm organizes data into a search structure called an *inverted file index*.
The index partitions data into clusters, each defined by a centroid, to enable fast approximate queries.

During training, *k*-means clustering is applied to the training data to compute :math:`n_{list}` centroids.
The `train fraction` option controls how much of the training data is used for *k*-means clustering; using a smaller value can speed up training for large datasets while still producing a reasonable partition.
The `k-means_iter` option sets the maximum number of *k*-means iterations.
Note that training only computes the centroids; no data is stored in the index at this stage.

To populate the index, data must be explicitly added using a separate add operation.
Each point is assigned to its nearest centroid, and a list of points is stored for each centroid.
Additional data can be added at any time after training without recomputing the centroids.

At search time, rather than comparing the query point against all indexed data, IVFFlat first identifies the :math:`n_{probe}` centroids closest to the query.
Only the points assigned to these centroids are then searched exhaustively.
This provides a speedup over brute-force search.
However, if one of the true nearest neighbors is assigned to a centroid not among the :math:`n_{probe}` closest, it will be missed.

The key parameters for controlling the accuracy-speed trade-off are :math:`n_{list}` (the number of centroids) and :math:`n_{probe}` (the number of centroids to search).
Larger :math:`n_{list}` creates more centroids with fewer points each.
Larger :math:`n_{probe}` improves recall at the cost of search speed; setting :math:`n_{probe} = n_{list}` recovers exact search.
A common guideline is to set :math:`n_{list} \approx \sqrt{N}` for a dataset of size :math:`N`, and :math:`n_{probe}` to a small fraction of :math:`n_{list}` depending on the desired recall.

The `metric` option determines how nearness is measured.
The available metrics are:

- **Euclidean distance**: :math:`d(x,y) = \sqrt{\sum_i (x_i - y_i)^2}`.
- **Squared Euclidean distance**: :math:`d(x,y) = \sum_i (x_i - y_i)^2`.
- **Cosine distance**: :math:`d(x,y) = 1 - \frac{x^\mathrm{T}y}{\|x\| \|y\|}`.
- **Inner product**: :math:`\langle x, y \rangle = x^\mathrm{T}y`.

The squared Euclidean distance omits the square root computation, making it slightly faster to compute while preserving the same ordering of neighbors as Euclidean distance.

For `euclidean` and `sqeuclidean` metrics, standard *k*-means clustering is used during training.
For `cosine` and `inner product`, spherical *k*-means clustering is used instead, which normalizes the centroids to unit length at each iteration (:cite:t:`da_dhdh01`).

For `euclidean`, `sqeuclidean`, and `cosine`, the points with the smallest distances to the query are returned.
For `inner product`, maximum inner product search is performed, returning the points with the largest inner products with the query.

Outputs from approximate nearest neighbors
------------------------------------------
After training and adding data to the index, the following results are stored:

- **cluster centroids** - the :math:`n_{list}` centroids computed by *k*-means clustering during training. Each centroid is a vector of length :math:`n_{features}`.
- **list sizes** - the number of data points assigned to each centroid. This is an array of length :math:`n_{list}`.
- **n_list** - the number of lists in the index.
- **n_index** - the number of data points that have been added to the index.
- **n_features** - the number of features (dimensions) of the data.
- **k-means iterations** - the number of *k*-means iterations performed during training.

After querying the index with :func:`kneighbors`, the following results are returned:

- **indices of nearest neighbors** - the indices of the approximate nearest neighbors for each query point, provided in order of distance (closest first).
- **distances to the nearest neighbors** (optional) - the corresponding distances from each query point to its neighbors.

.. note::

   When querying the index, if the total number of data points stored across the :math:`n_{probe}` closest
   centroids is less than the requested number of neighbors :math:`k`, fewer than :math:`k` valid neighbors
   will be returned. In this case, the remaining entries in the output arrays are filled with sentinel values:

   - **indices**: set to :math:`-1`
   - **distances**: set to positive infinity for `euclidean`, `sqeuclidean`, and `cosine` metrics, or negative infinity for `inner product`

   This situation can occur when :math:`n_{probe}` is small and the probed centroids happen to contain
   few data points. To avoid this, ensure that enough centroids are probed or that the index contains
   sufficient data. The ``list_sizes`` output can be used to check how many data points are assigned to each centroid
   after adding data to the index. If each list contains at least :math:`k` data points, then any query will be guaranteed to return :math:`k` valid neighbors.

Typical workflow for ANN
------------------------

The standard way of computing the approximate nearest neighbors using AOCL-DA is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize an :func:`aoclda.neighbors.approximate_neighbors` object with options set in the class constructor.
      2. Train the model using :func:`aoclda.neighbors.approximate_neighbors.train` (to compute centroids only) or :func:`aoclda.neighbors.approximate_neighbors.train_and_add` (to train and add data in one step).
      3. If using :func:`aoclda.neighbors.approximate_neighbors.train`, populate the index by calling :func:`aoclda.neighbors.approximate_neighbors.add`.
      4. Query the index using :func:`aoclda.neighbors.approximate_neighbors.kneighbors` to find the approximate nearest neighbors for query points.
      5. Optionally, extract results from the object via its class attributes, such as ``cluster_centroids`` or ``list_sizes``.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_approx_nn``.
      2. Set options for the algorithm using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <ann_options>`).
      3. Pass training data to the handle using :ref:`da_approx_nn_set_training_data_? <da_approx_nn_set_training_data>`.
      4. Train the index using :ref:`da_approx_nn_train_? <da_approx_nn_train>`, then add data with :ref:`da_approx_nn_add_? <da_approx_nn_add>`. Alternatively, use :ref:`da_approx_nn_train_and_add_? <da_approx_nn_train_and_add>` to train and add data in one step.
      5. Query the index using :ref:`da_approx_nn_kneighbors_? <da_approx_nn_kneighbors>` to find the approximate nearest neighbors for query points.
      6. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`.

.. note::

   The :ref:`da_approx_nn_train_and_add_? <da_approx_nn_train_and_add>` function in C
   (or :func:`~aoclda.neighbors.approximate_neighbors.train_and_add` method in Python)
   is a convenience function for the common case where the same data used for training is also to
   be added to the index. This avoids having to call ``train`` followed by ``add`` with the same data.

.. _ann_options:

Options
-------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.neighbors.approximate_neighbors` class constructor.

   .. tab-item:: C
      :sync: C

      The following options can be set using :ref:`da_options_set_? <da_options_set>`:

      .. update options using table _opts_approximatenearestneighbors

      .. csv-table:: :strong:`Table of options for Approximate Nearest Neighbors.`
         :escape: ~
         :header: "Option name", "Type", "Default", "Description", "Constraints"

         "number of neighbors", "integer", ":math:`i=5`", "Number of neighbors considered for k-nearest neighbors.", ":math:`1 \le i`"
         "n_list", "integer", ":math:`i=1`", "Number of lists to construct for inverted file indices", ":math:`1 \le i`"
         "n_probe", "integer", ":math:`i=1`", "Number of lists to probe at search time for inverted file indices", ":math:`1 \le i`"
         "k-means_iter", "integer", ":math:`i=10`", "Maximum number of k-means iterations to perform at train time", ":math:`1 \le i`"
         "seed", "integer", ":math:`i=0`", "Seed for random number generation; set to -1 for non-deterministic results.", ":math:`-1 \le i`"
         "train fraction", "real", ":math:`r=1`", "Fraction of training data to use for k-means clustering.", ":math:`0 < r \le 1`"
         "algorithm", "string", ":math:`s=` `ivfflat`", "Algorithm used to compute the approximate nearest neighbors.", ":math:`s=` `auto`, or `ivfflat`."
         "metric", "string", ":math:`s=` `sqeuclidean`", "Metric used to compute distances.", ":math:`s=` `cosine`, `euclidean`, `inner product`, or `sqeuclidean`."
         "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."
         "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."

      If `algorithm` is set to `auto`, it defaults to `ivfflat`, the only currently available algorithm.

Examples
--------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. dropdown:: Approximate Nearest Neighbors Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/approximate_neighbors_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The code below can be found in ``ann.cpp`` in the ``examples`` folder of your installation.

      .. dropdown:: Approximate Nearest Neighbors Example

          .. literalinclude:: ../../../tests/examples/ann.cpp
              :language: C++
              :linenos:

Approximate Nearest Neighbors APIs
----------------------------------

Approximate Nearest Neighbors
=============================
.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.neighbors.approximate_neighbors(n_neighbors=5, algorithm='ivfflat', metric='sqeuclidean', n_list=1, n_probe=1, kmeans_iter=10, train_fraction=1.0, seed=0)
         :members:

   .. tab-item:: C

      .. _da_approx_nn_set_training_data:

      .. doxygenfunction:: da_approx_nn_set_training_data_s
         :project: da
         :outline:
      .. doxygenfunction:: da_approx_nn_set_training_data_d
         :project: da

      .. _da_approx_nn_train:

      .. doxygenfunction:: da_approx_nn_train_s
         :project: da
         :outline:
      .. doxygenfunction:: da_approx_nn_train_d
         :project: da

      .. _da_approx_nn_add:

      .. doxygenfunction::  da_approx_nn_add_s
         :project: da
         :outline:
      .. doxygenfunction:: da_approx_nn_add_d
         :project: da

      .. _da_approx_nn_train_and_add:

      .. doxygenfunction:: da_approx_nn_train_and_add_s
         :project: da
         :outline:
      .. doxygenfunction:: da_approx_nn_train_and_add_d
         :project: da

      .. _da_approx_nn_kneighbors:

      .. doxygenfunction:: da_approx_nn_kneighbors_s
         :project: da
         :outline:
      .. doxygenfunction:: da_approx_nn_kneighbors_d
         :project: da
