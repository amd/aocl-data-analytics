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




Clustering
**********

This chapter contains functions for performing clustering computations.

.. _kmeans_intro:

*k*-means clustering
============================

*k*-means clustering aims to partition a set of :math:`n_{\mathrm{samples}}` data points :math:`\{x_1, x_2, \dots, x_{n_{\mathrm{samples}}}\}` into :math:`n_{\mathrm{clusters}}` groups. Each group is described by the mean of the data points assigned to it :math:`\{\mu_1, \mu_2, \dots, \mu_{n_{\mathrm{clusters}}}\}`.
These means are commonly known as the *cluster centres* or *centroids* and are not generally points from the original data matrix.

Various *k*-means algorithms are available, but each one proceeds by attempting to minimize a quantity known as the *inertia*, or the *within-cluster sum-of-squares*, which is the sum of the squared distances of each sample to its closest cluster centre.
In the case of standard *k*-means clustering, where the distance metric is the Euclidean distance, this can be expressed as

.. math::
   \sum^{n_{\mathrm{samples}}}_{i=1}\min_{1\le j\le n_{\mathrm{clusters}}}\left(\|x_i-\mu_j\|^2\right).

In the case of *spherical* *k*-means clustering, where the distance metric is the cosine distance, the inertia is expressed as

.. math::
   \sum^{n_{\mathrm{samples}}}_{i=1}\min_{1\le j\le n_{\mathrm{clusters}}}\left(1-\frac{x_i\cdot\mu_j}{\|x_i\|\|\mu_j\|}\right).

In spherical *k*-means clustering the cluster centres are normalized to have unit length.
Data points are also typically normalized to have unit length before performing the clustering, though this can be disabled by setting the `normalize_data` option to `False` (for example, if the data is already normalized).
Computing spherical *k*-means clustering with unnormalized data may lead to unexpected results.

Since *k*-means clustering is an NP-hard problem, algorithms are heuristic in nature and converge to local optima.
Therefore it is often desirable to run the algorithms several times and select the result with the smallest inertia.

Outputs from *k*-means clustering
---------------------------------
After a *k*-means clustering computation the following results are stored:

- **cluster centres** - the centre of the clusters.
- **labels** - the cluster each sample in the data matrix belongs to.
- **inertia** - the sum of the squared distances of each sample to its closest cluster centre.
- **iterations** - the number of iterations that were performed.

Two post-processing operations may be of interest:

- **transform** - given a data matrix :math:`X` in the same coordinates as the original data matrix :math:`A`, express :math:`X` in terms of new coordinates in which each dimension is the distance to the cluster centres previously computed for :math:`A`.
- **predict** - given a data matrix :math:`Y` find the closest cluster centre out of the clusters previously computed for :math:`A`.

Typical workflow for *k*-means clustering
-----------------------------------------

The standard way of using *k*-means clustering in AOCL-DA  is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.clustering.kmeans` object with options set in the class constructor.
      2. Compute the *k*-means clusters using :func:`aoclda.clustering.kmeans.fit`.
      3. Perform further transformations or predictions using :func:`aoclda.clustering.kmeans.transform` or :func:`aoclda.clustering.kmeans.predict`.
      4. Extract results from the :func:`aoclda.clustering.kmeans` object via its class attributes.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_kmeans``.
      2. Set the number of clusters required and other options using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <kmeans_options>`).
      3. Pass data to the handle using :ref:`da_kmeans_set_data_? <da_kmeans_set_data>`.
      4. Optionally set the initial centres using :ref:`da_kmeans_set_init_centres_? <da_kmeans_set_init_centres>`.
      5. Compute the *k*-means clusters using :ref:`da_kmeans_compute_? <da_kmeans_compute>`.
      6. Perform further computations as required, using :ref:`da_kmeans_transform_? <da_kmeans_transform>` or :ref:`da_kmeans_predict_? <da_kmeans_predict>`.
      7. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`.


.. _kmeans_options:

Options
-------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.clustering.kmeans` class constructor.

   .. tab-item:: C
      :sync: C

      The following options can be set using :ref:`da_options_set_? <da_options_set>`:

      .. update options using table _opts_k-meansclustering

      .. csv-table:: k-means options
         :header: "Option Name", "Type", "Default", "Description", "Constraints"

         "convergence tolerance", "real", ":math:`r=10^{-4}`", "Convergence tolerance.", ":math:`0 \le r`"
         "algorithm", "string", ":math:`s=` `lloyd`", "Choice of underlying k-means algorithm.", ":math:`s=` `elkan`, `hartigan-wong`, `lloyd`, or `macqueen`."
         "distance", "string", ":math:`s=` `euclidean`", "Distance metric used for clustering. Use 'euclidean' for standard k-means or 'cosine' for spherical k-means (not compatible with Hartigan-Wong).", ":math:`s=` `cosine`, or `euclidean`."
         "normalize data", "string", ":math:`s=` `yes`", "Whether to normalize the input data before clustering. This option is only used if distance is set to cosine.", ":math:`s=` `no`, or `yes`."
         "initialization method", "string", ":math:`s=` `k-means++`", "How to determine the initial cluster centres.", ":math:`s=` `afk-mc2`, `k-means++`, `random`, `random partitions`, or `supplied`."
         "seed", "integer", ":math:`i=0`", "Seed for random number generation; set to -1 for non-deterministic results.", ":math:`-1 \le i`"
         "max_iter", "integer", ":math:`i=300`", "Maximum number of iterations.", ":math:`1 \le i`"
         "n_init", "integer", ":math:`i=10`", "Number of runs with different random seeds (ignored if you have specified initial cluster centres).", ":math:`1 \le i`"
         "n_clusters", "integer", ":math:`i=1`", "Number of clusters required.", ":math:`1 \le i`"
         "empty clusters", "string", ":math:`s=` `ignore`", "How to deal with empty clusters at the end of a k-means iteration.", ":math:`s=` `error`, `ignore`, or `split`."
         "afk-mc2 samples", "integer", ":math:`i=50`", "Number of samples to take for the AFK-MC2 initialization method.", ":math:`1 \le i`"
         "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
         "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."
         "mixed precision", "string", ":math:`s=` `no`", "Whether to use mixed precision iterative refinement, in which lower precision arithmetic is used before switching to the working precision for the final iterations.", ":math:`s=` `no`, or `yes`."
         "low precision convergence tolerance", "real", ":math:`r=10^{-2}`", "If mixed precision iterative refinement is enabled, convergence tolerance for the low precision phase.", ":math:`0 \le r`"
         "low precision max_iter", "integer", ":math:`i=200`", "If mixed precision iterative refinement is enabled, maximum number of iterations for the low precision phase.", ":math:`1 \le i`"

      Note that if the initialization method is set to ``random`` then the initial cluster centres are chosen randomly from the sample points.
      If it is set to ``random partitions`` then the sample points are assigned to a random cluster and the corresponding cluster centres are computed and used as the starting point.
      If it is set to ``afk-mc2`` then the initial cluster centres are chosen using the AFK-MC\ :sup:`2` algorithm of Bachem et al. (2016) :cite:t:`da_bachem2016`, with the number of samples specified by the ``afk-mc2 samples`` option.

      The standard algorithm for solving *k*-means problems is Lloyd's algorithm. Elkan's algorithm can be faster on naturally clustered datasets but uses considerably more memory. For more information on the available algorithms see :cite:t:`da_elkan`, :cite:t:`da_hartigan1979algorithm`, :cite:t:`da_lloyd1982least` and :cite:t:`da_macqueen1967some`.

      The option ``empty clusters`` determines behaviour in the case that all sample points have been assigned to fewer than *k* clusters.
      If set to ``ignore`` then empty clusters are allowed and the algorithm proceeds as normal.
      If set to ``error`` then an error is raised when an empty cluster is encountered (if ``n_init`` > 1 then the next initialization is attempted, so an error will only be returned to the calling program if all initializations led to empty clusters).
      If set to ``split`` then the point farthest from its closest cluster centre is chosen and assigned to the empty cluster.
      Note that if the Hartigan-Wong algorithm is used then ``empty clusters`` will be set to ``error`` internally.

      The option ``mixed precision`` switches on an experimental mode in which an initial clustering is performed in lower precision before refining the result in the working precision.
      The option ``low precision max_iter`` sets the maximum number of iterations for the low-precision phase, and ``low precision convergence tolerance`` sets the convergence tolerance for the low-precision phase.

.. _dbscan_intro:

DBSCAN clustering
============================

DBSCAN clustering partitions a set of :math:`n_{\mathrm{samples}}` data points :math:`\{x_1, x_2, \dots, x_{n_{\mathrm{samples}}}\}` into an unspecified number of clusters, determined at runtime by the density of points.

The algorithm is governed by two parameters, ``eps`` and ``min_samples``.
The ``eps`` parameter is the maximum distance between two samples for one to be considered as in the neighborhood of the other. The ``min_samples`` parameter is the number of samples in a neighborhood for a point to be classed as a *core sample*.

The algorithm works as follows:

1. The neighborhood of each sample point (that is, the indices of the points within distance ``eps``) is computed.
2. The sample points are then considered in turn:

   - If a point is not already assigned to a cluster and its neighborhood contains fewer than ``min_samples`` points, it is classed as noise.
   - A point is classed as a *core sample* if its neighborhood contains at least ``min_samples`` points. A new cluster is created containing this point.
   - The neighborhood of the core sample is then explored and any points not already assigned to a cluster are added to the cluster.

3. This process is repeated until all points have been assigned to a cluster or classed as noise.

Outputs from DBSCAN clustering
---------------------------------
After a DBSCAN clustering computation the following results are stored:

- **n_clusters** - the number of clusters found.
- **labels** - the cluster each sample in the data matrix belongs to. A label of -1 indicates that the point has been classified as noise and has not been assigned to a cluster.
- **n_core_sample_indices** - the number of core samples found.
- **core sample indices** - the indices of the points that were classed as core samples.

Typical workflow for DBSCAN clustering
-----------------------------------------

The standard way of using DBSCAN clustering in AOCL-DA  is as follows.

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.clustering.DBSCAN` object with options set in the class constructor.
      2. Optionally standardize the data.
      3. Compute the DBSCAN clusters using :func:`aoclda.clustering.DBSCAN.fit`.
      4. Extract results from the :func:`aoclda.clustering.DBSCAN` object via its class attributes.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_dbscan``.
      2. Pass data to the handle using :ref:`da_dbscan_set_data_? <da_dbscan_set_data>`.
      3. Set the options using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <dbscan_options>`).
      4. Compute the DBSCAN clusters using :ref:`da_dbscan_compute_? <da_dbscan_compute>`.
      5. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`.


.. _dbscan_options:

Options
-------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.clustering.DBSCAN` class constructor.

   .. tab-item:: C
      :sync: C

      The following options can be set using :ref:`da_options_set_? <da_options_set>`:

      .. update options using table _opts_dbscanclustering

      .. csv-table:: DBSCAN options
         :header: "Option Name", "Type", "Default", "Description", "Constraints"

         "power", "real", ":math:`r=2.0`", "The power of the Minkowski metric used.", ":math:`0 \le r`"
         "metric", "string", ":math:`s=` `euclidean`", "Choice of metric used to compute pairwise distances.", ":math:`s=` `cityblock`, `cosine`, `euclidean`, `l1`, `l2`, `manhattan`, `minkowski`, or `sqeuclidean`."
         "algorithm", "string", ":math:`s=` `auto`", "Choice of algorithm.", ":math:`s=` `auto`, `ball tree`, `brute`, or `kd tree`."
         "leaf size", "integer", ":math:`i=30`", "Leaf size for k-d tree or ball tree.", ":math:`1 \le i`"
         "eps", "real", ":math:`r=10^{-4}`", "Maximum distance for two samples to be considered in each other's neighborhood.", ":math:`0 \le r`"
         "min samples", "integer", ":math:`i=5`", "Minimum number of neighborhood samples for a core point.", ":math:`1 \le i`"
         "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
         "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."

Note that k-d trees are likely to be fastest for lower dimensional datasets, and ball trees may be preferred when data is not aligned along the coordinate axes, but trees cannot be used with the cosine distance, the squared Euclidean distance, or the Minkowski distance with power less than 1.0.

Examples (clustering)
========================

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. dropdown:: k-means Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/kmeans_ex.py
              :language: Python
              :linenos:

      .. dropdown:: DBSCAN Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/dbscan_ex.py
              :language: Python
              :linenos:


   .. tab-item:: C
      :sync: C

      The code below can be found in the ``examples`` folder of your installation.

      .. dropdown:: k-means Example

         .. literalinclude:: ../../../tests/examples/kmeans.cpp
            :language: C++
            :linenos:

      .. dropdown:: DBSCAN Example

         .. literalinclude:: ../../../tests/examples/dbscan.cpp
            :language: C++
            :linenos:



Clustering APIs
=================

*k*-means
------------

.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.clustering.kmeans(n_clusters=1, initialization_method='k-means++', C=None, n_init=10, max_iter=300, seed=-1, algorithm='elkan', tol=1.0e-4, mixed_precision=False, low_precision_max_iter=None, low_precision_tol=None, check_data=False)
         :members:

   .. tab-item:: C

      .. _da_kmeans_set_data:

      .. doxygenfunction:: da_kmeans_set_data_s
         :project: da
         :outline:
      .. doxygenfunction:: da_kmeans_set_data_d
         :project: da

      .. _da_kmeans_set_init_centres:

      .. doxygenfunction:: da_kmeans_set_init_centres_s
         :project: da
         :outline:
      .. doxygenfunction:: da_kmeans_set_init_centres_d
         :project: da

      .. _da_kmeans_compute:

      .. doxygenfunction:: da_kmeans_compute_s
         :project: da
         :outline:
      .. doxygenfunction:: da_kmeans_compute_d
         :project: da

      .. _da_kmeans_transform:

      .. doxygenfunction:: da_kmeans_transform_s
         :project: da
         :outline:
      .. doxygenfunction:: da_kmeans_transform_d
         :project: da

      .. _da_kmeans_predict:

      .. doxygenfunction:: da_kmeans_predict_s
         :project: da
         :outline:
      .. doxygenfunction:: da_kmeans_predict_d
         :project: da

DBSCAN
---------

.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.clustering.DBSCAN(min_samples=5, metric='euclidean', algorithm='brute', leaf_size=30, eps=0.5, power=2.0, check_data=false)
         :members:

   .. tab-item:: C

      .. _da_dbscan_set_data:

      .. doxygenfunction:: da_dbscan_set_data_s
         :project: da
         :outline:
      .. doxygenfunction:: da_dbscan_set_data_d
         :project: da

      .. _da_dbscan_compute:

      .. doxygenfunction:: da_dbscan_compute_s
         :project: da
         :outline:
      .. doxygenfunction:: da_dbscan_compute_d
         :project: da
