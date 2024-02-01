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




Clustering
**********

This chapter contains functions for unsupervised clustering.

.. _kmeans_intro:

*k*-means Clustering
============================

Description of *k*-means clustering still to be written. Should contain details on choosing the number of clusters and the underlying algorithm.

Outputs from *k*-means clustering
---------------------------------
After a *k*-means clustering computation the following results are stored:

- **cluster centres** - the centre of the clusters.
- **labels** - the cluster each sample int he data matrix is in.
- **inertia** - the sum of the squared distance of each sample to its closest cluster centre.
- **iterations** - the number of iterations that were performed.

After the PCA has been computed, two post-processing operations may be of interest:

- **transform** - given a data matrix :math:`X` in the same coordinates as the original data matrix :math:`A`, express :math:`X` in terms of new coordinates in which each dimension is the distance to the cluster centres previously computed for :math:`A`.
- **predict** - given a data matrix :math:`Y` find the closest cluster centre out of the clusters previously computed for :math:`A`.

Typical workflow for *k*-means clustering
-----------------------------------------
The standard way of using *k*-means clustering in AOCL-DA  is as follows.

1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_kmeans``.
2. Pass data to the handle using either :ref:`da_kmeans_set_data_? <da_kmeans_set_data>`.
3. Set the number of clusters required and other options using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <kmeans_options>`).
4. Optionally set the initial centres using :ref:`da_kmeans_set_init_centres_? <da_kmeans_set_init_centres>`.
5. Compute the *k*-means clusters using :ref:`da_kmeans_compute_? <da_kmeans_compute>`.
6. Perform further computations as required, using :ref:`da_kmeans_transform_? <da_kmeans_transform>` or :ref:`da_kmeans_predict_? <da_kmeans_predict>`.
7. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`.


.. _kmeans_options:

Options
-------

The following options can be set using :ref:`da_options_set_? <da_options_set>`:

The following options are supported.

.. update options using table _opts_k-means

.. csv-table:: k-means options
   :header: "Option Name", "Type", "Default", "Description", "Constraints"

   "convergence tolerance", "real", ":math:`r=10^{-4}`", "Convergence tolerance", ":math:`0 < r`"
   "algorithm", "string", ":math:`s=` `lloyd`", "Choice of underlying k-means algorithm", ":math:`s=` `elkan`, `hartigan-wong`, or `lloyd`."
   "initialization method", "string", ":math:`s=` `random`", "How to determine the initial cluster centres", ":math:`s=` `k-means++`, `random`, or `supplied`."
   "seed", "integer", ":math:`i=0`", "Seed for random number generation; set to -1 for non-deterministic results", ":math:`-1 \le i`"
   "max_iter", "integer", ":math:`i=300`", "Maximum number of iterations", ":math:`1 \le i`"
   "n_init", "integer", ":math:`i=10`", "Number of runs with different random seeds (ignored if you have specified initial cluster centres)", ":math:`1 \le i`"
   "n_clusters", "integer", ":math:`i=1`", "Number of clusters required", ":math:`1 \le i`"


Examples
========

The code below can be found in ``kmeans.cpp`` in the ``examples`` folder of your installation.

.. collapse:: k-means Example Code

    .. literalinclude:: ../../tests/examples/kmeans.cpp
        :language: C++
        :linenos:
.. toctree::
    :maxdepth: 1
    :hidden:

    clustering_api
