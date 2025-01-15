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



Clustering APIs
*********************

*k*-means
========================================

.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.clustering.kmeans(n_clusters=1, initialization_method='k-means++', C=None, n_init=10, max_iter=300, seed=-1, algorithm='elkan', tol=1.0e-4, check_data=false)
         :members:

   .. tab-item:: C

      .. _da_kmeans_set_data:

      .. doxygenfunction:: da_kmeans_set_data_s
         :outline:
      .. doxygenfunction:: da_kmeans_set_data_d

      .. _da_kmeans_set_init_centres:

      .. doxygenfunction:: da_kmeans_set_init_centres_s
         :outline:
      .. doxygenfunction:: da_kmeans_set_init_centres_d

      .. _da_kmeans_compute:

      .. doxygenfunction:: da_kmeans_compute_s
         :outline:
      .. doxygenfunction:: da_kmeans_compute_d

      .. _da_kmeans_transform:

      .. doxygenfunction:: da_kmeans_transform_s
         :outline:
      .. doxygenfunction:: da_kmeans_transform_d

      .. _da_kmeans_predict:

      .. doxygenfunction:: da_kmeans_predict_s
         :outline:
      .. doxygenfunction:: da_kmeans_predict_d

DBSCAN
========================================

.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.clustering.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='brute', leaf_size=30, p=None, precision='double', check_data=false)
         :members:

   .. tab-item:: C

      .. _da_dbscan_set_data:

      .. doxygenfunction:: da_dbscan_set_data_s
         :outline:
      .. doxygenfunction:: da_dbscan_set_data_d

      .. _da_dbscan_compute:

      .. doxygenfunction:: da_dbscan_compute_s
         :outline:
      .. doxygenfunction:: da_dbscan_compute_d
