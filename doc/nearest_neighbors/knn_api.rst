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



*k*-Nearest Neighbors APIs
**************************

*k*-Nearest Neighbors for Classification
========================================
.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.nearest_neighbors.knn_classifier(n_neighbors=5, weights='uniform', algorithm='brute', metric='euclidean', check_data=false)
         :members:

   .. tab-item:: C

      .. _da_knn_set_training_data:

      .. doxygenfunction:: da_knn_set_training_data_s
         :outline:
      .. doxygenfunction:: da_knn_set_training_data_d

      .. _da_knn_kneighbors:

      .. doxygenfunction:: da_knn_kneighbors_s
         :outline:
      .. doxygenfunction:: da_knn_kneighbors_d

      .. _da_knn_classes:

      .. doxygenfunction:: da_knn_classes_s
         :outline:
      .. doxygenfunction:: da_knn_classes_d

      .. _da_knn_predict_proba:

      .. doxygenfunction:: da_knn_predict_proba_s
         :outline:
      .. doxygenfunction:: da_knn_predict_proba_d

      .. _da_knn_predict:

      .. doxygenfunction:: da_knn_predict_s
         :outline:
      .. doxygenfunction:: da_knn_predict_d
