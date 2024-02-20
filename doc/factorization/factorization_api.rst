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



Factorization APIs
*********************

Principal component analysis and the SVD
========================================
.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.factorization.PCA(n_components=1, bias='unbiased', method='covariance', solver='auto', precision="double")
         :members:

   .. tab-item:: C

      .. _da_pca_set_data:

      .. doxygenfunction:: da_pca_set_data_s
         :outline:
      .. doxygenfunction:: da_pca_set_data_d

      .. _da_pca_compute:

      .. doxygenfunction:: da_pca_compute_s
         :outline:
      .. doxygenfunction:: da_pca_compute_d

      .. _da_pca_transform:

      .. doxygenfunction:: da_pca_transform_s
         :outline:
      .. doxygenfunction:: da_pca_transform_d

      .. _da_pca_inverse_transform:

      .. doxygenfunction:: da_pca_inverse_transform_s
         :outline:
      .. doxygenfunction:: da_pca_inverse_transform_d
