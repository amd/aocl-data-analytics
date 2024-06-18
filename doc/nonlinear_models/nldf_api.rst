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



Nonlinear Data Fitting APIs
***************************


.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.nonlinear_model.nlls(n_coef, n_res, weights=None, lower_bounds=None, upper_bounds=None, order='c', prec='double', model='hybrid', method='galahad', glob_strategy='tr', reg_power='quadratic', verbose=0)
         :members:

   .. tab-item:: C

      .. _da_nlls_callbacks:

      .. _da_nlls_callbacks_r:

      .. doxygentypedef:: da_resfun_t_s
        :outline:
      .. doxygentypedef:: da_resfun_t_d

      .. _da_nlls_callbacks_j:

      .. doxygentypedef:: da_resgrd_t_s
        :outline:
      .. doxygentypedef:: da_resgrd_t_d

      .. _da_nlls_callbacks_hf:

      .. doxygentypedef:: da_reshes_t_s
        :outline:
      .. doxygentypedef:: da_reshes_t_d

      .. _da_nlls_callbacks_hp:

      .. doxygentypedef:: da_reshp_t_s
        :outline:
      .. doxygentypedef:: da_reshp_t_d

      .. _da_nlls_define_residuals:

      .. doxygenfunction:: da_nlls_define_residuals_s
         :outline:
      .. doxygenfunction:: da_nlls_define_residuals_d

      .. _da_nlls_define_weights:

      .. doxygenfunction:: da_nlls_define_weights_s
         :outline:
      .. doxygenfunction:: da_nlls_define_weights_d

      .. _da_nlls_define_bounds:

      .. doxygenfunction:: da_nlls_define_bounds_s
         :outline:
      .. doxygenfunction:: da_nlls_define_bounds_d

      .. _da_nlls_fit:

      .. doxygenfunction:: da_nlls_fit_s
         :outline:
      .. doxygenfunction:: da_nlls_fit_d

      .. _da_optim_info_t:

      .. doxygenenum:: da_optim_info_t_
