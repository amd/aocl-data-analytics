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



Support Vector Machine APIs
****************************

.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.svm.SVC(C=1.0, kernel="rbf", degree=3, gamma=-1.0, coef0=0.0, probability=False, tol=0.001, max_iter=-1, tau=1.0e-12, check_data=False)
         :members:
         :inherited-members:
      .. autoclass:: aoclda.svm.SVR(C=1.0, epsilon=0.1, kernel="rbf", degree=3, gamma=-1.0, coef0=0.0, tol=0.001, max_iter=-1, tau=1.0e-12, check_data=False)
         :members:
         :inherited-members:
      .. autoclass:: aoclda.svm.NuSVC(nu=0.5, kernel="rbf", degree=3, gamma=-1.0, coef0=0.0, probability=False, tol=0.001, max_iter=-1, tau=1.0e-12, check_data=False)
         :members:
         :inherited-members:
      .. autoclass:: aoclda.svm.NuSVR(nu=0.5, C=1.0, kernel="rbf", degree=3, gamma=-1.0, coef0=0.0, tol=0.001, max_iter=-1, tau=1.0e-12, check_data=False)
         :members:
         :inherited-members:

   .. tab-item:: C

      .. _da_svm_select_model:

      .. doxygenfunction:: da_svm_select_model_s
         :outline:
      .. doxygenfunction:: da_svm_select_model_d

      .. _da_svm_set_data:

      .. doxygenfunction:: da_svm_set_data_s
         :outline:
      .. doxygenfunction:: da_svm_set_data_d

      .. _da_svm_compute:

      .. doxygenfunction:: da_svm_compute_s
         :outline:
      .. doxygenfunction:: da_svm_compute_d

      .. _da_svm_predict:

      .. doxygenfunction:: da_svm_predict_s
         :outline:
      .. doxygenfunction:: da_svm_predict_d

      .. _da_svm_decision_function:

      .. doxygenfunction:: da_svm_decision_function_s
         :outline:
      .. doxygenfunction:: da_svm_decision_function_d

      .. _da_svm_score:

      .. doxygenfunction:: da_svm_score_s
         :outline:
      .. doxygenfunction:: da_svm_score_d

      .. doxygentypedef:: da_svm_model
      .. doxygenenum:: da_svm_model_

      .. doxygentypedef:: da_svm_decision_function_shape
      .. doxygenenum:: da_svm_decision_function_shape_
