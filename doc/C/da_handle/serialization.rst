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

.. _model_persistence:

Model Persistence
===============================

Once a model has been trained using AOCL-DA, it can be saved to disk using :cpp:func:`da_handle_save_model`. 
This allows the trained model to be reused at a later time without needing to retrain it, which is particularly useful 
for computationally expensive models or when deploying models in production environments.

A saved model can be restored using :cpp:func:`da_handle_load_model`, which reads the model 
from disk and reconstructs the :cpp:type:`da_handle` with all the trained parameters and internal 
state. This enables you to make predictions or perform transformations using the pre-trained model immediately after loading.

Both functions work with binary files to ensure efficient storage and fast loading times. The model 
serialization preserves only the essential trained parameters and internal state needed for inference, 
such as model coefficients, cluster centers, or trained hyperparameters. Original training data is only saved if necessary.

Supported Models
----------------

The following AOCL-DA models support serialization (saving and loading):

- **Approximate Nearest Neighbors**
- **Decision Forest**
- **Decision Tree**
- **K-Means Clustering**
- **Linear Models**
- **Nearest Neighbors**
- **Principal Component Analysis**
- **Support Vector Machines**

Compatibility and Limitations
------------------------------

When saving and loading models, the following compatibility requirements and limitations apply:

**Endianness**
   The system endianness (byte order) must be the same when saving and loading a model. 
   Loading a model saved on a system with different endianness is not supported and may result in corrupted data or undefined behavior.

**Integer Type Compatibility (da_int)**
   The ``da_int`` type may be configured as either ``int32_t`` or ``int64_t`` at compile time:
   
   - If a model was saved with ``da_int`` as ``int32_t``, it can be loaded with ``da_int`` configured as either ``int32_t`` or ``int64_t``.
   - If a model was saved with ``da_int`` as ``int64_t``, it can **only** be loaded with ``da_int`` configured as ``int64_t``. 
     Loading into a ``int32_t`` configuration is not supported and will fail.

**Library Version Compatibility**
   Model serialization is only guaranteed to work between compatible library versions. 
   Attempting to load a model saved with an older or newer version of AOCL-DA may fail if the serialization format has changed. 
   The library performs version checks during loading and will return an error if the saved model format is incompatible with the current version.

Example: PCA Model Persistence
-------------------------------

The following example demonstrates saving and loading a trained PCA model using the C API:

.. literalinclude:: ../../../tests/examples/model_persistence_pca.cpp
   :language: c++
   :linenos:

API Reference
--------------

.. doxygenfunction:: da_handle_save_model(da_handle, const char *)
   :project: da

.. doxygenfunction:: da_handle_load_model(da_handle *, const char *)
   :project: da