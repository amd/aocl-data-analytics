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
- **Nonnegative Matrix Factorization**
- **Principal Component Analysis**
- **Kernel Principal Component Analysis**
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

Inspecting Saved Models
------------------------

AOCL-DA provides utility functions to inspect metadata from saved models.
:cpp:func:`da_print_model_metadata` reads and prints the metadata header (version, precision, handle type)
from a saved file without fully loading the model, which is useful for diagnosing version mismatches or
verifying file contents. :cpp:func:`da_handle_print_model_versions` prints the AOCL-DA build version and
serialization format version recorded in an already-loaded model.

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

.. doxygenfunction:: da_print_model_metadata(const char *filename)
   :project: da

.. doxygenfunction:: da_handle_print_model_versions(da_handle)
   :project: da

.. only:: internal

   Serialization Developer Guide
   =========================================

   Binary File Layout
   ------------------

   A serialized model file is contiguous binary data divided into three sequential sections:

   .. csv-table::
      :header: "Section", "Contents"

      "**Metadata**", "Header keyword (``AOCLDA_STORED_MODEL``), ``da_int`` size (LP64/ILP64), serialization version, handle type, precision (``float``/``double``), AOCL-DA build version string"
      "**Options**", "Registry size (``int64``), then for each option: key string followed by the option value"
      "**Model Data**", "Trained parameters and internal state fields serialized in the order defined by the model's ``serialize()`` method"

   Each container (``std::vector``, ``da_vector``, ``std::string``) is length-prefixed (``int64``) followed by its elements.

   Serialization Phases
   ------------------------------------

   Saving and loading a model proceeds through three sequential phases, all driven by the same ``serialize()`` method
   via the ``buffer_mode`` enum on ``serialization_buffer``:

   1. **Reserve** (``buffer_mode::reserve``): Computes the total buffer size needed without writing
      any data. The metadata size is added when the buffer is initialized (``add_metadata_size()`` in
      ``set_buffer_data()``). Then the model's ``serialize()`` method is called in reserve mode — each
      ``dispatch_buffer_io()`` call accumulates the byte size of its argument via ``get_type_size()``.
      Finally, the options registry size is added via ``compute_serialized_size()``.
      After all three contributions are summed, ``buffer.reserve()`` pre-allocates the output vector
      to the exact total size, avoiding reallocations during the write phase.

   2. **Serialize** (``buffer_mode::serialize``): The buffer mode is switched to serialize and the
      sections are written in order: metadata first (``serialize_metadata()``), then the options registry
      (``save_registry()``), then ``serialize()`` is called, this time each ``dispatch_buffer_io()``
      call appends the binary representation of its argument to the output buffer.

   3. **Deserialize** (``buffer_mode::deserialize``): On load, the buffer is initialized with the raw
      file bytes and an offset counter. The sections are read in the same order: metadata first
      (``deserialize_metadata()``) to validate compatibility, then ``load_registry()`` to restore option
      values, and finally ``serialize()`` is called. Now each ``dispatch_buffer_io()`` reads bytes from
      the buffer at the current offset and reconstructs the field.

   This design means each model implements a **single** ``serialize()`` method that serves all three phases,
   ensuring the save and load field ordering can never diverge.

   User Data Serialization
   ------------------------

   Some models (e.g., Nearest Neighbors, Approximate Nearest Neighbors) need to retain the
   original training data for inference. Because user-provided data arrives as raw pointers with an
   arbitrary leading dimension (stride), it cannot be handled by ``dispatch_buffer_io()`` which operates
   on owned containers.

   The dedicated ``serialize_user_data(X, order, m, n, ldx)`` method handles this case:

   - **Reserve phase**: adds ``m * n * sizeof(save_type_t<T>)`` to the buffer size.
   - **Serialize phase**: iterates over the logical ``m × n`` region, skipping any padding 
     (when ``ldx > m`` for column-major or ``ldx > n`` for
     row-major). The data is stored contiguously as a flat length-prefixed vector.
   - **Deserialize phase**: not handled by this method. On load, the model's ``serialize()``
     method reads the data back into an owned ``std::vector<T>`` via ``dispatch_buffer_io``, sets
     its raw pointer to the vector's storage, and the leading dimension is set to the 
     value of ``m`` when ``order == column_major`` or ``n`` when ``order == row_major``, 
     effectively removing the original stride.

   A ``nullptr`` input is saved as a zero-length vector, enabling models to store optional 
   user data objects.

   Type Normalization
   ------------------

   To guarantee portability across ``da_int`` configurations and compiler representations,
   scalar types are normalized to fixed-width on-disk representations via ``save_type_t<T>``:

   .. csv-table::
      :header: "Runtime Type", "On-Disk Type"

      "``bool``", "``uint8_t``"
      "``da_int``", "``int64_t``"
      "Any ``enum``", "``int64_t``"
      "``_Float16``", "``float``"
      "``float``", "``float``"
      "``double``", "``double``"
      "``char``", "``char``"

   Versioning Strategy
   --------------------

   Model persistence uses a two-level versioning scheme:

   **Global minimum version** (``model_persistence_min_version`` in ``model_persistence.hpp``):
      A floor below which *no* saved model will be loaded regardless of model type.
      Set to the AOCL-DA version (encoded as ``Major*10000 + Minor*100 + Patch``) that introduced
      the last breaking infrastructure change to the serialization format itself (metadata layout,
      type normalization rules, etc.). Bump this when the common serialization infrastructure changes
      in a backwards-incompatible way.

   **Per-model version** (``serialization_version`` member in each derived handle class):
      Each model class can declare its own version integer (defaulting to ``model_persistence_min_version``). 
      On load, the version stored in the file
      (written as part of metadata) is compared to the version in the current code via
      ``basic_handle::load_model()``. If they do not match exactly, loading fails with
      ``da_status_version_mismatch``. Bump a model's version when its ``serialize()`` method
      changes (fields added, removed, or reordered) or when mandatory options are added or
      modified in the model's option registry.

   When bumping versions:

   - If only one model's serialized fields change → bump that model's ``serialization_version``.
   - If the metadata/options/infrastructure format changes → bump ``model_persistence_min_version``
     and all per-model versions.

   Developer Checklist
   --------------------

   **What to serialize**

   The ``serialize()`` method must include every internal member that is needed after loading for
   the model to function correctly (predict, transform, score, etc.). This includes:

   - Trained model parameters (coefficients, cluster centers, tree structure, etc.)
   - Derived state that is expensive to recompute (e.g., pre-computed norms, sorted indices)
   - Flags that control post-training behavior (e.g., ``model_trained``, algorithm selection enums)
   - User-provided training data if required for inference (via ``serialize_user_data()``)

   **Field ordering**

   Fields must appear in the same order in ``serialize()`` regardless of buffer mode. 

   **Testing requirements**

   - Round-trip test: train → save → load → verify predictions match the original model.
   - Re-training after load: ensure the model handles re-training correctly after being loaded.
   - Cross-precision: if a model supports both ``float`` and ``double``, test that each precision
     can be saved and loaded independently.
   - Null/empty state: if a model has optional members (e.g., user data that may be ``nullptr``),
     test serialization with and without those members present.
   - Save after load: Ensure model can be saved again after being loaded.

   New API step by step Guide
   ---------------------------

   **C++**

   1. Identify all internal members which need to be stored so that all algorithm methods 
      work after loading (This doesn't apply to user data which is handled differently).
   2. Create a ``serialize(serialization_buffer &buffer)`` member function where each internal 
      member that needs saving is passed as an argument to the ``buffer.dispatch_buffer_io()`` method. 
      Here a lambda function is used to make status verification cleaner (See any algorithm for example).
   3. Create ``save_model(serialization_buffer &buffer)`` member function which needs to verify model has 
      been trained (if needed) or any similar verification at the top, followed by a call to ``basic_handle<T>::save_model(buffer)``. 
      The call to the basic_handle's implementation does the hard work by modifying the buffer and saving all necessary data. 
      DO NOT call ``serialize()`` otherwise it will duplicate data and possibly fail the serialization.
   4. Create a ``load_model(serialization_buffer &buffer)`` member function which needs to call 
      ``basic_handle<T>::load_model(buffer)``. The latter will do all the loading for you with no other calls necessary.
   
   **User data**

   1. Create an empty ``std::vector`` internal member for each distinct user data object needed to be saved.
   2. In ``serialize()`` create a conditional statement which checks the buffer mode via ``buffer.get_mode()``.
      If the mode is ``deserialize`` make a call to ``buffer.deserialize_data()`` and pass the internal member vector as 
      an argument. If the mode is ``serialize`` or ``reserve`` make a call to ``buffer.serialize_user_data()`` (see above 
      for more details on what this method does). If multiple user objects need loading make sure that the order is consistent 
      in the two branches of the conditional statement.
   3. Last, preferably in ``load_model()`` after calling ``basic_handle<T>::load_model(buffer)`` update the pointers 
      to user data objects to point to the internally stored vectors and update any internal members that 
      obtain their values from the user objects (``ldx``, ``order``, etc.)

   **Pybind**

   1. Create an additional initializer for the class which accepts ``precision`` as the single argument and initializes the 
      internal member.
   2. If any internal members of the pybind class need storing override ``save_data(py::dict &state)``
      and ``load_data(py::dict &state)`` and save them to (or load them from) the dictionary. This does 
      not include ``order`` and ``precision`` as they are stored by default.

   **Python**

   1. Create a ``__getstate__(self)`` method which returns a dictionary. Only required entry is 
      ``pybind_state`` which serializes the C++ layer using ``pickle.dumps(self._model)``. All other 
      entries are optional.
   2. Create a ``__setstate__(self, state)`` method which loads and initializes any saved members. 
      Take caution when initializing the model's ``double`` and ``single`` members, so that the algorithm 
      works as expected. Any member that is not saved will not be initialized when the model is loaded 
      and will be inaccessible rather than ``None``. Thus, initialize to ``None`` (or something) any 
      member that wasn't saved but can be used.

