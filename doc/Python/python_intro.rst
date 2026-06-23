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



.. _chapter_python_intro:


Introduction to the Python APIs
********************************

This section contains general instructions for calling AOCL-DA using the Python APIs.

Installation
=============

The easiest way to install AOCL-DA is using ``pip``:

.. code-block::

   pip install aoclda

AOCL-DA Python wheels can also be downloaded from https://www.amd.com/en/developer/aocl.html.
To install wheels downloaded in this manner, simply use the command ``pip install aoclda-*.whl``, where ``*`` in the wheel name will depend on your particular system. This will install the necessary libraries and dependencies.
For Linux users, if you find that your system is incompatible with the supplied wheel, you can instead install the Python package using the Spack recipe at the following link: https://www.amd.com/en/developer/zen-software-studio/applications/spack/spack-aocl.html.

.. note::
   Python support on Windows is currently experimental.
   The Python wheels downloaded from https://www.amd.com/en/developer/aocl.html do not include the LBFGSB linear model solver or the nonlinear least squares solver.
   To access these, building from source is required. Source code and compilation instructions are available at https://github.com/amd/aocl-data-analytics/.
   If you encounter issues, please e-mail us on
   toolchainsupport@amd.com.

Arrays
=============

AOCL-DA Python interfaces typically expect data to be supplied as array-like objects. These include NumPy arrays, Pandas data frames, Python lists, etc.
The Python interfaces will try to convert the array-like object to a NumPy array, if it is not already one or has an unsupported data type.
However, it is recommended to supply NumPy arrays directly, as this will avoid additional copying or conversion of the data, which can affect performance.
If supported, the data objects can be supplied either with ``order='C'`` or ``order='F'`` for row- or column-major ordering respectively.
For best performance, it is generally recommended to use ``order='F'`` when supplying NumPy arrays to AOCL-DA functions since row-major arrays may be copied and transposed internally.

In order to provide the best possible performance, the algorithmic functions will not automatically check for
``NaN`` data. If a ``NaN`` is passed into an algorithmic function, its behavior is undefined.
It is therefore the user's responsibility to ensure data is sanitized (this can be done in Python, or by setting the ``check_data`` optional argument in the appropriate algorithmic APIs).

.. _python_examples:

Python examples
===============

The AOCL-DA Python API comes with numerous example scripts, which are installed when you invoke ``pip install aoclda-*.whl``.
To locate these examples, the following commands can be used in your Python interpreter:

.. code-block::

    >>> from aoclda.examples import info
    >>> info.examples_path()
    >>> info.examples_list()

Alternatively, from your command prompt, you can call ``aocl.examples.info`` as a module to obtain the same information:

.. code-block::

   python -m aoclda.examples.info

The examples can then be run as standard Python scripts from the command prompt. For example:

.. code-block::

   python path/to/examples/pca_ex.py

You can also inspect and run the examples from a Python interpreter. For example:

.. code-block::

    >>> from aoclda.examples import pca_ex
    >>> import inspect
    >>> print(''.join(inspect.getsourcelines(pca_ex)[0]))
    >>> pca_ex.pca_example()


Model Persistence
==========================

Once a model has been trained using AOCL-DA, it can be saved to disk for later 
reuse without needing to retrain. This is particularly useful for computationally 
expensive models or when deploying models in production environments.

The recommended way to save and load AOCL-DA models is using `joblib <https://joblib.readthedocs.io/>`_. 
Under the hood, joblib uses Python's pickle protocol for the Python layer objects, 
which wraps AOCL-DA's custom binary serialization format for the underlying C/C++ 
algorithm implementations. This ensures efficient storage while preserving all 
trained parameters and internal state.

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

The saved models preserve only the essential trained parameters and internal state needed for inference, 
such as model coefficients, cluster centers, or trained hyperparameters. Original training data is only 
saved if necessary, ensuring efficient storage and faster loading times.

.. note::
   For information on compatibility requirements and limitations (endianness, integer type compatibility, 
   and library version requirements), please refer to the :ref:`C API serialization documentation <model_persistence>`.

Example: Saving and Loading a K-Means Model
--------------------------------------------

.. literalinclude:: ../../python_interface/python_package/aoclda/examples/model_persistence_kmeans_ex.py
   :language: Python
   :linenos:


Example: Using sklearn Patching with Model Persistence
-------------------------------------------------------

AOCL-DA provides a patching mechanism that allows you to use AOCL-DA's optimized 
implementations while maintaining compatibility with scikit-learn's API. This is 
particularly useful for existing codebases that use scikit-learn.

.. literalinclude:: ../../python_interface/python_package/aoclda/examples/model_persistence_skpatch_pca_ex.py
   :language: Python
   :linenos:
