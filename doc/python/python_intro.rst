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



.. _chapter_python_intro:

Introduction
*******************

A Python API exists for the algorithmic functions in AOCL-DA.

Installation
=============

Your AOCL-DA package comes bundled with Python wheel, ``aoclda-*.whl``, where ``*`` depends on your particular system.
To install the AOCL-DA Python API simply use the command ``pip install aoclda-*.whl``.

The wheel, ``aoclda-*.whl``, will install the necessary libraries and dependencies.
However, on Windows a Fortran runtime library ``libifcore-mt.lib`` is also required, so you will need to install the Intel Fortran compiler and set the environment variable ``INTEL_FCOMPILER`` to point to its installation directory.

NumPy Arrays
=============

AOCL-DA Python interfaces typically expect data to be supplied as NumPy arrays. These can be supplied either with ``order='C'`` or ``order='F'`` for row- or column-major ordering respectively.
The interface will convert C-style numpy arrays to Fortran style, so for best performance, it is recommended to use ``order='F'`` when supplying NumPy arrays to AOCL-DA functions.

.. _python_examples:

Python Examples
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

The examples can then be run as standard Python scripts from the command prompt, for example:

.. code-block::

   python path/to/examples/pca_ex.py

You can also inspect and run the examples from a Python interpreter, for example:

.. code-block::

    >>> from aoclda.examples import pca_ex
    >>> import inspect
    >>> print(''.join(inspect.getsourcelines(pca_ex)[0]))
    >>> pca_ex.test_pca()

The AOCL-DA Extension for Scikit-learn
=======================================

In addition to the Python API, AOCL-DA offers an extension to enable existing users of Scikit-learn
to extract better performance while making minimal changes to their code.

To use the extension, you must *patch* your existing code to replace the Scikit-learn symbols with
AOCL-DA symbols. This can be done by inserting the following lines prior to your Scikit-learn import
statement.

.. code-block::

   from aoclda.sklearn import skpatch, undo_skpatch
   skpatch()

You can switch back to standard Scikit-learn using

.. code-block::

   undo_skpatch()

Alternatively, you may wish to use the ``aoclda.sklearn`` module from the command line, without
making any changes to your own code:

.. code-block::

   python -m aoclda.sklearn your_python_script.py
   python -m aoclda.sklearn -m your_python_module

The following Scikit-learn classes are currently available in the AOCL-DA extension.

.. list-table:: AOCL-DA Extension for Scikit-learn
   :header-rows: 1

   * - Scikit-learn class
     - Notes
   * - ``sklearn.dcomposition.PCA``
     - ``fit``, ``transform``, ``inverse_transform`` and ``fit_transform`` methods and various class attributes only

Note that only a subset of the AOCL-DA functionality is available in this manner, and if, after
patching, you attempt to call class member functions which have not been implemented by AOCL-DA,
then a ``RuntimeError`` will be thrown. It is recommended that for the full benefit of using AOCL-DA
you use the Python APIs described on the subsequent pages of this manual.

.. toctree::
    :maxdepth: 1
    :hidden:

    factorization_api
