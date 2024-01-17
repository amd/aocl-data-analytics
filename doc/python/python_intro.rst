..
    Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

.. toctree::
    :maxdepth: 1
    :hidden:

    factorization_api
