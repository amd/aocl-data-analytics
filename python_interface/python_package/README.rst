AOCL Data Analytics
*******************

**The latest AMD plugin for scikit-learn is here!**

AMD's AOCL Data Analytics Library provides optimized building blocks for data analysis and classical machine learning applications.
The package leverages the `AMD optimizing CPU libraries (AOCL) <https://www.amd.com/en/developer/aocl.html>`_ to provide outstanding performance, not just on AMD processors but on other x86 hardware too.

Existing scikit-learn users can benefit from the performance of the AOCL Data Analytics Library without making any code changes, by simply patching existing scikit-learn code so that it automatically calls the library.
The AOCL Data Analytics Library also comes with additional Python APIs, providing access to algorithms not included in scikit-learn, such as nonlinear least squares optimization.

Installation
=============

The easiest way to access the AOCL Data Analytics Library is via the ``pip install`` command, which will download and install an appropriate wheel directly from PyPI.

Python wheels can also be downloaded directly from `AMD's AOCL-DA page <https://www.amd.com/en/developer/aocl/data-analytics.html>`_ and installed using ``pip``.

For Linux users, Python packages can also be downloaded and built using `Spack <https://www.amd.com/en/developer/zen-software-studio/applications/spack/spack-aocl.html>`_.

The AOCL Data Analytics Source code and compilation instructions are available at https://github.com/amd/aocl-data-analytics/.

Using the scikit-learn extension
================================
Existing scikit-learn users can *patch* code to replace the scikit-learn symbols with
AOCL Data Analytics symbols. This can be done by inserting the following lines prior to the scikit-learn ``import``
statement.

.. code-block::

   from aoclda.sklearn import skpatch, undo_skpatch
   skpatch()

You can switch back to standard scikit-learn using

.. code-block::

   undo_skpatch()

Note that after calling ``undo_skpatch``, you must reimport scikit-learn.

The ``skpatch`` and ``undo_skpatch`` functions can also be called with string or list arguments, specifying which scikit-learn package should be patched, for example:

.. code-block::

   skpatch("PCA")
   skpatch(["LinearRegression", "Ridge"])

Alternatively, you may wish to use the ``aoclda.sklearn`` module from the command line, without
making any changes to your own code:

.. code-block::

   python -m aoclda.sklearn your_python_script.py
   python -m aoclda.sklearn -m your_python_module

Python APIs
===============

In addition to the scikit-learn patch, AOCL Data Analytics contains its own set of Python APIs providing additional functionality.
The package comes with numerous example scripts.
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

License
=========

The ``License`` file is included in the ``licenses`` directory of the Python package installation.
Copyrighted code in AOCL Data Analytics is subject to the licenses set forth in the source code file headers of such code.

Further information
======================

Full documentation of the Python APIs can be found in the `AMD technical information portal <https://docs.amd.com/r/en-US/68552-AOCL-api-guide/AOCL-Data-Analytics>`_.

Note that Windows packages do not include the LBFGSB linear model solver or the nonlinear least squares solver.
To access these, building from source is required.
Source code and compilation instructions are available at https://github.com/amd/aocl-data-analytics/.

The AOCL Data Analytics library is part of `AMD Zen Software Studio <https://www.amd.com/en/developer/zen-software-studio.html>`_.
It is developed and maintained by `AMD <https://www.amd.com/>`_.
For support or queries, you can e-mail us on toolchainsupport@amd.com.