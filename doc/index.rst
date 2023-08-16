.. AOCL-DA documentation master file, created by
   sphinx-quickstart on Mon May 22 10:35:28 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AOCL-DA's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   general-intro
   linear-models
   trees-forests

**How to document a new chapter / area of AOCL-DA functionality**

* Create a new ReStructuredText file in the ``doc`` directory (where ``index.rst`` is).
* Write a general introduction to the chapter / area
* Use the Breathe extension to Sphinx to document the API using Doxygen-formatted code comments

See ``linear-models.rst`` and ``aoclda_linmod.h`` for an example.

**How to build documentation via CMake**

Create a virtual Python environment in which to install ``sphinx-build`` and the Sphinx extensions used by AOCL-DA.
E.g.,

.. code-block::

   cd ~/DA-projects/aocl-da
   python3 -m venv ~/DA-projects/doc-3
   source ~/DA-projects/doc-3/bin/activate
   pip install -r doc/requirements.txt


Build the Sphinx HTML.  E.g.,

.. code-block::

   cd build
   cmake -DBUILD_DOC=ON ..
   make sphinx


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==========

.. bibliography::
