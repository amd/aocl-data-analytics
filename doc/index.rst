.. AOCL-DA documentation master file, created by
   sphinx-quickstart on Mon May 22 10:35:28 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AOCL-DA's documentation!
************************

Introduction and data handling
------------------------------

.. toctree::
   :maxdepth: 1
   :caption: General contents:

   general-intro
   datastore
   csv-files
   da-handle
   option-setting
   error-codes

.. toctree::
   :maxdepth: 1
   :caption: Algorithms:

   basic-statistics
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

Useful elements
===============

Numbered lists
--------------

1. explicitly numbered list
2. new element
#. implicitely numbered 
   element of the same list

New List:

#. new implicitely numbered list
   
   #. sublist
   #. with implicit
   #. elements

#. second element
#. `link to the doc <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#lists-and-quote-like-blocks>`_

tables
------

.. csv-table:: Example
   :header: "String1", "number", "String2"
   :widths: 15, 10, 30

   "Short description", 1.03, "`More online documentation for tables 
   <https://pandemic-overview.readthedocs.io/en/latest/myGuides/reStructuredText-Tables-Examples.html#csv-table-example>`_"
   "Another", 10.4, "This is a long description over
   several lines"
   "new line", 1.99, "longer line decscription"

Adding a picture
----------------

.. image:: pics/kitten.jpg
   :align: center

`link to more documentation <https://pandemic-overview.readthedocs.io/en/latest/myGuides/reStructuredText-Images-and-Figures-Examples.html>`_


Useful elements
===============

Numbered lists
--------------

1. explicitly numbered list
2. new element
#. implicitely numbered 
   element of the same list

New List:

#. new implicitely numbered list
   
   #. sublist
   #. with implicit
   #. elements

#. second element
#. `link to the doc <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#lists-and-quote-like-blocks>`_

tables
------

.. csv-table:: Example
   :header: "String1", "number", "String2"
   :widths: 15, 10, 30

   "Short description", 1.03, "`More online documentation for tables 
   <https://pandemic-overview.readthedocs.io/en/latest/myGuides/reStructuredText-Tables-Examples.html#csv-table-example>`_"
   "Another", 10.4, "This is a long description over
   several lines"
   "new line", 1.99, "longer line decscription"

Adding a picture
----------------

.. image:: pics/kitten.jpg
   :align: center

`link to more documentation <https://pandemic-overview.readthedocs.io/en/latest/myGuides/reStructuredText-Images-and-Figures-Examples.html>`_


References
==========

.. bibliography::
