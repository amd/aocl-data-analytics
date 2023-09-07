Documentation utilities
***********************

Creating and building new doc
=============================

**How to document a new chapter / area of AOCL-DA functionality**

* Create a new ReStructuredText file in the ``doc`` directory (where ``index.rst`` is).
* Write a general introduction to the chapter / area
* Use the Breathe extension to Sphinx to document the API using Doxygen-formatted code comments

See ``linear-models.rst`` and ``aoclda_linmod.h`` for an example.


**Installing the documentation dependencies** 

Create a virtual Python environment in which to install ``sphinx-build`` and the Sphinx extensions used by AOCL-DA.
E.g.,

.. code-block::

   cd ~/DA-projects/aocl-da
   python3 -m venv ~/DA-projects/doc-3
   source ~/DA-projects/doc-3/bin/activate
   pip install -r doc/requirements.txt

The presence of the requirements is checked in the build system.

**How to build documentation via CMake**

Build the doc target: it will build both the html and the pdf documentation

.. code-block::

   cmake -DBUILD_DOC=ON
   cmake --build . --target doc

targets doc_html and doc_pdf are also available to build only one of the two.

.. code-block::

   cmake --build . --target doc_[html|pdf]

To build the release version of the doc (excluding internal documentation), set the variable INTERNAL_DOC at configure time: 

.. code-block::

   cmake -DBUILD_DOC=ON -DINTERNAL_DOC=OFF
   cmake --build . --target doc

Adding internal documentation
=============================

Internal only doc can be added with the ``.. only`` directives. We tag every piece of internal doc as ``internal``

.. code-block::

   .. only:: internal

      Your documentation here


Documenting errors
==================
All public APIs returning da_status should document the error codes as:

.. code-block::

    * @returns @ref da_status
    * - @ref da_status_success Add description here
    * - @ref da_status_internal_errors Add description here
    * - @TODO add others.

Embeding links and equation in Doxygen comments
===============================================

A special doxygen command was made to be able to embed restructured text in Doxygen comments.

.. code-block::

    * @rst
    * write your rst code here such as references: `link to chapter introduction <chapter_gen_intro>`_
    * @rst

Note that the leading asterisk is madatory for this command to work. See ``aoclda-handle.h`` for an example.


Restructured text examples
==========================

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