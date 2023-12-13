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



Documentation Utilities
***********************

This page is for internal use only. The main documentation page can be found :ref:`here <chapter_gen_intro>`.

.. only:: internal

   Creating and Building New Doc
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

   It is also possible to add and pretty-print an entire file using

   .. code-block::

      .. literalinclude:: <rel/abs/file.cpp>
         :language: C++
         :linenos:

   Most extensions are understood: C, Python, etc.

   If the file is too long or ancillary, then in can be displayed in a collapsed form (html only),
   for this add the following

   .. code-block::

        .. collapse:: AOCL-DA Sphinx Configuration.

            .. literalinclude:: ../conf.py
                :language: Python
                :linenos:

   .. collapse:: AOCL-DA Sphinx Configuration.

      .. literalinclude:: ../conf.py
          :language: Python
          :linenos:


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

   Adding Internal Documentation
   =============================

   Internal only doc can be added with the ``.. only`` directives. We tag every piece of internal doc as ``internal``

   .. code-block::

      .. only:: internal

         Your documentation here


   Documenting Errors
   ==================
   All public APIs returning da_status should document the error codes as:

   .. code-block::

       * - @returns @ref da_status
       * - @ref da_status_success Add description here
       * - @ref da_status_internal_errors Add description here
       * - @TODO add others.

   Embedding Links and Equation in Doxygen Comments
   ================================================

   A special ``doxygen`` command was made to be able to embed restructured text in Doxygen comments.

   .. code-block::

       * @rst
       * write your rst code here such as references: `link to chapter introduction <chapter_gen_intro>`_
       * @rst

   Note that the leading asterisk is mandatory for this command to work. See ``aoclda-handle.h`` for an example.


   Restructured Text Examples
   ==========================

   Numbered Lists
   --------------

   1. explicitly numbered list
   2. new element
   #. implicitly numbered
      element of the same list

   New List:

   #. new implicitly numbered list

      #. sublist
      #. with implicit
      #. elements

   #. second element
   #. `link to the doc <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#lists-and-quote-like-blocks>`_

   Tables
   ------

   .. csv-table:: Example
      :header: "String1", "number", "String2"
      :widths: 15, 10, 30

      "Short description", 1.03, "`More online documentation for tables
      <https://pandemic-overview.readthedocs.io/en/latest/myGuides/reStructuredText-Tables-Examples.html#csv-table-example>`_"
      "Another", 10.4, "This is a long description over
      several lines"
      "new line", 1.99, "longer line description"

   Adding a Picture
   ----------------

   .. image:: ../pics/kitten.jpg
      :align: center

   `link to more documentation <https://pandemic-overview.readthedocs.io/en/latest/myGuides/reStructuredText-Images-and-Figures-Examples.html>`_