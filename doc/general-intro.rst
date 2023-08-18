
.. _chapter_gen_intro:

.. _chapter_gen_intro:

General introduction
********************
 

This section describes the common utilities and data structures of the AOCL-DA library:

* :ref:`DA handle <intro_handle>`, the main data structure used accross all chapters of the library.
    - :ref:`API <handle_api>`
* :ref:`Error handling <da_errors>`.
    - :ref:`API <da_errors_api>`
* :ref:`Option setting <da_options>`.
    - :ref:`API <da_options_api>`

.. _intro_handle:

Handle description
==================
TODO describe main handle structure

.. _handle_api:

Handle API Reference
--------------------
.. 
    .. doxygentypedef:: da_handle
    .. doxygenenum:: da_handle_type
    .. doxygenfunction:: da_handle_init_d
    .. doxygenfunction:: da_handle_init_s
    .. doxygenfunction:: da_handle_destroy
    .. doxygenfunction:: da_handle_print_error_message
    .. doxygenfunction:: da_check_handle_type

.. _da_errors:

Error handling in AOCL-DA
=========================
TODO describe errors

.. _da_errors_api:

Error specific API
------------------
.. 
    .. doxygenenum:: da_status_

.. _da_options:

Setting Optional parameters
===========================
TODO describe options

.. _da_options_api:


.. _da_int:

da_int
------

TODO: document da_int; this text is here as a placeholder so we can insert references to da_int throughout the code
Say something about ldx and column major ordering