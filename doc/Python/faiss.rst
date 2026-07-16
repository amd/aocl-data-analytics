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



.. _faiss:

Extension for FAISS
*******************

In addition to the Python API, AOCL-DA offers an experimental extension to enable existing users of FAISS to
extract better performance while making minimal changes to their code.

To use the extension, you must *patch* your existing code to replace the FAISS symbols with AOCL-DA
symbols. This can be done by inserting the following lines prior to your FAISS import statement.

Known limitations
=================

Only the classes listed in the table below are patched. All other FAISS index types are unaffected by
``faiss_patch`` and continue to use native FAISS.

The following limitations apply to patched objects:

* ``add_with_ids`` and ``range_search`` on ``IndexIVFFlat`` are not supported and raise
  ``NotImplementedError``.
* Non-``Flat`` quantizers for ``IndexIVFFlat`` are not supported; a warning is emitted and the
  quantizer object is ignored.

Patched objects are AOCL-DA objects and are not interchangeable with native FAISS objects. 
A patched object cannot be passed as an argument to an unpatched FAISS object, so
AOCL-DA patched indexes must be used as standalone, top-level indexes:

.. code-block:: python

   # After faiss_patch():
   quantizer = faiss.IndexFlatL2(d)           # AOCL-DA object
   hnsw = faiss.IndexHNSWFlat(quantizer, M)   # error: expects a native FAISS object

   # Patched indexes should be used as standalone, top-level indexes:
   ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
   ivf.train(X)
   ivf.add(X)
   D, I = ivf.search(Q, k)                    # works

Note that only a subset of the FAISS functionality is available in this manner, and if, after
patching, you attempt to call class member functions which have not been implemented in AOCL-DA,
then a ``NotImplementedError`` will be thrown. It is recommended that for the full benefit of using
AOCL-DA you use the Python APIs described on the subsequent pages of this manual.

.. code-block::

   from aoclda.faiss import faiss_patch, undo_faiss_patch
   faiss_patch()

You can switch back to standard FAISS using

.. code-block::

   undo_faiss_patch()

Note that ``undo_faiss_patch`` restores the original classes on the ``faiss`` module directly;
you do not need to reimport it. Names imported directly into local scope are not restored,
however:

.. code-block::

   faiss_patch()
   from faiss import IndexFlatL2   # binds the AOCL-DA replacement
   undo_faiss_patch()
   idx = IndexFlatL2(d)            # still AOCL-DA — rebind explicitly if needed:
   IndexFlatL2 = faiss.IndexFlatL2

The ``faiss_patch`` and ``undo_faiss_patch`` functions can also be called with string or list
arguments, specifying which FAISS symbols should be patched, for example:

.. code-block::

   faiss_patch("IndexIVFFlat")
   faiss_patch(["IndexFlatL2", "Kmeans"])

Alternatively, you may wish to use the ``aoclda.faiss`` module from the command line, without making
any changes to your own code:

.. code-block::

   python -m aoclda.faiss your_python_script.py

.. note::

   Objects created after patching are AOCL-DA objects, not native FAISS objects. They can be used
   as standalone top-level indexes but cannot be passed to unpatched FAISS index types that expect
   genuine FAISS objects internally.

The following FAISS classes are currently available in the AOCL-DA extension.

.. list-table:: AOCL-DA Extension for FAISS
   :header-rows: 1

   * - FAISS class
     - Notes
   * - ``faiss.IndexIVFFlat``
     - * ``train``, ``add``, ``search`` and ``reset`` methods and ``nlist``, ``nprobe``, ``d``, ``ntotal``, ``is_trained``, ``metric_type`` and ``cp`` (``niter``, ``seed``) attributes are supported.
       * ``add_with_ids``, ``range_search``, ``reconstruct`` and ``merge_from`` are not supported.
       * the quantizer argument is used only to validate that its dimension matches ``d``; AOCL-DA handles quantization internally and does not use the quantizer object. The ``quantizer`` property always returns ``None``.
       * a warning is emitted if the quantizer is not a ``Flat`` index (``IndexFlatL2`` or ``IndexFlatIP``).
   * - ``faiss.IndexFlatL2``
     - * ``add``, ``search``, ``assign``, ``reconstruct``, ``reconstruct_n``, ``reconstruct_batch``, ``search_and_reconstruct`` and ``reset`` methods and ``d``, ``ntotal`` and ``is_trained`` attributes are supported.
       * ``train`` is a no-op, retained for API compatibility.
   * - ``faiss.index_factory``
     - * the ``"IVF{n},Flat"`` and ``"Flat"`` pattern strings are supported.
       * other pattern strings fall through to native FAISS.
   * - ``faiss.Kmeans``
     - * ``train`` and ``assign`` methods and ``centroids``, ``d``, ``k``, ``cp`` (``niter``, ``nredo``, ``seed``, ``spherical``) attributes are supported.
       * ``spherical=True`` is supported and maps to cosine distance clustering.
       * ``nredo`` maps to running *k*-means multiple times and keeping the best result (``n_init`` in AOCL-DA).
       * ``verbose`` is accepted but has no effect; a warning is emitted.

Model persistence is supported via ``pickle``. The native ``faiss.write_index`` and
``faiss.read_index`` functions are not supported by the AOCL-DA backend.
